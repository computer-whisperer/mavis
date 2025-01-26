use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use candle_core::{Device, IndexOp, Tensor, Var};
use candle_core::DType;
use candle_core::quantized::{gguf_file};
use candle_transformers::generation::LogitsProcessor;
use crate::llama::{Llama, LlamaConfig, LoraConfig};
use crate::llama;
use tokenizers::tokenizer::Tokenizer;
use candle_nn::{ops, AdamW, Optimizer, VarBuilder, VarMap};
use crate::hybrid_var_map::{HybridVarMap, VarOrTensor};

#[derive(Clone)]
pub struct TextGenerationContext {
    kv_cache: llama::KVCache,
    tokens: Vec<u32>,
    bos_token_id: u32,
    unprocessed_text: String,
    tokenizer: Arc<Tokenizer>,
    last_logits: Option<Tensor>,
}

impl TextGenerationContext {
    pub fn add_unprocessed_text(&mut self, text: &str) {
        self.unprocessed_text.push_str(text);
        self.last_logits = None;
    }

    pub fn add_tokens(&mut self, tokens: &[u32]) -> candle_core::Result<()> {
        if !self.unprocessed_text.is_empty() {
            self.anneal()?;
        }
        self.tokens.extend_from_slice(tokens);
        self.last_logits = None;
        Ok(())
    }

    pub fn tokenize_and_append_unprocessed_text(&mut self) {
        let tokens = self.tokenizer.encode(self.unprocessed_text.as_str(), false).unwrap().get_ids().to_vec();
        self.unprocessed_text = String::new();
        self.tokens.extend_from_slice(&tokens);
        self.last_logits = None;
    }

    pub fn anneal(&mut self) -> candle_core::Result<()> {
        let text = self.to_string();
        let mut new_tokens = vec![self.bos_token_id];
        new_tokens.extend(self.tokenizer.encode(text, false).unwrap().get_ids());

        // Figure out how many tokens we can keep
        let mut tokens_to_keep = 0;
        let max_tokens_to_keep = self.tokens.len().min(new_tokens.len()).min(self.kv_cache.get_cached_token_count());
        while tokens_to_keep < max_tokens_to_keep {
            if new_tokens[tokens_to_keep] != self.tokens[tokens_to_keep] {
                break;
            }
            tokens_to_keep += 1;
        }

        self.tokens = new_tokens;
        self.unprocessed_text = String::new();
        //self.kv_cache.truncate(tokens_to_keep)?;
        self.kv_cache.reset();
        self.last_logits = None;
        Ok(())
    }

    pub fn get_unprocessed_tokens(&mut self) -> candle_core::Result<&[u32]> {
        if !self.unprocessed_text.is_empty() {
            self.anneal()?;
        }
        Ok(&self.tokens[self.kv_cache.get_cached_token_count()..])
    }

    pub fn get_tokens(&mut self) -> candle_core::Result<&[u32]> {
        if !self.unprocessed_text.is_empty() {
            self.anneal()?;
        }
        Ok(&self.tokens[1..])
    }

    pub fn to_string(&self) -> String {
        let mut ret = self.tokenizer.decode(&self.tokens, true).unwrap();
        ret.push_str(&self.unprocessed_text);
        ret
    }

    pub fn get_approximate_num_tokens_in_context(&mut self) -> usize {
        let unprocessed_token_count = if self.unprocessed_text.is_empty() {
            0
        }  else {
            self.tokenizer.encode(self.unprocessed_text.as_str(), false).unwrap().get_ids().to_vec().len()
        };
        self.tokens.len() + unprocessed_token_count
    }

    pub fn clear_kv_cache(&mut self)  {
        self.kv_cache.reset()
    }
}

pub struct TextGeneration {
    model: Llama,
    device: Device,
    tokenizer: Arc<Tokenizer>,
    use_flash_attn: bool,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token_ids: Vec<u32>,
    bos_token_id: u32,
    optimizer: AdamW,
    max_inference_context_tokens: usize,
    max_train_context_tokens: usize,
    verbose: bool,
    lora_vm: VarMap
}

#[derive(Debug)]
enum TextGenerationError {
    ContextTooLong
}


impl Display for TextGenerationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TextGenerationError::ContextTooLong => {write!(f, "Context Too Long")}
        }
    }
}

impl std::error::Error for TextGenerationError{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}


impl TextGeneration {

    pub(crate) fn new(device: Device, lora_safetensors_path: Option<std::path::PathBuf>) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path =  "/ceph-fuse/public/neural_models/llms/Meta-Llama-3-8B.Q4_K_M.gguf";
        //let model_path =  "/ceph-fuse/public/neural_models/llms/Meta-Llama-3.1-8B";
        // If it's a dir, it's probably a safetensors import
        let model_path = std::path::Path::new(model_path);
        let is_safetensors_import = model_path.is_dir();

        let dtype = if device.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        };


        let (mut model, tokenizer) = if is_safetensors_import {
            let tokenizer_filename = model_path.join("tokenizer.json");
            let model_filenames = [
                model_path.join("model-00001-of-00004.safetensors"),
                model_path.join("model-00002-of-00004.safetensors"),
                model_path.join("model-00003-of-00004.safetensors"),
                model_path.join("model-00004-of-00004.safetensors")
            ];
            let config_filename = model_path.join("config.json");

            let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let config = config.into_config();

            let mut var_map = HybridVarMap::new();
            {
                let mut ws = var_map.data().lock().unwrap();

                let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&model_filenames)? };
                for (name, _) in tensors.tensors() {
                    let tensor = tensors
                        .load(&name, &device)?
                        .to_device(&device)?
                        .to_dtype(dtype)?;
                    ws.insert(name.clone(), VarOrTensor::Tensor(tensor));
                }
            }
            let vb = VarBuilder::from_backend(Box::new(var_map.clone()), dtype, device.clone());

            let model = Llama::new_from_varbuilder(device.clone(), vb, config, dtype)?;

            (model, tokenizer)
        }
        else {
            let mut file = std::fs::File::open(&model_path)?;
            let gguf_content = gguf_file::Content::read(&mut file)
                .map_err(|e| e.with_path(model_path))?;

            /*
            let tokens = gguf_content.metadata.get("tokenizer.ggml.tokens").unwrap().to_vec().unwrap();
            for i in 0..100 {
                println!("token {} : {}", i, tokens[i].to_string().unwrap())
            }*/
            //println!("bos token id: {}", gguf_content.metadata.get("tokenizer.ggml.bos_token_id").unwrap().to_u32().unwrap());
            //println!("rope dimension count: {}", gguf_content.metadata.get("llama.rope.dimension_count").unwrap().to_u32().unwrap());

            let model = Llama::new_from_gguf(device.clone(), gguf_content, &mut file, DType::F32)?;

            // temp
            let tokenizer_filename = model_path.join("/ceph-fuse/public/neural_models/llms/Meta-Llama-3.1-8B/tokenizer.json");
            let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

            (model, tokenizer)
        };

        let r = 256;
        let lora_config = LoraConfig::new(r, (r as f64)*2.0);


        let mut lora_var_map = VarMap::new();

        if let Some(path) = &lora_safetensors_path {
            let mut ws = lora_var_map.data().lock().unwrap();

            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
            for (name, _) in tensors.tensors() {
                let tensor = tensors
                    .load(&name, &device)?
                    .to_device(&device)?
                    .to_dtype(DType::F32)?;
                ws.insert(name.clone(), Var::from_tensor(&tensor)?);
            }
        }

        let lora_vb = VarBuilder::from_backend(Box::new(lora_var_map.clone()), DType::F32, device.clone());
        model.add_lora(&lora_config, lora_vb)?;

        //let bos_token_id = config.bos_token_id.unwrap();
        let bos_token_id = 128000;
        let eos_token_ids = vec![128001, 198];

        let adamw_params = candle_nn::ParamsAdamW {
            lr: 0.00001,
            ..Default::default()
        };

        let optimizer = AdamW::new(lora_var_map.all_vars(), adamw_params).unwrap();
        Ok(Self {
            model,
            device,
            use_flash_attn: false,
            tokenizer: Arc::new(tokenizer),
            repeat_penalty: 1.0,
            repeat_last_n: 10,
            bos_token_id,
            eos_token_ids,
            optimizer,
            max_inference_context_tokens: 2048,
            max_train_context_tokens: 350,
            verbose: false,
            lora_vm: lora_var_map
        })
    }

    pub fn save_lora<P: AsRef<std::path::Path>>(&self, lora_safetensors_path: P) -> candle_core::Result<()> {
        self.lora_vm.save(lora_safetensors_path)
    }

    pub fn new_context(&self) -> candle_core::Result<TextGenerationContext> {
        let kv_cache = self.model.new_kv_cache()?;

        Ok(TextGenerationContext {
            kv_cache,
            tokenizer: self.tokenizer.clone(),
            bos_token_id: self.bos_token_id,
            tokens: vec![self.bos_token_id],
            unprocessed_text: String::new(),
            last_logits: None
        })
    }

    pub fn train_text(&mut self, context: Option<&mut TextGenerationContext>, trained_text: &str, learning_rate: f64) -> anyhow::Result<()> {
        let mut trained_tokens = if let Some(_) = &context {
            Vec::new()
        }
        else {
            vec![self.bos_token_id]
        };
        trained_tokens.extend(self.tokenizer.encode(trained_text, true).unwrap().get_ids().to_vec());
        self.train_tokens(context, &trained_tokens, learning_rate)
    }

    pub fn train_tokens(&mut self, context: Option<&mut TextGenerationContext>, training_tokens: &[u32], learning_rate: f64) -> anyhow::Result<()> {
        let context = if let Some(context) = context {
            self.process_context(context)?;
            Some(context)
        } else {
            None
        };
        self.optimizer.set_learning_rate(learning_rate);

        let input_tensor = Tensor::new(training_tokens, &self.device)?.unsqueeze(0)?;
        let context_tensor = input_tensor.i((.., ..input_tensor.dim(1)?-1))?;
        let cache = if let Some(context) = context {
            Some(&mut context.kv_cache)
        } else {
            None
        };
        let logits = self.model.forward(&context_tensor, cache, false, false, false)?.squeeze(0)?;
        let log_sm = ops::log_softmax(&logits, 1)?;
        let training_values = input_tensor.squeeze(0)?.i((1..))?;
        let loss = candle_nn::loss::cross_entropy(&log_sm, &training_values)?;
        self.optimizer.backward_step(&loss)?;
        let local_loss = loss.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        println!("Training loss: {}", local_loss);
        Ok(())
    }


    pub(crate) fn get_max_train_context_tokens(&self) -> usize {
        self.max_train_context_tokens
    }

    pub(crate) fn get_max_inference_context_tokens(&self) -> usize {
        self.max_inference_context_tokens
    }

    pub fn query_next_generation_logit(&self, context: &mut TextGenerationContext, test: &str) -> anyhow::Result<f32> {
        self.process_context(context)?;

        let filter_tokens = self.tokenizer.encode(test, true).unwrap().get_ids().to_vec();
        let input_tokens = Tensor::new(filter_tokens, &self.device)?;

        let logits = self.model.forward(&input_tokens.unsqueeze(0)?, Some(&mut context.kv_cache), false, false, false)?.squeeze(0)?;
        let last_new_logit = logits.i((logits.dim(0)?-1))?;
        let first_new_logits = logits.i((0..input_tokens.dim(0)?-1))?;

        let (output_logits, desired_tokens) = if let Some(last_logits) = &context.last_logits {
            let last_logits = last_logits.unsqueeze(0)?;
            let logits = Tensor::cat(&[&last_logits, &first_new_logits], 0)?;
            (logits, input_tokens)
        } else {
            let training_values = input_tokens.i((1..))?;
            (first_new_logits, training_values)
        };
        //println!("output tokens: {:?}, desired tokens: {:?}", output_logits.shape(), desired_tokens.shape());

        let log_sm = ops::log_softmax(&output_logits, 1)?;
        let loss = candle_nn::loss::cross_entropy(&log_sm, &desired_tokens)?;
        let loss = loss.to_dtype(DType::F32)?.to_vec0::<f32>()?;

        context.last_logits = Some(last_new_logit);
        Ok(loss)
    }

    pub fn process_context(&self, context: &mut TextGenerationContext) -> candle_core::Result<()> {
        let unprocessed_tokens = context.get_unprocessed_tokens()?;
        if unprocessed_tokens.is_empty() {
            return Ok(());
        }
        let input = Tensor::new(unprocessed_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, Some(&mut context.kv_cache), self.use_flash_attn, true, true)?.squeeze(0)?;
        context.last_logits = Some(logits);
        Ok(())
    }

    pub(crate) fn run(&self, context: &mut TextGenerationContext, logits_processor: &mut LogitsProcessor) -> anyhow::Result<String> {

        use std::io::Write;

        if context.tokens.len() > self.max_inference_context_tokens {
            return Err(TextGenerationError::ContextTooLong.into());
        }

        let mut generated_tokens = 0usize;
        {
            let unprocessed_tokens = context.get_unprocessed_tokens()?;
            println!("{} unprocessed tokens", unprocessed_tokens.len());
        }

        let sample_len = 100;
        let mut output = String::new();
        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let input = Tensor::new(context.get_unprocessed_tokens()?, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, Some(&mut context.kv_cache), self.use_flash_attn, true, true)?.squeeze(0)?.squeeze(0)?;
            context.last_logits = Some(logits.clone());

            let logits = logits.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = context.tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &context.tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;

            generated_tokens += 1;
            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            let token_as_text = self.tokenizer.decode(&[next_token], false).unwrap();
            if token_as_text.contains("\n") {
                // End token
                if !token_as_text.starts_with('\n') {
                    // Must retain first part
                    let rest = token_as_text.split('\n').next().unwrap();
                    context.add_unprocessed_text(rest);
                    output.push_str(rest);
                    if self.verbose {
                        print!("{rest}");
                        std::io::stdout().flush()?;
                    }
                }
                break;
            }
            output.push_str(&token_as_text);
            if self.verbose {
                print!("{token_as_text}");
                std::io::stdout().flush()?;
            }
            context.add_tokens(&[next_token]);
        }
        let dt = start_gen.elapsed();
        if self.verbose {
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }
        Ok(output)
    }
}