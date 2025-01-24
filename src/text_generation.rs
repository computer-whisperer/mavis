use std::fmt::{Debug, Display, Formatter};
use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_core::DType;
use candle_core::quantized::{gguf_file};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{LlamaEosToks};
use crate::llama::{Cache, Llama, LlamaConfig, LoraConfig};
use tokenizers::tokenizer::Tokenizer;
use candle_nn::{ops, AdamW, Optimizer, VarBuilder};
use crate::hybrid_var_map::{HybridVarMap, VarOrTensor};

pub struct TextGeneration {
    model: Llama,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    context_tokens: Vec<u32>,
    context_text: String,
    num_context_tokens_processed: usize,
    cache: Cache,
    use_flash_attn: bool,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token_id: LlamaEosToks,
    bos_token_id: u32,
    optimizer: AdamW,
    max_inference_context_tokens: usize,
    max_train_context_tokens: usize
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

    pub(crate) fn new(device: Device) -> Result<Self, Box<dyn std::error::Error>> {
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

            let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
            let tokenizer = TokenOutputStream::new(tokenizer);
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

            let model = Llama::new_from_varbuilder(vb, config, dtype)?;

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

            let model = Llama::new_from_gguf(&device, gguf_content, &mut file, DType::F32)?;

            // temp
            let tokenizer_filename = model_path.join("/ceph-fuse/public/neural_models/llms/Meta-Llama-3.1-8B/tokenizer.json");
            let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
            let tokenizer = TokenOutputStream::new(tokenizer);

            (model, tokenizer)
        };

        let r = 256;
        let lora_config = LoraConfig::new(r, (r as f64)*2.0);
        let mut lora_var_map = HybridVarMap::new();
        let vb = VarBuilder::from_backend(Box::new(lora_var_map.clone()), dtype, device.clone());
        model.add_lora(&lora_config, vb)?;

        let cache = model.new_cache(&device, true)?;

        let logits_processor = LogitsProcessor::new(299792458, Some(1.0), None);
        //let bos_token_id = config.bos_token_id.unwrap();
        let bos_token_id = 128000;
        let mut eos_vec = vec![128001];
        eos_vec.extend(tokenizer.get_token("\n"));
        let eos_token_id = LlamaEosToks::Multiple(eos_vec);

        let adamw_params = candle_nn::ParamsAdamW {
            lr: 0.00001,
            ..Default::default()
        };

        let optimizer = AdamW::new(lora_var_map.all_vars(), adamw_params).unwrap();
        Ok(Self {
            model,
            device,
            cache,
            use_flash_attn: false,
            context_tokens: vec![bos_token_id],
            context_text: String::new(),
            num_context_tokens_processed: 0,
            tokenizer,
            logits_processor,
            repeat_penalty: 1.0,
            repeat_last_n: 10,
            bos_token_id,
            eos_token_id,
            optimizer,
            max_inference_context_tokens: 2048,
            max_train_context_tokens: 350
        })
    }

    pub fn train(&mut self, context: &str, learning_rate: f64) -> Result<()> {
        self.tokenizer.clear();
        self.cache.reset();
        self.optimizer.set_learning_rate(learning_rate);
        let mut input_tokens = vec![self.bos_token_id];
        input_tokens.extend(self.tokenizer.tokenizer().encode(context, true).unwrap().get_ids().to_vec());
        if input_tokens.len() > self.max_train_context_tokens {
            input_tokens.truncate(self.max_train_context_tokens)
        }
        let input_tensor = Tensor::new(input_tokens, &self.device)?.unsqueeze(0)?;
        //println!("input tensor shape: {:?}", input_tensor.shape());
        let context_tensor = input_tensor.i((.., ..input_tensor.dim(1)?-1))?;
        //println!("context tensor shape: {:?}", context_tensor.shape());
        let logits = self.model.forward_for_training(&context_tensor, 0, &self.cache)?.squeeze(0)?;
        self.cache.reset();
        //println!("output logits shape: {:?}", logits.shape());
        let log_sm = ops::log_softmax(&logits, 1)?;
        //println!("output log_sm shape: {:?}", log_sm.shape());
        let training_values = input_tensor.squeeze(0)?.i((1..))?;
        //println!("training values shape: {:?}", log_sm.shape());
        let loss = candle_nn::loss::cross_entropy(&log_sm, &training_values)?;
        self.optimizer.backward_step(&loss)?;
        let local_loss = loss.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        println!("Training loss: {}", local_loss);
        self.cache.reset();
        self.num_context_tokens_processed = 0;
        Ok(())
    }

    pub fn add_context(&mut self, context: &str) -> Result<()> {
        use std::io::Write;
        use anyhow::Error as E;
        self.tokenizer.clear();
        self.context_text.push_str(context);
        let mut new_tokens = self.tokenizer.tokenizer().encode(context, true).unwrap().get_ids().to_vec();
        self.context_tokens.append(&mut new_tokens);
        Ok(())
    }

    pub fn clear_context(&mut self) {
        self.context_tokens.clear();
        self.context_tokens.push(self.bos_token_id);
        self.context_text.clear();
        self.num_context_tokens_processed = 0;
        self.cache.reset();
    }

    pub fn get_context_len(&self) -> usize {
        self.context_tokens.len()
    }

    pub(crate) fn get_max_train_context_tokens(&self) -> usize {
        self.max_train_context_tokens
    }

    pub(crate) fn get_max_inference_context_tokens(&self) -> usize {
        self.max_inference_context_tokens
    }

    pub fn get_context_string(&self) -> String {
        self.context_text.clone()
    }

    pub fn test_next_generation(&mut self, filter: &str) -> Result<bool> {
        use std::io::Write;
        use anyhow::Error as E;
        self.tokenizer.clear();
        let mut local_context_tokens = self.context_tokens.clone();

        self.cache.reset();
        self.num_context_tokens_processed = 0;

        if self.context_tokens.len() > self.max_inference_context_tokens {
            return Err(TextGenerationError::ContextTooLong.into());
        }

        let mut local_tokens_processed = 0usize;

        let mut generated_text = String::new();

        let sample_len = 100;
        let mut output = String::new();
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let logits = {
                if self.cache.use_kv_cache {
                    let ctxt = &local_context_tokens[local_tokens_processed..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let res = self.model.forward(&input, local_tokens_processed, &self.cache, self.use_flash_attn)?;
                    local_tokens_processed = local_context_tokens.len();
                    res
                }
                else {
                    let ctxt = &local_context_tokens[..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let res = self.model.forward(&input, 0, &self.cache, self.use_flash_attn)?;
                    res
                }
            };
            //println!("output logits shape: {:?}", logits.shape());
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = local_context_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &local_context_tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;

            match self.eos_token_id {
                LlamaEosToks::Single(eos_tok_id) if next_token == eos_tok_id => {
                    //print!("Got EOS token.");
                    std::io::stdout().flush()?;
                    break;
                }
                LlamaEosToks::Multiple(ref eos_ids) if eos_ids.contains(&next_token) => {
                    //print!("Got EOS token.");
                    std::io::stdout().flush()?;
                    break;
                }
                _ => (),
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                generated_text.push_str(t.as_str());

                if generated_text.len() > filter.len() {
                    break;
                }

                //print!("{t}");
                std::io::stdout().flush()?;
            }
            local_context_tokens.push(next_token);
        }

        self.cache.reset();
        self.num_context_tokens_processed = 0;

        let mut is_match = true;

        if generated_text.len() > filter.len() {
            for i in 0..filter.len() {
                if generated_text.chars().nth(i).unwrap() != filter.chars().nth(i).unwrap() {
                    is_match = false;
                    break;
                }
            }
        }
        else {
            is_match = false;
        }

        //println!("Looking for \"{}\", generated \"{}\"", filter, generated_text);

        Ok(is_match)
    }

    pub(crate) fn run(&mut self, prompt: &str) -> Result<String> {
        //println!("{}, {}, {}", self.num_context_tokens_processed, self.context_tokens.len(), self.cache.get_cached_token_count());

        use std::io::Write;
        use anyhow::Error as E;
        self.tokenizer.clear();
        self.context_text.push_str(prompt);
        let mut new_tokens = self.tokenizer.tokenizer().encode(prompt, true).unwrap().get_ids().to_vec();
        self.context_tokens.append(&mut new_tokens);

        if self.context_tokens.len() > self.max_inference_context_tokens {
            return Err(TextGenerationError::ContextTooLong.into());
        }

        if self.num_context_tokens_processed+1 < self.context_tokens.len() {
            // For now we do this, hopefully we can fix later
            self.cache.reset();
            self.num_context_tokens_processed = 0;
        }

        let mut generated_tokens = 0usize;

        let sample_len = 50;
        let mut output = String::new();
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let logits = {
                if self.cache.use_kv_cache {
                    let ctxt = &self.context_tokens[self.num_context_tokens_processed..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let res = self.model.forward(&input, self.num_context_tokens_processed, &self.cache, self.use_flash_attn)?;
                    self.num_context_tokens_processed = self.context_tokens.len();
                    res
                }
                else {
                    let ctxt = &self.context_tokens[..];
                    let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                    let res = self.model.forward(&input, 0, &self.cache, self.use_flash_attn)?;
                    res
                }
            };
            //println!("output logits shape: {:?}", logits.shape());
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = self.context_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &self.context_tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;

            generated_tokens += 1;
            match self.eos_token_id {
                LlamaEosToks::Single(eos_tok_id) if next_token == eos_tok_id => {
                    //print!("Got EOS token.");
                    std::io::stdout().flush()?;
                    break;
                }
                LlamaEosToks::Multiple(ref eos_ids) if eos_ids.contains(&next_token) => {
                    //print!("Got EOS token.");
                    std::io::stdout().flush()?;
                    break;
                }
                _ => (),
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                if t.contains("\n") {
                    // End token
                    if !t.starts_with('\n') {
                        // Must retain first part
                        let rest = t.split('\n').next().unwrap();
                        let mut new_tokens = self.tokenizer.tokenizer().encode(rest, true).unwrap().get_ids().to_vec();
                        self.context_tokens.append(&mut new_tokens);
                        self.context_text.push_str(rest);
                        output.push_str(rest);
                        print!("{rest}");
                        std::io::stdout().flush()?;
                        //println!("Got newline (partial token)");
                        break;
                    }
                    //println!("Got newline");
                    break;
                }
                output.push_str(&t);
                self.context_text.push_str(t.as_str());
                print!("{t}");
                std::io::stdout().flush()?;
            }
            self.context_tokens.push(next_token);
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            output.push_str(&rest);
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(output)
    }
}