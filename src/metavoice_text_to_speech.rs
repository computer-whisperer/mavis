use std::io::Write;
use bytes::Bytes;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::encodec;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::metavoice::{adapters, gpt, tokenizers, transformer};
use candle_transformers::models::quantized_metavoice::transformer as qtransformer;
use opus::{Application, Channels};
use rand::{distributions::Distribution, SeedableRng};
use tokio::sync::mpsc;

enum Transformer {
    Normal(transformer::Model),
    Quantized(qtransformer::Model),
}

pub struct MetavoiceTextToSpeech {
    device: Device,
    fs_tokenizer: tokenizers::BPE,
    first_stage_model: Transformer,
    spk_emb: Tensor,
    second_stage_model: gpt::Model,
    logits_processor: LogitsProcessor,
    encodec_model: encodec::Model,
    second_stage_config: gpt::Config,
}

pub const ENCODEC_NTOKENS: u32 = 1024;


impl MetavoiceTextToSpeech {
    pub fn new(device: Device) -> anyhow::Result<Self> {
        let model_path = "/ceph-fuse/public/neural_models/text_to_speech/candle-metavoice";
        let encodec_path = "/ceph-fuse/public/neural_models/text_to_speech/encodec_24khz";

        let first_stage_meta: serde_json::Value = serde_json::from_reader(&std::fs::File::open(format!("{}/first_stage.meta.json", model_path))?)?;
        let first_stage_tokenizer = match first_stage_meta.as_object() {
            None => anyhow::bail!("not a json object"),
            Some(j) => match j.get("tokenizer") {
                None => anyhow::bail!("no tokenizer key"),
                Some(j) => j,
            },
        };
        let fs_tokenizer = tokenizers::BPE::from_json(first_stage_tokenizer, 512)?;

        let second_stage_weights = format!("{}/second_stage.safetensors", model_path);
        let encodec_weights = format!("{}/model.safetensors", encodec_path);

        let dtype = DType::BF16;

        let first_stage_config = transformer::Config::cfg1b_v0_1();

        let first_stage_weights = format!("{}/first_stage.safetensors", model_path);
        let first_stage_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[first_stage_weights], dtype, &device)? };
        let first_stage_model = transformer::Model::new(&first_stage_config, first_stage_vb)?;
        let first_stage_model = Transformer::Normal(first_stage_model);

        let second_stage_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[second_stage_weights], dtype, &device)? };
        let second_stage_config = gpt::Config::cfg1b_v0_1();
        let second_stage_model = gpt::Model::new(second_stage_config.clone(), second_stage_vb)?;

        let encodec_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[encodec_weights], dtype, &device)? };
        let encodec_config = encodec::Config::default();
        let encodec_model = encodec::Model::new(&encodec_config, encodec_vb)?;

        let spk_emb_file = format!("{}/spk_emb.safetensors", model_path);

        let spk_emb = candle_core::safetensors::load(&spk_emb_file, &device)?;
        let spk_emb = match spk_emb.get("spk_emb") {
            None => anyhow::bail!("missing spk_emb tensor in {spk_emb_file:?}"),
            Some(spk_emb) => spk_emb.to_dtype(dtype)?,
        };
        let mut logits_processor = LogitsProcessor::new(12345, Some(1.0), Some(0.95));

        Ok(Self {
            device: device.clone(),
            fs_tokenizer,
            first_stage_model,
            second_stage_model,
            spk_emb,
            second_stage_config,
            encodec_model,
            logits_processor
        })
    }

    pub fn run(&mut self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let debug_mode = false;

        let prompt_tokens = self.fs_tokenizer.encode(prompt)?;
        let mut tokens = prompt_tokens.clone();

        let guidance_scale = 3.0;

        // First stage generation.
        for index in 0..2000 {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?;
            let input = Tensor::stack(&[&input, &input], 0)?;
            let logits = match &mut self.first_stage_model {
                Transformer::Normal(m) => m.forward(&input, &self.spk_emb, tokens.len() - context_size)?,
                Transformer::Quantized(m) => {
                    m.forward(&input, &self.spk_emb, tokens.len() - context_size)?
                }
            };
            let logits0 = logits.i((0, 0))?;
            let logits1 = logits.i((1, 0))?;
            let logits = ((logits0 * guidance_scale)? + logits1 * (1. - guidance_scale))?;
            let logits = logits.to_dtype(DType::F32)?;
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            print!(".");
            std::io::stdout().flush()?;
            if next_token == 2048 {
                break;
            }
        }
        println!();
        let fie2c = adapters::FlattenedInterleavedEncodec2Codebook::new(ENCODEC_NTOKENS);
        let (text_ids, ids1, ids2) = fie2c.decode(&tokens);
        if debug_mode {println!("text ids len: {}", text_ids.len());}
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345 + 1337);
        // TODO: Use the config rather than hardcoding the offset here.
        let encoded_text: Vec<_> = prompt_tokens.iter().map(|v| v - 1024).collect();
        let mut hierarchies_in1 =
            [encoded_text.as_slice(), ids1.as_slice(), &[ENCODEC_NTOKENS]].concat();
        let mut hierarchies_in2 = [
            vec![ENCODEC_NTOKENS; encoded_text.len()].as_slice(),
            ids2.as_slice(),
            &[ENCODEC_NTOKENS],
        ]
            .concat();
        hierarchies_in1.resize(self.second_stage_config.block_size, ENCODEC_NTOKENS);
        hierarchies_in2.resize(self.second_stage_config.block_size, ENCODEC_NTOKENS);
        let in_x1 = Tensor::new(hierarchies_in1, &self.device)?;
        let in_x2 = Tensor::new(hierarchies_in2, &self.device)?;
        let in_x = Tensor::stack(&[in_x1, in_x2], 0)?.unsqueeze(0)?;
        let logits = self.second_stage_model.forward(&in_x)?;

        if debug_mode{
            println!("sampling from logits...");
        }
        let mut codes = vec![];
        for logits in logits.iter() {
            let logits = logits.squeeze(0)?;
            let (seq_len, _) = logits.dims2()?;
            let mut codes_ = Vec::with_capacity(seq_len);
            for step in 0..seq_len {
                let logits = logits.i(step)?.to_dtype(DType::F32)?;
                let logits = &(&logits / 1.0)?;
                let prs = candle_nn::ops::softmax_last_dim(logits)?.to_vec1::<f32>()?;
                let distr = rand::distributions::WeightedIndex::new(prs.as_slice())?;
                let sample = distr.sample(&mut rng) as u32;
                codes_.push(sample)
            }
            codes.push(codes_)
        }

        let codes = Tensor::new(codes, &self.device)?.unsqueeze(0)?;
        let codes = Tensor::cat(&[in_x, codes], 1)?;
        if debug_mode{println!("codes: {codes}");}
        let tilted_encodec = adapters::TiltedEncodec::new(ENCODEC_NTOKENS);
        let codes = codes.i(0)?.to_vec2::<u32>()?;
        let (text_ids, audio_ids) = tilted_encodec.decode(&codes);
        if debug_mode{println!("text_ids len: {:?}", text_ids.len());}
        let audio_ids = Tensor::new(audio_ids, &self.device)?.unsqueeze(0)?;
        if debug_mode{println!("audio_ids shape: {:?}", audio_ids.shape());}
        let pcm = self.encodec_model.decode(&audio_ids)?;
        if debug_mode{println!("output pcm shape: {:?}", pcm.shape());}
        let pcm = pcm.i(0)?.i(0)?.to_dtype(DType::F32)?;
        let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
        let pcm = pcm.to_vec1::<f32>()?;
        if debug_mode{
            let mut output = std::fs::File::create("/ceph-fuse/public/k8s/mavis/data/debug/test.wav")?;
            candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
        }

        Ok(pcm)
    }

    pub fn reset(&mut self) {
        match self.first_stage_model {
            Transformer::Normal(ref mut m) => m.clear_kv_cache(),
            Transformer::Quantized(ref mut m) => m.clear_kv_cache(),
        }
    }

    pub async fn tts_task(mut self,
                          opus_sender: mpsc::Sender<(Vec<u8>, bool)>,
                          mut new_tts_rx: mpsc::Receiver<String> )
    {
        let mut opus_encoder = opus::Encoder::new(24000, Channels::Mono, Application::Voip).unwrap();

        loop {
            tokio::select! {
                Some(message) = new_tts_rx.recv() => {
                    let mut pcm_out = self.run(&message).unwrap();
                    self.reset();
                    while pcm_out.len() % 480 != 0 {
                        pcm_out.push(0.0);
                    }
                    if pcm_out.len() < 6000 {
                        println!("TTS only gave {} samples!", pcm_out.len());
                    }
                    else {
                        let mut samples_sent = 0;
                        while samples_sent < pcm_out.len() {
                            let new_samples_to_send = 480;
                            let opus_bytes_out = opus_encoder.encode_vec_float(&pcm_out[samples_sent..samples_sent+new_samples_to_send], 1024).unwrap();
                            let opus_buffer_bytes = Bytes::copy_from_slice(&opus_bytes_out);


                            let is_last_packet = samples_sent+new_samples_to_send == pcm_out.len();
                            opus_sender.send((opus_buffer_bytes.to_vec(), is_last_packet)).await.unwrap();

                            samples_sent += new_samples_to_send;
                        }
                        opus_encoder.reset_state().unwrap();
                    }

                }
            }
        }
    }
}