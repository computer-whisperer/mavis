use std::time::{Duration, Instant};
use bytes::Bytes;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use candle_transformers::models::parler_tts;
use opus::{Application, Channels};
use rubato::Resampler;
use tokio::sync::mpsc;
use tokio::time::sleep;

struct StreamingTTSJob {
    logits_processor: LogitsProcessor,
    audio_tokens: Vec<u32>,
    prompt_tokens: Tensor,
    all_audio_tokens: Vec<Vec<u32>>,
    prompt_len: usize,
    encoded: Tensor,
    prompt_hidden_states: Tensor,
    step: usize,
    last_decoded_position: usize
}

pub struct ParlerTextToSpeech {
    model: parler_tts::Model,
    config: parler_tts::Config,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl ParlerTextToSpeech {
    pub fn new(device: Device) -> anyhow::Result<Self> {

        let model_path = std::path::Path::new("/ceph-fuse/public/neural_models/text_to_speech/parler-tts-large-v1");

        let model_files = vec![
            model_path.join("model-00001-of-00002.safetensors"),
            model_path.join("model-00002-of-00002.safetensors")
        ];

        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;

        let dtype = if device.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        };

        let start = std::time::Instant::now();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device)? };
        let config: parler_tts::Config = serde_json::from_reader(std::fs::File::open(model_path.join("config.json"))?)?;
        let mut model = parler_tts::Model::new(&config, vb)?;
        println!("loaded the model in {:?}", start.elapsed());



        // Implementation goes here
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            dtype
        })
    }

    pub fn sanitise_inputs(input: &str) -> Option<String> {
        // Detect links and remove them
        let input = regex::Regex::new(r"https?://[^\s]+").unwrap().replace_all(input, "").to_string();
        // Convert all-uppercase acronyms to spaced-out characters
        let input = regex::Regex::new(r"(?i)([A-Z]{2,})")
           .unwrap()
           .replace_all(&input, " $0 ");
        // Remove all punctuation
        let input = input.replace(r"[^\w\s]", "");
        // Remove leading and trailing whitespace
        let input = input.trim();

        if input.is_empty() {
            None
        }
        else {
            Some(input.to_string())
        }
    }

    pub fn new_streaming_tts(&mut self, prompt: &str) -> anyhow::Result<StreamingTTSJob> {
        let description_tokens = self.tokenizer
            .encode("Laura delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let description_tokens = Tensor::new(description_tokens, &self.device)?.unsqueeze(0)?;
        let prompt_tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let prompt_tokens = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let logits_processor = LogitsProcessor::new(299792458, Some(1.0), None);

        self.model.decoder.clear_kv_cache();
        self.model.text_encoder.clear_kv_cache();
        let encoded = self.model.text_encoder.forward(&description_tokens)?;
        let encoded = match self.model.enc_to_dec_proj.as_ref() {
            None => encoded,
            Some(proj) => encoded.apply(proj)?,
        };
        let prompt_hidden_states = prompt_tokens.apply(&self.model.embed_prompts)?;
        let num_codebooks = self.model.decoder.num_codebooks;
        let mut audio_tokens = vec![self.model.decoder_start_token_id; num_codebooks];
        let mut all_audio_tokens = vec![vec![]; num_codebooks];
        let prompt_len = prompt_hidden_states.dim(1)?;

        Ok(StreamingTTSJob{
            logits_processor,
            audio_tokens,
            prompt_tokens,
            all_audio_tokens,
            prompt_len,
            encoded,
            prompt_hidden_states,
            step: 0,
            last_decoded_position: 0
        })
    }

    pub fn advance_streaming_tts_job(&mut self, job: &mut StreamingTTSJob) -> candle_core::Result<(Vec<Vec<u32>>, bool)> {
        let num_codebooks = self.model.decoder.num_codebooks;
        let input_ids = Tensor::from_slice(
            job.audio_tokens.as_slice(),
            (1, num_codebooks, 1),
            job.prompt_tokens.device(),
        )?;
        let (prompt_hidden_states, pos) = if job.step == 0 {
            (Some(&job.prompt_hidden_states), 0)
        } else {
            (None, job.step + job.prompt_len)
        };
        let causal_mask = if pos == 0 {
            self.model.prepare_causal_mask(job.prompt_len + 1, job.prompt_len + 1, input_ids.device())?
        } else {
            self.model.prepare_causal_mask(1, pos + 1, input_ids.device())?
        };
        let logits = self.model.decoder.forward(
            &input_ids,
            prompt_hidden_states,
            Some(&causal_mask),
            &job.encoded,
            None,
            pos,
        )?;
        for (logit_idx, logit) in logits.iter().enumerate() {
            if logit_idx > job.step {
                break;
            }
            if job.audio_tokens[logit_idx] != self.model.pad_token_id {
                let logit = logit.i((0, logit.dim(1)? - 1))?;
                let token = job.logits_processor.sample(&logit)?;
                job.audio_tokens[logit_idx] = token
            }
        }
        let done = job.audio_tokens.iter().all(|v| v == &self.model.pad_token_id) || job.step > 1024;
        for (cb_idx, &token) in job.audio_tokens.iter().enumerate() {
            if token != self.model.decoder_start_token_id && token != self.model.pad_token_id {
                job.all_audio_tokens[cb_idx].push(token)
            }
        }

        // Decode any finalized code sets
        let min_len = job.all_audio_tokens.iter().map(|v| v.len()).min().unwrap_or(0);

        let mut codes_out = vec![vec![]; num_codebooks];
        for i in job.last_decoded_position .. min_len {
            for j in 0..num_codebooks {
                codes_out[j].push(job.all_audio_tokens[j][i])
            }
        }
        job.last_decoded_position = min_len;

        job.step += 1;

        Ok((codes_out, done))
    }

    pub fn single_shot(&mut self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let processed_prompt = Self::sanitise_inputs(prompt);
        if let Some(prompt) = processed_prompt {
            let description_tokens = self.tokenizer
                .encode("Laura delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.", true)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            let description_tokens = Tensor::new(description_tokens, &self.device)?.unsqueeze(0)?;
            let prompt_tokens = self.tokenizer
                .encode(prompt, true)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            let prompt_tokens = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
            let logits_processor = LogitsProcessor::new(299792458, Some(0.8), None);

            //println!("starting generation...");
            let codes = self.model.generate(&prompt_tokens, &description_tokens, logits_processor, 512)?;
            //println!("generated codes\n{codes}");
            let codes = codes.to_dtype(DType::I64)?;
            //codes.save_safetensors("codes", "out.safetensors")?;
            let codes = codes.unsqueeze(0)?;
            let pcm = self.model
                .audio_encoder
                .decode_codes(&codes.to_device(&self.device)?)?;
            //println!("{pcm}");
            let pcm = pcm.i((0, 0))?;
            let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
            let pcm = pcm.to_vec1::<f32>()?;
            if false {
                let mut output = std::fs::File::create("/ceph-fuse/public/k8s/mavis/data/debug/test1.wav")?;
                candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, self.config.audio_encoder.sampling_rate)?;
            }

            Ok(pcm)
        }
        else {
            Ok(vec![])
        }
    }

    pub async fn tts_task_singleshot(mut self,
                                     opus_sender: mpsc::Sender<(Vec<u8>, bool, Duration)>,
                                     mut new_tts_rx: mpsc::Receiver<String>,
                                     may_talk_after_sender: mpsc::Sender<Instant>)
    {
        let (pcm_sender, pcm_receiver) = mpsc::channel::<(Vec<f32>, bool, u32)>(128);
        let (codes_sender, codes_receiver) = mpsc::channel::<(Vec<Vec<u32>>, bool)>(64);
        tokio::spawn(Self::opus_stream_encoder(pcm_receiver, opus_sender, may_talk_after_sender));

        loop {
            tokio::select! {
                Some(message) = new_tts_rx.recv() => {
                    let processed_prompt = Self::sanitise_inputs(&message);
                    if let Some(prompt) = processed_prompt {
                        let mut pcm_out = self.single_shot(&prompt).unwrap();
                        pcm_sender.send((pcm_out, true, self.config.audio_encoder.sampling_rate)).await.unwrap();
                    }
                }
            }
        }
    }

    pub async fn tts_task_streaming(mut self,
                                    opus_sender: mpsc::Sender<(Vec<u8>, bool, Duration)>,
                                    mut new_tts_rx: mpsc::Receiver<String>,
                                    may_talk_after_sender: mpsc::Sender<Instant>
    )
    {
        let (pcm_sender, pcm_receiver) = mpsc::channel::<(Vec<f32>, bool, u32)>(128);
        let (codes_sender, codes_receiver) = mpsc::channel::<(Vec<Vec<u32>>, bool)>(64);
        tokio::spawn(Self::tts_task_decoding(pcm_sender, codes_receiver, self.model.audio_encoder.clone(), self.device.clone(), self.model.decoder.num_codebooks));
        tokio::spawn(Self::opus_stream_encoder(pcm_receiver, opus_sender, may_talk_after_sender));

        loop {
            tokio::select! {
                Some(message) = new_tts_rx.recv() => {
                    let mut job = self.new_streaming_tts(&message).unwrap();
                    loop {
                        let (new_codes, is_done) = self.advance_streaming_tts_job(&mut job).unwrap();
                        codes_sender.send((new_codes, is_done)).await.unwrap();
                        if is_done {
                            break;
                        }
                    }
                }
                _ = sleep(Duration::from_millis(100)) => {

                }
            }
        }
    }

    pub async fn tts_task_decoding(
        pcm_sender: mpsc::Sender<(Vec<f32>, bool, u32)>,
        mut codes_receiver: mpsc::Receiver<(Vec<Vec<u32>>, bool)>,
        audio_encoder: candle_transformers::models::dac::Model,
        device: Device,
        num_codebooks: usize
    ) {
        let mut codes_buffer: Vec<Vec<u32>> = vec![vec![]; num_codebooks];
        let mut num_decoded = 0;
        loop {
            let is_done = loop {
                let (codes, is_done) = codes_receiver.recv().await.unwrap();
                for i in 0..num_codebooks {
                    codes_buffer[i].extend(&codes[i])
                }
                if is_done {
                    break true;
                }
                if codes_receiver.is_empty() {
                    break false;
                }
            };

            let num_codes_have = codes_buffer[0].len();

            let pcm = if ((num_codes_have - num_decoded) > 10 && num_codes_have > 200) || (is_done && (num_codes_have > num_decoded)) {
                let new_codes_buffer = codes_buffer.clone();
                let codes = Tensor::new(new_codes_buffer, &Device::Cpu).unwrap();
                let codes = codes.to_dtype(DType::I64).unwrap();
                let codes = codes.unsqueeze(0).unwrap();
                let pcm = audio_encoder.decode_codes(&codes.to_device(&device).unwrap()).unwrap();
                let new_pcm_start = num_decoded*512;
                let pcm = pcm.i((0, 0, new_pcm_start..)).unwrap();
                //let pcm = candle_examples::audio::normalize_loudness(&pcm, self.config.audio_encoder.sampling_rate, true)?;
                num_decoded = num_codes_have;

                let num_to_truncate = num_codes_have - 200.min(num_codes_have);

                for x in 0..num_codebooks {
                    codes_buffer[x].drain(..num_to_truncate);
                }
                num_decoded = codes_buffer[0].len();
                pcm.to_vec1::<f32>().unwrap()
            } else {
                vec![]
            };

            if pcm.len() > 0 || is_done {
                pcm_sender.send((pcm, is_done, 44100)).await.unwrap();
            }
        }
    }

    async fn opus_stream_encoder(
        mut pcm_receiver: mpsc::Receiver<(Vec<f32>, bool, u32)>,
        opus_sender: mpsc::Sender<(Vec<u8>, bool, Duration)>,
        may_talk_after_sender: mpsc::Sender<Instant>
    ) {
        let mut resampler = None;
        let mut last_sample_rate = 0;
        let mut tx_end_instant = Instant::now();
        let mut pcm_buffer = vec![];
        let mut resampled_pcm = vec![];
        let mut samples_sent = 0;
        let output_sample_rate = 24000;
        let mut opus_encoder = opus::Encoder::new(output_sample_rate, Channels::Mono, Application::Voip).unwrap();
        loop {
            let (pcm_data, is_last, sample_rate) = pcm_receiver.recv().await.unwrap();
            pcm_buffer.extend_from_slice(&pcm_data);
            if is_last {
                while pcm_buffer.len() % 1024 != 0 {
                    pcm_buffer.push(0.0);
                }
            }
            if sample_rate != last_sample_rate {
                resampler = None;
            }
            last_sample_rate = sample_rate;
            if resampler.is_none().clone() {
                // Initialize the resampler with the new sample rate
                resampler = Some(rubato::FastFixedIn::new(
                    (output_sample_rate as f64)/(sample_rate as f64),
                    10.,
                    rubato::PolynomialDegree::Septic,
                    1024,
                    1,
                ).unwrap());
            };
            let resampler = resampler.as_mut().unwrap();

            // resample the audio, one chunk of 1024 samples at a time.
            // in case the audio input failed to produce an exact multiple of 1024 samples,
            // process the remainder on the next iteration of the loop.
            let full_chunks = pcm_buffer.len() / 1024;
            let remainder = pcm_buffer.len() % 1024;
            for chunk in 0..full_chunks {
                let buffered_pcm = &pcm_buffer[chunk * 1024..(chunk + 1) * 1024];
                let pcm = resampler.process(&[&buffered_pcm], None).unwrap();
                resampled_pcm.extend_from_slice(&pcm[0]);
            }
            if remainder == 0 {
                pcm_buffer.clear();
            } else {
                // efficiently copy the remainder to the beginning of the `buffered_pcm` buffer and
                // truncate it.  That's more efficient then allocating a new vector and copying into it
                pcm_buffer.copy_within(full_chunks * 1024.., 0);
                pcm_buffer.truncate(remainder);
            }

            if is_last {
                while resampled_pcm.len() % 480 != 0 {
                    resampled_pcm.push(0.0);
                }
            }

            while samples_sent+480 <= resampled_pcm.len() {
                if samples_sent == 0 {
                    tx_end_instant = Instant::now();
                }
                let new_samples_to_send = 480;
                let opus_bytes_out = opus_encoder.encode_vec_float(&resampled_pcm[samples_sent..samples_sent+new_samples_to_send], 1024).unwrap();
                let opus_buffer_bytes = Bytes::copy_from_slice(&opus_bytes_out);


                let is_last_packet = is_last && (samples_sent+new_samples_to_send == resampled_pcm.len());
                let this_packet_duration = Duration::from_secs_f32(480.0/(output_sample_rate as f32));
                opus_sender.send((opus_buffer_bytes.to_vec(), is_last_packet, this_packet_duration)).await.unwrap();

                tx_end_instant += this_packet_duration;

                samples_sent += new_samples_to_send;
            }

            if is_last {
                may_talk_after_sender.send(tx_end_instant).await.unwrap();
                resampled_pcm.clear();
                samples_sent = 0;
                opus_encoder.reset_state().unwrap();
            }
        }
    }
}