use tokio::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::str::FromStr;
use std::time::Duration;
use candle_core::Device;
use candle_transformers::generation::LogitsProcessor;
use chrono::{DateTime, Utc};
use tokio::time::sleep;
use crate::mumble_connector::MumbleEvent;
use crate::text_generation::{TextGeneration, TextGenerationContext};

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct TextMessageRecord {
    time: DateTime<Utc>,
    username: String,
    text: String,
}

impl TextMessageRecord {
    fn to_llm_text(&self) -> String {
        let temp = self.text.replace("\n", " ");
        let stripped_text = temp.trim();
        format!("[{}]: {} \n", self.username, stripped_text)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct STTMessageRecord {
    time: DateTime<Utc>,
    username: String,
    text: String,
}

impl STTMessageRecord {
    fn to_llm_text(&self) -> String {
        format!("[{}]: {} \n", self.username, self.text.trim())
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct GeneratedMessageRecord {
    time: DateTime<Utc>,
    username: String,
    text: String,
}

impl GeneratedMessageRecord {
    fn to_llm_text(&self) -> String {
        format!("[{}]: {} \n", self.username, self.text.trim())
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct UserConnectedRecord {
    time: DateTime<Utc>,
    username: String,
}

impl UserConnectedRecord {
    fn to_llm_text(&self) -> String {
        format!("[SERVER]: {} connected \n", self.username)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct UserPresentRecord {
    time: DateTime<Utc>,
    username: String,
}

impl UserPresentRecord {
    fn to_llm_text(&self) -> String {
        format!("[SERVER]: {} present \n", self.username)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct UserDisconnectedRecord {
    time: DateTime<Utc>,
    username: String,
}

impl UserDisconnectedRecord {
    fn to_llm_text(&self) -> String {
        format!("[SERVER]: {} disconnected \n", self.username)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct ServerConnectedRecord {
    time: DateTime<Utc>,
}

impl ServerConnectedRecord {
    fn to_llm_text(&self) -> String {
        format!("[SERVER]: CONNECTED TO SERVER AT {} \n", self.time)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct ServerDisconnectedRecord {
    time: DateTime<Utc>,
}

impl ServerDisconnectedRecord {
    fn to_llm_text(&self) -> String {
        format!("[SERVER]: DISCONNECTED FROM SERVER AT {} \n", self.time)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct EnteredChannelRecord {
    time: DateTime<Utc>,
    channel: String
}

impl EnteredChannelRecord {
    fn to_llm_text(&self) -> String {
        format!("[SERVER]: Entered channel {} \n", self.channel)
    }
}


#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
enum Record {
    TextMessage(TextMessageRecord),
    STTMessage(STTMessageRecord),
    GeneratedMessage(GeneratedMessageRecord),
    UserConnected(UserConnectedRecord),
    UserPresent(UserPresentRecord),
    UserDisconnected(UserDisconnectedRecord),
    ServerConnected(ServerConnectedRecord),
    ServerDisconnected(ServerDisconnectedRecord),
    EnteredChannel(EnteredChannelRecord)
}

impl Record {
    fn to_llm_text(&self) -> String {
        match self {
            Record::TextMessage(record) => record.to_llm_text(),
            Record::STTMessage(record) => record.to_llm_text(),
            Record::GeneratedMessage(record) => record.to_llm_text(),
            Record::UserConnected(record) => {record.to_llm_text()}
            Record::UserPresent(record) => {record.to_llm_text()}
            Record::UserDisconnected(record) => {record.to_llm_text()}
            Record::ServerConnected(record) => {record.to_llm_text()}
            Record::ServerDisconnected(record) => {record.to_llm_text()}
            Record::EnteredChannel(record) => {record.to_llm_text()}
        }
    }

    fn get_time(&self) -> DateTime<Utc> {
        match self {
            Record::TextMessage(record) => record.time,
            Record::STTMessage(record) => record.time,
            Record::GeneratedMessage(record) => record.time,
            Record::UserConnected(record) => record.time,
            Record::UserPresent(record) => record.time,
            Record::UserDisconnected(record) => record.time,
            Record::ServerConnected(record) => record.time,
            Record::ServerDisconnected(record) => record.time,
            Record::EnteredChannel(record) => record.time,
        }
    }
}

pub(crate) struct ChatBot {
    loaded_context: Vec<Record>,
    text_generation: TextGeneration,
    text_generation_context: TextGenerationContext,
    context_log_file: Option<File>,
    data_directory: std::path::PathBuf,
    username: String,
    do_tts: bool,
    do_prompt: bool,
    last_trained_message_datetime: Option<DateTime<Utc>>,
    training_cycles_since_last_save: usize,
    prompt_threshold: f32,
    logits_processor: LogitsProcessor
}

impl ChatBot {
    pub fn new(device: Device) -> Self {
        let data_directory = std::path::PathBuf::from_str("/ceph-fuse/public/k8s/mavis/data/").unwrap();
        // Enumerate files in the lora directory
        let mut files: Vec<_> = fs::read_dir(data_directory.join("loras")).unwrap().map(|r| r.unwrap()).collect();
        files.sort_by_key(|dir| dir.path());
        let (lora_to_use, loaded_datetime) = if let Some(last_file) = files.last() {
            let last_file_path = last_file.path();
            let last_file_stem = last_file_path.file_stem().unwrap().to_str().unwrap();
            let timestamp_str = last_file_stem.split('_').last().unwrap();
            let timestamp = timestamp_str.parse::<i64>().unwrap();
            let loaded_datetime: DateTime<Utc> = DateTime::from_timestamp(timestamp, 0).unwrap();
            (Some(last_file_path), Some(loaded_datetime))
        } else {
            (None, None)
        };

        let mut text_generation = TextGeneration::new(device, lora_to_use).unwrap();

        let text_generation_context = text_generation.new_context().unwrap();
        let logits_processor = LogitsProcessor::new(299792458, Some(1.0), None);
        Self {
            loaded_context: Vec::new(),
            text_generation,
            text_generation_context,
            context_log_file: None,
            data_directory,
            username: "mavis".to_string(),
            do_tts: false,
            do_prompt: true,
            last_trained_message_datetime: loaded_datetime,
            training_cycles_since_last_save: 0,
            prompt_threshold: 1.0,
            logits_processor
        }
    }

    pub fn load_pre_context(&mut self) -> anyhow::Result<()> {
        let text = fs::read_to_string(self.data_directory.join("pre_context_train.txt"))?;

        for _ in 0..4 {
            let mut start_offset = 0;
            while start_offset > text.len() {
                self.text_generation.train_text(None, &text.as_str()[start_offset..], 0.0001)?;
                start_offset += 100;
            }
        }
        Ok(())
    }

    pub fn load_context(&mut self) {

        if self.last_trained_message_datetime.is_none() {
            self.load_pre_context().unwrap();
        }

        let mut files: Vec<_> = fs::read_dir(self.data_directory.join("context")).unwrap().map(|r| r.unwrap()).collect();
        files.sort_by_key(|dir| dir.path());
        for filename in files {
            let f = File::open(filename.path()).unwrap();
            let deserializer = serde_json::Deserializer::from_reader(f);
            let iterator = deserializer.into_iter::<Record>();
            let mut is_first_record = true;
            let mut last_record = None;
            for item in iterator {
                let new_record = item.unwrap();
                if is_first_record {
                    is_first_record = false;
                    if let Record::ServerConnected(_) = &new_record {
                        // All is well
                    }
                    else {
                        let injected_start_record = Record::ServerConnected(ServerConnectedRecord{
                            time: new_record.get_time()
                        });
                        self.process_new_record(injected_start_record, false, true);
                    }
                }
                last_record = Some(new_record.clone());
                self.process_new_record(new_record, false, true);
            }
            if let Some(last_record) = last_record {
                if let Record::ServerDisconnected(_) = &last_record {
                    // All is well
                }
                else {
                    let injected_end_record = Record::ServerDisconnected(ServerDisconnectedRecord{
                        time: last_record.get_time()
                    });
                    self.process_new_record(injected_end_record, false, true);
                }
            }
        }
    }

    fn export_new_record(&mut self, record: &Record) {
        if self.context_log_file.is_none() {
            let out_path = self.data_directory.join("context").join(format!("context_{}.json", Utc::now().timestamp()));
            self.context_log_file = Some(File::create(out_path).unwrap());
        }
        let out = serde_json::to_string(record).unwrap();
        if let Some(mut file) = self.context_log_file.as_mut() {
            file.write_all(out.as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
        }
    }

    fn roll_text_generation_context(&mut self) {
        let max_training_tokens = 256;
        let max_non_training_tokens = 700;
        let records_advance = 10;
        let save_every_n_cycles = 10;
        if self.text_generation_context.get_approximate_num_tokens_in_context() > max_non_training_tokens + max_training_tokens {
            let last_timestamp = self.loaded_context.last().unwrap().get_time();
            let do_train = if let Some(last_load_message_date) = self.last_trained_message_datetime {
                last_timestamp > last_load_message_date
            } else {
                true
            };

            if false && do_train {
                // Get tokens out of current context and clear existing context
                let mut temp_context = self.text_generation.new_context().unwrap();
                core::mem::swap(&mut temp_context, &mut self.text_generation_context);
                let context_tokens = temp_context.get_tokens().unwrap();

                // Split context into training and non-training parts
                let mut training_context = self.text_generation.new_context().unwrap();
                let split = context_tokens.len()-max_training_tokens;
                let pre_train_tokens = &context_tokens[..split];
                let training_tokens = &context_tokens[split..];
                training_context.add_tokens(pre_train_tokens).unwrap();

                println!("Training lora! {} tokens of context and {} tokens for training", pre_train_tokens.len(), training_tokens.len() );
                self.text_generation.train_tokens(Some(&mut training_context), training_tokens, 0.00001).unwrap();

                self.last_trained_message_datetime = Some(last_timestamp);
                self.training_cycles_since_last_save += 1;
            }
            else {
                self.text_generation_context = self.text_generation.new_context().unwrap();
            }

            if self.training_cycles_since_last_save > save_every_n_cycles {
                self.save_lora_checkpoint().unwrap();
            }

            // Delete oldest 10 records
            self.loaded_context.drain(..records_advance);

            // load the model with the remaining context
            for record in &self.loaded_context {
                self.text_generation_context.add_unprocessed_text(&record.to_llm_text());
            }
        }
    }

    fn save_lora_checkpoint(&mut self) -> candle_core::Result<()> {
        let out_path = self.data_directory.join("loras").join(format!("checkpoint_{}.safetensors", self.last_trained_message_datetime.unwrap().timestamp()));
        self.training_cycles_since_last_save = 0;
        self.text_generation.save_lora(out_path)
    }

    fn process_new_record(&mut self, record: Record, do_export: bool, add_to_llm_context: bool) {
        if do_export {
            self.export_new_record(&record);
        }
        let llm_text = record.to_llm_text();
        println!("{}", llm_text.replace("\n", " "));
        if llm_text.len() < 200 {
            self.loaded_context.push(record);
            if add_to_llm_context {
                self.text_generation_context.add_unprocessed_text(&llm_text);
                self.roll_text_generation_context();
            }
        }
    }

    pub async fn run(
        &mut self,
        mut mumble_event_receiver: mpsc::Receiver<MumbleEvent>,
        mut new_stt_rx: mpsc::Receiver<(u32, String)>,
        mut new_tts_tx: mpsc::Sender<String>,
        user_map: Arc::<Mutex<HashMap<u32, String>>>,
        new_tx_messages_tx: mpsc::Sender<String>
    ) {
        loop {
            let mut do_prompt_model = false;
            let mut force_model_to_talk = false;
            tokio::select! {
                Some(event) = mumble_event_receiver.recv() => {
                    match event {
                        MumbleEvent::UserConnected(session_id) => {
                            let username = user_map.lock().unwrap().get(&session_id).cloned().unwrap_or_else(|| String::from("Unknown"));
                            let new_record = Record::UserConnected(UserConnectedRecord {
                                time: Utc::now(),
                                username,
                            });
                            self.process_new_record(new_record, true, true);
                        },
                        MumbleEvent::UserPresent(session_id) => {
                            let username = user_map.lock().unwrap().get(&session_id).cloned().unwrap_or_else(|| String::from("Unknown"));
                            let new_record = Record::UserPresent(UserPresentRecord {
                                time: Utc::now(),
                                username,
                            });
                            self.process_new_record(new_record, true, true);
                        },
                        MumbleEvent::UserDisconnected(session_id) => {
                            let username = user_map.lock().unwrap().get(&session_id).cloned().unwrap_or_else(|| String::from("Unknown"));
                            let new_record = Record::UserDisconnected(UserDisconnectedRecord {
                                time: Utc::now(),
                                username,
                            });
                            self.process_new_record(new_record, true, true);
                        },
                        MumbleEvent::TextMessage((session_id, text)) => {
                            if text.starts_with("/mavis") {
                                let parts = text.split(" ").collect::<Vec<&str>>();
                                if parts.len() >= 2 {
                                    if parts[1] == "tts" {
                                        if parts.len() == 2 {
                                            self.do_tts = !self.do_tts;
                                        }
                                        else {
                                            if parts[2] == "on" || parts[2] == "yes" {
                                                self.do_tts = true;
                                            }
                                            else {
                                                self.do_tts = false;
                                            }
                                        }
                                    }
                                    if parts[1] == "prompt" {
                                        do_prompt_model = true;
                                        force_model_to_talk = true;
                                    }
                                    if parts[1] == "dump" {
                                        println!("{}", self.text_generation_context.to_string())
                                    }
                                    if parts[1] == "threshold" {
                                        if parts.len() > 2 {
                                            if let Ok(threshold) = parts[2].parse::<f32>() {
                                                self.prompt_threshold = threshold;
                                            }
                                        }
                                    }
                                }
                            }
                            else {
                                let username = user_map.lock().unwrap().get(&session_id).cloned().unwrap_or_else(|| String::from("Unknown"));
                                let new_record = Record::TextMessage(TextMessageRecord{
                                    time: Utc::now(),
                                    username,
                                    text
                                });
                                self.process_new_record(new_record, true, true);
                                do_prompt_model = true;
                                force_model_to_talk = true;
                            }
                        }
                        MumbleEvent::ServerConnected() => {
                                let new_record = Record::ServerConnected(ServerConnectedRecord{
                                    time: Utc::now()
                                });
                                self.process_new_record(new_record, true, true);
                        }
                        MumbleEvent::EnteredChannel(channel_name) => {
                            let new_record = Record::EnteredChannel(EnteredChannelRecord{
                                time: Utc::now(),
                                channel: channel_name
                            });
                            self.process_new_record(new_record, true, true);
                        }}
                }
                Some((session_id, text)) = new_stt_rx.recv() => {
                    let username = user_map.lock().unwrap().get(&session_id).cloned().unwrap_or_else(|| String::from("Unknown"));
                    let sanitized_text = String::from(text.trim());
                    let new_record = Record::STTMessage(STTMessageRecord{
                        time: Utc::now(),
                        username,
                        text: sanitized_text
                    });
                    self.process_new_record(new_record, true, true);

                    do_prompt_model = true;
                    force_model_to_talk = false;
                }
            }
            if do_prompt_model && self.do_prompt {
                for i in 0..4 {
                    if !mumble_event_receiver.is_empty() || !new_stt_rx.is_empty() {
                        break;
                    }

                    let do_prompt = if (i == 0 && force_model_to_talk) {
                        true
                    }
                    else {
                        let score = self.text_generation.query_next_generation_logit(&mut self.text_generation_context.clone(), format!("[{}]:", self.username).as_str()).unwrap();
                        println!("Mavis reply score: {}", score);
                        score < self.prompt_threshold
                    };

                    if do_prompt {
                        self.text_generation_context.add_unprocessed_text(format!("[{}]: ", self.username).as_str());

                        let result = self.text_generation.run(&mut self.text_generation_context, &mut self.logits_processor);
                        if let Ok(mut text_result) = result {
                            // Newline token
                            self.text_generation_context.add_unprocessed_text("\n");

                            if self.do_tts { // Voice
                                new_tts_tx.send(text_result.clone()).await.unwrap();
                                new_tx_messages_tx.send(text_result.clone()).await.unwrap();
                            }
                            else {
                                new_tx_messages_tx.send(text_result.clone()).await.unwrap();
                            }
                            let new_record = Record::GeneratedMessage(GeneratedMessageRecord{
                                time: Utc::now(),
                                username: self.username.clone(),
                                text: text_result.replace("\n", " ")
                            });
                            self.process_new_record(new_record, true, false);
                            sleep(Duration::from_secs(1)).await;
                        }
                        else {
                            println!("Error generating response: {:?}", result);
                        }
                    }
                    else {
                        break;
                    }
                }
            }
        }
    }
}
