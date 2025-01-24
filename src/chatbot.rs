use tokio::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::time::sleep;
use crate::text_generation::TextGeneration;

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
enum Record {
    TextMessage(TextMessageRecord),
    STTMessage(STTMessageRecord),
    GeneratedMessage(GeneratedMessageRecord)
}

impl Record {
    fn to_llm_text(&self) -> String {
        match self {
            Record::TextMessage(record) => record.to_llm_text(),
            Record::STTMessage(record) => record.to_llm_text(),
            Record::GeneratedMessage(record) => record.to_llm_text(),
        }
    }
}

pub(crate) struct ChatBot {
    loaded_context: Vec<Record>,
    pub(crate) text_generation: TextGeneration,
    context_log_file: Option<File>,
    data_directory: String,
    username: String,
    do_tts: bool,
    do_prompt: bool,
}

impl ChatBot {
    pub fn new(text_generation: TextGeneration) -> Self {
        Self {
            loaded_context: Vec::new(),
            text_generation,
            context_log_file: None,
            data_directory: "/ceph-fuse/public/k8s/mavis/data/".to_string(),
            username: "mavis".to_string(),
            do_tts: false,
            do_prompt: true
        }
    }

    pub fn load_pre_context(&mut self, path: &str) -> anyhow::Result<()> {
        let text = fs::read_to_string(path)?;

        for _ in 0..4 {
            let mut start_offset = 0;
            while start_offset > text.len() {
                self.text_generation.train(&text.as_str()[start_offset..], 0.0001)?;
                start_offset += 100;
            }
        }


        Ok(())
    }

    pub fn load_context(&mut self) {
        self.load_pre_context(format!("{}pre_context_train.txt", self.data_directory).as_str()).unwrap();

        let mut files: Vec<_> = fs::read_dir(format!("{}context/", self.data_directory)).unwrap().map(|r| r.unwrap()).collect();
        files.sort_by_key(|dir| dir.path());
        for filename in files {
            let f = File::open(filename.path()).unwrap();
            let deserializer = serde_json::Deserializer::from_reader(f);
            let iterator = deserializer.into_iter::<Record>();
            for item in iterator {
                let new_record = item.unwrap();
                println!("{}", new_record.to_llm_text().replace("\n", " "));
                self.process_new_record(new_record, false, true);
            }
        }
    }

    fn export_new_record(&mut self, record: &Record) {
        if self.context_log_file.is_none() {
            self.context_log_file = Some(File::create(format!("{}context/context_{}.json", self.data_directory, Utc::now().timestamp())).unwrap());
        }
        let out = serde_json::to_string(record).unwrap();
        if let Some(mut file) = self.context_log_file.as_mut() {
            file.write_all(out.as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
        }
    }

    fn roll_text_generation_context(&mut self) {
        if self.text_generation.get_context_len() > self.text_generation.get_max_inference_context_tokens()-128 {
            println!("Training lora!");
            let context_string = self.text_generation.get_context_string();
            self.text_generation.train(context_string.as_str(), 0.00002).unwrap();
            self.text_generation.clear_context();

            // Delete oldest 1/4 records
            self.loaded_context.drain(0..10);

            // load the model with the remaining context
            for record in &self.loaded_context {
                self.text_generation.add_context(record.to_llm_text().as_str()).unwrap();
            }
        }
    }

    fn process_new_record(&mut self, record: Record, do_export: bool, add_to_llm_context: bool) {
        if do_export {
            self.export_new_record(&record);
        }
        let llm_text = record.to_llm_text();
        if llm_text.len() < 200 {
            self.loaded_context.push(record);
            if add_to_llm_context {
                self.text_generation.add_context(llm_text.as_str()).unwrap();
                self.roll_text_generation_context();
            }
        }
    }

    pub async fn run(
        &mut self,
        mut new_rx_messages_rx: mpsc::Receiver<(u32, String)>,
        mut new_stt_rx: mpsc::Receiver<(u32, String)>,
        mut new_tts_tx: mpsc::Sender<String>,
        user_map: Arc::<Mutex<HashMap<u32, String>>>,
        new_tx_messages_tx: mpsc::Sender<String>
    ) {
        loop {
            let mut do_prompt_model = false;
            let mut force_model_to_talk = false;
            tokio::select! {
                Some((session_id, text)) = new_rx_messages_rx.recv() => {
                    if text.starts_with("/mavis") {
                        let text = text.to_lowercase();
                        let parts = text.split(" ").collect::<Vec<&str>>();
                        if parts.len() >= 2 {
                            if parts[1] == "tts" {
                                if parts.len() == 2 {
                                    self.do_tts = !self.do_tts;
                                }/
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
                                if parts.len() == 2 {
                                    self.do_prompt = !self.do_prompt;
                                }
                                else {
                                    if parts[2] == "on" || parts[2] == "yes" {
                                        self.do_prompt = true;
                                    }
                                    else {
                                        self.do_prompt = false;
                                    }
                                }
                            }
                            if parts[1] == "dump" {
                                println!("{}", self.text_generation.get_context_string())
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
                        println!("{}", new_record.to_llm_text().replace("\n", " "));
                        self.process_new_record(new_record, true, true);
                        do_prompt_model = true;
                        force_model_to_talk = true;
                    }
                }
                Some((session_id, text)) = new_stt_rx.recv() => {
                    let username = user_map.lock().unwrap().get(&session_id).cloned().unwrap_or_else(|| String::from("Unknown"));
                    let sanitized_text = String::from(text.trim());
                    let new_record = Record::STTMessage(STTMessageRecord{
                        time: Utc::now(),
                        username,
                        text: sanitized_text
                    });
                    println!("{}", new_record.to_llm_text().replace("\n", " "));
                    self.process_new_record(new_record, true, true);

                    do_prompt_model = true;
                    force_model_to_talk = false;
                }
            }
            if do_prompt_model && self.do_prompt {
                for i in 0..4 {
                    if !new_rx_messages_rx.is_empty() || !new_stt_rx.is_empty() {
                        break;
                    }
                    if (i == 0 && force_model_to_talk) || self.text_generation.test_next_generation("[mavis]:").unwrap() {
                        let result = self.text_generation.run(format!("[{}]:", self.username).as_str());
                        if let Ok(mut text_result) = result {
                            self.text_generation.add_context("\n").unwrap();

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
