use tokio::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::time::sleep;
use crate::mumble_connector::MumbleEvent;
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
        format!("[MAVIS_SYSTEM]: CONNECTED TO SERVER AT {} \n", self.time)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct ServerDisconnectedRecord {
    time: DateTime<Utc>,
}

impl ServerDisconnectedRecord {
    fn to_llm_text(&self) -> String {
        format!("[MAVIS_SYSTEM]: DISCONNECTED FROM SERVER AT {} \n", self.time)
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
struct EnteredChannelRecord {
    time: DateTime<Utc>,
    channel: String
}

impl EnteredChannelRecord {
    fn to_llm_text(&self) -> String {
        format!("[MAVIS_SYSTEM]: Entered channel {} \n", self.channel)
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
        println!("{}", llm_text.replace("\n", " "));
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
