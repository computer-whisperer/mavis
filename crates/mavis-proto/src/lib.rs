
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatLogMessage{
  pub username: String,
  pub message_body: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketServerClientMessage {
  Ping,
  NewChatMessages(Vec<ChatLogMessage>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WebsocketClientServerMessage {
  Pong,
  GetFullChatLog,
}
