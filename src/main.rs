use std::collections::HashMap;
use axum::{
    Router,
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};

use argparse::{ArgumentParser, StoreOption};
use argparse::Store;
use argparse::StoreTrue;
use futures::join;
use futures::StreamExt;
use futures::SinkExt;
use mumble_protocol_2x::crypt::ClientCryptState;
use std::convert::Into;
use std::convert::TryInto;
use std::net::ToSocketAddrs;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use axum::extract::ws::Message;
use candle_core::Device;
use candle_core::utils::cuda_is_available;
use candle_transformers::generation::LogitsProcessor;
use tokio::sync::{mpsc, watch, Notify};
use tokio::sync::watch::{Receiver, Sender};
use tokio_util::codec::Decoder;
mod text_generation;
mod speech_to_text;
mod mumble_connector;
mod chatbot;
mod hybrid_var_map;
mod llama;
mod metavoice_text_to_speech;
mod parler_text_to_speech;
//mod f5_text_to_speech;

use chatbot::ChatBot;
use mavis_proto::{ChatLogMessage, WebsocketServerClientMessage};
use text_generation::TextGeneration;
use crate::mumble_connector::{MumbleEvent};
use crate::speech_to_text::SpeechToText;
use crate::metavoice_text_to_speech::MetavoiceTextToSpeech;
use crate::parler_text_to_speech::ParlerTextToSpeech;

async fn stt_task(device: Device,
                  opus_receiver: mpsc::Receiver<(u32, Vec<u8>, bool)>,
                  new_stt_tx: mpsc::Sender<(u32, String)>,
                  notify_ready: Arc<Notify>
) {
    let mut speech_to_text = SpeechToText::new(device).unwrap();
    notify_ready.notify_one();
    speech_to_text.stt_task(opus_receiver, new_stt_tx).await
}

async fn tts_task(device: Device,
                  opus_sender: mpsc::Sender<(Vec<u8>, bool, Duration)>,
                  new_tts_rx: mpsc::Receiver<String>,
                  may_talk_after_sender: mpsc::Sender<Instant>
) {
    let mut text_to_speech = ParlerTextToSpeech::new(device).unwrap();
    //text_to_speech.tts_task_streaming(opus_sender, new_tts_rx, may_talk_after_sender).await
    text_to_speech.tts_task_singleshot(opus_sender, new_tts_rx, may_talk_after_sender).await
}

struct AppData {
    full_message_log: Arc<Mutex<Vec<ChatLogMessage>>>,
    new_message_event: Receiver<ChatLogMessage>
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    full_message_log: Arc<Mutex<Vec<ChatLogMessage>>>,
    new_message_sender: Sender<()>
) -> impl IntoResponse {
    ws.on_upgrade(move |socket: WebSocket| handle_socket(socket, full_message_log.clone(), new_message_sender.subscribe()))
}

async fn send_message(socket: &mut WebSocket, message: WebsocketServerClientMessage) {
    let mut data = Vec::<u8>::new();
    ciborium::into_writer(&message, &mut data).unwrap();
    socket.send(Message::Binary(data.into())).await.unwrap();
}

async fn handle_socket(
    mut socket: WebSocket,
    full_message_log: Arc<Mutex<Vec<ChatLogMessage>>>,
    mut new_message_receiver: Receiver<()>
) {
    let initial_messages = {
        full_message_log.lock().unwrap().clone()
    };
    let mut num_messages_sent = initial_messages.len();
    send_message(
        &mut socket,
        WebsocketServerClientMessage::NewChatMessages(initial_messages)
    ).await;

    loop {
        tokio::select! {
            Ok(_) = new_message_receiver.changed() => {
                let new_messages = {
                    let messages = full_message_log.lock().unwrap();
                    messages[num_messages_sent..].to_vec()
                };
                num_messages_sent += new_messages.len();
                send_message(&mut socket, WebsocketServerClientMessage::NewChatMessages(new_messages)).await;
            }
        }
    }
}


#[tokio::main]
async fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    //#[cfg(feature = "cuda")]
    //candle_core::quantized::cuda::set_force_dmmv(args.force_dmmv);

    candle_core::cuda::set_gemm_reduced_precision_f16(true);
    candle_core::cuda::set_gemm_reduced_precision_bf16(true);

    // construct a subscriber that prints formatted traces to stdout
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_target(false)
        .init();


    let (device_a, device_b) = if cuda_is_available() {
        (Device::new_cuda(0).unwrap(),
        Device::new_cuda(1).unwrap())
    }
    else {
        (Device::Cpu,
         Device::Cpu)
    };

    if false {
        // Text generation testing
        let mut text_generation = TextGeneration::new(device_a.clone(), None).unwrap();
        let mut logits_processor = LogitsProcessor::new(299792458, Some(1.0), None);

        let mut context = text_generation.new_context().unwrap();
        context.add_unprocessed_text("2 + 2 = ");
        for x in ["0", "1", "2", "3", "4", "5", "6", "fred"] {
            println!("{}: {}", x, text_generation.query_next_generation_logit(&mut context.clone(), x).unwrap());
        }

        println!("\n\n\n");
        let mut logits_processor = LogitsProcessor::new(299792428, Some(1.0), None);
        let mut context = text_generation.new_context().unwrap();
        context.add_unprocessed_text("Incrementing integers: 1 2");
        let out = text_generation.run(&mut context, &mut logits_processor).unwrap();
        println!("Generated text: {}", out);
        context.add_unprocessed_text("\nFibonacci sequence: 1 1");
        let out = text_generation.run(&mut context, &mut logits_processor).unwrap();
        println!("Generated text: {}", out);
        context.add_unprocessed_text("\nPi: ");
        let mut context_b = context.clone();
        let mut context_c = context.clone();

        let mut logits_processor = LogitsProcessor::new(299792428, Some(1.0), None);
        let out = text_generation.run(&mut context, &mut logits_processor).unwrap();
        println!("Generated text a: {}", out);

        context_b.clear_kv_cache();
        let mut logits_processor = LogitsProcessor::new(299792428, Some(1.0), None);
        let out = text_generation.run(&mut context_b, &mut logits_processor).unwrap();
        println!("Generated text b: {}", out);

        let mut logits_processor = LogitsProcessor::new(299792428, Some(1.0), None);
        let out = text_generation.run(&mut context_c, &mut logits_processor).unwrap();
        println!("Generated text c: {}", out);

        println!("Full context a: {:?}", context.get_tokens());
        println!("Full context b: {:?}", context_b.get_tokens());
        println!("Full context c: {:?}", context_c.get_tokens());

        if false {

        }
        return;
    }

    if false {
        let mut chat_bot = ChatBot::new(device_a);
        chat_bot.load_context();

        println!("\n\n\n");
        let mut context = chat_bot.text_generation.new_context().unwrap();



        let new_text = "[SERVER]: CONNECTED TO SERVER AT 2025-01-26 03:19:06.489246485 UTC
[SERVER]: Entered channel Root of the gecko tree
[SERVER]: thefiregecko present
[SERVER]: thebluegecko present
[SERVER]: u6bkep present
[SERVER]: computer-whisperer present
[computer-whisperer]: hello mavis
";

        chat_bot.text_generation.process_context(&mut chat_bot.text_generation_context).unwrap();
        chat_bot.text_generation_context.add_unprocessed_text(new_text);
        //println!("{}", chat_bot.text_generation_context.to_string());
        chat_bot.text_generation_context.add_unprocessed_text("[mavis]:");
        //chat_bot.text_generation_context.anneal().unwrap();
        //println!("{}", chat_bot.text_generation_context.to_string());
        println!("Generated text: {}", chat_bot.text_generation.run(&mut chat_bot.text_generation_context, &mut chat_bot.logits_processor).unwrap());

        return;

    }

    let (mumble_event_sender, mumble_event_receiver) = mpsc::channel::<MumbleEvent>(32);
    let (new_tx_messages_sender, new_tx_messages_receiver) = mpsc::channel::<String>(32);

    let (new_stt_sender, mut new_stt_receiver) = mpsc::channel::<(u32, String)>(32);
    let (new_tts_sender, new_tts_receiver) = mpsc::channel::<String>(32);

    let (incoming_opus_sender, incoming_opus_receiver) = mpsc::channel::<(u32, Vec<u8>, bool)>(256);
    let (outgoing_opus_sender, outgoing_opus_receiver) = mpsc::channel::<(Vec<u8>, bool, Duration)>(256);

    let (may_talk_after_sender, may_talk_after_receiver) = mpsc::channel::<Instant>(8);

    let user_map = Arc::new(Mutex::<HashMap<u32, String>>::new(HashMap::new()));

    let notify_stt_ready = Arc::new(tokio::sync::Notify::new());

    tokio::spawn(stt_task(device_b.clone(), incoming_opus_receiver, new_stt_sender, notify_stt_ready.clone()));
    tokio::spawn(tts_task(device_b.clone(), outgoing_opus_sender, new_tts_receiver, may_talk_after_sender));

    let messages = Arc::new(Mutex::new(Vec::<ChatLogMessage>::new()));
    let (message_broadcast_tx, message_broadcast_rx) = watch::channel(());
    //let mut chat_bot = ChatBot::new(device_a);
    //chat_bot.load_context();

    notify_stt_ready.notified().await;
    // Handle command line arguments
    let mut server_host = "".to_string();
    let mut server_port = 64738u16;
    let mut user_name = "EchoBot".to_string();
    let mut pass_word = None;
    let mut accept_invalid_cert = false;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Run the echo client example");
        ap.refer(&mut server_host)
            .add_option(&["--host"], Store, "Hostname of mumble server")
            .required();
        ap.refer(&mut server_port)
            .add_option(&["--port"], Store, "Port of mumble server");
        ap.refer(&mut user_name)
            .add_option(&["--username"], Store, "User name used to connect");
        ap.refer(&mut pass_word)
            .add_option(&["--password"], StoreOption, "Password used to connect");
        ap.refer(&mut accept_invalid_cert).add_option(
            &["--accept-invalid-cert"],
            StoreTrue,
            "Accept invalid TLS certificates",
        );
        ap.parse_args_or_exit();
    }
    let server_addr = (server_host.as_ref(), server_port)
        .to_socket_addrs()
        .expect("Failed to parse server address")
        .next()
        .expect("Failed to resolve server address");

    // Oneshot channel for setting UDP CryptState from control task
    // For simplicity we don't deal with re-syncing, real applications would have to.

    tokio::spawn(mumble_connector::connect(
        server_addr,
        server_host,
        user_name,
        pass_word,
        accept_invalid_cert,
        mumble_event_sender,
        new_tx_messages_receiver,
        incoming_opus_sender,
        outgoing_opus_receiver,
        user_map.clone()
    ));

    let messages_b = messages.clone();
    let sender_b = message_broadcast_tx.clone();
    tokio::spawn(async move {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/ws",
                get(move |ws: WebSocketUpgrade| {
                    websocket_handler(ws, messages_b.clone(), sender_b.clone())
                }),
            ) // Add WebSocket endpoint
            .nest_service("/pkg", ServeDir::new("./crates/mavis-webui/pkg/"))
            .nest_service(
                "/assets",
                ServeDir::new("./crates/mavis-webui/assets/"),
            )
            .route_service(
                "/index.html",
                ServeFile::new("./crates/mavis-webui/assets/index.html"),
            )
            .route_service(
                "/",
                ServeFile::new("./crates/mavis-webui/assets/index.html"),
            )
            .layer(TraceLayer::new_for_http());

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        tracing::info!("WebUI server listening on port 3000");
        axum::serve(listener, app).await.unwrap();
    });

    loop {
        if let Some((a, b)) = new_stt_receiver.recv().await {
            let user_map = user_map.lock().unwrap();
            let user = if let Some(x) = user_map.get(&a) {
                x.clone()
            } else {
                format!("({a})")
            };
            let mut messages = messages.lock().unwrap();
            messages.push(
                ChatLogMessage{
                    username: user.clone(),
                    message_body: b.clone(),
                }
            );
            message_broadcast_tx.send(()).unwrap();
            println!("{user}, {b}");
        }
    }
    //chat_bot.run(mumble_event_receiver, new_stt_receiver, new_tts_sender, user_map.clone(), new_tx_messages_sender, may_talk_after_receiver).await
}