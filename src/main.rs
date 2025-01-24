use std::collections::HashMap;
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
use candle_core::Device;
use candle_core::utils::cuda_is_available;
use tokio::sync::mpsc;
use tokio_util::codec::Decoder;
mod text_generation;
mod speech_to_text;
mod mumble_connector;
mod chatbot;
mod hybrid_var_map;
mod llama;
mod text_to_speech;

use chatbot::ChatBot;
use text_generation::TextGeneration;
use crate::speech_to_text::SpeechToText;
use crate::text_to_speech::TextToSpeech;

async fn stt_task(device: Device,
                  mut opus_receiver: mpsc::Receiver<(u32, Vec<u8>, bool)>,
                  new_stt_tx: mpsc::Sender<(u32, String)>) {
    let mut speech_to_text = SpeechToText::new(device).unwrap();
    speech_to_text.stt_task(opus_receiver, new_stt_tx).await
}

async fn tts_task(device: Device,
                  opus_sender: mpsc::Sender<(Vec<u8>, bool)>,
                  mut new_tts_rx: mpsc::Receiver<String> ) {
    let mut text_to_speech = TextToSpeech::new(device).unwrap();
    text_to_speech.tts_task(opus_sender, new_tts_rx).await
}

#[tokio::main()]
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

    let (crypt_state_sender, crypt_state_receiver) = mpsc::channel::<ClientCryptState>(8);

    let (mut new_rx_messages_tx, mut new_rx_messages_rx) = mpsc::channel::<(u32, String)>(32);
    let (mut new_tx_messages_tx, mut new_tx_messages_rx) = mpsc::channel::<String>(32);
    let (mut new_stt_tx, mut new_stt_rx) = mpsc::channel::<(u32, String)>(32);
    let (mut new_tts_tx, mut new_tts_rx) = mpsc::channel::<String>(32);
    let (mut must_resync_crypt_tx, mut must_resync_crypt_rx) = mpsc::channel::<()>(1);
    let (mut incoming_opus_tx, mut incoming_opus_rx) = mpsc::channel::<(u32, Vec<u8>, bool)>(32);
    let (mut outgoing_opus_tx, mut outgoing_opus_rx) = mpsc::channel::<(Vec<u8>, bool)>(32);

    let user_map = Arc::new(Mutex::<HashMap<u32, String>>::new(HashMap::new()));

    tokio::spawn(stt_task(device_b.clone(), incoming_opus_rx, new_stt_tx));
    tokio::spawn(tts_task(device_b.clone(), outgoing_opus_tx, new_tts_rx));

    let mut text_generation = TextGeneration::new(device_a.clone()).unwrap();


    //text_generation.run("fibonacci sequence: 1 1 2 3").unwrap();
    //text_generation.add_context("This is a long").unwrap();
    //text_generation.run("This is a different").unwrap();
    //return;
/*
    text_generation.clear_context();
    for _ in 0..10 {
        text_generation.train("This is a long, mysterious string that I want to burn hard into the llm's memory. What happens now?", 0.0001).unwrap();
    }
    text_generation.run("This is a long");
    text_generation.clear_context();
    text_generation.run("This is a different");
    text_generation.clear_context();
    return;*/

    let mut chat_bot = ChatBot::new(text_generation);
    chat_bot.load_context();

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
        crypt_state_sender,
        new_rx_messages_tx,
        new_tx_messages_rx,
        must_resync_crypt_rx,
        user_map.clone()
    ));

    tokio::spawn(mumble_connector::handle_udp(
        server_addr,
        crypt_state_receiver,
        incoming_opus_tx,
        outgoing_opus_rx,
        must_resync_crypt_tx,
    ));

    chat_bot.run(new_rx_messages_rx, new_stt_rx, new_tts_tx, user_map.clone(), new_tx_messages_tx).await
}