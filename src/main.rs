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
use std::sync::Mutex;
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

use chatbot::ChatBot;
use text_generation::TextGeneration;
use crate::speech_to_text::SpeechToText;


#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
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
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    // use that subscriber to process traces emitted after this point
    tracing::subscriber::set_global_default(subscriber).unwrap();


    let (device_a, device_b) = if cuda_is_available() {
        (Device::new_cuda(0).unwrap(),
        Device::new_cuda(1).unwrap())
    }
    else {
        (Device::Cpu,
         Device::Cpu)
    };
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

    //chat_bot.text_generation.clear_context();

    let mut speech_to_text = SpeechToText::new(device_b).unwrap();

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
    let (crypt_state_sender, crypt_state_receiver) = mpsc::channel::<ClientCryptState>(8);

    let (mut new_rx_messages_tx, mut new_rx_messages_rx) = mpsc::channel::<(u32, String)>(32);
    let (mut new_tx_messages_tx, mut new_tx_messages_rx) = mpsc::channel::<String>(32);
    let (mut new_stt_tx, mut new_stt_rx) = mpsc::channel::<(u32, String)>(32);
    let (mut must_resync_crypt_tx, mut must_resync_crypt_rx) = mpsc::channel::<()>(1);

    let user_map = Mutex::<HashMap<u32, String>>::new(HashMap::new());

    // Run it
    join!(
        mumble_connector::connect(
            server_addr,
            server_host,
            user_name,
            pass_word,
            accept_invalid_cert,
            crypt_state_sender,
            new_rx_messages_tx,
            new_tx_messages_rx,
            must_resync_crypt_rx,
            &user_map
        ),
        mumble_connector::handle_udp(
            server_addr, crypt_state_receiver, speech_to_text, new_stt_tx,
            must_resync_crypt_tx,
        ),
        chat_bot.run(new_rx_messages_rx, new_stt_rx, &user_map, new_tx_messages_tx)
    );
}