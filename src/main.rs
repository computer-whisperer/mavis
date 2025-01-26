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
use candle_transformers::generation::LogitsProcessor;
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
use crate::mumble_connector::MumbleEvent;
use crate::speech_to_text::SpeechToText;
use crate::text_to_speech::TextToSpeech;

async fn stt_task(device: Device,
                  pcm_receiver: mpsc::Receiver<(u32, Vec<f32>)>,
                  new_stt_tx: mpsc::Sender<(u32, String)>) {
    let mut speech_to_text = SpeechToText::new(device).unwrap();
    speech_to_text.stt_task(pcm_receiver, new_stt_tx).await
}

async fn stt_opus_task(
                  opus_receiver: mpsc::Receiver<(u32, Vec<u8>, bool)>,
                  pcm_sender: mpsc::Sender<(u32, Vec<f32>)>) {
    SpeechToText::stt_opus_task(opus_receiver, pcm_sender).await
}


async fn tts_task(device: Device,
                  opus_sender: mpsc::Sender<(Vec<u8>, bool)>,
                  mut new_tts_rx: mpsc::Receiver<String> ) {
    let mut text_to_speech = TextToSpeech::new(device).unwrap();
    text_to_speech.tts_task(opus_sender, new_tts_rx).await
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

    let (crypt_state_sender, crypt_state_receiver) = mpsc::channel::<ClientCryptState>(8);

    let (mumble_event_sender, mumble_event_receiver) = mpsc::channel::<MumbleEvent>(32);
    let (new_tx_messages_tx, new_tx_messages_rx) = mpsc::channel::<String>(32);
    let (new_stt_tx, new_stt_rx) = mpsc::channel::<(u32, String)>(32);
    let (new_tts_tx, new_tts_rx) = mpsc::channel::<String>(32);
    let (must_resync_crypt_tx, must_resync_crypt_rx) = mpsc::channel::<()>(1);
    let (incoming_opus_tx, incoming_opus_rx) = mpsc::channel::<(u32, Vec<u8>, bool)>(256);
    let (outgoing_opus_tx, outgoing_opus_rx) = mpsc::channel::<(Vec<u8>, bool)>(32);
    let (incoming_pcm_tx, incoming_pcm_rx) = mpsc::channel::<(u32, Vec<f32>)>(64);

    let user_map = Arc::new(Mutex::<HashMap<u32, String>>::new(HashMap::new()));

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

        println!("\n\n\n");
        let mut logits_processor = LogitsProcessor::new(299792458, Some(1.0), None);
        for _ in 0..10 {
            text_generation.train_text(None,"This is a long, mysterious string that I want to burn hard into the llm's memory. What happens now?", 0.0001).unwrap();
        }
        let mut context = text_generation.new_context().unwrap();
        context.add_unprocessed_text("This is a long");
        println!("Generated text: {}", text_generation.run(&mut context, &mut logits_processor).unwrap());

        context = text_generation.new_context().unwrap();
        context.add_unprocessed_text("1 2 3 4");
        println!("Generated text: {}", text_generation.run(&mut context, &mut logits_processor).unwrap());
        return;

    }

    tokio::spawn(stt_opus_task(incoming_opus_rx, incoming_pcm_tx));
    tokio::spawn(stt_task(device_b.clone(), incoming_pcm_rx, new_stt_tx));
    tokio::spawn(tts_task(device_b.clone(), outgoing_opus_tx, new_tts_rx));

    let mut chat_bot = ChatBot::new(device_a);
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
        mumble_event_sender,
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

    chat_bot.run(mumble_event_receiver, new_stt_rx, new_tts_tx, user_map.clone(), new_tx_messages_tx).await
}