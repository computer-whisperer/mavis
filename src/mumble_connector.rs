use std::net::{Ipv6Addr, SocketAddr};
use bytes::Bytes;
use futures::channel::oneshot;
use crate::speech_to_text::SpeechToText;
use mumble_protocol::crypt::ClientCryptState;
use mumble_protocol::voice::{VoicePacket, VoicePacketPayload};
use std::collections::HashMap;
use opus::Channels;
use std::sync::Mutex;
use std::fs;
use std::time::{Duration};
use futures::{SinkExt, StreamExt};
use mumble_protocol::control::{msgs, ClientControlCodec, ControlPacket};
use regex::Regex;
use tokio::net::TcpStream;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::{sleep, sleep_until, Instant};
use tokio_native_tls::{native_tls, TlsConnector};
use tokio_util::codec::Decoder;
use tokio_util::udp::UdpFramed;

pub(crate) async fn connect(
    server_addr: SocketAddr,
    server_host: String,
    user_name: String,
    pass_word: Option<String>,
    accept_invalid_cert: bool,
    crypt_state_sender: mpsc::Sender<ClientCryptState>,
    new_rx_messages_tx: mpsc::Sender<(u32, String)>,
    mut new_tx_messages_rx: mpsc::Receiver<String>,
    user_map: &Mutex<HashMap<u32, String>>,
) {

    let file_contents = fs::read("mumble_certs/mavis.p12").unwrap();

    let identity = native_tls::Identity::from_pkcs12(&file_contents, "").unwrap();

    // Connect to server via TCP
    let stream = TcpStream::connect(&server_addr).await.expect("Failed to connect to server:");
    println!("TCP connected..");

    // Wrap the connection in TLS
    let mut builder = native_tls::TlsConnector::builder();
    builder.danger_accept_invalid_certs(accept_invalid_cert);
    builder.identity(identity);
    let connector: TlsConnector = builder
        .build()
        .expect("Failed to create TLS connector")
        .into();
    let tls_stream = connector
        .connect(&server_host, stream)
        .await
        .expect("Failed to connect TLS: {}");
    println!("TLS connected..");

    // Wrap the TLS stream with Mumble's client-side control-channel codec
    let (mut sink, mut stream) = ClientControlCodec::new().framed(tls_stream).split();

    // Version
    let mut msg = mumble_protocol::control::msgs::Version::new();
    msg.set_version(0x01000000);
    sink.send(msg.into()).await.unwrap();

    // Authenticate
    let mut msg = mumble_protocol::control::msgs::Authenticate::new();
    msg.set_username(user_name);
    if let Some(pass_word) = pass_word {
        msg.set_password(pass_word);
    }
    msg.set_opus(true);
    sink.send(msg.into()).await.unwrap();

    println!("Logging in..");
    let mut crypt_state = None;

    let mut current_channel_id = 0;
    let mut current_session_id = 0;

    // Handle incoming packets
    loop {
        tokio::select! {
            Some(packet) = stream.next() => {
                match packet.unwrap() {
                    ControlPacket::TextMessage(mut msg) => {
                        let text = msg.take_message();
                        if text.len() > 512 {
                            // Long message is probably an image
                            println!("Long message received ({} chars), skipping", text.len());
                        }
                        else {
                            new_rx_messages_tx.send((msg.get_actor(), text)).await.unwrap();
                        }
                    }
                    ControlPacket::CryptSetup(msg) => {
                        // Wait until we're fully connected before initiating UDP voice
                        println!("CryptSetup received");
                        crypt_state = Some(ClientCryptState::new_from(
                            msg.get_key()
                                .try_into()
                                .expect("Server sent private key with incorrect size"),
                            msg.get_client_nonce()
                                .try_into()
                                .expect("Server sent client_nonce with incorrect size"),
                            msg.get_server_nonce()
                                .try_into()
                                .expect("Server sent server_nonce with incorrect size"),
                        ));

                        crypt_state_sender.send(
                            crypt_state
                                .take()
                                .expect("Server didn't send us any CryptSetup packet!")).await.unwrap();
                    }
                    ControlPacket::ServerSync(sync_data) => {
                        current_session_id = sync_data.get_session();
                        println!("Logged in!");
                    }
                    ControlPacket::ChannelState(state) => {
                        println!("ChannelState received");
                    }
                    ControlPacket::UserState(state) => {
                        println!("UserState received");
                        if state.has_name() {
                            let user_name = state.get_name().to_string();
                            let user_id = state.get_session();
                            user_map.lock().unwrap().insert(user_id, user_name);
                        }
                    }
                    ControlPacket::Reject(msg) => {
                        println!("Login rejected: {:?}", msg);
                    }
                    _ => {},
                }
            }
            _ = sleep(Duration::from_secs(5)) => {
                let mut ping = msgs::Ping::new();
                sink.send(ControlPacket::Ping(Box::new(ping))).await.unwrap();
            }
            Some(text) = new_tx_messages_rx.recv() => {
                let mut message = msgs::TextMessage::new();
                message.set_message(String::from(text));
                message.set_channel_id(vec!(current_channel_id));
                sink.send(ControlPacket::TextMessage (Box::new(message))).await.unwrap();
                sink.flush().await.unwrap();
            }
        }
    }
}

pub(crate) async fn handle_udp(
    server_addr: SocketAddr,
    mut crypt_state_recv: mpsc::Receiver<ClientCryptState>,
    mut speech_to_text: SpeechToText,
    mut new_stt_tx: mpsc::Sender<(u32, String)>,
) {
    let re = Regex::new(r"<[^>]*>").unwrap();

    // Bind UDP socket
    let udp_socket = UdpSocket::bind((Ipv6Addr::from(0u128), 0u16))
        .await
        .expect("Failed to bind UDP socket");

    // Wait for initial CryptState
    let crypt_state = match crypt_state_recv.recv().await {
        Some(crypt_state) => crypt_state,
        // disconnected before we received the CryptSetup packet, oh well
        None => return,
    };
    println!("UDP ready!");

    // Wrap the raw UDP packets in Mumble's crypto and voice codec (CryptState does both)
    let mut udp_framed = UdpFramed::new(udp_socket, crypt_state);

    // Note: A normal application would also send periodic Ping packets, and its own audio
    //       via UDP. We instead trick the server into accepting us by sending it one
    //       dummy voice packet.
    udp_framed.send((
        VoicePacket::Audio {
            _dst: std::marker::PhantomData,
            target: 0,
            session_id: (),
            seq_num: 0,
            payload: VoicePacketPayload::Opus(Bytes::from([0u8; 128].as_ref()), true),
            position_info: None,
        },
        server_addr,
    )).await.unwrap();

    let mut opus_decoders = HashMap::<u32, opus::Decoder>::new();
    let mut pcm_buffers = HashMap::<u32, Vec::<f32>>::new();

    let mut next_ping_time = Instant::now();
    loop {
        tokio::select! {
            Some(new_crypt_state) = crypt_state_recv.recv() => {
                *udp_framed.codec_mut() = new_crypt_state;
            }
            Some(packet) = udp_framed.next() => {
                let (packet, src_addr) = match packet {
                    Ok(packet) => packet,
                    Err(err) => {
                        eprintln!("Got an invalid UDP packet: {}", err);
                        // To be expected, considering this is the internet, just ignore it
                        return
                        continue
                    }
                };
                match packet {
                    VoicePacket::Ping { .. } => {
                        // Note: A normal application would handle these and only use UDP for voice
                        //       once it has received one.
                        continue
                    }
                    VoicePacket::Audio {
                        seq_num,
                        payload,
                        position_info,
                        session_id,
                        ..
                    } => {
                        // Got audio
                        match payload {
                            VoicePacketPayload::Opus(data, end_of_transmission) => {

                                if !opus_decoders.contains_key(&session_id) {
                                    opus_decoders.insert(session_id, opus::Decoder::new(16000, Channels::Mono).unwrap());
                                    pcm_buffers.insert(session_id, Vec::new());
                                }

                                let mut output_buffer_i16 = [0i16; 32000];
                                let decoded = opus_decoders.get_mut(&session_id).unwrap().decode(
                                    data.get(..).unwrap(),
                                    &mut output_buffer_i16,
                                    false
                                ).unwrap();
                                let mut output_buffer_f32 = [0.0f32; 32000];
                                for i in 0..decoded {
                                    output_buffer_f32[i] = (output_buffer_i16[i] as f32 / 32768.0)*2.0 - 1.0;
                                }
                                let pcm_buffer = pcm_buffers.get_mut(&session_id).unwrap();
                                pcm_buffer.extend(&output_buffer_f32[0..decoded]);

                                if pcm_buffer.len() > 16000*10 || end_of_transmission {
                                    let mel = speech_to_text.pcm_to_mel(&pcm_buffer).unwrap();
                                    pcm_buffer.clear();
                                    let res = speech_to_text.run(&mel, None).unwrap();
                                    speech_to_text.reset_kv_cache();
                                    for segment in res {
                                        if segment.dr.no_speech_prob < 0.3 {
                                            // Strip all special tokens (like <thing>)
                                            let text = re.replace_all(segment.dr.text.as_str(), "").into_owned();
                                            new_stt_tx.send((session_id, text)).await.unwrap();
                                       }
                                    }
                                }
                            }
                            _ => {
                                // Unknown format
                            }
                        }
                    }
                }
            }
            _ = sleep_until(next_ping_time) => {
                // Send a Ping packet
                udp_framed.send((VoicePacket::Ping{timestamp: 10}, server_addr)).await.unwrap();
                next_ping_time = Instant::now() + Duration::from_secs(2);
            }
        }
    }
}