use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6};
use bytes::Bytes;
use futures::channel::oneshot;
use crate::speech_to_text::SpeechToText;
use mumble_protocol_2x::crypt::ClientCryptState;
use mumble_protocol_2x::voice::{VoicePacket, VoicePacketPayload};
use std::collections::{HashMap, HashSet};
use opus::{Application, Channels};
use std::sync::{Arc, Mutex};
use std::fs;
use std::time::{Duration, Instant};
use futures::{SinkExt, StreamExt};
use mumble_protocol_2x::control::{msgs, ClientControlCodec, ControlPacket};
use regex::Regex;
use socket2::{Domain, SockAddr, Socket, Type};
use tokio::net::TcpStream;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::{sleep, sleep_until};
use tokio_native_tls::{native_tls, TlsConnector};
use tokio_util::codec::Decoder;
use tokio_util::udp::UdpFramed;
use crate::text_to_speech::TextToSpeech;

pub(crate) async fn connect(
    server_addr: SocketAddr,
    server_host: String,
    user_name: String,
    pass_word: Option<String>,
    accept_invalid_cert: bool,
    crypt_state_sender: mpsc::Sender<ClientCryptState>,
    new_rx_messages_tx: mpsc::Sender<(u32, String)>,
    mut new_tx_messages_rx: mpsc::Receiver<String>,
    mut must_resync_crypt_rx: mpsc::Receiver<()>,
    user_map: Arc<Mutex<HashMap<u32, String>>>,
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
    let mut msg = mumble_protocol_2x::control::msgs::Version::new();
    msg.set_version_v1(0x01000000);
    sink.send(msg.into()).await.unwrap();

    // Authenticate
    let mut msg = mumble_protocol_2x::control::msgs::Authenticate::new();
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

    let mut stored_key = None;
    let mut stored_client_nonce = None;
    let mut stored_server_nonce = None;



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
                            new_rx_messages_tx.send((msg.actor(), text)).await.unwrap();
                        }
                    }
                    ControlPacket::CryptSetup(msg) => {
                        // Wait until we're fully connected before initiating UDP voice
                        println!("CryptSetup received: {}, {}, {}", msg.key().len(), msg.client_nonce().len(), msg.server_nonce().len() );
                        if msg.key().len() > 0 {
                            stored_key = Some(msg.key().try_into().expect("Server sent private key with incorrect size"))
                        }
                        if msg.client_nonce().len() > 0 {
                            stored_client_nonce = Some(msg.client_nonce().try_into().expect("Server sent client_nonce with incorrect size"))
                        }
                        if msg.server_nonce().len() > 0 {
                            stored_server_nonce = Some(msg.server_nonce().try_into().expect("Server sent server_nonce with incorrect size"))
                        }
                        crypt_state = Some(ClientCryptState::new_from(
                            stored_key.unwrap(),
                            stored_client_nonce.unwrap(),
                            stored_server_nonce.unwrap(),
                        ));

                        crypt_state_sender.send(
                            crypt_state
                                .take()
                                .expect("Server didn't send us any CryptSetup packet!")).await.unwrap();
                    }
                    ControlPacket::ServerSync(sync_data) => {
                        current_session_id = sync_data.session();
                        println!("Logged in!");
                        let mut inner_user_state = msgs::UserState::new();
                        inner_user_state.set_channel_id(0);
                        inner_user_state.set_session(current_session_id);
                        inner_user_state.set_actor(current_session_id);
                        sink.send(ControlPacket::UserState(Box::new(inner_user_state))).await.unwrap();
                        sink.flush().await.unwrap();
                    }
                    ControlPacket::ChannelState(state) => {
                        println!("ChannelState received");
                    }
                    ControlPacket::UserState(state) => {
                        println!("UserState received");
                        if state.has_name() {
                            let user_name = state.name().to_string();
                            let user_id = state.session();
                            user_map.lock().unwrap().insert(user_id, user_name);
                        }
                    }
                    ControlPacket::Reject(msg) => {
                        println!("Login rejected: {:?}", msg);
                    }
                    ControlPacket::Version(_) => {println!("Received version packet");}
                    ControlPacket::UDPTunnel(_) => {println!("Received UDP Tunnel Packet");}
                    ControlPacket::Authenticate(_) => {println!("Received Authenticate Packet");}
                    ControlPacket::Ping(_) => {
                        //println!("Received Ping Packet");
                    }
                    ControlPacket::ChannelRemove(_) => {println!("Received Channel Remove Packet");}
                    ControlPacket::UserRemove(_) => {println!("Received User Remove Packet");}
                    ControlPacket::BanList(_) => {println!("Received BanList Packet");}
                    ControlPacket::PermissionDenied(_) => {println!("Received Permission Denied Packet");}
                    ControlPacket::ACL(_) => {println!("Received ACL Packet");}
                    ControlPacket::QueryUsers(_) => {println!("Received Query Users Packet");}
                    ControlPacket::ContextActionModify(_) => {println!("Received Context Action Modify Packet");}
                    ControlPacket::ContextAction(_) => {println!("Received Context Action Packet");}
                    ControlPacket::UserList(_) => {println!("Received User List Packet");}
                    ControlPacket::VoiceTarget(_) => {println!("Received Voice Target Packet");}
                    ControlPacket::PermissionQuery(_) => {println!("Received Permission Query Packet");}
                    ControlPacket::CodecVersion(_) => {println!("Received Codec Version Packet");}
                    ControlPacket::UserStats(_) => {println!("Received User Stats Packet");}
                    ControlPacket::RequestBlob(_) => {println!("Received Request Blob Packet");}
                    ControlPacket::ServerConfig(_) => {println!("Received Server Config Packet");}
                    ControlPacket::SuggestConfig(_) => {println!("Received Suggest Config Packet");}
                    ControlPacket::Other(_) => {println!("Received Other Packet");}
                    _ => {}
                }
            }
            _ = sleep(Duration::from_secs(3)) => {
                let ping = msgs::Ping::new();
                sink.send(ControlPacket::Ping(Box::new(ping))).await.unwrap();
            }
            Some(text) = new_tx_messages_rx.recv() => {
                let mut message = msgs::TextMessage::new();
                message.set_message(String::from(text));
                message.set_channel_id(vec!(current_channel_id));
                sink.send(ControlPacket::TextMessage (Box::new(message))).await.unwrap();
                sink.flush().await.unwrap();
            }
            _ = must_resync_crypt_rx.recv() => {
                let crypt_setup = msgs::CryptSetup::new();
                println!("Requesting new crypt state!");
                sink.send(ControlPacket::CryptSetup(Box::new(crypt_setup))).await.unwrap();
                sink.flush().await.unwrap();
                println!("Requested new crypt state!");
            }
        }
    }
}

pub(crate) async fn handle_udp(
    server_addr: SocketAddr,
    mut crypt_state_recv: mpsc::Receiver<ClientCryptState>,
    mut incoming_opus_tx: mpsc::Sender<(u32, Vec<u8>, bool)>,
    mut outgoing_opus_rx: mpsc::Receiver<(Vec<u8>, bool)>,
    must_resync_crypt_tx: mpsc::Sender<()>,
) {


    let inner_socket = Socket::new(Domain::ipv6(), Type::dgram(), None).unwrap();
    let addr = SocketAddrV6::new(Ipv6Addr::from(0u128), 0u16, 0, 0).into();
    inner_socket.bind(&addr).expect("Failed to bind UDP socket");
    inner_socket.set_recv_buffer_size(1000000usize).unwrap();
    inner_socket.set_nonblocking(true).unwrap();
    let udp_socket = UdpSocket::from_std(inner_socket.into_udp_socket()).unwrap();


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

    let mut error_count = 0;
    let mut seq_num = 0;

    let mut last_resync_request_time = Instant::now();

    let mut next_ping_time = Instant::now();
    loop {
        tokio::select! {
            Some(new_crypt_state) = crypt_state_recv.recv() => {
                println!("Adding new crypt state");
                *udp_framed.codec_mut() = new_crypt_state;
                last_resync_request_time = Instant::now();
            }
            Some(packet) = udp_framed.next() => {
                let (packet, src_addr) = match packet {
                    Ok(packet) => packet,
                    Err(err) => {
                        //eprintln!("Got an invalid UDP packet: {:?}", err);
                        // To be expected, considering this is the internet, just ignore it
                        if last_resync_request_time.elapsed() > Duration::from_secs(15) {
                            println!("Requesting crypt resync...");
                            must_resync_crypt_tx.try_send(()).ok();
                            last_resync_request_time = Instant::now();
                        }
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
                                incoming_opus_tx.send((session_id, data.to_vec(), end_of_transmission)).await.unwrap();
                            }
                            _ => {
                                // Unknown format
                            }
                        }
                    }
                }
            }
            Some((data, is_last)) = outgoing_opus_rx.recv() => {
                let data_bytes = Bytes::copy_from_slice(&data);
                udp_framed.send((
                    VoicePacket::Audio {
                        _dst: std::marker::PhantomData,
                        target: 0,
                        session_id: (),
                        seq_num,
                        payload: VoicePacketPayload::Opus(data_bytes, is_last),
                        position_info: None,
                    },
                    server_addr,
                )).await.unwrap();
                seq_num += 1;
            }
            _ = sleep_until(next_ping_time.into()) => {
                // Send a Ping packet
                udp_framed.send((VoicePacket::Ping{timestamp: 10}, server_addr)).await.unwrap();
                next_ping_time = Instant::now() + Duration::from_secs(1);
            }
        }


    }
}