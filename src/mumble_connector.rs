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
use rubato::Resampler;
use socket2::{Domain, SockAddr, Socket, Type};
use tokio::net::TcpStream;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::{sleep, sleep_until};
use tokio_native_tls::{native_tls, TlsConnector};
use tokio_util::codec::Decoder;
use tokio_util::udp::UdpFramed;
use crate::metavoice_text_to_speech::MetavoiceTextToSpeech;


pub enum MumbleEvent {
    ServerConnected(),
    UserConnected(u32),
    UserPresent(u32),
    UserDisconnected(u32),
    EnteredChannel(String),
    TextMessage((u32, String))
}

pub(crate) async fn connect(
    server_addr: SocketAddr,
    server_host: String,
    user_name: String,
    pass_word: Option<String>,
    accept_invalid_cert: bool,
    mumble_event_sender: mpsc::Sender<MumbleEvent>,
    mut new_tx_messages_rx: mpsc::Receiver<String>,
    incoming_opus_sender: mpsc::Sender<(u32, Vec<u8>, bool)>,
    outgoing_opus_receiver: mpsc::Receiver<(Vec<u8>, bool, Duration)>,
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
    let major = 1u64;
    let minor = 5u64;
    let patch = 0u64;
    msg.set_version_v1((major<<16 | minor<<8 | patch) as u32);
    msg.set_version_v2(major<<32 | minor<<16 | patch);
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

    let mut current_channel_id = None;
    let mut current_session_id = None;

    let mut stored_key = None;
    let mut stored_client_nonce = None;
    let mut stored_server_nonce = None;

    let mut user_channels = HashMap::<u32, u32>::new();
    let mut current_users_in_channel = HashSet::<u32>::new();
    let mut channel_names = HashMap::<u32, String>::new();

    mumble_event_sender.send(MumbleEvent::ServerConnected()).await.unwrap();

    let (crypt_state_sender, crypt_state_receiver) = mpsc::channel::<ClientCryptState>(8);
    let (must_resync_crypt_tx, mut must_resync_crypt_rx) = mpsc::channel::<()>(1);
    tokio::spawn(handle_udp(
        server_addr,
        crypt_state_receiver,
        incoming_opus_sender,
        outgoing_opus_receiver,
        must_resync_crypt_tx,
    ));

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
                            mumble_event_sender.send(MumbleEvent::TextMessage((msg.actor(), text))).await.unwrap();
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
                        let session_id = sync_data.session();
                        current_session_id = Some(session_id);
                        println!("Logged in!");
                        let mut inner_user_state = msgs::UserState::new();
                        inner_user_state.set_channel_id(0);
                        inner_user_state.set_session(session_id);
                        inner_user_state.set_actor(session_id);
                        sink.send(ControlPacket::UserState(Box::new(inner_user_state))).await.unwrap();
                        sink.flush().await.unwrap();
                    }
                    ControlPacket::ChannelState(state) => {
                        println!("ChannelState received");
                        let channel_id = if state.has_channel_id() {
                            state.channel_id()
                        } else {
                            0
                        };

                        if state.has_name() {
                             let channel_name = state.name().to_string();
                             channel_names.insert(channel_id, channel_name);
                        }
                    }
                    ControlPacket::UserState(state) => {
                        println!("UserState received");
                        //println!("{:?}", state);

                        if state.has_session() {
                            let session_id = state.session();
                            let channel_id = if state.has_channel_id() {
                                state.channel_id()
                            }
                            else {
                                0
                            };

                            if state.has_name() {
                                let user_name = state.name().to_string();
                                let user_id = session_id;
                                user_map.lock().unwrap().insert(user_id, user_name);
                            }
                            if let Some(current_channel_id) = current_channel_id {
                                if let Some(current_session_id) = current_session_id {
                                    if session_id != current_session_id && channel_id == current_channel_id {
                                        if user_channels.contains_key(&session_id) {
                                            if user_channels.get(&session_id).unwrap() != &current_channel_id {
                                                // User moved in from another channel
                                                mumble_event_sender.send(MumbleEvent::UserConnected(session_id)).await.unwrap();
                                                current_users_in_channel.insert(session_id);
                                            }
                                        }
                                        else {
                                            // New user connection
                                            mumble_event_sender.send(MumbleEvent::UserConnected(session_id)).await.unwrap();
                                            current_users_in_channel.insert(session_id);
                                        }
                                    }
                                }
                            }
                            user_channels.insert(session_id, channel_id);
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
                    ControlPacket::UserRemove(msg) => {
                        if msg.has_session() {
                            user_channels.remove(&msg.session());
                        }
                    }
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
                if let Some(current_session_id) = current_session_id {
                    if let Some(new_channel_id) = user_channels.get(&current_session_id) {
                        let new_channel_name = if let Some(name) = channel_names.get(&new_channel_id) {
                            name.clone()
                        }
                        else {
                            "Unknown Channel".to_string()
                        };

                        if let Some(prev_channel_id) = current_channel_id {
                            if prev_channel_id != *new_channel_id {
                                mumble_event_sender.send(MumbleEvent::EnteredChannel(new_channel_name)).await.unwrap();
                            }
                        }
                        else {
                            mumble_event_sender.send(MumbleEvent::EnteredChannel(new_channel_name)).await.unwrap();
                        }
                        current_channel_id = Some(*new_channel_id);
                    }

                    // Update connect/disconnect messages
                    if let Some(current_channel_id) = current_channel_id {
                        for (actor, channel) in &user_channels {
                            if *channel == current_channel_id {
                                if !current_users_in_channel.contains(actor) {
                                    if *actor != current_session_id {
                                        mumble_event_sender.send(MumbleEvent::UserPresent(*actor)).await.unwrap();
                                    }
                                    current_users_in_channel.insert(*actor);
                                }
                            }
                        }
                        for actor in current_users_in_channel.clone() {
                            if !user_channels.contains_key(&actor) || *user_channels.get(&actor).unwrap() != current_channel_id {
                                if actor != current_session_id {
                                    mumble_event_sender.send(MumbleEvent::UserDisconnected(actor)).await.unwrap();
                                }
                                current_users_in_channel.remove(&actor);
                            }
                        }
                    }
                }
            }
            _ = sleep(Duration::from_secs(3)) => {
                let ping = msgs::Ping::new();
                sink.send(ControlPacket::Ping(Box::new(ping))).await.unwrap();
            }
            Some(text) = new_tx_messages_rx.recv() => {
                if let Some(current_channel_id) = current_channel_id {
                    let mut message = msgs::TextMessage::new();
                    message.set_message(String::from(text));
                    message.set_channel_id(vec!(current_channel_id));
                    sink.send(ControlPacket::TextMessage (Box::new(message))).await.unwrap();
                    sink.flush().await.unwrap();
                }
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
    mut outgoing_opus_rx: mpsc::Receiver<(Vec<u8>, bool, Duration)>,
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

    let mut seq_num = 0;

    let mut last_resync_request_time = Instant::now();

    let mut next_ping_time = Instant::now();
    let mut next_sendable_audio_time = Instant::now();
    loop {
        let may_send_audio = next_sendable_audio_time <= Instant::now();

        tokio::select! {
            Some(new_crypt_state) = crypt_state_recv.recv() => {
                println!("Adding new crypt state");
                *udp_framed.codec_mut() = new_crypt_state;
                last_resync_request_time = Instant::now();
            }
            Some(packet) = udp_framed.next() => {
                let (packet, _) = match packet {
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
                                if incoming_opus_tx.capacity() == 0 {
                                    eprintln!("Incoming Opus queue is full! dropping packet");
                                    continue;
                                }
                                else {
                                    incoming_opus_tx.send((session_id, data.to_vec(), end_of_transmission)).await.unwrap();
                                }
                            }
                            _ => {
                                // Unknown format
                            }
                        }
                    }
                }
            }
            Some((data, is_last, duration)) = async {if may_send_audio {outgoing_opus_rx.recv().await} else {None}} => {
                let data_bytes = Bytes::copy_from_slice(&data);
                next_sendable_audio_time = Instant::now() + duration;
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