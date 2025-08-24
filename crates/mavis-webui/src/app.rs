use crate::widgets::toggle::toggle_ui;
use egui::epaint::{CubicBezierShape, QuadraticBezierShape, RectShape};
use egui::{
    Color32, CursorIcon, Event, EventFilter, Label, Layout, Margin, Rect, Response, RichText,
    Sense, Shape, Stroke, StrokeKind, UiBuilder, Vec2, Widget, WidgetText, vec2,
};
use futures::SinkExt;
use log::{debug, info};
use rand::{random, random_range};
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, max};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use strum::IntoEnumIterator;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::js_sys;
use wasm_bindgen_futures::js_sys::ArrayBuffer;
use web_sys::WebSocket;
use mavis_proto::{
    WebsocketClientServerMessage, WebsocketServerClientMessage, ChatLogMessage
};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AppState {
    dummy: u32
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            dummy: 0
        }
    }
}

pub struct WebUIApp {
    websocket_client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
    app_state: AppState,
    message_log: Vec<ChatLogMessage>,
}

impl WebUIApp {
    /// Called once before the first frame.
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        websocket_server_client_receiver: mpsc::UnboundedReceiver<WebsocketServerClientMessage>,
        websocket_client_server_sender: mpsc::UnboundedSender<WebsocketClientServerMessage>,
    ) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        cc.egui_ctx.set_zoom_factor(1.2);

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        let app_state = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        Self {
            websocket_server_client_receiver,
            websocket_client_server_sender,
            app_state,
            message_log: Vec::new(),
        }
    }
}

impl eframe::App for WebUIApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.app_state);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        loop {
            match self.websocket_server_client_receiver.try_recv() {
                Ok(msg) => {
                    match msg {
                        WebsocketServerClientMessage::NewChatMessages(x) => {
                                self.message_log.extend(x)
                        }
                        _ => {
                            log::debug!("Unhandled message: {:?}", msg);
                        }
                    }
                }
                Err(_) => {
                    break;
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                egui::widgets::global_theme_preference_switch(ui);
                ui.heading("Awesome Project");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for record in &self.message_log {
                    ui.horizontal(|ui| {
                        ui.label(record.username.clone());
                        ui.label(record.message_body.clone());
                    });
                }
            });
        });
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}
