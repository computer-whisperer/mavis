#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(target_arch = "wasm32")]
mod websockets;
#[cfg(target_arch = "wasm32")]
mod widgets;

#[cfg(target_arch = "wasm32")]
pub use app::WebUIApp;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use tokio::sync::mpsc;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    let (server_client_sender, server_client_receiver) = mpsc::unbounded_channel();
    let (client_server_sender, client_server_receiver) = mpsc::unbounded_channel();

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        websockets::websocket_task(server_client_sender, client_server_receiver).await;
    });

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| {
                    Ok(Box::new(WebUIApp::new(
                        cc,
                        server_client_receiver,
                        client_server_sender,
                    )))
                }),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}
