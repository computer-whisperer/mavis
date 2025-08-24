
use std::fs;
use base64::Engine;
use ollama_rs::{
    generation::{
        completion::{request::GenerationRequest, GenerationResponse},
        images::Image,
    },
    Ollama,
};
use reqwest::get;
use tokio::runtime::Runtime;

const IMAGE_URL: &str = "http://frigate.cjbal.com:5000/api/main/latest.jpg?h=1480";
const PROMPT: &str = "You are an AI assistant responsible for managing an automated home. A man named Christian lives here, along with his two cats. Here is a still from a security camera in the main room of the house. You are in the combined living room/office space. The camera is equipped with a night-vision mode, so don't be surprised when it looks weird. Please fill out this json structure with what room he is in, what he is doing, and how confident you are: {room: \"\", confidence: \"\", currently_doing: \"\"}. ";

fn main() {

    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        loop {
            // Download the image and encode it to base64
            let bytes = match download_image(IMAGE_URL).await {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Failed to download image: {}", e);
                    return;
                }
            };

            let base64_image = base64::engine::general_purpose::STANDARD.encode(&bytes);

            // Create an Image struct from the base64 string
            let camera_image = Image::from_base64(&base64_image);

            // Create a GenerationRequest with the model and prompt, adding the image
            let request =
                GenerationRequest::new("llama3.2-vision:90b".to_string(), PROMPT.to_string()).add_image(camera_image);

            // Send the request to the model and get the response
            let response = match send_request(request).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to get response: {}", e);
                    return;
                }
            };

            // Print the response
            println!("{}\n\n", response.response);
            //break;
        }

    });
}

// Function to download the image
async fn download_image(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    let response = get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

// Function to send the request to the model
async fn send_request(
    request: GenerationRequest,
) -> Result<GenerationResponse, Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://ollama.cjbal.com".to_string(), 80);
    let response = ollama.generate(request).await?;
    Ok(response)
}