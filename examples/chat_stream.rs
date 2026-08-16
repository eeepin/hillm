//! Streaming chat completion example.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example chat_stream
//! ```

use futures_util::StreamExt;
use hillm::{
    ChatCompletionRequest, ClientBuilder, LlmClient, Message, MessageContent, UserMessage,
};

#[tokio::main]
async fn main() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");

    let client = ClientBuilder::new()
        .api_key(api_key)
        .provider("openai")
        .build()
        .expect("client should build");

    let req = ChatCompletionRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![Message::User(UserMessage {
            content: MessageContent::Text("Write a haiku about Rust.".into()),
            name: None,
        })],
        ..Default::default()
    };

    let mut stream = client.chat_stream(req).await.expect("stream should start");

    print!("Assistant: ");
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                for choice in chunk.choices {
                    if let Some(delta) = choice.delta.content {
                        print!("{}", delta);
                    }
                }
            }
            Err(e) => {
                eprintln!("\nstream error: {e}");
                break;
            }
        }
    }
    println!();
}
