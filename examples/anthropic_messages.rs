//! Native Anthropic Messages API example.
//!
//! Unlike the chat completion compat adapter, this uses the native
//! Anthropic Messages protocol — including cache usage, native stop
//! reasons, and content block events.
//!
//! Run with:
//! ```bash
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_messages
//! ```

use hillm::{
    client::AnthropicMessagesClient,
    provider::ApiType,
    types::anthropic::{
        AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
        AnthropicResponseContentBlock, AnthropicRole,
    },
};

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

    // Select the native Anthropic Messages API type explicitly.
    let client = hillm::ClientBuilder::new()
        .api_key(api_key)
        .provider("anthropic")
        .api_type(ApiType::AnthropicMessages)
        .build()
        .expect("client should build");

    let req = AnthropicMessagesRequest {
        model: "claude-sonnet-4-20250514".into(),
        max_tokens: 1024,
        messages: vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicContentBlock::Text {
                text: "What is the capital of France? Answer in one sentence.".into(),
                cache_control: None,
            }],
        }],
        ..Default::default()
    };

    match client.create_message(req).await {
        Ok(resp) => {
            println!("Model: {}", resp.model);
            println!("Stop reason: {:?}", resp.stop_reason);
            for block in resp.content {
                if let AnthropicResponseContentBlock::Text { text, .. } = block {
                    println!("\n{text}");
                }
            }
            println!(
                "\n[usage: {} input + {} output = {} total]",
                resp.usage.input_tokens,
                resp.usage.output_tokens,
                resp.usage.input_tokens + resp.usage.output_tokens
            );
            if let Some(cache) = resp.usage.cache_creation_input_tokens {
                println!("  cache creation: {cache}");
            }
            if let Some(cache) = resp.usage.cache_read_input_tokens {
                println!("  cache read: {cache}");
            }
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
