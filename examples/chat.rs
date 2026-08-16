//! Basic chat completion example.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example chat
//! ```

use hillm::{
    AssistantMessage, ChatCompletionRequest, ClientBuilder, LlmClient, Message, MessageContent,
    UserMessage,
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
            content: MessageContent::Text("What is the capital of France?".into()),
            name: None,
        })],
        ..Default::default()
    };

    match client.chat(req).await {
        Ok(resp) => {
            for choice in resp.choices {
                if let AssistantMessage {
                    content: Some(MessageContent::Text(text)),
                    ..
                } = choice.message
                {
                    println!("{}", text);
                }
            }
            if let Some(usage) = resp.usage {
                println!(
                    "\n[usage: {} prompt + {} completion = {} total]",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => eprintln!("error: {e}"),
    }
}
