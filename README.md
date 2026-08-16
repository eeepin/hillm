# hillm

A unified Rust client abstraction for multiple LLM providers.

`hillm` provides a single, ergonomic API surface (`LLMClient`, `ResponseClient`, `FileClient`, `BatchClient`) that routes requests to **OpenAI**, **Anthropic**, **AWS Bedrock**, and any custom OpenAI-compatible endpoint — with first-class support for streaming, caching, circuit breaking, fallbacks, hedging, rate limiting, budgets, multi-tenant credential resolution, and CEL-expression guardrails.

## Status

Pre-1.0. The core architecture (three explicit API routes: OpenAI Chat, OpenAI Responses, Anthropic Messages) is stable, but public APIs may still change.

## Features

- **Multi-provider** — OpenAI, Anthropic (native Messages + Chat compat adapter), Bedrock (Converse), and data-driven / custom OpenAI-compatible endpoints
- **Three explicit API routes** — `OpenAIChatCompletions`, `OpenAIResponses`, `AnthropicMessages`; provider instances are bound to one route for their lifetime, no silent fallback
- **Streaming** — robust byte-boundary-independent SSE decoder with per-event size bounds, UTF-8 split handling, and protocol-specific stream event types
- **Tower middleware** — cache (exact + semantic), negative cache, singleflight, circuit breaker, fallback / fallback chain, hedging, cooldown, budget, rate limit, router, health, idempotency, guardrails, hooks, metrics, tracing
- **Multi-tenant** — per-tenant credential resolution (in-memory or etcd), per-tenant outbound policies, per-tenant budget ledgers
- **Security defaults** — `OutboundPolicy::DenyPrivate` + `HILLM_OUTBOUND_POLICY` env var; per-client policy validators isolate tenants
- **WASM** — `wasm32-unknown-unknown` supported via the `wasm-http` feature

## Quick start

```rust
use hillm::client::ClientBuilder;
use hillm::types::{ChatCompletionRequest, Message, UserMessage, MessageContent};

#[tokio::main]
async fn main() {
    let client = ClientBuilder::new()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .provider("openai")
        .build()
        .unwrap();

    let req = ChatCompletionRequest {
        model: "gpt-4o-mini".into(),
        messages: vec![Message::User(UserMessage {
            content: MessageContent::Text("Hello!".into()),
            name: None,
        })],
        ..Default::default()
    };

    let resp = client.chat(req).await.unwrap();
    println!("{:?}", resp.choices[0].message);
}
```

## Feature flags

| Flag | Description |
|------|-------------|
| `default-http` (default) | reqwest + tokio + rustls |
| `wasm-http` | reqwest + gloo-timers for `wasm32-unknown-unknown` |
| `tower` | Tower middleware (pulls in `tracing`) |
| `tracing` | `tracing` instrumentation |
| `otel` | OpenTelemetry exporter |
| `tokenizer` | HuggingFace tokenizers |
| `opendal` | OpenDAL-backed cache and vector store |
| `bedrock` | AWS Bedrock provider |
| `etcd` | etcd-backed tenant key resolver |
| `guardrail` | CEL-expression guardrails |
| `full` | Everything except `wasm-http` |

## Development

```bash
cargo test --locked                     # default features
cargo test --locked --features tower    # with tower middleware
cargo clippy --locked --all-targets --features tower -- -D warnings
cargo fmt --all -- --check
cargo check --locked --no-default-features
```

## License

Licensed under the [MIT license](LICENSE-MIT).
