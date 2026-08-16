/// Provider authentications like api-keys, OAuth tokens...
pub mod auth;
/// Client traits for make llm requests with reqwest-backend [`client::DefaultClient`].
pub mod client;
#[cfg(feature = "tower")]
pub mod embedding;
pub mod error;
pub mod guardrail;
pub mod http;
pub mod image;
pub mod observability;
/// Providers like OpenAI, Anthropic and custom providers...
pub mod provider;
pub mod realtime;
pub mod sse;
pub mod tenant;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
#[cfg(feature = "tower")]
/// `tower` middleware layers like retries, rate limiting, observability, and so on...
pub mod tower;
/// Requests/Response Data Transfer Objects.
pub mod types;
pub mod util;
#[cfg(feature = "tower")]
pub mod vectorstore;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub use client::{
    AnthropicMessagesClient, BatchClient, BatchWaitError, BoxFuture, BoxStream, ClientBuilder,
    ClientConfig, ClientConfigBuilder, DefaultClient, FileClient, FileConfig, LlmClient,
    LlmClientRaw, ResponseClient, WaitForBatchConfig,
};
#[cfg(not(any(feature = "default-http", feature = "wasm-http")))]
pub use client::{
    AnthropicMessagesClient, BatchClient, BatchWaitError, BoxFuture, BoxStream, ClientConfig,
    ClientConfigBuilder, FileConfig, LlmClient, LlmClientRaw, WaitForBatchConfig,
};
pub use error::{HiLlmError, HiLlmResult};
pub use http::transport::TransportConfig;
pub use provider::{
    AuthConfig, AuthType, ModelCapabilities, ProviderConfig, StreamFormat, all_providers,
    capabilities,
    cost::{completion_cost, completion_cost_with_cache},
    custom::{
        AuthHeaderFormat, CustomProviderConfig, CustomProviderRegistry, register_custom_provider,
        unregister_custom_provider,
    },
};
pub use realtime::{
    OpenAiRealtimeTranslator, RealtimeEnvelope, RealtimeEvent, RealtimeTranslator, ResponseStatus,
};
pub use tenant::{
    InMemoryKeyResolver, KeyResolver, KeyResolverError, ResolvedKey, TenantContext, TenantId,
};
#[cfg(feature = "tokenizer")]
pub use tokenizer::{count_request_tokens, count_tokens};
pub use types::*;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub fn ensure_crypto_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

#[cfg(not(any(feature = "default-http", feature = "wasm-http")))]
pub fn ensure_crypto_provider() {
    // No-op when no HTTP backend is enabled
}
