pub mod auth;
pub mod client;
#[cfg(feature = "tower")]
pub mod embedding;
pub mod error;
pub mod guardrail;
pub mod http;
pub mod image;
pub mod observability;
pub mod provider;
pub mod realtime;
pub mod tenant;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
#[cfg(feature = "tower")]
pub mod tower;
/// Requests/Response Data Transfer Objects.
pub mod types;
pub mod util;
#[cfg(feature = "tower")]
pub mod vectorstore;

pub use client::{
    BatchClient, BatchWaitError, BoxFuture, BoxStream, ClientBuilder, ClientConfig,
    ClientConfigBuilder, DefaultClient, FileClient, FileConfig, LlmClient, LlmClientRaw,
    ResponseClient, WaitForBatchConfig,
};
pub use error::{HiLlmError, HiLlmResult};
pub use http::transport::TransportConfig;
pub use provider::{
    AuthConfig, AuthType, ModelCapabilities, ProviderConfig, StreamFormat, all_providers,
    capabilities,
    cost::{completion_cost, completion_cost_with_cache},
    custom::{
        AuthHeaderFormat, CustomProviderConfig, register_custom_provider,
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

pub fn ensure_crypto_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
