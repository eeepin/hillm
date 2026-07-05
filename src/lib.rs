pub mod auth;
pub mod client;
pub mod error;
pub mod http;
pub mod image;
pub mod provider;
/// Requests/Response Data Transfer Objects.
pub mod types;

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
    custom::{
        AuthHeaderFormat, CustomProviderConfig, register_custom_provider,
        unregister_custom_provider,
    },
};
pub use types::*;

pub fn ensure_crypto_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
