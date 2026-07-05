pub mod auth;
pub mod client;
pub mod error;
pub mod http;
pub mod image;
pub mod provider;
/// Requests/Response Data Transfer Objects.
pub mod types;

pub fn ensure_crypto_provider() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
