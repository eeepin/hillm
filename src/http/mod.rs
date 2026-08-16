#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub(crate) mod eventstream;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub(crate) mod request;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub(crate) mod retry;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub(crate) mod stream;
pub mod transport;
