#[cfg(any(feature = "default-http", feature = "wasm-http"))]
mod chat;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
mod raw;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
mod anthropic;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
mod batch;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
mod file;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
mod response;
