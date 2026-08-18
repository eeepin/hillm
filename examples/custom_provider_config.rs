//! Example demonstrating the new `custom_provider` configuration approach.
//!
//! This shows how to use `CustomProviderConfig` directly with `ClientBuilder`,
//! which is cleaner than setting individual fields like `base_url` and `api_type`.
//!
//! This example requires HTTP features to be enabled.

// This example requires HTTP transport features
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use hillm::{AuthHeaderFormat, ClientBuilder, CustomProviderConfig, provider::APIType};

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Old approach (still supported for backward compatibility):
    // let client = ClientBuilder::new()
    //     .api_key("sk-...")
    //     .base_url("https://api.example.com/v1")
    //     .api_type(APIType::OpenAIChatCompletions)
    //     .build()?;

    // New approach: use CustomProviderConfig directly
    let custom_config = CustomProviderConfig {
        name: "my-provider".to_string(),
        base_url: "https://api.example.com/v1".to_string(),
        auth_header: AuthHeaderFormat::Bearer,
        models: vec!["gpt-4".to_string(), "gpt-3.5-turbo".to_string()],
        env_vars: [("api_key".to_string(), "MY_PROVIDER_API_KEY".to_string())]
            .into_iter()
            .collect(),
        available_api_types: vec![APIType::OpenAIChatCompletions],
        default_api_type: Some(APIType::OpenAIChatCompletions),
    };

    let _client = ClientBuilder::new()
        .api_key("sk-...")
        .custom_provider(custom_config)
        .build()?;

    println!("Client built successfully with custom provider config!");

    // The custom_provider config takes precedence over base_url/api_type
    // This is useful when loading config from TOML/JSON files

    Ok(())
}

#[cfg(not(any(feature = "default-http", feature = "wasm-http")))]
fn main() {
    println!("This example requires HTTP features (default-http or wasm-http)");
}
