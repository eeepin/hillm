use crate::types::Modality;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::OnceCell;

// Fetch Providers and models info from models.dev

// Error of provider registry failed
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("failed to fetch data: {0}")]
    FetchError(String),
    #[error("failed to parse data: {0}")]
    ParseError(String),
}

static PROVIDER_REGISTRY: OnceCell<Arc<ProviderRegistry>> = OnceCell::const_new();
const PROVIDER_API_URL: &str = "https://models.dev/api.json";
pub(crate) const TOKENS_PER_MILLION: f64 = 1_000_000.0;

type ProviderRegistry = HashMap<String, ProviderEntry>;

#[derive(Debug, Deserialize)]
pub struct ProviderEntry {
    id: String,
    env: Vec<String>,
    #[serde(default)]
    api: String,
    pub(crate) models: HashMap<String, Model>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Model {
    id: String,
    #[serde(default)]
    attachment: bool,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    temperature: bool,
    #[serde(default)]
    modalities: ModelModality,
    #[serde(default)]
    limit: ModelLimit,
    #[serde(default)]
    pub(crate) cost: Option<ModelPrice>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelModality {
    #[serde(default)]
    input: Vec<Modality>,
    #[serde(default)]
    output: Vec<Modality>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelLimit {
    #[serde(default)]
    context: u64,
    #[serde(default)]
    output: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelPrice {
    pub(crate) input: f64,
    pub(crate) output: f64,
    pub(crate) cache_read: Option<f64>,
    pub(crate) cache_write: Option<f64>,
}

async fn fetch_provider() -> Result<ProviderRegistry, ProviderError> {
    let client = reqwest::Client::new();
    let response = client
        .get(PROVIDER_API_URL)
        .send()
        .await
        .map_err(|e| ProviderError::FetchError(e.to_string()))?;

    let text = response
        .text()
        .await
        .map_err(|e| ProviderError::FetchError(e.to_string()))?;

    parse_provider(&text)
}

fn parse_provider(json: &str) -> Result<ProviderRegistry, ProviderError> {
    let providers: ProviderRegistry =
        serde_json::from_str(json).map_err(|e| ProviderError::ParseError(e.to_string()))?;
    Ok(providers)
}

pub async fn registry() -> Result<Arc<ProviderRegistry>, ProviderError> {
    PROVIDER_REGISTRY
        .get_or_try_init(|| async {
            let registry = fetch_provider().await?;
            Ok(Arc::new(registry))
        })
        .await
        .map(Arc::clone)
}

/// Return the current Unix epoch timestamp in seconds.
pub(crate) fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Streaming wire format of providers response stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StreamFormat {
    #[default]
    Sse,
    AwsEventStream,
}

/// Provider capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProviderCapabilities {
    pub vision: bool,
    pub reasoning: bool,
    pub structured_output: bool,
    pub function_calling: bool,
    pub audio_in: bool,
    pub audio_out: bool,
    pub video_in: bool,
}

impl ProviderCapabilities {
    fn from_entry(provider_entry: ProviderEntry) -> Self {
        // TODO
        ProviderCapabilities::default()
    }
}

/// Default is all false.
static DEFAULT_CAPABILITIES: ProviderCapabilities = ProviderCapabilities {
    vision: false,
    reasoning: false,
    structured_output: false,
    function_calling: false,
    audio_in: false,
    audio_out: false,
    video_in: false,
};

pub fn capabilities(provider_name: &str) -> ProviderCapabilities {
    let Some(reg) = PROVIDER_REGISTRY.get() else {
        return DEFAULT_CAPABILITIES;
    };
    if let Some(provider_entry) = &reg.get(provider_name) {
        ProviderCapabilities::from_entry(provider_entry)
    } else {
        DEFAULT_CAPABILITIES
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub display_name: Option<String>,
    pub base_url: Option<String>,
    pub auth: Option<AuthConfig>,
    pub endpoints: Option<Vec<String>>,
    pub param_mappings: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AuthType {
    Bearer,
    #[serde(alias = "header", alias = "x-api-key")]
    ApiKey,
    None,
    #[serde(other)]
    Unknown,
}

// Auth configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(rename = "type")]
    pub auth_type: AuthType,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub env_var: Option<String>,
}

// Provider trait

pub(crate) trait Provider: Send + Sync {
    fn name(&self) -> &str;

    fn base_url(&self) -> &str;

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)>;

    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn dynamic_headers(&self, _body: &serde_json::Value) -> Vec<(String, String)> {
        vec![]
    }

    fn matches_model(&self, model: &str) -> bool;

    fn chat_completions_path(&self) -> &str {
        "/chat/completions"
    }

    fn embeddings_path(&self) -> &str {
        "/embeddings"
    }

    fn models_path(&self) -> &str {
        "/models"
    }

    fn image_generations_path(&self) -> &str {
        "/images/generations"
    }

    fn audio_speech_path(&self) -> &str {
        "/audio/speech"
    }

    fn audio_transcriptions_path(&self) -> &str {
        "/audio/transcriptions"
    }

    fn moderations_path(&self) -> &str {
        "/moderations"
    }

    fn rerank_path(&self) -> &str {
        "/rerank"
    }

    fn files_path(&self) -> &str {
        "/files"
    }

    fn batches_path(&self) -> &str {
        "/batches"
    }

    fn responses_path(&self) -> &str {
        "/responses"
    }

    fn search_path(&self) -> &str {
        "/search"
    }

    fn ocr_path(&self) -> &str {
        "/ocr"
    }

    #[allow(dead_code)]
    fn supports_streaming(&self) -> bool {
        true
    }

    fn transform_request(&self, body: &mut serde_json::Value) -> Result<(), ()> {
        let _ = body;
        Ok(())
    }

    fn transform_response(&self, _body: &mut serde_json::Value) -> Result<(), ()> {
        Ok(())
    }

    fn build_url(&self, endpoint_path: &str, _model: &str) -> String {
        format!("{}{}", self.base_url(), endpoint_path)
    }

    fn parse_stream_event(
        &self,
        event_data: &str,
    ) -> Result<Option<crate::types::ChatCompletionChunk>, ()> {
        serde_json::from_str::<crate::types::ChatCompletionChunk>(event_data)
            .map(Some)
            .map_err(|e| LiterLlmError::Streaming {
                message: format!("failed to parse SSE data: {e}"),
            })
    }

    fn stream_format(&self) -> StreamFormat {
        StreamFormat::Sse
    }

    fn build_stream_url(&self, endpoint_path: &str, model: &str) -> String {
        self.build_url(endpoint_path, model)
    }

    fn signing_headers(&self, method: &str, url: &str, body: &[u8]) -> Vec<(String, String)> {
        let _ = (method, url, body);
        vec![]
    }

    fn validate(&self) -> Result<(), ()> {
        Ok(())
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn env_var(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_provider_price_extracts_models_correctly() {
        let json = r#"{
                            "openai": {
                                "id": "openai",
                                "env": [],
                                "api": "",
                                "models": {
                                    "gpt-4": {
                                        "id": "gpt-4",
                                        "attachment": false,
                                        "reasoning": false,
                                        "tool_call": false,
                                        "temperature": false,
                                        "modalities": {"input": [], "output": []},
                                        "limit": {"context": 0, "output": 0},
                                        "cost": {
                                            "input": 30.0,
                                            "output": 60.0,
                                            "cache_read": 15.0
                                        }
                                    }
                                }
                            }
                        }"#;
        let registry = parse_provider(json).unwrap();
        assert!(registry.get("openai").unwrap().models.contains_key("gpt-4"));
        let price = &registry.get("openai").unwrap().models["gpt-4"]
            .cost
            .as_ref()
            .unwrap();
        assert!((price.input / TOKENS_PER_MILLION - 0.00003).abs() < 1e-10);
        assert!((price.output / TOKENS_PER_MILLION - 0.00006).abs() < 1e-10);
        assert_eq!(price.cache_read.unwrap() / TOKENS_PER_MILLION, 0.000015);
    }

    #[test]
    fn parse_provider_price_handles_missing_cost() {
        let json = r#"{
            "openai": {
                "id": "openai",
                "env": [],
                "api": "",
                "models": {
                    "gpt-4": {
                        "id": "gpt-4",
                        "attachment": false,
                        "reasoning": false,
                        "tool_call": false,
                        "temperature": false,
                        "modalities": {"input": [], "output": []},
                        "limit": {"context": 0, "output": 0}
                    }
                }
            }
        }"#;
        let registry = parse_provider(json).unwrap();
        assert!(
            registry.get("openai").unwrap().models["gpt-4"]
                .cost
                .is_none()
        );
    }

    #[test]
    fn parse_provider_price_handles_partial_cost() {
        let json = r#"{
            "test": {
                "id": "test",
                "env": [],
                "api": "",
                "models": {
                    "model": {
                        "id": "model",
                        "attachment": false,
                        "reasoning": false,
                        "tool_call": false,
                        "temperature": false,
                        "modalities": {"input": [], "output": []},
                        "limit": {"context": 0, "output": 0},
                        "cost": {
                            "input": 10.0,
                            "output": 0.0
                        }
                    }
                }
            }
        }"#;
        let registry = parse_provider(json).unwrap();
        let price = &registry.get("test").unwrap().models["model"]
            .cost
            .as_ref()
            .unwrap();
        assert!((price.input / TOKENS_PER_MILLION - 0.00001).abs() < 1e-10);
        assert_eq!(price.output / TOKENS_PER_MILLION, 0.0);
        assert_eq!(price.cache_read, None);
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn fetch_provider_returns_valid_registry() {
        let result = fetch_provider().await;
        assert!(
            result.is_ok(),
            "fetch_provider should succeed: {:?}",
            result.err()
        );
        let registry = result.unwrap();
        assert!(
            !registry.values().collect::<Vec<_>>()[0].models.is_empty(),
            "registry should have models"
        );
    }
}
