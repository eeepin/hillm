pub(crate) mod anthropic;
pub mod api_type;
pub(crate) mod bedrock;
pub mod codec;
pub mod cost;
pub(crate) mod custom;
pub(crate) mod datadriven;
pub(crate) mod openai;
pub(crate) mod openai_compatible;
pub mod outbound_policy;

pub use api_type::APIType;
pub use codec::APITypeCodec;
pub use outbound_policy::{
    OutboundPolicy, current_policy, set_outbound_policy, validate_outbound_url,
    validate_outbound_url_sync,
};

use anthropic::AnthropicProvider;
use bedrock::BedrockProvider;
use datadriven::ConfigDrivenProvider;
use openai::OpenAIProvider;

use crate::error::{HiLlmError, HiLlmResult};
use crate::types::{ChatCompletionChunk, Modality};
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
    #[serde(default)]
    name: String,
    pub(crate) models: HashMap<String, ModelEntry>,
}

impl ProviderEntry {
    pub fn to_config(&self) -> ProviderConfig {
        let mut auth = None;
        for e in &self.env {
            if e.contains("API_KEY") || e.contains("API_TOKEN") {
                auth = Some(AuthConfig {
                    auth_type: AuthType::Bearer,
                    env_var: Some(e.clone()),
                });
                break;
            }
        }
        let models = self.models.keys().cloned().collect();
        ProviderConfig {
            name: self.id.clone(),
            display_name: Some(self.name.clone()),
            base_url: Some(self.api.clone()),
            auth,
            endpoints: None,
            models,
            param_mappings: None,
            available_api_types: vec![], // Will use default [OpenAIChatCompletions]
            default_api_type: None,
        }
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct ModelEntry {
    #[allow(dead_code)]
    id: String,
    #[serde(default, flatten)]
    capabilities: ModelCapabilities,
    #[serde(default)]
    #[allow(dead_code)]
    limit: ModelLimit,
    #[serde(default)]
    pub(crate) cost: Option<ModelPrice>,
}

/// Provider model capability
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    attachment: bool,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    structured_output: bool,
    #[serde(default)]
    temperature: bool,
    #[serde(default)]
    modalities: ModelModality,
}

pub fn capabilities(provider_name: &str, model_name: &str) -> ModelCapabilities {
    let Some(reg) = PROVIDER_REGISTRY.get() else {
        return ModelCapabilities::default();
    };
    if let Some(model_entry) = reg
        .get(provider_name)
        .and_then(|provider_entry| provider_entry.models.get(model_name))
    {
        model_entry.capabilities.clone()
    } else {
        ModelCapabilities::default()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ModelModality {
    #[serde(default)]
    input: Vec<Modality>,
    #[serde(default)]
    output: Vec<Modality>,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct ModelLimit {
    #[serde(default)]
    #[allow(dead_code)]
    context: u64,
    #[serde(default)]
    #[allow(dead_code)]
    output: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelPrice {
    #[serde(flatten)]
    pub(crate) token_price: TokenPrice,
    pub(crate) tiers: Option<Vec<TokenPriceTier>>,
}

impl ModelPrice {
    pub fn token_price_by_tier(&self, context: u64) -> &TokenPrice {
        if let Some(tiers) = &self.tiers {
            let best = tiers
                .iter()
                .filter(|t| context >= t.tier.size)
                .max_by_key(|t| t.tier.size);
            if let Some(tier) = best {
                return &tier.token_price;
            }
        }
        &self.token_price
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenPrice {
    pub(crate) input: f64,
    pub(crate) output: f64,
    pub(crate) cache_read: Option<f64>,
    pub(crate) cache_write: Option<f64>,
}

impl TokenPrice {
    pub fn cost(
        &self,
        input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
    ) -> Result<Option<f64>, ProviderError> {
        let cache_read_tokens = cache_read_tokens.min(input_tokens);
        let cache_write_tokens = cache_write_tokens.min(input_tokens - cache_read_tokens);
        let uncached_input_tokens = input_tokens - cache_read_tokens - cache_write_tokens;
        let cost_per_million = (uncached_input_tokens as f64) * self.input
            + (cache_read_tokens as f64) * self.cache_read.unwrap_or(self.input)
            + (cache_write_tokens as f64) * self.cache_write.unwrap_or(self.input)
            + (output_tokens as f64) * self.output;
        Ok(Some(cost_per_million / TOKENS_PER_MILLION))
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenPriceTier {
    #[serde(flatten)]
    pub(crate) token_price: TokenPrice,
    pub(crate) tier: ContextTier,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(tag = "type", rename = "context")]
pub struct ContextTier {
    pub(crate) size: u64,
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

/// Synchronously check if the registry has been initialized.
pub(crate) fn registry_get() -> Option<&'static Arc<ProviderRegistry>> {
    PROVIDER_REGISTRY.get()
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
    SSE,
    AwsEventStream,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub display_name: Option<String>,
    pub base_url: Option<String>,
    pub auth: Option<AuthConfig>,
    pub endpoints: Option<Vec<String>>,
    pub models: Vec<String>,
    pub param_mappings: Option<HashMap<String, String>>,
    /// API types this provider supports. Empty defaults to `[OpenAIChatCompletions]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_api_types: Vec<APIType>,
    /// The default API type to use when creating a provider instance.
    /// Must be one of `available_api_types` if both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_api_type: Option<APIType>,
}

impl ProviderConfig {
    /// Returns the effective available API types, falling back to `[OpenAIChatCompletions]` if empty.
    pub fn effective_api_types(&self) -> Vec<APIType> {
        if self.available_api_types.is_empty() {
            vec![APIType::OpenAIChatCompletions]
        } else {
            self.available_api_types.clone()
        }
    }

    /// Returns the effective default API type, falling back to the first available type.
    pub fn effective_default_api_type(&self) -> APIType {
        self.default_api_type
            .unwrap_or_else(|| self.effective_api_types()[0])
    }

    /// Validates the API type configuration.
    /// Returns an error if `default_api_type` is set but not in `available_api_types`.
    pub fn validate_api_types(&self) -> HiLlmResult<()> {
        if let Some(default) = self.default_api_type {
            let available = self.effective_api_types();
            if !available.contains(&default) {
                return Err(HiLlmError::BadRequest {
                    message: format!(
                        "default_api_type '{default}' is not in available_api_types {:?}",
                        available
                    ),
                    status: 400,
                });
            }
        }
        Ok(())
    }
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

    fn matches_model(&self, model: &str) -> bool;

    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn dynamic_headers(&self, _body: &serde_json::Value) -> Vec<(String, String)> {
        vec![]
    }

    #[allow(dead_code)]
    fn available_api_types(&self) -> Vec<APIType> {
        vec![APIType::OpenAIChatCompletions]
    }

    #[allow(dead_code)]
    fn codec_for(&self, api_type: APIType) -> Option<Box<dyn codec::APITypeCodec>> {
        let _ = api_type;
        None
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn env_var(&self) -> Option<&str> {
        None
    }

    fn validate(&self) -> HiLlmResult<()> {
        Ok(())
    }

    // --- Legacy endpoint path methods (used by DefaultClient for non-codec paths) ---

    fn chat_completions_path(&self) -> &str {
        "/chat/completions"
    }

    fn embeddings_path(&self) -> &str {
        "/embeddings"
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

    fn search_path(&self) -> &str {
        "/search"
    }

    fn ocr_path(&self) -> &str {
        "/ocr"
    }

    fn models_path(&self) -> &str {
        "/models"
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

    // --- Legacy URL building ---

    fn build_url(&self, endpoint_path: &str, _model: &str) -> String {
        format!("{}{}", self.base_url(), endpoint_path)
    }

    fn build_stream_url(&self, endpoint_path: &str, model: &str) -> String {
        self.build_url(endpoint_path, model)
    }

    // --- Legacy request/response transforms ---

    fn transform_request(&self, _body: &mut serde_json::Value) -> HiLlmResult<()> {
        Ok(())
    }

    fn transform_response(&self, _body: &mut serde_json::Value) -> HiLlmResult<()> {
        Ok(())
    }

    // --- Legacy streaming ---

    fn stream_format(&self) -> StreamFormat {
        StreamFormat::SSE
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<ChatCompletionChunk>> {
        if data == "[DONE]" {
            return Ok(None);
        }
        serde_json::from_str(data)
            .map(Some)
            .map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse stream event: {e}"),
            })
    }

    // --- Legacy signing ---

    fn signing_headers(&self, _method: &str, _url: &str, _body: &[u8]) -> Vec<(String, String)> {
        vec![]
    }
}

pub(crate) fn get_provider(name: &str) -> Option<Box<dyn Provider>> {
    if let Some(provider) = custom::detect_custom_provider(name, "") {
        return Some(provider);
    }

    match name {
        "openai" => {
            return Some(Box::new(OpenAIProvider));
        }
        "anthropic" => {
            return Some(Box::new(AnthropicProvider));
        }
        "bedrock" => return Some(Box::new(BedrockProvider::from_env())),
        _ => {
            let reg = PROVIDER_REGISTRY.get()?;
            if let Some(entry) = reg
                .values()
                .collect::<Vec<&ProviderEntry>>()
                .iter()
                .find(|e| *e.id == *name)
            {
                return Some(Box::new(ConfigDrivenProvider::new(entry.to_config())));
            }
        }
    }

    None
}

pub async fn all_providers() -> HiLlmResult<Vec<ProviderConfig>> {
    let registry = registry().await.map_err(|e| HiLlmError::InternalError {
        message: e.to_string(),
    })?;
    Ok(registry
        .values()
        .map(|provider_entry| provider_entry.to_config())
        .collect())
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
            .unwrap()
            .token_price;
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
            .unwrap()
            .token_price;
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

    #[test]
    fn provider_config_defaults_to_chat_completions() {
        let config = ProviderConfig {
            name: "test".to_string(),
            display_name: None,
            base_url: Some("https://example.com".to_string()),
            auth: None,
            endpoints: None,
            models: vec!["model-a".to_string()],
            param_mappings: None,
            available_api_types: vec![],
            default_api_type: None,
        };
        assert_eq!(config.effective_api_types(), vec![APIType::OpenAIChatCompletions]);
        assert_eq!(config.effective_default_api_type(), APIType::OpenAIChatCompletions);
    }

    #[test]
    fn provider_config_validates_default_in_available() {
        let config = ProviderConfig {
            name: "test".to_string(),
            display_name: None,
            base_url: Some("https://example.com".to_string()),
            auth: None,
            endpoints: None,
            models: vec!["model-a".to_string()],
            param_mappings: None,
            available_api_types: vec![APIType::OpenAIChatCompletions],
            default_api_type: Some(APIType::AnthropicMessages),
        };
        let result = config.validate_api_types();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("default_api_type"));
    }

    #[test]
    fn provider_config_accepts_valid_default() {
        let config = ProviderConfig {
            name: "test".to_string(),
            display_name: None,
            base_url: Some("https://example.com".to_string()),
            auth: None,
            endpoints: None,
            models: vec!["model-a".to_string()],
            param_mappings: None,
            available_api_types: vec![APIType::OpenAIChatCompletions, APIType::OpenAIResponses],
            default_api_type: Some(APIType::OpenAIResponses),
        };
        assert!(config.validate_api_types().is_ok());
        assert_eq!(config.effective_default_api_type(), APIType::OpenAIResponses);
    }
}
