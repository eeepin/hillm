pub(crate) mod anthropic;
pub mod api_type;
pub(crate) mod bedrock;
pub mod codec;
pub mod cost;
pub(crate) mod custom;
pub(crate) mod datadriven;
pub(crate) mod openai;
pub mod outbound_policy;

pub use api_type::ApiType;
pub use codec::ApiTypeCodec;
#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
pub use outbound_policy::validate_outbound_url;
pub use outbound_policy::{
    OutboundPolicy, current_policy, set_outbound_policy, validate_outbound_url_sync,
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

// Fetch Providers and models info from models.dev

// Error of provider registry failed
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("failed to fetch data: {0}")]
    FetchError(String),
    #[error("failed to parse data: {0}")]
    ParseError(String),
}

type ProviderRegistry = HashMap<String, ProviderEntry>;

// ---------------------------------------------------------------------------
// Versioned provider registry snapshots
// ---------------------------------------------------------------------------

/// Source of a [`ProviderRegistrySnapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrySource {
    /// Fetched from the remote `models.dev` endpoint.
    Remote,
    /// No network available; the snapshot is empty or derived from local data.
    Offline,
}

/// A versioned snapshot of the provider registry.
///
/// Holds the registry data together with metadata about when it was fetched
/// and where it came from. Supports explicit refresh — calling
/// [`refresh_registry`] replaces the current snapshot with a freshly fetched
/// one while preserving the old data until the new one is fully parsed.
#[derive(Debug, Clone)]
pub struct ProviderRegistrySnapshot {
    /// The actual registry data.
    pub data: Arc<ProviderRegistry>,
    /// Unix epoch timestamp (seconds) when this snapshot was created.
    pub fetched_at: u64,
    /// Where this snapshot's data came from.
    pub source: RegistrySource,
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
static PROVIDER_REGISTRY: std::sync::RwLock<Option<ProviderRegistrySnapshot>> =
    std::sync::RwLock::new(None);

#[cfg(not(any(feature = "default-http", feature = "wasm-http")))]
static PROVIDER_REGISTRY: std::sync::RwLock<Option<ProviderRegistrySnapshot>> =
    std::sync::RwLock::new(None);

#[allow(dead_code)]
const PROVIDER_API_URL: &str = "https://models.dev/api.json";
pub(crate) const TOKENS_PER_MILLION: f64 = 1_000_000.0;

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
        let mut env_vars = HashMap::new();
        for e in &self.env {
            if e.contains("API_KEY") || e.contains("API_TOKEN") {
                env_vars.insert("api_key".to_string(), e.clone());
            } else {
                env_vars.insert(e.to_lowercase(), e.clone());
            }
        }
        let auth = if env_vars.is_empty() {
            None
        } else {
            Some(AuthConfig {
                auth_type: AuthType::Bearer,
                env_vars,
            })
        };
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
    let Ok(guard) = PROVIDER_REGISTRY.read() else {
        return ModelCapabilities::default();
    };
    let Some(snapshot) = guard.as_ref() else {
        return ModelCapabilities::default();
    };
    if let Some(model_entry) = snapshot
        .data
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

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
async fn fetch_provider() -> Result<ProviderRegistry, ProviderError> {
    use crate::util::bound::{RESPONSE_BODY_MAX_BYTES, check_bound};

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

    check_bound("provider registry", 0, text.len(), RESPONSE_BODY_MAX_BYTES)
        .map_err(|e| ProviderError::FetchError(e.to_string()))?;

    parse_provider(&text)
}

#[allow(dead_code)]
fn parse_provider(json: &str) -> Result<ProviderRegistry, ProviderError> {
    let providers: ProviderRegistry =
        serde_json::from_str(json).map_err(|e| ProviderError::ParseError(e.to_string()))?;
    Ok(providers)
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub async fn registry() -> Result<Arc<ProviderRegistry>, ProviderError> {
    // Fast path: return existing snapshot if present.
    {
        let guard = PROVIDER_REGISTRY
            .read()
            .map_err(|e| ProviderError::ParseError(format!("registry lock poisoned: {e}")))?;
        if let Some(snapshot) = guard.as_ref() {
            return Ok(Arc::clone(&snapshot.data));
        }
    }

    // Slow path: fetch and install a new snapshot.
    let fetched = fetch_provider().await?;
    let snapshot = ProviderRegistrySnapshot {
        data: Arc::new(fetched),
        fetched_at: unix_timestamp_secs(),
        source: RegistrySource::Remote,
    };

    let mut guard = PROVIDER_REGISTRY
        .write()
        .map_err(|e| ProviderError::ParseError(format!("registry lock poisoned: {e}")))?;
    // Double-check: another task may have initialized while we were fetching.
    if guard.is_none() {
        *guard = Some(snapshot.clone());
    } else if let Some(existing) = guard.as_ref() {
        // Use whichever is newer (should not normally happen, but be safe).
        if snapshot.fetched_at >= existing.fetched_at {
            *guard = Some(snapshot.clone());
        }
    }
    Ok(Arc::clone(&guard.as_ref().expect("just initialized").data))
}

#[cfg(not(any(feature = "default-http", feature = "wasm-http")))]
pub fn registry() -> Result<Arc<ProviderRegistry>, ProviderError> {
    // Without HTTP features, we can't fetch from remote; return an empty
    // registry so capability lookups degrade to defaults.
    let mut guard = PROVIDER_REGISTRY
        .write()
        .map_err(|e| ProviderError::ParseError(format!("registry lock poisoned: {e}")))?;
    if guard.is_none() {
        *guard = Some(ProviderRegistrySnapshot {
            data: Arc::new(HashMap::new()),
            fetched_at: unix_timestamp_secs(),
            source: RegistrySource::Offline,
        });
    }
    Ok(Arc::clone(&guard.as_ref().expect("just initialized").data))
}

/// Refresh the provider registry by re-fetching from the remote source.
///
/// Returns the new snapshot on success. The old snapshot is replaced
/// atomically — concurrent readers see either the old or the new data,
/// never a partially-updated state.
///
/// Without HTTP features, this is a no-op that returns the current
/// snapshot (or an empty offline snapshot if none exists).
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub async fn refresh_registry() -> Result<ProviderRegistrySnapshot, ProviderError> {
    let fetched = fetch_provider().await?;
    let snapshot = ProviderRegistrySnapshot {
        data: Arc::new(fetched),
        fetched_at: unix_timestamp_secs(),
        source: RegistrySource::Remote,
    };

    let mut guard = PROVIDER_REGISTRY
        .write()
        .map_err(|e| ProviderError::ParseError(format!("registry lock poisoned: {e}")))?;
    *guard = Some(snapshot.clone());
    Ok(snapshot)
}

/// Return the current registry snapshot, if one has been loaded.
///
/// Returns `None` if [`registry`] has not been called yet.
#[must_use]
pub fn registry_snapshot() -> Option<ProviderRegistrySnapshot> {
    PROVIDER_REGISTRY
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Return the unix timestamp (seconds) when the registry was last loaded,
/// or `None` if it has not been loaded yet.
#[must_use]
pub fn registry_fetched_at() -> Option<u64> {
    registry_snapshot().map(|s| s.fetched_at)
}

/// Return the source of the current registry snapshot, or `None` if the
/// registry has not been loaded yet.
#[must_use]
pub fn registry_source() -> Option<RegistrySource> {
    registry_snapshot().map(|s| s.source)
}

/// Synchronously check if the registry has been initialized and return a
/// clone of its data.
///
/// Returns `None` if the registry has not been loaded yet (i.e. [`registry`]
/// has not been called).
pub(crate) fn registry_get() -> Option<Arc<ProviderRegistry>> {
    PROVIDER_REGISTRY
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().map(|s| Arc::clone(&s.data)))
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
    pub available_api_types: Vec<ApiType>,
    /// The default API type to use when creating a provider instance.
    /// Must be one of `available_api_types` if both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_api_type: Option<ApiType>,
}

impl ProviderConfig {
    /// Returns the effective available API types, falling back to `[OpenAIChatCompletions]` if empty.
    pub fn effective_api_types(&self) -> Vec<ApiType> {
        if self.available_api_types.is_empty() {
            vec![ApiType::OpenAIChatCompletions]
        } else {
            self.available_api_types.clone()
        }
    }

    /// Returns the effective default API type, falling back to the first available type.
    pub fn effective_default_api_type(&self) -> ApiType {
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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env_vars: HashMap<String, String>,
}

// Provider trait

#[allow(dead_code)]
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
    fn available_api_types(&self) -> Vec<ApiType> {
        vec![ApiType::OpenAIChatCompletions]
    }

    /// The API type this provider instance is bound to.
    ///
    /// Fixed at creation time and must not change for the lifetime of the
    /// instance. Instances returned by [`create_provider`] report the API
    /// type that was explicitly selected; other instances report their
    /// effective default.
    #[allow(dead_code)]
    fn api_type(&self) -> ApiType {
        self.available_api_types()
            .first()
            .copied()
            .unwrap_or(ApiType::OpenAIChatCompletions)
    }

    #[allow(dead_code)]
    fn codec_for(&self, api_type: ApiType) -> Option<Box<dyn codec::ApiTypeCodec>> {
        let _ = api_type;
        None
    }

    /// Returns the complete map of environment variable names for this provider.
    ///
    /// Keys are semantic names (e.g., "api_key", "org_id"), values are the
    /// actual environment variable names (e.g., "OPENAI_API_KEY").
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn env_vars(&self) -> HashMap<&str, &str> {
        HashMap::new()
    }

    /// Returns the environment variable name for the given key.
    ///
    /// Common keys: "api_key", "org_id", "project_id".
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn env_var(&self, key: &str) -> Option<&str> {
        self.env_vars().get(key).copied()
    }

    fn validate(&self) -> HiLlmResult<()> {
        Ok(())
    }

    // --- Legacy endpoint path methods (used by Client for non-codec paths) ---

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

/// A provider wrapper that overrides the base URL.
///
/// Used when a named provider (e.g. "openai") is selected but the user wants
/// to point at a different endpoint (e.g. a proxy or OpenAI-compatible service).
/// All [`Provider`] methods delegate to the inner provider except [`Provider::base_url`].
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub(crate) struct BaseUrlOverride {
    inner: Arc<dyn Provider>,
    base_url: String,
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl BaseUrlOverride {
    pub(crate) fn new(inner: Arc<dyn Provider>, base_url: String) -> Self {
        Self { inner, base_url }
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl Provider for BaseUrlOverride {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        self.inner.auth_header(api_key)
    }

    fn matches_model(&self, model: &str) -> bool {
        self.inner.matches_model(model)
    }

    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        self.inner.extra_headers()
    }

    fn dynamic_headers(&self, body: &serde_json::Value) -> Vec<(String, String)> {
        self.inner.dynamic_headers(body)
    }

    fn available_api_types(&self) -> Vec<ApiType> {
        self.inner.available_api_types()
    }

    fn api_type(&self) -> ApiType {
        self.inner.api_type()
    }

    fn codec_for(&self, api_type: ApiType) -> Option<Box<dyn codec::ApiTypeCodec>> {
        self.inner.codec_for(api_type)
    }

    fn env_vars(&self) -> HashMap<&str, &str> {
        self.inner.env_vars()
    }

    fn validate(&self) -> HiLlmResult<()> {
        self.inner.validate()
    }

    fn chat_completions_path(&self) -> &str {
        self.inner.chat_completions_path()
    }

    fn embeddings_path(&self) -> &str {
        self.inner.embeddings_path()
    }

    fn image_generations_path(&self) -> &str {
        self.inner.image_generations_path()
    }

    fn audio_speech_path(&self) -> &str {
        self.inner.audio_speech_path()
    }

    fn audio_transcriptions_path(&self) -> &str {
        self.inner.audio_transcriptions_path()
    }

    fn moderations_path(&self) -> &str {
        self.inner.moderations_path()
    }

    fn rerank_path(&self) -> &str {
        self.inner.rerank_path()
    }

    fn search_path(&self) -> &str {
        self.inner.search_path()
    }

    fn ocr_path(&self) -> &str {
        self.inner.ocr_path()
    }

    fn models_path(&self) -> &str {
        self.inner.models_path()
    }

    fn files_path(&self) -> &str {
        self.inner.files_path()
    }

    fn batches_path(&self) -> &str {
        self.inner.batches_path()
    }

    fn responses_path(&self) -> &str {
        self.inner.responses_path()
    }

    fn build_url(&self, endpoint_path: &str, _model: &str) -> String {
        // Use the overridden base_url, not the inner provider's
        format!("{}{}", self.base_url(), endpoint_path)
    }

    fn build_stream_url(&self, endpoint_path: &str, _model: &str) -> String {
        // Use the overridden base_url, not the inner provider's
        self.build_url(endpoint_path, _model)
    }

    fn transform_request(&self, body: &mut serde_json::Value) -> HiLlmResult<()> {
        self.inner.transform_request(body)
    }

    fn transform_response(&self, body: &mut serde_json::Value) -> HiLlmResult<()> {
        self.inner.transform_response(body)
    }

    fn stream_format(&self) -> StreamFormat {
        self.inner.stream_format()
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<ChatCompletionChunk>> {
        self.inner.parse_stream_event(data)
    }

    fn signing_headers(&self, method: &str, url: &str, body: &[u8]) -> Vec<(String, String)> {
        self.inner.signing_headers(method, url, body)
    }
}

pub(crate) fn get_provider(name: &str) -> Option<Box<dyn Provider>> {
    if let Ok(Some(provider)) = custom::detect_custom_provider(name, "", custom::ApiTypeFilter::Any)
    {
        return Some(provider);
    }

    match name {
        "openai" => {
            return Some(Box::new(OpenAIProvider::default()));
        }
        "anthropic" => {
            return Some(Box::new(AnthropicProvider::default()));
        }
        "bedrock" => return Some(Box::new(BedrockProvider::from_env())),
        _ => {
            let snapshot = registry_get()?;
            if let Some(entry) = snapshot
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

/// Create a provider instance validated against a specific API type.
///
/// Unlike [`get_provider`], this function checks that the provider supports
/// the requested `api_type` and returns a structured error if not. The
/// returned instance is bound to `api_type` for its lifetime
/// ([`Provider::api_type`] reports it).
#[allow(dead_code)] // Part of ApiType routing infrastructure, not yet consumed by client
pub(crate) fn create_provider(
    name: &str,
    api_type: ApiType,
) -> Result<Box<dyn Provider>, HiLlmError> {
    // Custom providers are detected with an API type filter so that a
    // registered custom provider which does not support the requested API
    // type is not silently returned (and, when the name does not match a
    // built-in, surfaces as ApiTypeUnsupported instead of ProviderNotFound).
    if let Some(provider) =
        custom::detect_custom_provider(name, "", custom::ApiTypeFilter::Exact(api_type))?
    {
        return Ok(provider);
    }

    // Built-in providers are bound to the explicitly requested API type.
    match name {
        "openai" => return Ok(Box::new(openai::OpenAIProvider::with_api_type(api_type)?)),
        "anthropic" => {
            // The native protocol for Anthropic is Messages. Selecting OpenAI
            // Chat Completions for it yields the explicit compatibility
            // adapter, never a hidden default.
            if api_type == ApiType::OpenAIChatCompletions {
                return Ok(Box::new(
                    anthropic::compat::AnthropicChatCompatProvider::new(),
                ));
            }
            return Ok(Box::new(anthropic::AnthropicProvider::with_api_type(
                api_type,
            )?));
        }
        _ => {}
    }

    let provider = get_provider(name).ok_or_else(|| HiLlmError::ProviderNotFound {
        name: name.to_string(),
    })?;

    if !provider.available_api_types().contains(&api_type) {
        return Err(HiLlmError::ApiTypeUnsupported {
            api_type: api_type.to_string(),
            provider: provider.name().to_string(),
        });
    }

    Ok(provider)
}

/// Returns the codec implementation for an API type, if one exists.
///
/// Used by providers whose wire format is fully determined by the API type
/// (custom, config-driven and OpenAI-compatible providers), so they do not
/// need to duplicate the codec mapping.
#[allow(dead_code)]
pub(crate) fn codec_for_api_type(api_type: ApiType) -> Option<Box<dyn codec::ApiTypeCodec>> {
    match api_type {
        ApiType::OpenAIChatCompletions => Some(Box::new(openai::OpenAIChatCompletionsCodec)),
        ApiType::OpenAIResponses => Some(Box::new(openai::OpenAIResponsesCodec)),
        ApiType::AnthropicMessages => Some(Box::new(anthropic::codec::AnthropicMessagesCodec)),
        ApiType::BedrockConverse => Some(Box::new(bedrock::codec::BedrockConverseCodec)),
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub async fn all_providers() -> HiLlmResult<Vec<ProviderConfig>> {
    let registry = registry().await.map_err(|e| HiLlmError::InternalError {
        message: e.to_string(),
    })?;
    Ok(registry
        .values()
        .map(|provider_entry| provider_entry.to_config())
        .collect())
}

#[cfg(not(any(feature = "default-http", feature = "wasm-http")))]
pub fn all_providers() -> HiLlmResult<Vec<ProviderConfig>> {
    let registry = registry().map_err(|e| HiLlmError::InternalError {
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

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
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
        assert_eq!(
            config.effective_api_types(),
            vec![ApiType::OpenAIChatCompletions]
        );
        assert_eq!(
            config.effective_default_api_type(),
            ApiType::OpenAIChatCompletions
        );
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
            available_api_types: vec![ApiType::OpenAIChatCompletions],
            default_api_type: Some(ApiType::AnthropicMessages),
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
            available_api_types: vec![ApiType::OpenAIChatCompletions, ApiType::OpenAIResponses],
            default_api_type: Some(ApiType::OpenAIResponses),
        };
        assert!(config.validate_api_types().is_ok());
        assert_eq!(
            config.effective_default_api_type(),
            ApiType::OpenAIResponses
        );
    }

    #[test]
    fn provider_entry_to_config_scans_all_env_vars() {
        let entry = ProviderEntry {
            id: "test".into(),
            env: vec!["NON_KEY_VAR".into(), "MY_API_KEY".into()],
            api: Default::default(),
            name: Default::default(),
            models: Default::default(),
        };
        let config = entry.to_config();
        assert_eq!(
            config
                .auth
                .as_ref()
                .unwrap()
                .env_vars
                .get("api_key")
                .map(|s| s.as_str()),
            Some("MY_API_KEY")
        );
    }

    #[test]
    fn create_provider_returns_provider_for_supported_api_type() {
        let result = create_provider("openai", ApiType::OpenAIChatCompletions);
        assert!(result.is_ok());
        let provider = result.unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.api_type(), ApiType::OpenAIChatCompletions);
    }

    #[test]
    fn create_provider_returns_provider_for_responses_api_type() {
        let result = create_provider("openai", ApiType::OpenAIResponses);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().api_type(),
            ApiType::OpenAIResponses,
            "instance must be bound to the requested api type"
        );
    }

    #[test]
    fn create_provider_rejects_unsupported_api_type() {
        let result = create_provider("openai", ApiType::AnthropicMessages);
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(matches!(err, HiLlmError::ApiTypeUnsupported { .. }));
            let msg = err.to_string();
            assert!(msg.contains("openai"));
        }
    }

    #[test]
    fn create_provider_anthropic_chat_completions_returns_explicit_compat_adapter() {
        // Selecting Chat Completions for Anthropic is allowed, but only via
        // the explicit compatibility adapter — the instance reports the Chat
        // API type and a Chat codec, while the wire endpoint stays /messages.
        let provider = create_provider("anthropic", ApiType::OpenAIChatCompletions)
            .expect("anthropic supports chat completions through the compat adapter");
        assert_eq!(provider.api_type(), ApiType::OpenAIChatCompletions);
        let codec = provider
            .codec_for(ApiType::OpenAIChatCompletions)
            .expect("compat codec");
        assert_eq!(codec.endpoint_path(), "/messages");
    }

    #[test]
    fn create_provider_rejects_responses_api_type_for_anthropic() {
        let result = create_provider("anthropic", ApiType::OpenAIResponses);
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(matches!(err, HiLlmError::ApiTypeUnsupported { .. }));
        }
    }

    #[test]
    fn create_provider_anthropic_binds_messages_api_type() {
        let provider = create_provider("anthropic", ApiType::AnthropicMessages)
            .expect("anthropic supports messages");
        assert_eq!(provider.api_type(), ApiType::AnthropicMessages);
        assert!(provider.codec_for(ApiType::AnthropicMessages).is_some());
    }

    #[test]
    fn create_provider_returns_not_found_for_unknown() {
        let result = create_provider("nonexistent_provider", ApiType::OpenAIChatCompletions);
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(matches!(err, HiLlmError::ProviderNotFound { .. }));
        }
    }
}
