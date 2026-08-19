pub mod builder;
pub mod config;
pub mod config_file;
mod impls;
#[cfg(all(feature = "default-http", feature = "tower"))]
pub mod managed;

use std::future::Future;
use std::pin::Pin;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use std::sync::Arc;

use futures_core::Stream;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::{HiLlmError, HiLlmResult};
use crate::types::audio::{CreateSpeechRequest, CreateTranscriptionRequest, TranscriptionResponse};
use crate::types::batch::{
    BatchListQuery, BatchListResponse, BatchObject, BatchStatus, CreateBatchRequest,
};
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::types::embedding::{EmbeddingRequest, EmbeddingResponse};
use crate::types::file::{
    CreateFileRequest, DeleteResponse, FileListQuery, FileListResponse, FileObject,
};
use crate::types::image::{CreateImageRequest, ImagesResponse};
use crate::types::model::ModelsListResponse;
use crate::types::moderation::{ModerationRequest, ModerationResponse};
use crate::types::ocr::{OcrRequest, OcrResponse};
use crate::types::raw::{RawExchange, RawStreamExchange};
use crate::types::rerank::{RerankRequest, RerankResponse};
use crate::types::response::{CreateResponseRequest, ResponseObject, ResponsesStreamEvent};
use crate::types::search::{SearchRequest, SearchResponse};

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::auth::Credential;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::provider::{
    self, Provider,
    custom::{ApiTypeFilter, CustomProvider},
    openai::OpenAIProvider,
};

pub use builder::{ClientBuilder, NoApiKey, NoProvider, WithApiKey, WithProvider};
pub use config::{ClientConfig, ClientConfigBuilder};
pub use config_file::FileConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaitForBatchConfig {
    pub initial_interval_secs: f64,
    pub max_interval_secs: f64,
    pub backoff_multiplier: f32,
    pub timeout_secs: Option<f64>,
}

impl Default for WaitForBatchConfig {
    fn default() -> Self {
        Self {
            initial_interval_secs: 5.0,
            max_interval_secs: 60.0,
            backoff_multiplier: 1.5,
            timeout_secs: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum BatchWaitError {
    #[error("Batch reached terminal failure state: {status:?}")]
    Failed { status: BatchStatus },
    #[error("Polling timed out after {timeout_secs:.1}s")]
    Timeout { timeout_secs: f64 },
    #[error("Client error (code {code}): {message}")]
    Client { message: String, code: u32 },
}

impl From<HiLlmError> for BatchWaitError {
    fn from(err: HiLlmError) -> Self {
        Self::Client {
            code: u32::from(err.status_code()),
            message: err.to_string(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(target_arch = "wasm32")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(not(target_arch = "wasm32"))]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

#[cfg(target_arch = "wasm32")]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
struct PreparedRequest {
    url: String,
    provider: Arc<dyn Provider>,
    body_json: serde_json::Value,
    body_bytes: bytes::Bytes,
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub(crate) fn str_pair(pair: &(String, String)) -> (&str, &str) {
    (pair.0.as_str(), pair.1.as_str())
}

/// OpenAI Chat Completions API client.
///
/// This trait is specifically for the Chat Completions API route.
/// For other APIs (embeddings, images, audio, etc.), use their dedicated traits.
#[cfg(not(target_arch = "wasm32"))]
pub trait ChatCompletionClient: Send + Sync {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>>;

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>;

    fn chat_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ChatCompletionResponse>>>;

    fn chat_stream_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<
        '_,
        HiLlmResult<RawStreamExchange<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>,
    >;
}

#[cfg(target_arch = "wasm32")]
pub trait ChatCompletionClient {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>>;

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>;

    fn chat_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ChatCompletionResponse>>>;

    fn chat_stream_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<
        '_,
        HiLlmResult<RawStreamExchange<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>,
    >;
}

/// OpenAI Embeddings API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait EmbeddingClient: Send + Sync {
    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>>;

    fn embed_raw(
        &self,
        req: EmbeddingRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<EmbeddingResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait EmbeddingClient {
    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>>;

    fn embed_raw(
        &self,
        req: EmbeddingRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<EmbeddingResponse>>>;
}

/// OpenAI Images API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait ImageClient: Send + Sync {
    fn image_generate(&self, req: CreateImageRequest)
    -> BoxFuture<'_, HiLlmResult<ImagesResponse>>;

    fn image_generate_raw(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ImagesResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait ImageClient {
    fn image_generate(&self, req: CreateImageRequest)
    -> BoxFuture<'_, HiLlmResult<ImagesResponse>>;

    fn image_generate_raw(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ImagesResponse>>>;
}

/// OpenAI Audio API client (speech and transcription).
#[cfg(not(target_arch = "wasm32"))]
pub trait AudioClient: Send + Sync {
    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>>;

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>>;

    fn transcribe_raw(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<TranscriptionResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait AudioClient {
    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>>;

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>>;

    fn transcribe_raw(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<TranscriptionResponse>>>;
}

/// OpenAI Moderations API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait ModerationClient: Send + Sync {
    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>>;

    fn moderate_raw(
        &self,
        req: ModerationRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ModerationResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait ModerationClient {
    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>>;

    fn moderate_raw(
        &self,
        req: ModerationRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ModerationResponse>>>;
}

/// Rerank API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait RerankClient: Send + Sync {
    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>>;

    fn rerank_raw(
        &self,
        req: RerankRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<RerankResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait RerankClient {
    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>>;

    fn rerank_raw(
        &self,
        req: RerankRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<RerankResponse>>>;
}

/// Search API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait SearchClient: Send + Sync {
    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>>;

    fn search_raw(
        &self,
        req: SearchRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<SearchResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait SearchClient {
    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>>;

    fn search_raw(
        &self,
        req: SearchRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<SearchResponse>>>;
}

/// OCR API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait OcrClient: Send + Sync {
    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>>;

    fn ocr_raw(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<RawExchange<OcrResponse>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait OcrClient {
    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>>;

    fn ocr_raw(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<RawExchange<OcrResponse>>>;
}

/// Models API client.
#[cfg(not(target_arch = "wasm32"))]
pub trait ModelClient: Send + Sync {
    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>>;
}

#[cfg(target_arch = "wasm32")]
pub trait ModelClient {
    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>>;
}

#[cfg(not(target_arch = "wasm32"))]
pub trait FileClient: Send + Sync {
    fn create_file(&self, req: CreateFileRequest) -> BoxFuture<'_, HiLlmResult<FileObject>>;

    fn retrieve_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<FileObject>>;

    fn delete_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<DeleteResponse>>;

    fn list_files(
        &self,
        query: Option<FileListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<FileListResponse>>;

    fn file_content(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>>;
}

#[cfg(target_arch = "wasm32")]
pub trait FileClient {
    fn create_file(&self, req: CreateFileRequest) -> BoxFuture<'_, HiLlmResult<FileObject>>;

    fn retrieve_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<FileObject>>;

    fn delete_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<DeleteResponse>>;

    fn list_files(
        &self,
        query: Option<FileListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<FileListResponse>>;

    fn file_content(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>>;
}

#[cfg(not(target_arch = "wasm32"))]
pub trait BatchClient: Send + Sync {
    fn create_batch(&self, req: CreateBatchRequest) -> BoxFuture<'_, HiLlmResult<BatchObject>>;

    fn retrieve_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>>;

    fn list_batches(
        &self,
        query: Option<BatchListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<BatchListResponse>>;

    fn cancel_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>>;
}

#[cfg(target_arch = "wasm32")]
pub trait BatchClient {
    fn create_batch(&self, req: CreateBatchRequest) -> BoxFuture<'_, HiLlmResult<BatchObject>>;

    fn retrieve_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>>;

    fn list_batches(
        &self,
        query: Option<BatchListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<BatchListResponse>>;

    fn cancel_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>>;
}

#[cfg(not(target_arch = "wasm32"))]
pub trait ResponseClient: Send + Sync {
    fn create_response(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    /// Creates a response through the OpenAI Responses API and streams the
    /// native Responses SSE events. Events are delivered as
    /// [`ResponsesStreamEvent`] — they are not converted into Chat
    /// Completions chunks.
    fn create_response_stream(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ResponsesStreamEvent>>>>;

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;
}

#[cfg(target_arch = "wasm32")]
pub trait ResponseClient {
    fn create_response(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    /// Creates a response through the OpenAI Responses API and streams the
    /// native Responses SSE events. Events are delivered as
    /// [`ResponsesStreamEvent`] — they are not converted into Chat
    /// Completions chunks.
    fn create_response_stream(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ResponsesStreamEvent>>>>;

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;
}

use crate::types::anthropic::{
    AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent,
};

/// Client for the Anthropic Messages API with its native request/response
/// and stream event types.
///
/// This trait only routes through [`provider::APIType::AnthropicMessages`].
/// Calling it against a provider instance bound to another API type fails
/// with [`HiLlmError::EndpointNotSupported`] before any request is sent.
#[cfg(not(target_arch = "wasm32"))]
pub trait AnthropicMessagesClient: Send + Sync {
    fn create_message(
        &self,
        req: AnthropicMessagesRequest,
    ) -> BoxFuture<'_, HiLlmResult<AnthropicMessagesResponse>>;

    /// Creates a message and streams the native Anthropic SSE events
    /// (`message_start`, `content_block_delta`, …). Events are delivered as
    /// [`AnthropicStreamEvent`] — they are not converted into Chat
    /// Completions chunks.
    fn create_message_stream(
        &self,
        req: AnthropicMessagesRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<AnthropicStreamEvent>>>>;
}

#[cfg(target_arch = "wasm32")]
pub trait AnthropicMessagesClient {
    fn create_message(
        &self,
        req: AnthropicMessagesRequest,
    ) -> BoxFuture<'_, HiLlmResult<AnthropicMessagesResponse>>;

    /// Creates a message and streams the native Anthropic SSE events
    /// (`message_start`, `content_block_delta`, …). Events are delivered as
    /// [`AnthropicStreamEvent`] — they are not converted into Chat
    /// Completions chunks.
    fn create_message_stream(
        &self,
        req: AnthropicMessagesRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<AnthropicStreamEvent>>>>;
}

/// Default client based on `reqwest`.
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[derive(Clone)]
pub struct Client {
    config: ClientConfig,
    http_client: reqwest::Client,
    provider: Arc<dyn Provider>,
    cached_auth_header: Option<(String, String)>,
    cached_extra_headers: Vec<(String, String)>,
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl Client {
    pub fn new(config: ClientConfig, provider: Option<String>) -> HiLlmResult<Self> {
        let provider = build_provider(&config, provider)?;

        provider.validate()?;

        #[cfg(not(target_arch = "wasm32"))]
        let mut config = config;
        #[cfg(not(target_arch = "wasm32"))]
        if config.load_env
            && config.api_key.expose_secret().is_empty()
            && let Some(env_var_name) = provider.env_var("api_key")
        {
            match std::env::var(env_var_name) {
                Ok(val) if !val.is_empty() => {
                    config.api_key = secrecy::SecretString::from(val);
                }
                _ => {
                    return Err(HiLlmError::Authentication {
                        message: format!(
                            "no API key provided and environment variable {env_var_name} is not set"
                        ),
                        status: 401,
                    });
                }
            }
        }

        let mut header_map = reqwest::header::HeaderMap::new();
        for (k, v) in config.headers() {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|_| {
                HiLlmError::InvalidHeader {
                    name: k.clone(),
                    reason: "pre-validated header name became invalid".into(),
                }
            })?;
            let val = reqwest::header::HeaderValue::from_str(v).map_err(|_| {
                HiLlmError::InvalidHeader {
                    name: k.clone(),
                    reason: "pre-validated header value became invalid".into(),
                }
            })?;
            header_map.insert(name, val);
        }
        let http_client = {
            #[cfg(feature = "default-http")]
            crate::ensure_crypto_provider();
            let builder = reqwest::Client::builder().default_headers(header_map);
            #[cfg(all(feature = "default-http", not(target_arch = "wasm32")))]
            let builder = {
                // Use the per-client outbound policy if set, otherwise fall
                // back to the process-global policy.
                let is_off = match &config.outbound_policy {
                    Some(validator) => validator.current_policy().is_off(),
                    None => crate::provider::current_policy().is_off(),
                };
                if !is_off {
                    let resolver = match &config.outbound_policy {
                        Some(validator) => crate::provider::outbound_policy::GuardedResolver::new(
                            Arc::clone(validator),
                        ),
                        None => crate::provider::outbound_policy::GuardedResolver::from_global(),
                    };
                    builder.dns_resolver(Arc::new(resolver))
                } else {
                    builder
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            let builder = builder.timeout(config.timeout);
            #[cfg(not(target_arch = "wasm32"))]
            let builder = config.transport.apply_to_builder(builder);
            builder.build().map_err(HiLlmError::from)?
        };

        let cached_auth_header = provider
            .auth_header(config.api_key.expose_secret())
            .map(|(name, value)| (name.into_owned(), value.into_owned()));

        let cached_extra_headers = provider
            .extra_headers()
            .iter()
            .map(|&(name, value)| (name.to_owned(), value.to_owned()))
            .collect();

        Ok(Self {
            config,
            http_client,
            provider,
            cached_auth_header,
            cached_extra_headers,
        })
    }

    async fn resolve_auth_header_for_provider(
        &self,
        prov: &dyn Provider,
    ) -> HiLlmResult<Option<(String, String)>> {
        if let Some(ref cp) = self.config.credential_provider {
            let credential = cp.resolve().await?;
            match credential {
                Credential::BearerToken(token) => Ok(Some((
                    "Authorization".to_owned(),
                    format!("Bearer {}", token.expose_secret()),
                ))),
                Credential::AwsCredentials { .. } => Ok(None),
            }
        } else {
            Ok(prov
                .auth_header(self.config.api_key.expose_secret())
                .map(|(name, value)| (name.into_owned(), value.into_owned())))
        }
    }

    fn all_headers_for_provider(
        &self,
        prov: &dyn Provider,
        method: &str,
        url: &str,
        body_json: &serde_json::Value,
        body_bytes: &[u8],
    ) -> Vec<(String, String)> {
        let mut headers = prov.signing_headers(method, url, body_bytes);
        headers.extend(
            prov.extra_headers()
                .iter()
                .map(|&(name, value)| (name.to_owned(), value.to_owned())),
        );
        headers.extend(prov.dynamic_headers(body_json));
        headers
    }

    fn prepare_request(
        &self,
        serializable: &impl serde::Serialize,
        endpoint_fn: impl FnOnce(&dyn Provider) -> &str,
        model: &str,
        stream: Option<bool>,
    ) -> HiLlmResult<PreparedRequest> {
        if model.is_empty() {
            return Err(HiLlmError::BadRequest {
                message: "model must not be empty".into(),
                status: 400,
            });
        }

        let provider = self.provider.clone();
        if !provider.matches_model(model) {
            return Err(HiLlmError::BadRequest {
                message: format!("{} has no model named {}", provider.name(), model),
                status: 400,
            });
        }
        let endpoint_path = endpoint_fn(provider.as_ref());
        let url = provider.build_url(endpoint_path, model);

        let mut body = serde_json::to_value(serializable)?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".into(), serde_json::Value::String(model.to_string()));
            if let Some(s) = stream {
                obj.insert("stream".into(), serde_json::Value::Bool(s));
            }
        }
        provider.transform_request(&mut body)?;

        let body_bytes = bytes::Bytes::from(serde_json::to_vec(&body)?);

        Ok(PreparedRequest {
            url,
            provider,
            body_json: body,
            body_bytes,
        })
    }

    async fn resolve_auth_header(&self) -> HiLlmResult<Option<(String, String)>> {
        if let Some(ref cp) = self.config.credential_provider {
            let credential = cp.resolve().await?;
            match credential {
                Credential::BearerToken(token) => Ok(Some((
                    "Authorization".to_owned(),
                    format!("Bearer {}", token.expose_secret()),
                ))),
                Credential::AwsCredentials { .. } => Ok(None),
            }
        } else {
            Ok(self.cached_auth_header.clone())
        }
    }

    fn all_headers(
        &self,
        method: &str,
        url: &str,
        body_json: &serde_json::Value,
        body_bytes: &[u8],
    ) -> Vec<(String, String)> {
        let mut headers = self.provider.signing_headers(method, url, body_bytes);
        headers.extend(self.cached_extra_headers.iter().cloned());
        headers.extend(self.provider.dynamic_headers(body_json));
        headers
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
fn build_provider(
    config: &ClientConfig,
    provider_name: Option<String>,
) -> HiLlmResult<Arc<dyn Provider>> {
    match provider_name {
        // 1. Explicit name → try to resolve by name first
        Some(name) => {
            // Try to resolve the provider
            let resolved_result = if let Some(api_type) = config.api_type {
                // Explicit API type: validate and bind
                provider::create_provider(&name, api_type).map(Some)
            } else {
                // No API type specified: use get_provider which returns default binding
                Ok(provider::get_provider(&name))
            };

            match (resolved_result?, &config.base_url) {
                // Found + no base_url → use as-is
                (Some(provider), None) => Ok(Arc::from(provider)),

                // Found + base_url → wrap with URL override
                (Some(provider), Some(url)) => Ok(Arc::new(provider::BaseUrlOverride::new(
                    Arc::from(provider),
                    url.clone(),
                ))),

                // Not found + custom_provider set → use custom_provider as fallback
                (None, _) if config.custom_provider.is_some() => {
                    let custom_config = config.custom_provider.as_ref().unwrap();
                    let api_type = config
                        .api_type
                        .unwrap_or_else(|| custom_config.effective_default_api_type());
                    Ok(Arc::new(CustomProvider::from_config(
                        custom_config.clone(),
                        ApiTypeFilter::Exact(api_type),
                    )))
                }

                // Not found + no custom_provider → error
                (None, _) => Err(HiLlmError::ProviderNotFound { name }),
            }
        }

        // 2. No name → check custom_provider, then base_url, then default
        None => {
            if let Some(ref custom_config) = config.custom_provider {
                let api_type = config
                    .api_type
                    .unwrap_or_else(|| custom_config.effective_default_api_type());
                Ok(Arc::new(CustomProvider::from_config(
                    custom_config.clone(),
                    ApiTypeFilter::Exact(api_type),
                )))
            } else if let Some(ref url) = config.base_url {
                // No name + base_url → use OpenAI with URL override
                let provider = if let Some(api_type) = config.api_type {
                    OpenAIProvider::with_api_type(api_type)?
                } else {
                    OpenAIProvider::default()
                };
                Ok(Arc::new(provider::BaseUrlOverride::new(
                    Arc::new(provider),
                    url.clone(),
                )))
            } else {
                Ok(Arc::new(OpenAIProvider::default()))
            }
        }
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait BatchRetriever {
    async fn fetch_batch_for_polling(&self, batch_id: &str) -> HiLlmResult<BatchObject>;
}
