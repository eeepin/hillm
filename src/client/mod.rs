pub mod builder;
pub mod config;
pub mod config_file;
#[cfg(all(feature = "default-http", feature = "tower"))]
pub mod managed;

use std::future::Future;
use std::pin::Pin;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use std::sync::Arc;
use std::time::Duration;

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
use crate::types::response::{CreateResponseRequest, ResponseObject};
use crate::types::search::{SearchRequest, SearchResponse};

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::auth::Credential;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::http;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::provider::{
    self, Provider, openai::OpenAiProvider, openai_compatible::OpenAiCompatibleProvider,
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
fn str_pair(pair: &(String, String)) -> (&str, &str) {
    (pair.0.as_str(), pair.1.as_str())
}

/// The LLM Client trait
#[cfg(not(target_arch = "wasm32"))]
pub trait LlmClient: Send + Sync {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>>;

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>;

    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>>;

    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>>;

    fn image_generate(&self, req: CreateImageRequest)
    -> BoxFuture<'_, HiLlmResult<ImagesResponse>>;

    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>>;

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>>;

    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>>;

    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>>;

    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>>;

    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>>;
}

#[cfg(target_arch = "wasm32")]
pub trait LlmClient {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>>;

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>;

    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>>;

    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>>;

    fn image_generate(&self, req: CreateImageRequest)
    -> BoxFuture<'_, HiLlmResult<ImagesResponse>>;

    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>>;

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>>;

    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>>;

    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>>;

    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>>;

    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>>;
}

pub trait LlmClientRaw: LlmClient {
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

    fn embed_raw(
        &self,
        req: EmbeddingRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<EmbeddingResponse>>>;

    fn image_generate_raw(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ImagesResponse>>>;

    fn transcribe_raw(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<TranscriptionResponse>>>;

    fn moderate_raw(
        &self,
        req: ModerationRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ModerationResponse>>>;

    fn rerank_raw(
        &self,
        req: RerankRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<RerankResponse>>>;

    fn search_raw(
        &self,
        req: SearchRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<SearchResponse>>>;

    fn ocr_raw(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<RawExchange<OcrResponse>>>;
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

    fn retrieve_batch(&self, batch_id: &str) -> BoxFuture<'_, ReHiLlmResultsult<BatchObject>>;

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

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;
}

#[cfg(target_arch = "wasm32")]
pub trait ResponseClient {
    fn create_response(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>>;
}

/// Default client based on `reqwest`.
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[derive(Clone)]
pub struct DefaultClient {
    config: ClientConfig,
    http_client: reqwest::Client,
    provider: Arc<dyn Provider>,
    cached_auth_header: Option<(String, String)>,
    cached_extra_headers: Vec<(String, String)>,
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl DefaultClient {
    pub fn new(config: ClientConfig, provider: Option<String>) -> HiLlmResult<Self> {
        let provider = build_provider(&config, provider);

        provider.validate()?;

        #[cfg(not(target_arch = "wasm32"))]
        let mut config = config;
        #[cfg(not(target_arch = "wasm32"))]
        if config.load_env
            && config.api_key.expose_secret().is_empty()
            && let Some(env_var_name) = provider.env_var()
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
                if !matches!(
                    crate::provider::current_policy(),
                    crate::provider::OutboundPolicy::Off
                ) {
                    builder.dns_resolver(crate::provider::outbound_policy::guarded_resolver())
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
            provider: provider,
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
fn build_provider(config: &ClientConfig, provider_name: Option<String>) -> Arc<dyn Provider> {
    if let Some(ref base_url) = config.base_url {
        // TODO: make different special provider match
        return Arc::new(OpenAiCompatibleProvider {
            name: "custom".into(),
            base_url: base_url.clone(),
            env_var: None,
            models: vec![],
        });
    }

    if let Some(name) = provider_name
        && let Some(p) = provider::get_provider(&name)
    {
        return Arc::from(p);
    }

    Arc::new(OpenAiProvider)
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl LlmClient for DefaultClient {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.chat_completions_path(), &req.model, Some(false))?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<ChatCompletionResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.chat_completions_path(), &req.model, Some(true))?;

            let url = prepared
                .provider
                .build_stream_url(prepared.provider.chat_completions_path(), &req.model);

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            match prepared.provider.stream_format() {
                provider::StreamFormat::Sse => {
                    let provider = Arc::clone(&prepared.provider);
                    let parse_event = move |data: &str| provider.parse_stream_event(data);
                    let stream = http::stream::post_stream(
                        &self.http_client,
                        &url,
                        auth,
                        &extra,
                        prepared.body_bytes,
                        self.config.max_retries,
                        parse_event,
                    )
                    .await?;
                    Ok(stream)
                }
            }
        })
    }

    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.embeddings_path(), &req.model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<EmbeddingResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>> {
        Box::pin(async move {
            let url = self.provider.build_url(self.provider.models_path(), "");
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let mut raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            self.provider.transform_response(&mut raw)?;
            serde_json::from_value::<ModelsListResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn image_generate(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<ImagesResponse>> {
        Box::pin(async move {
            let model = req.model.as_deref().unwrap_or_default();
            let prepared =
                self.prepare_request(&req, |p| p.image_generations_path(), model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<ImagesResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.audio_speech_path(), &req.model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            http::request::post_binary(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await
        })
    }

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.audio_transcriptions_path(), &req.model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<TranscriptionResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>> {
        Box::pin(async move {
            let model = req.model.as_deref().unwrap_or_default();
            let prepared = self.prepare_request(&req, |p| p.moderations_path(), model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<ModerationResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.rerank_path(), &req.model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<RerankResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.search_path(), &req.model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<SearchResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.ocr_path(), &req.model, None)?;

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;
            prepared.provider.transform_response(&mut raw)?;
            serde_json::from_value::<OcrResponse>(raw).map_err(HiLlmError::from)
        })
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl LlmClientRaw for DefaultClient {
    fn chat_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ChatCompletionResponse>>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.chat_completions_path(), &req.model, Some(false))?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data =
                serde_json::from_value::<ChatCompletionResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn chat_stream_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<
        '_,
        HiLlmResult<RawStreamExchange<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>,
    > {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.chat_completions_path(), &req.model, Some(true))?;
            let raw_request = prepared.body_json.clone();
            let url = prepared
                .provider
                .build_stream_url(prepared.provider.chat_completions_path(), &req.model);

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let stream = match prepared.provider.stream_format() {
                provider::StreamFormat::Sse => {
                    let provider = Arc::clone(&prepared.provider);
                    let parse_event = move |data: &str| provider.parse_stream_event(data);
                    http::stream::post_stream(
                        &self.http_client,
                        &url,
                        auth,
                        &extra,
                        prepared.body_bytes,
                        self.config.max_retries,
                        parse_event,
                    )
                    .await?
                }
            };

            Ok(RawStreamExchange {
                stream,
                raw_request,
            })
        })
    }

    fn embed_raw(
        &self,
        req: EmbeddingRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<EmbeddingResponse>>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.embeddings_path(), &req.model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data =
                serde_json::from_value::<EmbeddingResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn image_generate_raw(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ImagesResponse>>> {
        Box::pin(async move {
            let model = req.model.as_deref().unwrap_or_default();
            let prepared =
                self.prepare_request(&req, |p| p.image_generations_path(), model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data = serde_json::from_value::<ImagesResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn transcribe_raw(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<TranscriptionResponse>>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.audio_transcriptions_path(), &req.model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data =
                serde_json::from_value::<TranscriptionResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn moderate_raw(
        &self,
        req: ModerationRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ModerationResponse>>> {
        Box::pin(async move {
            let model = req.model.as_deref().unwrap_or_default();
            let prepared = self.prepare_request(&req, |p| p.moderations_path(), model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data =
                serde_json::from_value::<ModerationResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn rerank_raw(
        &self,
        req: RerankRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<RerankResponse>>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.rerank_path(), &req.model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data = serde_json::from_value::<RerankResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn search_raw(
        &self,
        req: SearchRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<SearchResponse>>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.search_path(), &req.model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data = serde_json::from_value::<SearchResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn ocr_raw(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<RawExchange<OcrResponse>>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.ocr_path(), &req.model, None)?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data = serde_json::from_value::<OcrResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl FileClient for DefaultClient {
    fn create_file(&self, req: CreateFileRequest) -> BoxFuture<'_, HiLlmResult<FileObject>> {
        Box::pin(async move {
            let url = self.provider.build_url(self.provider.files_path(), "");
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("POST", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            use base64::Engine;
            let file_bytes = base64::engine::general_purpose::STANDARD
                .decode(&req.file)
                .map_err(|e| HiLlmError::BadRequest {
                    message: format!("invalid base64 file data: {e}"),
                    status: 400,
                })?;

            let filename = req.filename.unwrap_or_else(|| "upload".to_owned());
            let file_part = reqwest::multipart::Part::bytes(file_bytes).file_name(filename);
            let purpose_str = serde_json::to_value(&req.purpose)?
                .as_str()
                .unwrap_or_default()
                .to_owned();
            let form = reqwest::multipart::Form::new()
                .part("file", file_part)
                .text("purpose", purpose_str);

            let raw =
                http::request::post_multipart(&self.http_client, &url, auth, &extra, form).await?;
            serde_json::from_value::<FileObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn retrieve_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<FileObject>> {
        let file_id = file_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}",
                self.provider.build_url(self.provider.files_path(), ""),
                file_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<FileObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn delete_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<DeleteResponse>> {
        let file_id = file_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}",
                self.provider.build_url(self.provider.files_path(), ""),
                file_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("DELETE", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::delete_json(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<DeleteResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn list_files(
        &self,
        query: Option<FileListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<FileListResponse>> {
        Box::pin(async move {
            let base_url = self.provider.build_url(self.provider.files_path(), "");
            let url = if let Some(ref q) = query {
                let mut params = Vec::new();
                if let Some(ref purpose) = q.purpose {
                    params.push(format!("purpose={purpose}"));
                }
                if let Some(limit) = q.limit {
                    params.push(format!("limit={limit}"));
                }
                if let Some(ref after) = q.after {
                    params.push(format!("after={after}"));
                }
                if params.is_empty() {
                    base_url
                } else {
                    format!("{base_url}?{}", params.join("&"))
                }
            } else {
                base_url
            };
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<FileListResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn file_content(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>> {
        let file_id = file_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}/content",
                self.provider.build_url(self.provider.files_path(), ""),
                file_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            http::request::get_binary(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await
        })
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl BatchClient for DefaultClient {
    fn create_batch(&self, req: CreateBatchRequest) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        Box::pin(async move {
            let url = self.provider.build_url(self.provider.batches_path(), "");
            let body_bytes = bytes::Bytes::from(serde_json::to_vec(&req)?);
            let body_json = serde_json::to_value(&req)?;

            let auth_header = self.resolve_auth_header().await?;
            let all_headers = self.all_headers("POST", &url, &body_json, &body_bytes);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let raw = http::request::post_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn retrieve_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        let batch_id = batch_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}",
                self.provider.build_url(self.provider.batches_path(), ""),
                batch_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn list_batches(
        &self,
        query: Option<BatchListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<BatchListResponse>> {
        Box::pin(async move {
            let base_url = self.provider.build_url(self.provider.batches_path(), "");
            let url = if let Some(ref q) = query {
                let mut params = Vec::new();
                if let Some(limit) = q.limit {
                    params.push(format!("limit={limit}"));
                }
                if let Some(ref after) = q.after {
                    params.push(format!("after={after}"));
                }
                if params.is_empty() {
                    base_url
                } else {
                    format!("{base_url}?{}", params.join("&"))
                }
            } else {
                base_url
            };
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchListResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn cancel_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        let batch_id = batch_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}/cancel",
                self.provider.build_url(self.provider.batches_path(), ""),
                batch_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let body_json = serde_json::Value::Null;
            let body_bytes = bytes::Bytes::new();
            let all_headers = self.all_headers("POST", &url, &body_json, &body_bytes);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let raw = http::request::post_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchObject>(raw).map_err(HiLlmError::from)
        })
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait BatchRetriever {
    async fn fetch_batch_for_polling(&self, batch_id: &str) -> HiLlmResult<BatchObject>;
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl BatchRetriever for DefaultClient {
    async fn fetch_batch_for_polling(&self, batch_id: &str) -> HiLlmResult<BatchObject> {
        self.retrieve_batch(batch_id).await
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub async fn wait_for_batch_impl<R: BatchRetriever>(
    retriever: &R,
    batch_id: &str,
    config: WaitForBatchConfig,
) -> Result<BatchObject, BatchWaitError> {
    #[cfg(not(target_arch = "wasm32"))]
    let started = tokio::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    let started = web_time::Instant::now();
    let mut interval_secs = config.initial_interval_secs;

    loop {
        let batch = retriever.fetch_batch_for_polling(batch_id).await?;

        match batch.status {
            BatchStatus::Completed => return Ok(batch),
            BatchStatus::Failed | BatchStatus::Expired | BatchStatus::Cancelled => {
                return Err(BatchWaitError::Failed {
                    status: batch.status,
                });
            }
            BatchStatus::Validating
            | BatchStatus::InProgress
            | BatchStatus::Finalizing
            | BatchStatus::Cancelling => {
                if let Some(timeout_secs) = config.timeout_secs {
                    let timeout = Duration::from_secs_f64(timeout_secs);
                    if started.elapsed() >= timeout {
                        return Err(BatchWaitError::Timeout { timeout_secs });
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                tokio::time::sleep(Duration::from_secs_f64(interval_secs)).await;
                #[cfg(target_arch = "wasm32")]
                gloo_timers::future::sleep(Duration::from_secs_f64(interval_secs)).await;
                let next = (interval_secs as f32 * config.backoff_multiplier)
                    .min(config.max_interval_secs as f32) as f64;
                interval_secs = next;
            }
        }
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl DefaultClient {
    pub async fn wait_for_batch(
        &self,
        batch_id: &str,
        config: WaitForBatchConfig,
    ) -> std::result::Result<BatchObject, BatchWaitError> {
        wait_for_batch_impl(self, batch_id, config).await
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ResponseClient for DefaultClient {
    fn create_response(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<ResponseObject>> {
        Box::pin(async move {
            let url = self.provider.build_url(self.provider.responses_path(), "");
            let body_bytes = bytes::Bytes::from(serde_json::to_vec(&req)?);
            let body_json = serde_json::to_value(&req)?;

            let auth_header = self.resolve_auth_header().await?;
            let all_headers = self.all_headers("POST", &url, &body_json, &body_bytes);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let raw = http::request::post_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<ResponseObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>> {
        let response_id = response_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}",
                self.provider.build_url(self.provider.responses_path(), ""),
                response_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<ResponseObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>> {
        let response_id = response_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}/cancel",
                self.provider.build_url(self.provider.responses_path(), ""),
                response_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let body_json = serde_json::Value::Null;
            let body_bytes = bytes::Bytes::new();
            let all_headers = self.all_headers("POST", &url, &body_json, &body_bytes);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let raw = http::request::post_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<ResponseObject>(raw).map_err(HiLlmError::from)
        })
    }
}
