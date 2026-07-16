use std::sync::{Arc, Mutex};

use tower::{Layer, Service};

use super::config::ClientConfig;
use super::{
    BatchClient, BoxFuture, BoxStream, DefaultClient, FileClient, LlmClient, ResponseClient,
};
use crate::error::{HiLlmError, HiLlmResult};
#[cfg(feature = "opendal")]
use crate::tower::OpenDalCacheStore;
use crate::tower::types::{LlmRequest, LlmResponse};
use crate::tower::{
    BudgetLayer, BudgetState, CacheBackend, CacheLayer, CooldownLayer, CostTrackingLayer,
    HealthCheckLayer, HooksLayer, LlmService, ModelRateLimitLayer, TracingLayer,
};
use crate::types::audio::{CreateSpeechRequest, CreateTranscriptionRequest, TranscriptionResponse};
use crate::types::batch::{BatchListQuery, BatchListResponse, BatchObject, CreateBatchRequest};
use crate::types::file::{
    CreateFileRequest, DeleteResponse, FileListQuery, FileListResponse, FileObject,
};
use crate::types::image::{CreateImageRequest, ImagesResponse};
use crate::types::moderation::{ModerationRequest, ModerationResponse};
use crate::types::ocr::{OcrRequest, OcrResponse};
use crate::types::rerank::{RerankRequest, RerankResponse};
use crate::types::response::{CreateResponseRequest, ResponseObject};
use crate::types::search::{SearchRequest, SearchResponse};
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, ModelsListResponse,
};

struct SyncService {
    inner: Mutex<tower::util::BoxCloneService<LlmRequest, LlmResponse, HiLlmError>>,
}

impl SyncService {
    fn clone_service(&self) -> tower::util::BoxCloneService<LlmRequest, LlmResponse, HiLlmError> {
        match self.inner.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

pub struct ManagedClient {
    inner: Arc<DefaultClient>,
    service: Option<SyncService>,
    budget_state: Option<Arc<BudgetState>>,
}

impl ManagedClient {
    pub fn new(config: ClientConfig, provider: Option<String>) -> HiLlmResult<Self> {
        let client = DefaultClient::new(config.clone(), provider.clone())?;
        let inner = Arc::new(client);

        let (service, budget_state) =
            build_service_stack(&config, Arc::clone(&inner), provider.unwrap_or_default());

        Ok(Self {
            inner,
            service,
            budget_state,
        })
    }

    #[must_use]
    pub fn inner(&self) -> &DefaultClient {
        &self.inner
    }

    #[must_use]
    pub fn budget_state(&self) -> Option<&Arc<BudgetState>> {
        self.budget_state.as_ref()
    }

    #[must_use]
    pub fn has_middleware(&self) -> bool {
        self.service.is_some()
    }

    fn call_service(&self, req: LlmRequest) -> BoxFuture<'static, HiLlmResult<LlmResponse>> {
        let mut svc = match self.service.as_ref() {
            Some(s) => s.clone_service(),
            None => {
                return Box::pin(async {
                    Err(HiLlmError::InternalError {
                        message: "call_service called without middleware stack".into(),
                    })
                });
            }
        };
        Box::pin(async move { svc.call(req).await })
    }
}

fn build_service_stack(
    config: &ClientConfig,
    client: Arc<DefaultClient>,
    provider: impl Into<String>,
) -> (Option<SyncService>, Option<Arc<BudgetState>>) {
    let has_cache = config.cache_config.is_some();
    let has_budget = config.budget_config.is_some();
    let has_hooks = !config.hooks.is_empty();
    let has_cooldown = config.cooldown_duration.is_some();
    let has_rate_limit = config.rate_limit_config.is_some();
    let has_health_check = config.health_check_interval.is_some();
    let has_cost = config.enable_cost_tracking;
    let has_tracing = config.enable_tracing;
    let provider = provider.into();

    if !has_cache
        && !has_budget
        && !has_hooks
        && !has_cooldown
        && !has_rate_limit
        && !has_health_check
        && !has_cost
        && !has_tracing
    {
        return (None, None);
    }

    let base = LlmService::new_from_arc(client);

    let mut budget_state: Option<Arc<BudgetState>> = None;

    type Bcs = tower::util::BoxCloneService<LlmRequest, LlmResponse, HiLlmError>;

    let svc: Bcs = tower::util::BoxCloneService::new(base);

    let svc = if let Some(ref cache_cfg) = config.cache_config {
        let layer = if let Some(ref store) = config.cache_store {
            CacheLayer::with_store(Arc::clone(store))
        } else {
            match &cache_cfg.backend {
                CacheBackend::Memory => CacheLayer::new(cache_cfg.clone()),
                #[cfg(feature = "opendal")]
                CacheBackend::OpenDal {
                    scheme,
                    config: backend_config,
                } => {
                    match OpenDalCacheStore::from_config(
                        &scheme,
                        backend_config.clone(),
                        "llm-cache/",
                        cache_cfg.ttl,
                    ) {
                        Ok(store) => CacheLayer::with_store(Arc::new(store)),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create OpenDAL cache store, falling back to in-memory: {e}"
                            );
                            CacheLayer::new(cache_cfg.clone())
                        }
                    }
                }
            }
        };
        tower::util::BoxCloneService::new(layer.layer(svc))
    } else {
        svc
    };

    let svc = if let Some(interval) = config.health_check_interval {
        let layer = HealthCheckLayer::new(interval);
        tower::util::BoxCloneService::new(layer.layer(svc))
    } else {
        svc
    };

    let svc = if let Some(duration) = config.cooldown_duration {
        let layer = CooldownLayer::new(duration);
        tower::util::BoxCloneService::new(layer.layer(svc))
    } else {
        svc
    };

    let svc = if let Some(ref rl_cfg) = config.rate_limit_config {
        let layer = ModelRateLimitLayer::new(rl_cfg.clone());
        tower::util::BoxCloneService::new(layer.layer(svc))
    } else {
        svc
    };

    let svc = if has_cost {
        tower::util::BoxCloneService::new(CostTrackingLayer::new(provider.clone()).layer(svc))
    } else {
        svc
    };

    let svc = if let Some(ref budget_cfg) = config.budget_config {
        let state = Arc::new(BudgetState::new());
        budget_state = Some(Arc::clone(&state));
        let layer = BudgetLayer::new(budget_cfg.clone(), state, provider.clone());
        tower::util::BoxCloneService::new(layer.layer(svc))
    } else {
        svc
    };

    let svc = if has_hooks {
        let layer = HooksLayer::new(config.hooks.clone(), provider.clone());
        tower::util::BoxCloneService::new(layer.layer(svc))
    } else {
        svc
    };

    let svc = if has_tracing {
        tower::util::BoxCloneService::new(TracingLayer.layer(svc))
    } else {
        svc
    };

    (
        Some(SyncService {
            inner: Mutex::new(svc),
        }),
        budget_state,
    )
}

impl LlmClient for ManagedClient {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>> {
        if self.service.is_none() {
            return self.inner.chat(req);
        }
        let fut = self.call_service(LlmRequest::Chat(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Chat(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Chat response, got {other:?}"),
                }),
            }
        })
    }

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>> {
        if self.service.is_none() {
            return self.inner.chat_stream(req);
        }
        let fut = self.call_service(LlmRequest::ChatStream(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::ChatStream(s) => Ok(s),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected ChatStream response, got {other:?}"),
                }),
            }
        })
    }

    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>> {
        if self.service.is_none() {
            return self.inner.embed(req);
        }
        let fut = self.call_service(LlmRequest::Embed(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Embed(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Embed response, got {other:?}"),
                }),
            }
        })
    }

    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>> {
        if self.service.is_none() {
            return self.inner.list_models();
        }
        let fut = self.call_service(LlmRequest::ListModels());
        Box::pin(async move {
            match fut.await? {
                LlmResponse::ListModels(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected ListModels response, got {other:?}"),
                }),
            }
        })
    }

    fn image_generate(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<ImagesResponse>> {
        if self.service.is_none() {
            return self.inner.image_generate(req);
        }
        let fut = self.call_service(LlmRequest::ImageGenerate(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::ImageGenerate(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected ImageGenerate response, got {other:?}"),
                }),
            }
        })
    }

    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>> {
        if self.service.is_none() {
            return self.inner.speech(req);
        }
        let fut = self.call_service(LlmRequest::Speech(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Speech(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Speech response, got {other:?}"),
                }),
            }
        })
    }

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>> {
        if self.service.is_none() {
            return self.inner.transcribe(req);
        }
        let fut = self.call_service(LlmRequest::Transcribe(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Transcribe(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Transcribe response, got {other:?}"),
                }),
            }
        })
    }

    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>> {
        if self.service.is_none() {
            return self.inner.moderate(req);
        }
        let fut = self.call_service(LlmRequest::Moderate(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Moderate(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Moderate response, got {other:?}"),
                }),
            }
        })
    }

    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>> {
        if self.service.is_none() {
            return self.inner.rerank(req);
        }
        let fut = self.call_service(LlmRequest::Rerank(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Rerank(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Rerank response, got {other:?}"),
                }),
            }
        })
    }

    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>> {
        if self.service.is_none() {
            return self.inner.search(req);
        }
        let fut = self.call_service(LlmRequest::Search(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Search(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Search response, got {other:?}"),
                }),
            }
        })
    }

    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>> {
        if self.service.is_none() {
            return self.inner.ocr(req);
        }
        let fut = self.call_service(LlmRequest::Ocr(req));
        Box::pin(async move {
            match fut.await? {
                LlmResponse::Ocr(r) => Ok(r),
                other => Err(HiLlmError::InternalError {
                    message: format!("expected Ocr response, got {other:?}"),
                }),
            }
        })
    }
}

impl FileClient for ManagedClient {
    fn create_file(&self, req: CreateFileRequest) -> BoxFuture<'_, HiLlmResult<FileObject>> {
        self.inner.create_file(req)
    }

    fn retrieve_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<FileObject>> {
        self.inner.retrieve_file(file_id)
    }

    fn delete_file(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<DeleteResponse>> {
        self.inner.delete_file(file_id)
    }

    fn list_files(
        &self,
        query: Option<FileListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<FileListResponse>> {
        self.inner.list_files(query)
    }

    fn file_content(&self, file_id: &str) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>> {
        self.inner.file_content(file_id)
    }
}

impl BatchClient for ManagedClient {
    fn create_batch(&self, req: CreateBatchRequest) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        self.inner.create_batch(req)
    }

    fn retrieve_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        self.inner.retrieve_batch(batch_id)
    }

    fn list_batches(
        &self,
        query: Option<BatchListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<BatchListResponse>> {
        self.inner.list_batches(query)
    }

    fn cancel_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        self.inner.cancel_batch(batch_id)
    }
}

impl ResponseClient for ManagedClient {
    fn create_response(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLlmResult<ResponseObject>> {
        self.inner.create_response(req)
    }

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>> {
        self.inner.retrieve_response(response_id)
    }

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLlmResult<ResponseObject>> {
        self.inner.cancel_response(response_id)
    }
}
