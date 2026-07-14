use std::cell::Cell;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::task::{Context, Poll};
use std::time::Instant;

use futures_util::FutureExt as _;
use tower::Layer;
use tower::Service;

use super::cache::CACHE_STATE_CELL;
use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::observability::usage::{
    CacheState, UsageEvent, UsageEventOutcome, UsageSink, UsageSinkErased,
};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

// Hook trait
/// Callback trait
pub trait LlmHook: Send + Sync + 'static {
    /// Called before the request
    fn on_request(
        &self,
        _req: &LlmRequest,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    /// Called after the response
    fn on_response(
        &self,
        _req: &LlmRequest,
        _resp: &LlmResponse,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }

    /// Called when errors occur
    fn on_error(
        &self,
        _req: &LlmRequest,
        _err: &HiLlmError,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}

#[derive(Clone)]
pub struct HooksLayer {
    hooks: Arc<Vec<Arc<dyn LlmHook>>>,
    usage_sink: Option<Arc<dyn UsageSinkErased>>,
    provider: String,
}

impl HooksLayer {
    #[must_use]
    pub fn new(hooks: Vec<Arc<dyn LlmHook>>, provider: impl Into<String>) -> Self {
        Self {
            hooks: Arc::new(hooks),
            usage_sink: None,
            provider: provider.into(),
        }
    }

    #[must_use]
    pub fn single(hook: Arc<dyn LlmHook>, provider: impl Into<String>) -> Self {
        Self::new(vec![hook], provider)
    }

    #[must_use]
    pub fn with_usage_sink<S: UsageSink>(mut self, sink: Arc<S>) -> Self {
        self.usage_sink = Some(sink as Arc<dyn UsageSinkErased>);
        self
    }
}

impl<S> Layer<S> for HooksLayer {
    type Service = HooksService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HooksService {
            inner,
            hooks: Arc::clone(&self.hooks),
            usage_sink: self.usage_sink.clone(),
            provider: self.provider.clone(),
        }
    }
}

pub struct HooksService<S> {
    inner: S,
    hooks: Arc<Vec<Arc<dyn LlmHook>>>,
    usage_sink: Option<Arc<dyn UsageSinkErased>>,
    provider: String,
}

impl<S: Clone> Clone for HooksService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            hooks: Arc::clone(&self.hooks),
            usage_sink: self.usage_sink.clone(),
            provider: self.provider.clone(),
        }
    }
}

impl<S> Service<LlmRequest> for HooksService<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let hooks = Arc::clone(&self.hooks);
        let usage_sink = self.usage_sink.clone();
        let req_clone = req.clone();
        let fut = self.inner.call(req);
        let provider = self.provider.clone();

        Box::pin(async move {
            for hook in hooks.iter() {
                let result = AssertUnwindSafe(hook.on_request(&req_clone))
                    .catch_unwind()
                    .await;
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(e),
                    Err(_panic) => {
                        tracing::error!("hook panicked during on_request");
                        return Err(HiLlmError::HookRejected {
                            message: "hook panicked".into(),
                        });
                    }
                }
            }

            let start = Instant::now();

            let mut cancel_guard = usage_sink.as_ref().map(|s| {
                CancellationGuard::new(Arc::clone(s), req_clone.clone(), start, &provider)
            });

            let (inner_result, cache_state) = CACHE_STATE_CELL
                .scope(Cell::new(CacheState::Bypass), async {
                    let result = fut.await;
                    let state = CACHE_STATE_CELL.with(|c| c.get());
                    (result, state)
                })
                .await;

            match inner_result {
                Ok(resp) => {
                    let latency_ms = start.elapsed().as_millis() as u64;

                    for hook in hooks.iter() {
                        if AssertUnwindSafe(hook.on_response(&req_clone, &resp))
                            .catch_unwind()
                            .await
                            .is_err()
                        {
                            tracing::error!("hook panicked during on_response");
                        }
                    }

                    if let Some(guard) = cancel_guard.take() {
                        guard.disarm();
                    }

                    if let Some(sink) = usage_sink {
                        let event = build_usage_event(
                            &provider,
                            &req_clone,
                            &resp,
                            latency_ms,
                            UsageEventOutcome::Success,
                            cache_state,
                        );
                        tokio::spawn(async move {
                            if let Err(err) = sink.emit_erased(event).await {
                                tracing::warn!(
                                    target: "ai.usage",
                                    error = %err,
                                    "usage sink emit failed"
                                );
                            }
                        });
                    }

                    Ok(resp)
                }
                Err(err) => {
                    let latency_ms = start.elapsed().as_millis() as u64;

                    for hook in hooks.iter() {
                        if AssertUnwindSafe(hook.on_error(&req_clone, &err))
                            .catch_unwind()
                            .await
                            .is_err()
                        {
                            tracing::error!("hook panicked during on_error");
                        }
                    }

                    if let Some(guard) = cancel_guard.take() {
                        guard.disarm();
                    }

                    if let Some(sink) = usage_sink {
                        let outcome = classify_error_outcome(&err);
                        let event = build_error_usage_event(
                            &provider,
                            &req_clone,
                            latency_ms,
                            outcome,
                            cache_state,
                        );
                        tokio::spawn(async move {
                            if let Err(sink_err) = sink.emit_erased(event).await {
                                tracing::warn!(
                                    target: "ai.usage",
                                    error = %sink_err,
                                    "usage sink emit failed on error path"
                                );
                            }
                        });
                    }

                    Err(err)
                }
            }
        })
    }
}

struct CancellationGuard {
    inner: Option<CancellationGuardInner>,
    provider: String,
}

struct CancellationGuardInner {
    sink: Arc<dyn UsageSinkErased>,
    req: LlmRequest,
    start: Instant,
}

impl CancellationGuard {
    fn new(
        sink: Arc<dyn UsageSinkErased>,
        req: LlmRequest,
        start: Instant,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            inner: Some(CancellationGuardInner { sink, req, start }),
            provider: provider.into(),
        }
    }

    fn disarm(mut self) {
        self.inner = None;
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };

        let latency_ms = inner.start.elapsed().as_millis() as u64;
        let event = build_error_usage_event(
            &self.provider,
            &inner.req,
            latency_ms,
            UsageEventOutcome::Cancelled,
            CacheState::Bypass,
        );
        let sink = inner.sink;

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(err) = sink.emit_erased(event).await {
                    tracing::warn!(
                        target: "ai.usage",
                        error = %err,
                        "usage sink emit failed for cancelled request"
                    );
                }
            });
        }
    }
}

// - Helpers -----

fn request_id(req: &LlmRequest) -> String {
    req.idempotency_key.clone().unwrap_or_else(|| {
        REQUEST_COUNTER
            .fetch_add(1, AtomicOrdering::Relaxed)
            .to_string()
    })
}

fn classify_error_outcome(err: &HiLlmError) -> UsageEventOutcome {
    match err {
        HiLlmError::Timeout => UsageEventOutcome::TimedOut,
        _ => UsageEventOutcome::Error,
    }
}

fn effective_model_from_response(resp: &LlmResponse) -> Option<String> {
    match resp {
        LlmResponse::Chat(r) => Some(r.model.clone()),
        LlmResponse::Embed(r) => Some(r.model.clone()),
        LlmResponse::Moderate(r) => Some(r.model.clone()),
        LlmResponse::Ocr(r) => Some(r.model.clone()),
        LlmResponse::Search(r) => Some(r.model.clone()),
        LlmResponse::ChatStream(_)
        | LlmResponse::Speech(_)
        | LlmResponse::Transcribe(_)
        | LlmResponse::Rerank(_)
        | LlmResponse::ListModels(_)
        | LlmResponse::ImageGenerate(_) => None,
    }
}

fn build_usage_event(
    provider: &str,
    req: &LlmRequest,
    resp: &LlmResponse,
    latency_ms: u64,
    outcome: UsageEventOutcome,
    cache_state: CacheState,
) -> UsageEvent {
    let model = req.model().unwrap_or("").to_owned();

    let (prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens, total_tokens) = resp
        .usage()
        .map(|u| {
            let (cached, cache_write) = u.prompt_tokens_details.as_ref().map_or((0, 0), |d| {
                (d.cached_tokens, d.cache_write_tokens.unwrap_or(0))
            });
            (
                u.prompt_tokens,
                u.completion_tokens,
                cached,
                cache_write,
                u.total_tokens,
            )
        })
        .unwrap_or((0, 0, 0, 0, 0));

    let cost = crate::provider::cost::completion_cost_with_cache(
        provider,
        &model,
        prompt_tokens,
        cached_tokens,
        cache_write_tokens,
        completion_tokens,
    )
    .unwrap_or(None)
    .and_then(|f| rust_decimal::Decimal::try_from(f).ok())
    .unwrap_or(rust_decimal::Decimal::ZERO);

    let finish_reason = match resp {
        LlmResponse::Chat(r) => r
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_ref())
            .map(|fr| format!("{fr:?}").to_lowercase()),
        _ => None,
    };

    let effective_model = effective_model_from_response(resp);

    UsageEvent {
        tenant_id: req.tenant_id.clone(),
        request_id: request_id(req),
        model,
        provider: provider.to_string(),
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        total_tokens,
        cost,
        cache_state,
        effective_model,
        finish_reason,
        outcome,
        latency_ms,
        metadata: std::collections::HashMap::new(),
        received_at: std::time::SystemTime::now(),
    }
}

fn build_error_usage_event(
    provider: &str,
    req: &LlmRequest,
    latency_ms: u64,
    outcome: UsageEventOutcome,
    cache_state: CacheState,
) -> UsageEvent {
    let model = req.model().unwrap_or("").to_owned();

    UsageEvent {
        tenant_id: req.tenant_id.clone(),
        request_id: request_id(req),
        model,
        provider: provider.to_string(),
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        total_tokens: 0,
        cost: rust_decimal::Decimal::ZERO,
        cache_state,
        effective_model: None,
        finish_reason: None,
        outcome,
        latency_ms,
        metadata: std::collections::HashMap::new(),
        received_at: std::time::SystemTime::now(),
    }
}
