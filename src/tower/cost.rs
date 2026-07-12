use std::task::{Context, Poll};

use tower::Layer;
use tower::Service;

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::cost;

pub struct CostTrackingLayer {
    provider: String,
}

impl CostTrackingLayer {
    #[must_use]
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
        }
    }
}

impl<S> Layer<S> for CostTrackingLayer {
    type Service = CostTrackingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CostTrackingService {
            inner,
            provider: self.provider.clone(),
        }
    }
}

pub struct CostTrackingService<S> {
    inner: S,
    provider: String,
}

impl<S> Clone for CostTrackingService<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            provider: self.provider.clone(),
        }
    }
}

impl<S> Service<LlmRequest> for CostTrackingService<S>
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
        let model = req.model().map(ToOwned::to_owned);
        let fut = self.inner.call(req);
        let provider = self.provider.clone();

        Box::pin(async move {
            let resp = fut.await?;
            record_cost(&provider, &model, &resp);
            Ok(resp)
        })
    }
}

fn record_cost(provider: &str, model: &Option<String>, resp: &LlmResponse) {
    let Some(model) = model else { return };
    let Some(usage) = resp.usage() else { return };

    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |d| d.cached_tokens);
    let cache_write = usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |d| d.cache_write_tokens.unwrap_or_default());
    if let Ok(Some(cost)) = cost::completion_cost_with_cache(
        provider,
        model,
        usage.prompt_tokens,
        cached,
        cache_write,
        usage.completion_tokens,
    ) {
        tracing::Span::current().record("ai.usage.cost", cost);
    }
}
