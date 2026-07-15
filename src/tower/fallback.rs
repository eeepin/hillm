use std::task::{Context, Poll};

use tower::Layer;
use tower::Service;

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

pub struct FallbackLayer<F> {
    fallback: F,
}

impl<F> FallbackLayer<F> {
    #[must_use]
    pub fn new(fallback: F) -> Self {
        Self { fallback }
    }
}

impl<S, F> Layer<S> for FallbackLayer<F>
where
    F: Clone,
{
    type Service = FallbackService<S, F>;

    fn layer(&self, primary: S) -> Self::Service {
        FallbackService {
            primary,
            fallback: self.fallback.clone(),
        }
    }
}

pub struct FallbackService<S, F> {
    primary: S,
    fallback: F,
}

impl<S, F> Clone for FallbackService<S, F>
where
    S: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            primary: self.primary.clone(),
            fallback: self.fallback.clone(),
        }
    }
}

impl<S, F> Service<LlmRequest> for FallbackService<S, F>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + 'static,
    S::Future: Send + 'static,
    F: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Clone + Send + 'static,
    F::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        match self.primary.poll_ready(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
            Poll::Ready(Ok(())) => {}
        }
        self.fallback.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let fallback_req = req.clone();
        let primary_fut = self.primary.call(req);

        let fresh = self.fallback.clone();
        let mut readied_fallback = std::mem::replace(&mut self.fallback, fresh);

        Box::pin(async move {
            match primary_fut.await {
                Ok(resp) => Ok(resp),
                Err(e) if e.is_transient() => {
                    tracing::warn!(
                        error = %e,
                        "primary service failed with transient error; trying fallback"
                    );
                    readied_fallback.call(fallback_req).await
                }
                Err(e) => Err(e),
            }
        })
    }
}
