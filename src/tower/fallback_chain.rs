use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service, ServiceExt as _};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryClass {
    Transient,
    Terminal,
}

pub trait RetryPolicy: Send + Sync + 'static {
    fn classify(&self, error: &HiLlmError) -> RetryClass;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultRetryPolicy;

impl RetryPolicy for DefaultRetryPolicy {
    fn classify(&self, error: &HiLlmError) -> RetryClass {
        if error.is_transient() {
            RetryClass::Transient
        } else {
            RetryClass::Terminal
        }
    }
}

pub struct FallbackChainLayer<S, R: RetryPolicy = DefaultRetryPolicy> {
    chain: Arc<Vec<S>>,
    policy: Arc<R>,
}

impl<S> FallbackChainLayer<S, DefaultRetryPolicy> {
    #[must_use]
    pub fn new(chain: Vec<S>) -> Self {
        Self {
            chain: Arc::new(chain),
            policy: Arc::new(DefaultRetryPolicy),
        }
    }
}

impl<S, R: RetryPolicy> FallbackChainLayer<S, R> {
    #[must_use]
    pub fn with_policy(chain: Vec<S>, policy: R) -> Self {
        Self {
            chain: Arc::new(chain),
            policy: Arc::new(policy),
        }
    }
}

impl<S: Clone, R: RetryPolicy> Clone for FallbackChainLayer<S, R> {
    fn clone(&self) -> Self {
        Self {
            chain: Arc::clone(&self.chain),
            policy: Arc::clone(&self.policy),
        }
    }
}

impl<S: Clone, R: RetryPolicy> Layer<()> for FallbackChainLayer<S, R> {
    type Service = FallbackChainService<S, R>;

    fn layer(&self, _inner: ()) -> Self::Service {
        FallbackChainService {
            chain: Arc::clone(&self.chain),
            policy: Arc::clone(&self.policy),
        }
    }
}

impl<S: Clone, R: RetryPolicy> FallbackChainLayer<S, R> {
    #[must_use]
    pub fn prepend(mut self, head: S) -> Self {
        let chain = Arc::make_mut(&mut self.chain);
        chain.insert(0, head);
        self
    }
}

pub struct FallbackChainService<S, R: RetryPolicy = DefaultRetryPolicy> {
    chain: Arc<Vec<S>>,
    policy: Arc<R>,
}

impl<S: Clone, R: RetryPolicy> Clone for FallbackChainService<S, R> {
    fn clone(&self) -> Self {
        Self {
            chain: Arc::clone(&self.chain),
            policy: Arc::clone(&self.policy),
        }
    }
}

impl<S, R> Service<LlmRequest> for FallbackChainService<S, R>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError>
        + Clone
        + Send
        + Sync
        + 'static,
    S::Future: Send + 'static,
    R: RetryPolicy,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: LlmRequest) -> Self::Future {
        let chain = Arc::clone(&self.chain);
        let policy = Arc::clone(&self.policy);

        Box::pin(async move {
            let chain_len = chain.len();
            tracing::debug!(chain_len, "fallback chain: starting walk");

            if chain.is_empty() {
                return Err(HiLlmError::ServerError {
                    message: "fallback chain is empty".into(),
                    status: 500,
                });
            }

            let mut last_err: Option<HiLlmError> = None;

            for (attempt, svc_template) in chain.iter().enumerate() {
                let mut svc = svc_template.clone();
                let span = tracing::debug_span!(
                    "fallback_chain.attempt",
                    chain_len,
                    attempt,
                    outcome = tracing::field::Empty,
                );
                let _guard = span.enter();
                let svc = match svc.ready().await {
                    Ok(s) => s,
                    Err(e) => match policy.classify(&e) {
                        RetryClass::Terminal => {
                            tracing::debug!(
                                attempt,
                                error = %e,
                                "fallback chain: terminal error in poll_ready, aborting"
                            );
                            return Err(e);
                        }
                        RetryClass::Transient => {
                            tracing::warn!(
                                attempt,
                                chain_len,
                                error = %e,
                                "fallback chain: transient error in poll_ready, trying next service"
                            );
                            last_err = Some(e);
                            continue;
                        }
                    },
                };

                match svc.call(request.clone()).await {
                    Ok(resp) => {
                        tracing::debug!(attempt, "fallback chain: success");
                        span.record("outcome", "success");
                        return Ok(resp);
                    }
                    Err(err) => match policy.classify(&err) {
                        RetryClass::Terminal => {
                            tracing::debug!(
                                attempt,
                                error = %err,
                                "fallback chain: terminal error, aborting"
                            );
                            span.record("outcome", "terminal");
                            return Err(err);
                        }
                        RetryClass::Transient => {
                            tracing::warn!(
                                attempt,
                                chain_len,
                                error = %err,
                                "fallback chain: transient error, trying next service"
                            );
                            span.record("outcome", "transient");
                            last_err = Some(err);
                        }
                    },
                }
            }

            Err(last_err.unwrap_or(HiLlmError::ServerError {
                message: "fallback chain exhausted all services".into(),
                status: 503,
            }))
        })
    }
}
