use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

pub trait HedgePolicy: Send + Sync + 'static {
    fn delay_for_attempt(&self, attempt: u32, latency_so_far: Duration) -> Option<Duration>;

    fn max_attempts(&self) -> u32;
}

pub struct FixedDelayHedge {
    delay: Duration,
    max_attempts: u32,
}

impl FixedDelayHedge {
    #[must_use]
    pub fn new(delay: Duration, max_attempts: u32) -> Self {
        Self {
            delay,
            max_attempts: max_attempts.max(1),
        }
    }
}

impl HedgePolicy for FixedDelayHedge {
    fn delay_for_attempt(&self, attempt: u32, _latency_so_far: Duration) -> Option<Duration> {
        if attempt > self.max_attempts {
            return None;
        }
        Some(self.delay * (attempt - 1))
    }

    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

pub struct HedgeLayer<P> {
    policy: Arc<P>,
}

impl<P: HedgePolicy> HedgeLayer<P> {
    #[must_use]
    pub fn new(policy: Arc<P>) -> Self {
        Self { policy }
    }
}

impl<P: HedgePolicy, S> Layer<S> for HedgeLayer<P> {
    type Service = HedgeService<P, S>;

    fn layer(&self, inner: S) -> Self::Service {
        HedgeService {
            inner,
            policy: Arc::clone(&self.policy),
        }
    }
}

pub struct HedgeService<P, S> {
    inner: S,
    policy: Arc<P>,
}

impl<P: HedgePolicy, S: Clone> Clone for HedgeService<P, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            policy: Arc::clone(&self.policy),
        }
    }
}

impl<P, S> Service<LlmRequest> for HedgeService<P, S>
where
    P: HedgePolicy + 'static,
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let policy = Arc::clone(&self.policy);
        let max_attempts = policy.max_attempts();

        let standby = self.inner.clone();
        let primary = std::mem::replace(&mut self.inner, standby);

        let inner_for_hedges = self.inner.clone();

        Box::pin(async move {
            tracing::debug!(hedge.max_attempts = max_attempts, "starting hedged request");
            hedge_race(req, primary, inner_for_hedges, policy, max_attempts).await
        })
    }
}

async fn hedge_race<S>(
    req: LlmRequest,
    mut primary: S,
    inner_for_hedges: S,
    policy: Arc<impl HedgePolicy>,
    max_attempts: u32,
) -> HiLlmResult<LlmResponse>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    use std::time::Instant;

    use tower::ServiceExt as _;

    let dispatch_time = Instant::now();

    if max_attempts == 1 {
        tracing::debug!("hedge fast path: max_attempts=1, calling primary directly");
        return primary.call(req).await;
    }

    let mut join_set: tokio::task::JoinSet<(u32, HiLlmResult<LlmResponse>)> =
        tokio::task::JoinSet::new();

    {
        let req_clone = req.clone();
        join_set.spawn(async move {
            let result = primary.call(req_clone).await;
            (1u32, result)
        });
    }

    for attempt in 2..=max_attempts {
        let latency_so_far = dispatch_time.elapsed();
        let Some(hedge_delay) = policy.delay_for_attempt(attempt, latency_so_far) else {
            break;
        };

        let req_clone = req.clone();
        let mut svc_clone = inner_for_hedges.clone();
        join_set.spawn(async move {
            if hedge_delay > Duration::ZERO {
                tokio::time::sleep(hedge_delay).await;
            }
            tracing::debug!(attempt, "launching hedged request");

            let model = req_clone.model().unwrap_or("").to_owned();
            let system = model
                .split_once('/')
                .map(|(p, _)| p.to_owned())
                .unwrap_or_default();
            super::metrics::record_retry_attempt(&system, &model, req_clone.operation_name());

            let ready_result = svc_clone.ready().await;
            let result = match ready_result {
                Ok(ready_svc) => ready_svc.call(req_clone).await,
                Err(e) => Err(e),
            };
            (attempt, result)
        });
    }

    let mut last_err: Option<HiLlmError> = None;

    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok((attempt, Ok(resp))) => {
                tracing::debug!(attempt, "hedged request succeeded first");
                join_set.abort_all();
                return Ok(resp);
            }
            Ok((attempt, Err(e))) => {
                tracing::debug!(attempt, error = %e, "hedged attempt failed");
                last_err = Some(e);
            }
            Err(join_err) if join_err.is_cancelled() => {}
            Err(join_err) => {
                tracing::error!(error = %join_err, "hedged task panicked");
                last_err = Some(HiLlmError::InternalError {
                    message: format!("hedge task panicked: {join_err}"),
                });
            }
        }
    }

    Err(last_err.unwrap_or(HiLlmError::InternalError {
        message: "all hedged attempts failed with no error recorded".into(),
    }))
}
