use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLLMError, HiLLMResult};

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
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLLMError> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
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
) -> HiLLMResult<LlmResponse>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLLMError> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    use std::time::Instant;

    use tower::ServiceExt as _;

    let dispatch_time = Instant::now();

    if max_attempts == 1 {
        tracing::debug!("hedge fast path: max_attempts=1, calling primary directly");
        return primary.call(req).await;
    }

    let mut join_set: tokio::task::JoinSet<(u32, HiLLMResult<LlmResponse>)> =
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

    let mut last_err: Option<HiLLMError> = None;

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
                last_err = Some(HiLLMError::InternalError {
                    message: format!("hedge task panicked: {join_err}"),
                });
            }
        }
    }

    Err(last_err.unwrap_or(HiLLMError::InternalError {
        message: "all hedged attempts failed with no error recorded".into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::types::{LlmRequest, LlmResponse};
    use crate::types::{
        AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, Message,
        MessageContent, Usage,
    };
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::task::{Context, Poll};
    use tokio::time::Duration;
    use tower::ServiceExt;

    fn create_chat_request(content: &str) -> LlmRequest {
        LlmRequest {
            kind: crate::tower::types::LlmRequestKind::Chat(ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![Message::User(crate::types::UserMessage {
                    content: MessageContent::Text(content.to_string()),
                    name: None,
                })],
                ..Default::default()
            }),
            tenant_id: None,
            idempotency_key: None,
        }
    }

    fn create_chat_response(content: &str) -> LlmResponse {
        LlmResponse::Chat(ChatCompletionResponse {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage {
                    content: Some(MessageContent::Text(content.to_string())),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Mock service that succeeds after a configurable delay.
    #[derive(Clone)]
    struct DelayedMockService {
        delay: Duration,
        should_fail: bool,
        call_count: Arc<AtomicU32>,
    }

    impl DelayedMockService {
        fn new(delay: Duration, should_fail: bool) -> Self {
            Self {
                delay,
                should_fail,
                call_count: Arc::new(AtomicU32::new(0)),
            }
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl Service<LlmRequest> for DelayedMockService {
        type Response = LlmResponse;
        type Error = HiLLMError;
        type Future = BoxFuture<'static, HiLLMResult<LlmResponse>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: LlmRequest) -> Self::Future {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let delay = self.delay;
            let should_fail = self.should_fail;
            Box::pin(async move {
                if delay > Duration::ZERO {
                    tokio::time::sleep(delay).await;
                }
                if should_fail {
                    Err(HiLLMError::ServiceUnavailable {
                        message: "mock fail".to_string(),
                        status: 503,
                    })
                } else {
                    Ok(create_chat_response("ok"))
                }
            })
        }
    }

    #[tokio::test]
    async fn hedge_fast_path_single_attempt() {
        // max_attempts=1 should call primary directly without spawning hedges.
        let mock = DelayedMockService::new(Duration::ZERO, false);
        let policy = Arc::new(FixedDelayHedge::new(Duration::from_millis(10), 1));
        let layer = HedgeLayer::new(policy);
        let mut service = layer.layer(mock.clone());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;
        assert!(result.is_ok());
        assert_eq!(mock.call_count(), 1, "only one call for single attempt");
    }

    #[tokio::test]
    async fn hedge_returns_first_success() {
        // All attempts succeed quickly with a fast inner service.
        let policy = Arc::new(FixedDelayHedge::new(Duration::from_millis(10), 3));
        let layer = HedgeLayer::new(policy);
        let fast = DelayedMockService::new(Duration::ZERO, false);
        let mut service = layer.layer(fast.clone());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;
        assert!(result.is_ok());
        // At least one call was made.
        assert!(fast.call_count() >= 1);
    }

    #[tokio::test]
    async fn hedge_all_fail_returns_error() {
        let mock = DelayedMockService::new(Duration::ZERO, true);
        let policy = Arc::new(FixedDelayHedge::new(Duration::from_millis(5), 3));
        let layer = HedgeLayer::new(policy);
        let mut service = layer.layer(mock.clone());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;
        assert!(result.is_err(), "all failing should return error");
        // At least some attempts should have been made.
        assert!(
            mock.call_count() >= 1,
            "at least one attempt should be made"
        );
    }

    #[test]
    fn fixed_delay_hedge_clamps_max_attempts_to_one() {
        let policy = FixedDelayHedge::new(Duration::from_millis(10), 0);
        assert_eq!(
            policy.max_attempts(),
            1,
            "max_attempts(0) should clamp to 1"
        );
    }

    #[test]
    fn fixed_delay_hedge_delay_scales_with_attempt() {
        let policy = FixedDelayHedge::new(Duration::from_millis(100), 5);
        assert_eq!(
            policy.delay_for_attempt(1, Duration::ZERO),
            Some(Duration::ZERO)
        );
        assert_eq!(
            policy.delay_for_attempt(2, Duration::ZERO),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            policy.delay_for_attempt(3, Duration::ZERO),
            Some(Duration::from_millis(200))
        );
        assert_eq!(policy.delay_for_attempt(6, Duration::ZERO), None);
    }
}
