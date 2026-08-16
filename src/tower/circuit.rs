use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

impl CircuitState {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Open,
            2 => Self::HalfOpen,
            _ => Self::Closed,
        }
    }
}

enum ProbeGuard<P: CircuitPolicy> {
    None,
    Half(Arc<P>),
}

impl<P: CircuitPolicy> ProbeGuard<P> {
    fn disarm(&mut self) {
        *self = Self::None;
    }
}

impl<P: CircuitPolicy> Drop for ProbeGuard<P> {
    fn drop(&mut self) {
        if let Self::Half(policy) = self {
            policy.release_probe_slot();
        }
    }
}

pub trait CircuitPolicy: Send + Sync + 'static {
    fn record_success(&self);

    fn record_failure(&self);

    fn should_allow(&self) -> bool;

    fn state(&self) -> CircuitState;

    fn release_probe_slot(&self) {}
}

struct CircuitInner {
    state: AtomicU8,
    consecutive_failures: AtomicU32,
    open_since: Mutex<Option<Instant>>,
    probe_in_flight: AtomicBool,
}

pub struct ExponentialBackoffCircuit {
    failure_threshold: u32,
    base_backoff: Duration,
    max_backoff: Duration,
    inner: Arc<CircuitInner>,
    open_count: AtomicU32,
}

impl ExponentialBackoffCircuit {
    #[must_use]
    pub fn new(failure_threshold: u32, base_backoff: Duration) -> Self {
        Self {
            failure_threshold,
            base_backoff,
            max_backoff: Duration::from_secs(120),
            inner: Arc::new(CircuitInner {
                state: AtomicU8::new(CircuitState::Closed as u8),
                consecutive_failures: AtomicU32::new(0),
                open_since: Mutex::new(None),
                probe_in_flight: AtomicBool::new(false),
            }),
            open_count: AtomicU32::new(0),
        }
    }

    fn current_backoff(&self) -> Duration {
        let count = self.open_count.load(Ordering::Relaxed);
        let shift = count.min(62) as u64;
        let factor = 1u64.checked_shl(shift as u32).unwrap_or(u64::MAX);
        let nanos = self.base_backoff.as_nanos().saturating_mul(factor as u128);
        let computed = Duration::from_nanos(nanos.min(u64::MAX as u128) as u64);
        computed.min(self.max_backoff)
    }

    fn maybe_half_open(&self) -> bool {
        let backoff = self.current_backoff();
        let guard = self
            .inner
            .open_since
            .lock()
            .expect("open_since mutex poisoned");
        if let Some(open_at) = *guard
            && open_at.elapsed() >= backoff
        {
            drop(guard);
            self.inner
                .state
                .store(CircuitState::HalfOpen as u8, Ordering::Release);
            tracing::info!(backoff = ?backoff, "circuit breaker entering half-open");
            return true;
        }
        false
    }
}

impl CircuitPolicy for ExponentialBackoffCircuit {
    fn record_success(&self) {
        self.inner.consecutive_failures.store(0, Ordering::Relaxed);
        let prev = self
            .inner
            .state
            .swap(CircuitState::Closed as u8, Ordering::Release);
        // Release the probe slot so a future HalfOpen round can happen.
        self.inner.probe_in_flight.store(false, Ordering::Release);
        if CircuitState::from_u8(prev) != CircuitState::Closed {
            tracing::info!("circuit breaker closed after successful probe");
        }
    }

    fn record_failure(&self) {
        let failures = self
            .inner
            .consecutive_failures
            .fetch_add(1, Ordering::AcqRel)
            + 1;

        let current_u8 = self.inner.state.load(Ordering::Acquire);
        let current = CircuitState::from_u8(current_u8);
        if current == CircuitState::Open {
            return;
        }

        let should_open = failures >= self.failure_threshold || current == CircuitState::HalfOpen;
        if !should_open {
            return;
        }

        let result = self.inner.state.compare_exchange(
            current_u8,
            CircuitState::Open as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        if result.is_ok() {
            let backoff = self.current_backoff();
            let open_count = self.open_count.fetch_add(1, Ordering::Relaxed) + 1;
            {
                let mut guard = self
                    .inner
                    .open_since
                    .lock()
                    .expect("open_since mutex poisoned");
                *guard = Some(Instant::now());
            }
            self.inner.probe_in_flight.store(false, Ordering::Release);
            tracing::warn!(
                consecutive_failures = failures,
                backoff = ?backoff,
                open_count,
                "circuit breaker opened"
            );
        }
    }

    fn should_allow(&self) -> bool {
        match CircuitState::from_u8(self.inner.state.load(Ordering::Acquire)) {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if self.maybe_half_open() {
                    self.inner
                        .probe_in_flight
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => self
                .inner
                .probe_in_flight
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
        }
    }

    fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.inner.state.load(Ordering::Acquire))
    }

    fn release_probe_slot(&self) {
        self.inner.probe_in_flight.store(false, Ordering::Release);
    }
}

pub struct CircuitLayer<P> {
    policy: Arc<P>,
    provider: String,
}

impl<P: CircuitPolicy> CircuitLayer<P> {
    #[must_use]
    pub fn new(policy: Arc<P>, provider: impl Into<String>) -> Self {
        Self {
            policy,
            provider: provider.into(),
        }
    }
}

impl<P: CircuitPolicy, S> Layer<S> for CircuitLayer<P> {
    type Service = CircuitService<P, S>;

    fn layer(&self, inner: S) -> Self::Service {
        CircuitService {
            inner,
            policy: Arc::clone(&self.policy),
            provider: self.provider.clone(),
        }
    }
}

pub struct CircuitService<P, S> {
    inner: S,
    policy: Arc<P>,
    provider: String,
}

impl<P: CircuitPolicy, S: Clone> Clone for CircuitService<P, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            policy: Arc::clone(&self.policy),
            provider: self.provider.clone(),
        }
    }
}

impl<P, S> Service<LlmRequest> for CircuitService<P, S>
where
    P: CircuitPolicy + 'static,
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
        let provider = self.provider.clone();
        let model = req.model().unwrap_or("").to_owned();
        let system = model
            .split_once('/')
            .map(|(p, _)| p.to_owned())
            .unwrap_or_default();
        let state = self.policy.state();

        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        Box::pin(async move {
            let (allowed, is_probe) = {
                let _span = tracing::debug_span!(
                    "circuit_breaker",
                    gen_ai.circuit.state = ?state,
                    provider = %provider,
                )
                .entered();
                let probe = state != CircuitState::Closed;
                (policy.should_allow(), probe)
            };

            if !allowed {
                tracing::debug!(provider = %provider, "circuit open -- rejecting request");

                super::metrics::record_circuit_trip(&system, &model);

                return Err(HiLlmError::ServiceUnavailable {
                    message: format!("circuit breaker open for provider '{provider}'"),
                    status: 503,
                });
            }
            let mut probe_guard: ProbeGuard<P> = if is_probe {
                ProbeGuard::Half(Arc::clone(&policy))
            } else {
                ProbeGuard::None
            };

            tracing::debug!(provider = %provider, state = ?policy.state(), "circuit allowing request through");

            match inner.call(req).await {
                Ok(resp) => {
                    probe_guard.disarm();
                    policy.record_success();
                    Ok(resp)
                }
                Err(e) => {
                    if e.is_transient() {
                        probe_guard.disarm();
                        policy.record_failure();
                    } else {
                        probe_guard.disarm();
                        if is_probe {
                            policy.release_probe_slot();
                        }
                    }
                    Err(e)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::types::{LlmRequest, LlmResponse};
    use crate::types::{AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, Message, MessageContent, Usage};
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};
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

    #[derive(Clone)]
    struct MockService {
        should_fail: bool,
    }

    impl MockService {
        fn new(should_fail: bool) -> Self {
            Self { should_fail }
        }
    }

    impl Service<LlmRequest> for MockService {
        type Response = LlmResponse;
        type Error = HiLlmError;
        type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: LlmRequest) -> Self::Future {
            let should_fail = self.should_fail;
            Box::pin(async move {
                if should_fail {
                    Err(HiLlmError::ServiceUnavailable {
                        message: "mock failure".to_string(),
                        status: 503,
                    })
                } else {
                    Ok(create_chat_response("test response"))
                }
            })
        }
    }

    #[tokio::test]
    async fn circuit_starts_closed() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(3, Duration::from_secs(1)));
        assert_eq!(policy.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn circuit_opens_after_threshold_failures() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(3, Duration::from_secs(1)));
        let mock = MockService::new(true);
        let service = CircuitService {
            inner: mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        // First 2 failures should not open circuit
        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let result = svc.ready().await.unwrap().call(req).await;
            assert!(result.is_err());
            assert_eq!(policy.state(), CircuitState::Closed, "Circuit should still be closed after {} failures", i + 1);
        }

        // Third failure should open circuit
        let mut svc = service.clone();
        let req = create_chat_request("test 3");
        let result = svc.ready().await.unwrap().call(req).await;
        assert!(result.is_err());
        assert_eq!(policy.state(), CircuitState::Open, "Circuit should be open after 3 failures");
    }

    #[tokio::test]
    async fn circuit_rejects_requests_when_open() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(2, Duration::from_secs(10)));
        let mock = MockService::new(true);
        let service = CircuitService {
            inner: mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        // Open the circuit
        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
        }
        assert_eq!(policy.state(), CircuitState::Open);

        // Next request should be rejected immediately
        let mut svc = service.clone();
        let req = create_chat_request("test");
        let result = svc.ready().await.unwrap().call(req).await;
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("circuit breaker open"));
        }
    }

    #[tokio::test]
    async fn circuit_enters_half_open_after_backoff() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(2, Duration::from_millis(100)));
        let mock = MockService::new(true);
        let service = CircuitService {
            inner: mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        // Open the circuit
        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
        }
        assert_eq!(policy.state(), CircuitState::Open);

        // Wait for backoff period (open_count=1 means backoff is base_backoff * 2^1 = 200ms)
        sleep(Duration::from_millis(250)).await;

        // Circuit should transition to HalfOpen on next check
        assert!(policy.should_allow(), "Circuit should allow probe request after backoff");
        assert_eq!(policy.state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn circuit_closes_after_successful_probe() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(2, Duration::from_millis(100)));

        // Start with failing service to open circuit
        let failing_mock = MockService::new(true);
        let service = CircuitService {
            inner: failing_mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
        }
        assert_eq!(policy.state(), CircuitState::Open);

        // Wait for backoff (open_count=1 means backoff is base_backoff * 2^1 = 200ms)
        sleep(Duration::from_millis(250)).await;

        // Now use successful service for probe
        let success_mock = MockService::new(false);
        let service = CircuitService {
            inner: success_mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        let mut svc = service.clone();
        let req = create_chat_request("probe");
        let result = svc.ready().await.unwrap().call(req).await;
        assert!(result.is_ok());
        assert_eq!(policy.state(), CircuitState::Closed, "Circuit should close after successful probe");
    }

    #[tokio::test]
    async fn circuit_reopens_after_failed_probe() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(2, Duration::from_millis(100)));
        let mock = MockService::new(true);
        let service = CircuitService {
            inner: mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        // Open the circuit
        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
        }
        assert_eq!(policy.state(), CircuitState::Open);

        // Wait for backoff
        sleep(Duration::from_millis(150)).await;

        // Probe should fail and reopen circuit
        let mut svc = service.clone();
        let req = create_chat_request("probe");
        let result = svc.ready().await.unwrap().call(req).await;
        assert!(result.is_err());

        // Circuit should be open again
        assert_eq!(policy.state(), CircuitState::Open, "Circuit should reopen after failed probe");
    }

    #[tokio::test]
    async fn circuit_success_resets_failure_count() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(3, Duration::from_secs(1)));

        // Start with failing service
        let failing_mock = MockService::new(true);
        let service = CircuitService {
            inner: failing_mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        // 2 failures (not enough to open)
        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
        }

        // One success should reset failure count
        let success_mock = MockService::new(false);
        let service = CircuitService {
            inner: success_mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        let mut svc = service.clone();
        let req = create_chat_request("success");
        let result = svc.ready().await.unwrap().call(req).await;
        assert!(result.is_ok());

        // Now we need 3 more failures to open circuit
        let failing_mock = MockService::new(true);
        let service = CircuitService {
            inner: failing_mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
            assert_eq!(policy.state(), CircuitState::Closed, "Circuit should still be closed after {} failures post-reset", i + 1);
        }

        // Third failure should open
        let mut svc = service.clone();
        let req = create_chat_request("test 3");
        let _ = svc.ready().await.unwrap().call(req).await;
        assert_eq!(policy.state(), CircuitState::Open, "Circuit should open after 3 failures post-reset");
    }

    #[tokio::test]
    async fn circuit_half_open_allows_only_one_probe() {
        let policy = Arc::new(ExponentialBackoffCircuit::new(2, Duration::from_millis(100)));
        let mock = MockService::new(true);
        let service = CircuitService {
            inner: mock,
            policy: policy.clone(),
            provider: "test".to_string(),
        };

        // Open the circuit
        for i in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            let _ = svc.ready().await.unwrap().call(req).await;
        }

        // Wait for backoff (open_count=1 means backoff is base_backoff * 2^1 = 200ms)
        sleep(Duration::from_millis(250)).await;

        // First probe should be allowed
        assert!(policy.should_allow());
        assert_eq!(policy.state(), CircuitState::HalfOpen);

        // Second probe should be rejected (probe slot taken)
        assert!(!policy.should_allow(), "Second probe should be rejected");
    }
}
