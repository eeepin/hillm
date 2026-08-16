use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service, ServiceExt as _};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLLMError, HiLLMResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryClass {
    Transient,
    Terminal,
}

pub trait RetryPolicy: Send + Sync + 'static {
    fn classify(&self, error: &HiLLMError) -> RetryClass;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultRetryPolicy;

impl RetryPolicy for DefaultRetryPolicy {
    fn classify(&self, error: &HiLLMError) -> RetryClass {
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
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLLMError>
        + Clone
        + Send
        + Sync
        + 'static,
    S::Future: Send + 'static,
    R: RetryPolicy,
{
    type Response = LlmResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LlmResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: LlmRequest) -> Self::Future {
        let chain = Arc::clone(&self.chain);
        let policy = Arc::clone(&self.policy);

        Box::pin(async move {
            let chain_len = chain.len();
            tracing::debug!(chain_len, "fallback chain: starting walk");

            if chain.is_empty() {
                return Err(HiLLMError::ServerError {
                    message: "fallback chain is empty".into(),
                    status: 500,
                });
            }

            let mut last_err: Option<HiLLMError> = None;

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

            Err(last_err.unwrap_or(HiLLMError::ServerError {
                message: "fallback chain exhausted all services".into(),
                status: 503,
            }))
        })
    }
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
    use tower::{Service, ServiceExt};

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

    /// Mock service that tracks calls and can be configured to succeed or fail
    /// with transient or terminal errors.
    #[derive(Clone)]
    struct MockService {
        behavior: MockBehavior,
        call_count: Arc<AtomicU32>,
        label: String,
    }

    #[derive(Clone)]
    enum MockBehavior {
        Success,
        TransientFail,
        TerminalFail,
    }

    impl MockService {
        fn success(label: &str) -> Self {
            Self {
                behavior: MockBehavior::Success,
                call_count: Arc::new(AtomicU32::new(0)),
                label: label.to_string(),
            }
        }

        fn transient_fail(label: &str) -> Self {
            Self {
                behavior: MockBehavior::TransientFail,
                call_count: Arc::new(AtomicU32::new(0)),
                label: label.to_string(),
            }
        }

        fn terminal_fail(label: &str) -> Self {
            Self {
                behavior: MockBehavior::TerminalFail,
                call_count: Arc::new(AtomicU32::new(0)),
                label: label.to_string(),
            }
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl Service<LlmRequest> for MockService {
        type Response = LlmResponse;
        type Error = HiLLMError;
        type Future = BoxFuture<'static, HiLLMResult<LlmResponse>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: LlmRequest) -> Self::Future {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let behavior = self.behavior.clone();
            let label = self.label.clone();
            Box::pin(async move {
                match behavior {
                    MockBehavior::Success => Ok(create_chat_response(&format!("{label} ok"))),
                    MockBehavior::TransientFail => Err(HiLLMError::ServiceUnavailable {
                        message: format!("{label} transient"),
                        status: 503,
                    }),
                    MockBehavior::TerminalFail => Err(HiLLMError::BadRequest {
                        message: format!("{label} terminal"),
                        status: 400,
                    }),
                }
            })
        }
    }

    #[tokio::test]
    async fn chain_first_succeeds() {
        let s1 = MockService::success("s1");
        let s2 = MockService::success("s2");
        let layer = FallbackChainLayer::new(vec![s1.clone(), s2.clone()]);
        let mut service = layer.layer(());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;

        assert!(result.is_ok());
        assert_eq!(s1.call_count(), 1, "first service should be called");
        assert_eq!(s2.call_count(), 0, "second service should not be called");
    }

    #[tokio::test]
    async fn chain_falls_through_transient_to_success() {
        let s1 = MockService::transient_fail("s1");
        let s2 = MockService::success("s2");
        let s3 = MockService::success("s3");
        let layer = FallbackChainLayer::new(vec![s1.clone(), s2.clone(), s3.clone()]);
        let mut service = layer.layer(());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;

        assert!(result.is_ok());
        assert_eq!(s1.call_count(), 1, "s1 should be tried");
        assert_eq!(s2.call_count(), 1, "s2 should be tried after s1 transient");
        assert_eq!(
            s3.call_count(),
            0,
            "s3 should not be tried since s2 succeeded"
        );
    }

    #[tokio::test]
    async fn chain_terminal_error_aborts_immediately() {
        let s1 = MockService::transient_fail("s1");
        let s2 = MockService::terminal_fail("s2");
        let s3 = MockService::success("s3");
        let layer = FallbackChainLayer::new(vec![s1.clone(), s2.clone(), s3.clone()]);
        let mut service = layer.layer(());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;

        assert!(result.is_err(), "terminal error should propagate");
        assert_eq!(s1.call_count(), 1);
        assert_eq!(s2.call_count(), 1, "s2 terminal error should abort chain");
        assert_eq!(s3.call_count(), 0, "s3 should not be tried after terminal");
    }

    #[tokio::test]
    async fn chain_all_transient_returns_last_error() {
        let s1 = MockService::transient_fail("s1");
        let s2 = MockService::transient_fail("s2");
        let layer = FallbackChainLayer::new(vec![s1.clone(), s2.clone()]);
        let mut service = layer.layer(());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;

        assert!(result.is_err(), "all failing should return error");
        assert_eq!(s1.call_count(), 1);
        assert_eq!(s2.call_count(), 1);
    }

    #[tokio::test]
    async fn chain_empty_returns_server_error() {
        let layer: FallbackChainLayer<MockService> = FallbackChainLayer::new(vec![]);
        let mut service = layer.layer(());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn chain_prepend_adds_service_to_front() {
        let s1 = MockService::transient_fail("s1");
        let s2 = MockService::success("s2");
        let s0 = MockService::success("s0");

        let layer = FallbackChainLayer::new(vec![s1.clone(), s2.clone()]).prepend(s0.clone());
        let mut service = layer.layer(());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;

        assert!(result.is_ok());
        assert_eq!(s0.call_count(), 1, "prepended s0 should be tried first");
        assert_eq!(
            s1.call_count(),
            0,
            "s1 should not be tried since s0 succeeded"
        );
    }

    #[test]
    fn default_retry_policy_classifies_transient() {
        let policy = DefaultRetryPolicy;
        let transient = HiLLMError::ServiceUnavailable {
            message: "t".into(),
            status: 503,
        };
        let terminal = HiLLMError::BadRequest {
            message: "t".into(),
            status: 400,
        };
        assert_eq!(policy.classify(&transient), RetryClass::Transient);
        assert_eq!(policy.classify(&terminal), RetryClass::Terminal);
    }
}
