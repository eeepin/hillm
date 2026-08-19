use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

struct CooldownState {
    cooldown_start: Option<Instant>,
}

pub struct CooldownLayer {
    duration: Duration,
}

impl CooldownLayer {
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

impl<S> Layer<S> for CooldownLayer {
    type Service = CooldownService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CooldownService {
            inner,
            duration: self.duration,
            state: Arc::new(RwLock::new(CooldownState {
                cooldown_start: None,
            })),
        }
    }
}

pub struct CooldownService<S> {
    inner: S,
    duration: Duration,
    state: Arc<RwLock<CooldownState>>,
}

impl<S: Clone> Clone for CooldownService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            duration: self.duration,
            state: Arc::clone(&self.state),
        }
    }
}

impl<S> Service<LlmRequest> for CooldownService<S>
where
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
        let state = Arc::clone(&self.state);
        let duration = self.duration;
        let mut inner = self.inner.clone();

        Box::pin(async move {
            {
                let read = state.read().await;
                if let Some(start) = read.cooldown_start {
                    if start.elapsed() < duration {
                        return Err(HiLlmError::ServiceUnavailable {
                            message: format!(
                                "service is cooling down for {:.0}s after a transient error",
                                duration.as_secs_f64()
                            ),
                            status: 503,
                        });
                    }
                    drop(read);
                    let mut write = state.write().await;
                    if let Some(s) = write.cooldown_start
                        && s.elapsed() >= duration
                    {
                        write.cooldown_start = None;
                    }
                }
            }

            match inner.call(req).await {
                Ok(resp) => Ok(resp),
                Err(e) if e.is_transient() => {
                    // Enter cooldown.
                    let mut write = state.write().await;
                    write.cooldown_start = Some(Instant::now());
                    Err(e)
                }
                Err(e) => Err(e),
            }
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
    use tokio::time::{Duration, sleep};
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
        should_fail_transient: bool,
        call_count: Arc<AtomicU32>,
    }

    impl MockService {
        fn new(should_fail_transient: bool) -> Self {
            Self {
                should_fail_transient,
                call_count: Arc::new(AtomicU32::new(0)),
            }
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
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
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let should_fail = self.should_fail_transient;
            Box::pin(async move {
                if should_fail {
                    Err(HiLlmError::ServiceUnavailable {
                        message: "transient".to_string(),
                        status: 503,
                    })
                } else {
                    Ok(create_chat_response("ok"))
                }
            })
        }
    }

    #[tokio::test]
    async fn cooldown_success_passes_through() {
        let mock = MockService::new(false);
        let layer = CooldownLayer::new(Duration::from_millis(100));
        let mut service = layer.layer(mock.clone());

        let req = create_chat_request("hello");
        let result = service.ready().await.unwrap().call(req).await;
        assert!(result.is_ok());
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn cooldown_transient_error_enters_cooldown() {
        let mock = MockService::new(true);
        let layer = CooldownLayer::new(Duration::from_millis(200));
        let mut service = layer.layer(mock.clone());

        // First call: inner fails transiently, enters cooldown.
        let req = create_chat_request("hello");
        let result = service.ready().await.unwrap().call(req).await;
        assert!(result.is_err());
        assert_eq!(mock.call_count(), 1);

        // Second call: should be rejected with 503 without calling inner.
        let req = create_chat_request("hello again");
        let result = service.ready().await.unwrap().call(req).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cooling down"),
            "expected cooling down error, got: {err_msg}"
        );
        assert_eq!(
            mock.call_count(),
            1,
            "inner should not be called during cooldown"
        );
    }

    #[tokio::test]
    async fn cooldown_expires_and_allows_retry() {
        let mock = MockService::new(true);
        let layer = CooldownLayer::new(Duration::from_millis(50));
        let mut service = layer.layer(mock.clone());

        // First call: enters cooldown.
        let req = create_chat_request("hello");
        let _ = service.ready().await.unwrap().call(req).await;
        assert_eq!(mock.call_count(), 1);

        // Wait for cooldown to expire.
        sleep(Duration::from_millis(100)).await;

        // Next call: should reach inner again.
        let req = create_chat_request("retry");
        let _ = service.ready().await.unwrap().call(req).await;
        assert_eq!(
            mock.call_count(),
            2,
            "inner should be called after cooldown expires"
        );
    }

    #[tokio::test]
    async fn cooldown_terminal_error_does_not_enter_cooldown() {
        // Inner that fails with a terminal (non-transient) error.
        #[derive(Clone)]
        struct TerminalFailService;

        impl Service<LlmRequest> for TerminalFailService {
            type Response = LlmResponse;
            type Error = HiLlmError;
            type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _req: LlmRequest) -> Self::Future {
                Box::pin(async move {
                    Err(HiLlmError::BadRequest {
                        message: "terminal".to_string(),
                        status: 400,
                    })
                })
            }
        }

        let mock = TerminalFailService;
        let layer = CooldownLayer::new(Duration::from_millis(100));
        let mut service = layer.layer(mock);

        // First call: terminal error, should not enter cooldown.
        let req = create_chat_request("hello");
        let _ = service.ready().await.unwrap().call(req).await;

        // Second call: should reach inner again (no cooldown).
        let req = create_chat_request("retry");
        let result = service.ready().await.unwrap().call(req).await;
        assert!(result.is_err(), "terminal error should propagate");
        assert!(
            !result.unwrap_err().to_string().contains("cooling down"),
            "should not be in cooldown after terminal error"
        );
    }
}
