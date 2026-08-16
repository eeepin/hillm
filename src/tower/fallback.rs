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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::types::{LlmRequest, LlmResponse};
    use crate::types::{
        AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, Message,
        MessageContent, Usage,
    };
    use std::sync::Arc;
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

    #[derive(Clone)]
    struct MockService {
        should_fail: bool,
        call_count: Arc<AtomicU32>,
    }

    impl MockService {
        fn new(should_fail: bool) -> Self {
            Self {
                should_fail,
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
            let should_fail = self.should_fail;
            Box::pin(async move {
                if should_fail {
                    Err(HiLlmError::ServiceUnavailable {
                        message: "mock transient failure".to_string(),
                        status: 503,
                    })
                } else {
                    Ok(create_chat_response("mock response"))
                }
            })
        }
    }

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
                    message: "terminal failure".to_string(),
                    status: 400,
                })
            })
        }
    }

    #[tokio::test]
    async fn fallback_primary_success_skips_fallback() {
        let primary = MockService::new(false);
        let fallback = MockService::new(false);

        let layer = FallbackLayer::new(fallback.clone());
        let mut service = layer.layer(primary.clone());

        let req = create_chat_request("hello");
        let result = service.ready().await.unwrap().call(req).await;

        assert!(result.is_ok(), "should succeed");
        assert_eq!(primary.call_count(), 1, "primary should be called once");
        assert_eq!(fallback.call_count(), 0, "fallback should not be called");
    }

    #[tokio::test]
    async fn fallback_primary_transient_error_triggers_fallback() {
        let primary = MockService::new(true); // fails transiently
        let fallback = MockService::new(false);

        let layer = FallbackLayer::new(fallback.clone());
        let mut service = layer.layer(primary.clone());

        let req = create_chat_request("hello");
        let result = service.ready().await.unwrap().call(req).await;

        assert!(result.is_ok(), "fallback should succeed");
        assert_eq!(primary.call_count(), 1, "primary should be called once");
        assert_eq!(fallback.call_count(), 1, "fallback should be called once");
    }

    #[tokio::test]
    async fn fallback_primary_terminal_error_skips_fallback() {
        let primary = TerminalFailService;
        let fallback = MockService::new(false);

        let layer = FallbackLayer::new(fallback.clone());
        let mut service = layer.layer(primary);

        let req = create_chat_request("hello");
        let result = service.ready().await.unwrap().call(req).await;

        assert!(result.is_err(), "terminal error should propagate");
        assert_eq!(
            fallback.call_count(),
            0,
            "fallback should not be called for terminal errors"
        );
    }

    #[tokio::test]
    async fn fallback_both_fail_returns_fallback_error() {
        let primary = MockService::new(true); // transient
        let fallback = MockService::new(true); // also transient

        let layer = FallbackLayer::new(fallback.clone());
        let mut service = layer.layer(primary.clone());

        let req = create_chat_request("hello");
        let result = service.ready().await.unwrap().call(req).await;

        assert!(result.is_err(), "both failing should return error");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
    }
}
