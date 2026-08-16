use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tower::{Layer, Service};

use super::cache::{CacheStore, CachedResponse, InMemoryStore, hash_key};
use super::types::{LLMRequest, LLMResponse};
use crate::client::BoxFuture;
use crate::error::{HiLLMError, HiLLMResult};

pub trait NegativeCachePolicy: Send + Sync + 'static {
    fn cache_for(&self, error: &HiLLMError) -> Option<Duration>;
}

pub struct FixedWindowNegativeCache {
    window: Duration,
    retryable_only: bool,
}

impl FixedWindowNegativeCache {
    #[must_use]
    pub fn new(window: Duration, retryable_only: bool) -> Self {
        Self {
            window,
            retryable_only,
        }
    }
}

impl Default for FixedWindowNegativeCache {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(5),
            retryable_only: true,
        }
    }
}

impl NegativeCachePolicy for FixedWindowNegativeCache {
    fn cache_for(&self, error: &HiLLMError) -> Option<Duration> {
        let eligible = if self.retryable_only {
            error.is_transient()
        } else {
            true
        };
        eligible.then_some(self.window)
    }
}

pub struct NegativeCacheLayer<P: NegativeCachePolicy = FixedWindowNegativeCache> {
    store: Arc<dyn CacheStore>,
    policy: Arc<P>,
}

impl NegativeCacheLayer<FixedWindowNegativeCache> {
    #[must_use]
    pub fn default_in_memory() -> Self {
        use crate::tower::cache::CacheConfig;
        Self {
            store: Arc::new(InMemoryStore::new(&CacheConfig::default())),
            policy: Arc::new(FixedWindowNegativeCache::default()),
        }
    }
}

impl Default for NegativeCacheLayer<FixedWindowNegativeCache> {
    fn default() -> Self {
        Self::default_in_memory()
    }
}

impl<P: NegativeCachePolicy> NegativeCacheLayer<P> {
    #[must_use]
    pub fn new(store: Arc<dyn CacheStore>, policy: Arc<P>) -> Self {
        Self { store, policy }
    }
}

impl<P: NegativeCachePolicy, S> Layer<S> for NegativeCacheLayer<P> {
    type Service = NegativeCacheService<P, S>;

    fn layer(&self, inner: S) -> Self::Service {
        NegativeCacheService {
            store: Arc::clone(&self.store),
            policy: Arc::clone(&self.policy),
            inner,
        }
    }
}

pub struct NegativeCacheService<P: NegativeCachePolicy, S> {
    store: Arc<dyn CacheStore>,
    policy: Arc<P>,
    inner: S,
}

impl<P: NegativeCachePolicy, S: Clone> Clone for NegativeCacheService<P, S> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            policy: Arc::clone(&self.policy),
            inner: self.inner.clone(),
        }
    }
}

impl<P, S> Service<LLMRequest> for NegativeCacheService<P, S>
where
    P: NegativeCachePolicy,
    S: Service<LLMRequest, Response = LLMResponse, Error = HiLLMError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LLMResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LLMResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LLMRequest) -> Self::Future {
        let key_and_body = hash_key(&req);
        let store = Arc::clone(&self.store);
        let policy = Arc::clone(&self.policy);
        let fut = self.inner.call(req);

        Box::pin(async move {
            let result = fut.await;
            if let Err(ref err) = result
                && let Some(window) = policy.cache_for(err)
                && let Some((key, body)) = key_and_body
            {
                let expires_at = Instant::now() + window;
                let cached_err = CachedResponse::Error {
                    error: Arc::new(HiLLMError::InternalError {
                        message: err.to_string(),
                    }),
                    expires_at,
                };
                store.put(key, body, cached_err).await;
            }
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::types::{LLMRequest, LLMResponse};
    use crate::types::{
        AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, Message,
        MessageContent, Usage,
    };
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::task::{Context, Poll};
    use tokio::time::Duration;
    use tower::{Service, ServiceExt};

    fn create_chat_request(content: &str) -> LLMRequest {
        LLMRequest {
            kind: crate::tower::types::LLMRequestKind::Chat(ChatCompletionRequest {
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

    fn create_chat_response() -> LLMResponse {
        LLMResponse::Chat(ChatCompletionResponse {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "test-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage {
                    content: Some(MessageContent::Text("ok".to_string())),
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

    impl Service<LLMRequest> for MockService {
        type Response = LLMResponse;
        type Error = HiLLMError;
        type Future = BoxFuture<'static, HiLLMResult<LLMResponse>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: LLMRequest) -> Self::Future {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let should_fail = self.should_fail;
            Box::pin(async move {
                if should_fail {
                    Err(HiLLMError::ServiceUnavailable {
                        message: "transient".to_string(),
                        status: 503,
                    })
                } else {
                    Ok(create_chat_response())
                }
            })
        }
    }

    #[test]
    fn fixed_window_policy_caches_transient_errors() {
        let policy = FixedWindowNegativeCache::new(Duration::from_secs(10), true);
        let transient = HiLLMError::ServiceUnavailable {
            message: "t".into(),
            status: 503,
        };
        assert_eq!(policy.cache_for(&transient), Some(Duration::from_secs(10)));
    }

    #[test]
    fn fixed_window_policy_skips_terminal_when_retryable_only() {
        let policy = FixedWindowNegativeCache::new(Duration::from_secs(10), true);
        let terminal = HiLLMError::BadRequest {
            message: "t".into(),
            status: 400,
        };
        assert_eq!(policy.cache_for(&terminal), None);
    }

    #[test]
    fn fixed_window_policy_caches_all_when_not_retryable_only() {
        let policy = FixedWindowNegativeCache::new(Duration::from_secs(10), false);
        let terminal = HiLLMError::BadRequest {
            message: "t".into(),
            status: 400,
        };
        assert_eq!(policy.cache_for(&terminal), Some(Duration::from_secs(10)));
    }

    #[test]
    fn default_policy_is_retryable_only_with_five_second_window() {
        let policy = FixedWindowNegativeCache::default();
        assert_eq!(policy.window, Duration::from_secs(5));
        assert!(policy.retryable_only);
    }

    #[tokio::test]
    async fn negative_cache_does_not_affect_success() {
        let mock = MockService::new(false);
        let layer = NegativeCacheLayer::default_in_memory();
        let mut service = layer.layer(mock.clone());

        let req = create_chat_request("hello");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;
        assert!(result.is_ok());
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn negative_cache_stores_transient_error() {
        let mock = MockService::new(true);
        let layer = NegativeCacheLayer::default_in_memory();
        let store = Arc::clone(&layer.store);
        let mut service = layer.layer(mock.clone());

        let req = create_chat_request("will fail");
        let result = ServiceExt::ready(&mut service)
            .await
            .unwrap()
            .call(req)
            .await;
        assert!(result.is_err());

        // The error should have been cached.
        let (key, body) = hash_key(&create_chat_request("will fail")).unwrap();
        let cached = store.get(key, &body).await;
        assert!(cached.is_some(), "transient error should be cached");
    }
}
