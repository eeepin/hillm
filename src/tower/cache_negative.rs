use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tower::{Layer, Service};

use super::cache::{CacheStore, CachedResponse, InMemoryStore, hash_key};
use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

pub trait NegativeCachePolicy: Send + Sync + 'static {
    fn cache_for(&self, error: &HiLlmError) -> Option<Duration>;
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
    fn cache_for(&self, error: &HiLlmError) -> Option<Duration> {
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

impl<P, S> Service<LlmRequest> for NegativeCacheService<P, S>
where
    P: NegativeCachePolicy,
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
                    error: Arc::new(HiLlmError::InternalError {
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
