use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tower::{Layer, Service};

use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::tower::cache::CachedResponse;
use crate::tower::types::{LlmRequest, LlmRequestKind, LlmResponse};

const IDEM_HASH_SEED_0: u64 = 0x5f72_616e_646f_6d5f; // _random_
const IDEM_HASH_SEED_1: u64 = 0x7676_7669_705f_7631; // vvvip_v1
const IDEM_HASH_SEED_2: u64 = 0x6861_7368_5f6b_6579; // hash_key
const IDEM_HASH_SEED_3: u64 = 0x6865_6c6c_6f6c_6c6d; // hellollm

fn idem_random_state() -> &'static ahash::RandomState {
    use std::sync::OnceLock;
    static STATE: OnceLock<ahash::RandomState> = OnceLock::new();
    STATE.get_or_init(|| {
        ahash::RandomState::generate_with(
            IDEM_HASH_SEED_0,
            IDEM_HASH_SEED_1,
            IDEM_HASH_SEED_2,
            IDEM_HASH_SEED_3,
        )
    })
}

fn compute_body_hash(request: &LlmRequest) -> Option<String> {
    let json = serde_json::to_string(&request.kind).ok()?;

    let h = idem_random_state().hash_one(&json);
    Some(format!("{h:016x}:{}", &json[..json.len().min(64)]))
}

#[derive(Clone)]
pub struct IdempotencyEntry {
    pub body_hash: String,
    pub response: Option<CachedResponse>,
    pub inserted_at: Instant,
    pub ttl: Duration,
}

impl std::fmt::Debug for IdempotencyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdempotencyEntry")
            .field("body_hash", &self.body_hash)
            .field("has_response", &self.response.is_some())
            .field("inserted_at", &self.inserted_at)
            .field("ttl", &self.ttl)
            .finish()
    }
}

impl IdempotencyEntry {
    fn is_expired(&self) -> bool {
        self.inserted_at.elapsed() > self.ttl
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IdempotencyStoreError {
    #[error("idempotency store backend error: {0}")]
    Backend(String),
}

pub trait IdempotencyStore: Send + Sync + 'static {
    fn get<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<IdempotencyEntry>, IdempotencyStoreError>>
                + Send
                + 'a,
        >,
    >;

    fn try_insert<'a>(
        &'a self,
        key: &'a str,
        body_hash: &'a str,
        ttl: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, IdempotencyStoreError>> + Send + 'a>>;

    fn store_response<'a>(
        &'a self,
        key: &'a str,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>>;

    fn remove<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>>;
}

#[derive(Default)]
pub struct InMemoryIdempotencyStore {
    map: DashMap<String, IdempotencyEntry>,
}

impl InMemoryIdempotencyStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl IdempotencyStore for InMemoryIdempotencyStore {
    fn get<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<IdempotencyEntry>, IdempotencyStoreError>>
                + Send
                + 'a,
        >,
    > {
        let result = self.map.get(key).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(entry.clone())
            }
        });
        Box::pin(std::future::ready(Ok(result)))
    }

    fn try_insert<'a>(
        &'a self,
        key: &'a str,
        body_hash: &'a str,
        ttl: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<bool, IdempotencyStoreError>> + Send + 'a>> {
        use dashmap::mapref::entry::Entry;

        let inserted = match self.map.entry(key.to_owned()) {
            Entry::Vacant(slot) => {
                slot.insert(IdempotencyEntry {
                    body_hash: body_hash.to_owned(),
                    response: None,
                    inserted_at: Instant::now(),
                    ttl,
                });
                true
            }
            Entry::Occupied(entry) => {
                if entry.get().is_expired() {
                    entry.replace_entry(IdempotencyEntry {
                        body_hash: body_hash.to_owned(),
                        response: None,
                        inserted_at: Instant::now(),
                        ttl,
                    });
                    true
                } else {
                    false
                }
            }
        };
        Box::pin(std::future::ready(Ok(inserted)))
    }

    fn store_response<'a>(
        &'a self,
        key: &'a str,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>> {
        if let Some(mut entry) = self.map.get_mut(key) {
            entry.response = Some(response);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn remove<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), IdempotencyStoreError>> + Send + 'a>> {
        self.map.remove(key);
        Box::pin(std::future::ready(Ok(())))
    }
}

pub struct IdempotencyLayer<S: IdempotencyStore> {
    store: Arc<S>,
    ttl: Duration,
}

impl<S: IdempotencyStore> IdempotencyLayer<S> {
    #[must_use]
    pub fn new(store: S) -> Self {
        Self::with_ttl(store, Duration::from_secs(24 * 60 * 60))
    }

    #[must_use]
    pub fn with_ttl(store: S, ttl: Duration) -> Self {
        Self {
            store: Arc::new(store),
            ttl,
        }
    }
}

impl<S: IdempotencyStore> Clone for IdempotencyLayer<S> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            ttl: self.ttl,
        }
    }
}

impl<I, S: IdempotencyStore> Layer<I> for IdempotencyLayer<S> {
    type Service = IdempotencyService<I, S>;

    fn layer(&self, inner: I) -> Self::Service {
        IdempotencyService {
            inner,
            store: Arc::clone(&self.store),
            ttl: self.ttl,
        }
    }
}

pub struct IdempotencyService<I, S: IdempotencyStore> {
    inner: I,
    store: Arc<S>,
    ttl: Duration,
}

impl<I: Clone, S: IdempotencyStore> Clone for IdempotencyService<I, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            store: Arc::clone(&self.store),
            ttl: self.ttl,
        }
    }
}

impl<I, S> Service<LlmRequest> for IdempotencyService<I, S>
where
    I: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Clone + Send + 'static,
    I::Future: Send + 'static,
    S: IdempotencyStore,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: LlmRequest) -> Self::Future {
        let standby = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, standby);

        let store = Arc::clone(&self.store);
        let ttl = self.ttl;

        Box::pin(async move {
            let Some(ref raw_key) = request.idempotency_key.clone() else {
                return inner.call(request).await;
            };

            let tenant_prefix = request
                .tenant_id
                .as_ref()
                .map(|t| t.as_ref())
                .unwrap_or("_");
            let key = format!("{tenant_prefix}:{raw_key}");

            let body_hash = match compute_body_hash(&request) {
                Some(h) => h,
                None => {
                    return inner.call(request).await;
                }
            };

            if let Some(entry) = store.get(&key).await.map_err(store_err)? {
                if entry.body_hash != body_hash {
                    return Err(HiLlmError::IdempotencyConflict {
                        key: raw_key.clone(),
                    });
                }
                if let Some(cached) = entry.response {
                    return cached.into_llm_response();
                }
                return Err(HiLlmError::IdempotencyInFlight {
                    key: raw_key.clone(),
                });
            }

            let inserted = store
                .try_insert(&key, &body_hash, ttl)
                .await
                .map_err(store_err)?;

            if !inserted {
                if let Some(entry) = store.get(&key).await.map_err(store_err)? {
                    if entry.body_hash != body_hash {
                        return Err(HiLlmError::IdempotencyConflict {
                            key: raw_key.clone(),
                        });
                    }
                    if let Some(cached) = entry.response {
                        return cached.into_llm_response();
                    }
                    return Err(HiLlmError::IdempotencyInFlight {
                        key: raw_key.clone(),
                    });
                }
            }

            // ── Call inner service ────────────────────────────────────────
            let result = inner.call(request).await;

            match &result {
                Ok(resp) => {
                    let cached = match resp {
                        LlmResponse::Chat(r) => Some(CachedResponse::Chat(r.clone())),
                        LlmResponse::Embed(r) => Some(CachedResponse::Embed(r.clone())),
                        _ => None,
                    };
                    if let Some(cached_resp) = cached {
                        let _ = store.store_response(&key, cached_resp).await;
                    } else {
                        let _ = store.remove(&key).await;
                    }
                }
                Err(_) => {
                    let _ = store.remove(&key).await;
                }
            }

            result
        })
    }
}

#[inline]
fn store_err(e: IdempotencyStoreError) -> HiLlmError {
    HiLlmError::InternalError {
        message: format!("idempotency store: {e}"),
    }
}

#[must_use]
#[allow(dead_code)]
pub(crate) fn is_cacheable_kind(kind: &LlmRequestKind) -> bool {
    matches!(kind, LlmRequestKind::Chat(_) | LlmRequestKind::Embed(_))
}
