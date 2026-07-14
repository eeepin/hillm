use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tower::{Layer, Service};

use super::hash::{ExactHashStrategy, HashKeyInput, HashKeyStrategy};
use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::BoxFuture;
use crate::embedding::EmbeddingProvider;
use crate::error::{HiLlmError, HiLlmResult};
use crate::observability::usage::CacheState;
use crate::tower::cache_policy::{
    CacheDecision, CachePolicy, CachePolicyContext, StandardCachePolicy,
};
use crate::types::{ChatCompletionResponse, EmbeddingResponse};
use crate::vectorstore::VectorStore;

tokio::task_local! {
    pub static CACHE_STATE_CELL: Cell<CacheState>;
}

pub fn record_cache_state(state: CacheState) {
    let _ = CACHE_STATE_CELL.try_with(|c| c.set(state));
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheBackend {
    #[default]
    Memory,
    #[cfg(feature = "opendal")]
    OpenDal {
        scheme: String,
        config: std::collections::HashMap<String, String>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub ttl: Duration,
    pub backend: CacheBackend,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            ttl: Duration::from_secs(300),
            backend: CacheBackend::Memory,
        }
    }
}

#[derive(Clone, Debug)]
pub enum CachedResponse {
    Chat(ChatCompletionResponse),
    Embed(EmbeddingResponse),
    Error {
        error: Arc<HiLlmError>,
        expires_at: Instant,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CachedResponseRepr {
    Chat(ChatCompletionResponse),
    Embed(EmbeddingResponse),
}

impl Serialize for CachedResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Chat(r) => CachedResponseRepr::Chat(r.clone()).serialize(serializer),
            Self::Embed(r) => CachedResponseRepr::Embed(r.clone()).serialize(serializer),
            Self::Error { .. } => Err(serde::ser::Error::custom(
                "CachedResponse::Error is not serialisable; convert to a serialisable form before writing to an external store",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for CachedResponse {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        match CachedResponseRepr::deserialize(deserializer)? {
            CachedResponseRepr::Chat(r) => Ok(Self::Chat(r)),
            CachedResponseRepr::Embed(r) => Ok(Self::Embed(r)),
        }
    }
}

impl CachedResponse {
    pub fn into_llm_response(self) -> HiLlmResult<LlmResponse> {
        match self {
            Self::Chat(r) => Ok(LlmResponse::Chat(r)),
            Self::Embed(r) => Ok(LlmResponse::Embed(r)),
            Self::Error { error, .. } => {
                Err(
                    Arc::try_unwrap(error).unwrap_or_else(|arc| HiLlmError::InternalError {
                        message: arc.to_string(),
                    }),
                )
            }
        }
    }

    #[must_use]
    pub fn is_expired_error(&self) -> bool {
        matches!(self, Self::Error { expires_at, .. } if Instant::now() >= *expires_at)
    }
}

#[derive(Debug, Clone)]
pub struct CacheMetadata {
    pub inserted_at: Instant,
    pub ttl: Duration,
    pub size_bytes: usize,
    pub hit_count: u64,
}

pub trait CacheStore: Send + Sync + 'static {
    fn get(
        &self,
        key: u64,
        request_body: &str,
    ) -> Pin<Box<dyn Future<Output = Option<CachedResponse>> + Send + '_>>;

    fn put(
        &self,
        key: u64,
        request_body: String,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    fn remove(&self, key: u64) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    fn set_ttl(&self, _key: u64, _ttl: Duration) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(std::future::ready(()))
    }

    fn iter_keys(&self) -> Pin<Box<dyn Future<Output = Vec<u64>> + Send + '_>> {
        Box::pin(std::future::ready(Vec::new()))
    }

    fn metadata(
        &self,
        _key: u64,
    ) -> Pin<Box<dyn Future<Output = Option<CacheMetadata>> + Send + '_>> {
        Box::pin(std::future::ready(None))
    }
}

#[derive(Clone)]
struct CacheEntry {
    request_body: String,
    response: CachedResponse,
    inserted_at: Instant,
    ttl_override: Option<Duration>,
    hit_count: u64,
    size_bytes: usize,
}

struct InnerCache {
    map: HashMap<u64, CacheEntry>,
    order: VecDeque<u64>,
    max_entries: usize,
    ttl: Duration,
}

impl InnerCache {
    fn new(config: &CacheConfig) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_entries: config.max_entries,
            ttl: config.ttl,
        }
    }

    fn effective_ttl(&self, entry: &CacheEntry) -> Duration {
        entry.ttl_override.unwrap_or(self.ttl)
    }

    fn get_if_valid(&self, key: u64, request_body: &str) -> Option<CachedResponse> {
        let entry = self.map.get(&key)?;
        if entry.request_body != request_body {
            return None;
        }
        let is_expired = match &entry.response {
            CachedResponse::Error { expires_at, .. } => Instant::now() >= *expires_at,
            _ => entry.inserted_at.elapsed() > self.effective_ttl(entry),
        };
        if is_expired {
            return None;
        }
        Some(entry.response.clone())
    }

    fn remove_expired(&mut self, key: u64) {
        let ttl = self.ttl;
        let expired = self.map.get(&key).is_some_and(|e| {
            let eff = e.ttl_override.unwrap_or(ttl);
            match &e.response {
                CachedResponse::Error { expires_at, .. } => Instant::now() >= *expires_at,
                _ => e.inserted_at.elapsed() > eff,
            }
        });
        if expired {
            self.map.remove(&key);
        }
    }

    fn insert(&mut self, key: u64, request_body: String, response: CachedResponse) {
        if self.map.contains_key(&key) {
            self.order.retain(|k| *k != key);
        }

        while self.map.len() >= self.max_entries {
            if let Some(oldest_key) = self.order.pop_front() {
                self.map.remove(&oldest_key);
            } else {
                break;
            }
        }

        let size_bytes = serde_json::to_string(&response)
            .map(|s| s.len())
            .unwrap_or(0);
        self.map.insert(
            key,
            CacheEntry {
                request_body,
                response,
                inserted_at: Instant::now(),
                ttl_override: None,
                hit_count: 0,
                size_bytes,
            },
        );
        self.order.push_back(key);
    }

    fn record_hit(&mut self, key: u64) {
        if let Some(entry) = self.map.get_mut(&key) {
            entry.hit_count = entry.hit_count.saturating_add(1);
        }
    }
}

pub struct InMemoryStore {
    inner: RwLock<InnerCache>,
}

impl InMemoryStore {
    #[must_use]
    pub fn new(config: &CacheConfig) -> Self {
        Self {
            inner: RwLock::new(InnerCache::new(config)),
        }
    }
}

impl CacheStore for InMemoryStore {
    fn get(
        &self,
        key: u64,
        request_body: &str,
    ) -> Pin<Box<dyn Future<Output = Option<CachedResponse>> + Send + '_>> {
        let hit = self.inner.write().ok().and_then(|mut cache| {
            let hit = cache.get_if_valid(key, request_body);
            if hit.is_none() {
                cache.remove_expired(key);
            } else {
                cache.record_hit(key);
            }
            hit
        });
        Box::pin(std::future::ready(hit))
    }

    fn put(
        &self,
        key: u64,
        request_body: String,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        if let Ok(mut cache) = self.inner.write() {
            cache.insert(key, request_body, response);
        }
        Box::pin(std::future::ready(()))
    }

    fn remove(&self, key: u64) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        if let Ok(mut cache) = self.inner.write() {
            cache.map.remove(&key);
        }
        Box::pin(std::future::ready(()))
    }

    fn set_ttl(&self, key: u64, ttl: Duration) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        if let Ok(mut cache) = self.inner.write()
            && let Some(entry) = cache.map.get_mut(&key)
        {
            entry.ttl_override = Some(ttl);
        }
        Box::pin(std::future::ready(()))
    }

    fn iter_keys(&self) -> Pin<Box<dyn Future<Output = Vec<u64>> + Send + '_>> {
        let keys = self
            .inner
            .read()
            .map(|cache| cache.map.keys().copied().collect())
            .unwrap_or_default();
        Box::pin(std::future::ready(keys))
    }

    fn metadata(
        &self,
        key: u64,
    ) -> Pin<Box<dyn Future<Output = Option<CacheMetadata>> + Send + '_>> {
        let result = self.inner.read().ok().and_then(|cache| {
            let entry = cache.map.get(&key)?;
            Some(CacheMetadata {
                inserted_at: entry.inserted_at,
                ttl: cache.effective_ttl(entry),
                size_bytes: entry.size_bytes,
                hit_count: entry.hit_count,
            })
        });
        Box::pin(std::future::ready(result))
    }
}

pub struct CacheLayer {
    store: Arc<dyn CacheStore>,
    key_strategy: Arc<dyn HashKeyStrategy>,
    cache_policy: Arc<dyn CachePolicy>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    vector_store: Option<Arc<dyn VectorStore>>,
}

impl CacheLayer {
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            store: Arc::new(InMemoryStore::new(&config)),
            key_strategy: Arc::new(ExactHashStrategy),
            cache_policy: Arc::new(StandardCachePolicy::default()),
            embedding_provider: None,
            vector_store: None,
        }
    }

    #[must_use]
    pub fn with_store(store: Arc<dyn CacheStore>) -> Self {
        Self {
            store,
            key_strategy: Arc::new(ExactHashStrategy),
            cache_policy: Arc::new(StandardCachePolicy::default()),
            embedding_provider: None,
            vector_store: None,
        }
    }

    #[must_use]
    pub fn with_key_strategy(mut self, strategy: Arc<dyn HashKeyStrategy>) -> Self {
        self.key_strategy = strategy;
        self
    }

    #[must_use]
    pub fn with_policy(mut self, policy: Arc<dyn CachePolicy>) -> Self {
        self.cache_policy = policy;
        self
    }

    #[must_use]
    pub fn with_semantic_cache(
        mut self,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        self.embedding_provider = Some(embedding_provider);
        self.vector_store = Some(vector_store);
        self
    }
}

impl<S> Layer<S> for CacheLayer {
    type Service = CacheService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CacheService {
            inner,
            store: Arc::clone(&self.store),
            key_strategy: Arc::clone(&self.key_strategy),
            cache_policy: Arc::clone(&self.cache_policy),
            embedding_provider: self.embedding_provider.clone(),
            vector_store: self.vector_store.clone(),
        }
    }
}

pub struct CacheService<S> {
    inner: S,
    store: Arc<dyn CacheStore>,
    key_strategy: Arc<dyn HashKeyStrategy>,
    cache_policy: Arc<dyn CachePolicy>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    vector_store: Option<Arc<dyn VectorStore>>,
}

impl<S: Clone> Clone for CacheService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            store: Arc::clone(&self.store),
            key_strategy: Arc::clone(&self.key_strategy),
            cache_policy: Arc::clone(&self.cache_policy),
            embedding_provider: self.embedding_provider.clone(),
            vector_store: self.vector_store.clone(),
        }
    }
}

impl<S> CacheService<S> {
    pub async fn warm<'a>(&self, requests: impl Iterator<Item = HashKeyInput<'a>>) {
        for input in requests {
            let (key, body) = self.key_strategy.key_for(&input);
            if self.store.get(key, &body).await.is_none() {
                let _ = (key, body);
            }
        }
    }
}

pub(crate) fn hash_key(req: &LlmRequest) -> Option<(u64, String)> {
    let json = match &req.kind {
        LlmRequestKind::Chat(r) => serde_json::to_string(r).ok()?,
        LlmRequestKind::Embed(r) => serde_json::to_string(r).ok()?,
        _ => return None,
    };

    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    Some((hasher.finish(), json))
}

fn strategy_key(strategy: &dyn HashKeyStrategy, req: &LlmRequest) -> Option<(u64, String)> {
    let req_tenant = req.tenant_id().map(|t| t.as_ref().to_owned());
    let (model, messages_json, params_json, tenant_id, system_prompt) = match &req.kind {
        LlmRequestKind::Chat(r) => {
            let msgs = serde_json::to_string(&r.messages).ok()?;
            let params = serde_json::json!({
                "temperature": r.temperature,
                "top_p": r.top_p,
                "max_tokens": r.max_tokens,
                "n": r.n,
                "stop": r.stop,
            });
            let tenant_id: Option<String> = req_tenant.or_else(|| {
                r.user
                    .as_deref()
                    .and_then(|u| u.strip_prefix("tenant:"))
                    .map(str::to_owned)
            });
            let system_prompt: Option<String> = r.messages.iter().find_map(|m| {
                if let crate::types::Message::System(s) = m {
                    s.content.as_text()
                } else {
                    None
                }
            });
            (
                r.model.as_str().to_owned(),
                msgs,
                params.to_string(),
                tenant_id,
                system_prompt,
            )
        }
        LlmRequestKind::Embed(r) => {
            let input = serde_json::to_string(&r.input).ok()?;
            let params = serde_json::json!({
                "dimensions": r.dimensions,
                "encoding_format": r.encoding_format,
            });
            (
                r.model.as_str().to_owned(),
                input,
                params.to_string(),
                req_tenant,
                None,
            )
        }
        _ => return None,
    };

    let input = HashKeyInput {
        model: &model,
        messages_json: &messages_json,
        params_json: &params_json,
        tenant_id: tenant_id.as_deref(),
        system_prompt: system_prompt.as_deref(),
    };
    Some(strategy.key_for(&input))
}

impl<S> Service<LlmRequest> for CacheService<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        static EMPTY_METADATA: OnceLock<HashMap<String, String>> = OnceLock::new();
        let empty_meta = EMPTY_METADATA.get_or_init(HashMap::new);

        let stream = matches!(req.kind, LlmRequestKind::ChatStream(_));
        let model = req.model().unwrap_or("").to_owned();
        let tenant_id_str: Option<String> = req.tenant_id().map(|t| t.as_ref().to_owned());
        let ctx = CachePolicyContext {
            model: &model,
            tenant_id: tenant_id_str.as_deref(),
            stream,
            metadata: empty_meta,
        };
        let decision: CacheDecision = self.cache_policy.decide(&ctx);

        let key_and_body = if decision.bypass {
            None
        } else {
            strategy_key(self.key_strategy.as_ref(), &req)
        };

        let store = Arc::clone(&self.store);
        let embedding_provider = self.embedding_provider.clone();
        let vector_store = self.vector_store.clone();

        let standby = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, standby);
        let fut = inner.call(req);

        Box::pin(async move {
            if decision.use_exact
                && let Some((k, ref body)) = key_and_body
                && let Some(cached) = store.get(k, body).await
            {
                #[cfg(feature = "otel")]
                crate::tower::metrics::record_cache_tier_hit("", &model, "exact");
                record_cache_state(CacheState::ExactHit);
                return cached.into_llm_response();
            }
            #[cfg(feature = "otel")]
            if decision.use_exact && key_and_body.is_some() {
                crate::tower::metrics::record_cache_tier_miss("", &model, "exact");
            }

            if decision.use_semantic
                && let (Some(ep), Some(vs)) = (&embedding_provider, &vector_store)
                && let Some((_, ref body)) = key_and_body
            {
                let maybe_cached = async {
                    let query_vec = ep.embed(body).await.ok()?;
                    let best = vs
                        .search(&query_vec, 1, decision.similarity_threshold)
                        .await
                        .into_iter()
                        .next()?;
                    store
                        .get(
                            best.metadata.cache_key,
                            &best.metadata.original_request_body,
                        )
                        .await
                }
                .await;
                if let Some(cached) = maybe_cached {
                    #[cfg(feature = "otel")]
                    crate::tower::metrics::record_cache_tier_hit("", &model, "semantic");
                    record_cache_state(CacheState::SemanticHit);
                    return cached.into_llm_response();
                }
                #[cfg(feature = "otel")]
                crate::tower::metrics::record_cache_tier_miss("", &model, "semantic");
            }
            record_cache_state(CacheState::Miss);
            let resp = fut.await?;
            if let Some((k, body)) = key_and_body {
                let cached = match &resp {
                    LlmResponse::Chat(r) => Some(CachedResponse::Chat(r.clone())),
                    LlmResponse::Embed(r) => Some(CachedResponse::Embed(r.clone())),
                    _ => None,
                };
                if let Some(cached_resp) = cached {
                    store.put(k, body.clone(), cached_resp).await;
                    if let Some(ttl) = decision.ttl_override {
                        store.set_ttl(k, ttl).await;
                    }

                    if decision.use_semantic
                        && let (Some(ep), Some(vs)) = (&embedding_provider, &vector_store)
                        && let Ok(vec) = ep.embed(&body).await
                    {
                        let metadata = crate::vectorstore::VectorMetadata {
                            cache_key: k,
                            original_request_body: body.clone(),
                            tenant_id: None,
                            inserted_at: std::time::SystemTime::now(),
                            extra: HashMap::new(),
                        };
                        let _ = vs.update(format!("{k}"), vec, metadata).await;
                    }
                }
            }

            Ok(resp)
        })
    }
}
