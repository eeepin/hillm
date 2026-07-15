use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::DashMap;
use tokio::sync::broadcast;
use tower::{Layer, Service};

use super::cache::{CachedResponse, record_cache_state};
use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::observability::usage::CacheState;

type InFlightMap = Arc<DashMap<u64, broadcast::Sender<SingleflightResult>>>;

pub type SingleflightResult = std::result::Result<CachedResponse, Arc<HiLlmError>>;

pub enum SingleflightHandle {
    Leader {
        complete: Box<dyn FnOnce(SingleflightResult) + Send>,
    },
    Follower {
        recv: broadcast::Receiver<SingleflightResult>,
    },
}

pub trait SingleflightCoordinator: Send + Sync + 'static {
    fn join<'a>(
        &'a self,
        key: u64,
    ) -> Pin<Box<dyn Future<Output = SingleflightHandle> + Send + 'a>>;
}

pub struct InMemorySingleflight {
    in_flight: InFlightMap,
}

impl Default for InMemorySingleflight {
    fn default() -> Self {
        Self {
            in_flight: Arc::new(DashMap::new()),
        }
    }
}

impl InMemorySingleflight {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl SingleflightCoordinator for InMemorySingleflight {
    fn join<'a>(
        &'a self,
        key: u64,
    ) -> Pin<Box<dyn Future<Output = SingleflightHandle> + Send + 'a>> {
        Box::pin(async move {
            use dashmap::mapref::entry::Entry;

            match self.in_flight.entry(key) {
                Entry::Vacant(slot) => {
                    let (tx, _) = broadcast::channel::<SingleflightResult>(1);
                    let tx_for_map = tx.clone();
                    slot.insert(tx_for_map);
                    let map = Arc::clone(&self.in_flight);

                    let guard = LeaderDropGuard {
                        map: Arc::clone(&map),
                        key,
                        disarmed: false,
                    };

                    let complete = Box::new(move |result: SingleflightResult| {
                        let mut g = guard;
                        g.disarmed = true;
                        let _ = tx.send(result);
                        map.remove(&key);
                    });

                    SingleflightHandle::Leader { complete }
                }
                Entry::Occupied(entry) => {
                    let recv = entry.get().subscribe();
                    SingleflightHandle::Follower { recv }
                }
            }
        })
    }
}

struct LeaderDropGuard {
    map: InFlightMap,
    key: u64,
    disarmed: bool,
}

impl Drop for LeaderDropGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            self.map.remove(&self.key);
        }
    }
}

pub struct SingleflightLayer<C: SingleflightCoordinator> {
    coordinator: Arc<C>,
}

impl<C: SingleflightCoordinator> SingleflightLayer<C> {
    #[must_use]
    pub fn new(coordinator: Arc<C>) -> Self {
        Self { coordinator }
    }
}

impl<C: SingleflightCoordinator, S> Layer<S> for SingleflightLayer<C> {
    type Service = SingleflightService<C, S>;

    fn layer(&self, inner: S) -> Self::Service {
        SingleflightService {
            coordinator: Arc::clone(&self.coordinator),
            inner,
        }
    }
}

pub struct SingleflightService<C: SingleflightCoordinator, S> {
    coordinator: Arc<C>,
    inner: S,
}

impl<C: SingleflightCoordinator, S: Clone> Clone for SingleflightService<C, S> {
    fn clone(&self) -> Self {
        Self {
            coordinator: Arc::clone(&self.coordinator),
            inner: self.inner.clone(),
        }
    }
}

fn singleflight_key(req: &LlmRequest) -> Option<u64> {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let json = match &req.kind {
        LlmRequestKind::Chat(r) => serde_json::to_string(r).ok()?,
        LlmRequestKind::Embed(r) => serde_json::to_string(r).ok()?,
        _ => return None,
    };
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    Some(hasher.finish())
}

impl<C, S> Service<LlmRequest> for SingleflightService<C, S>
where
    C: SingleflightCoordinator,
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
        let key = singleflight_key(&req);
        let Some(key) = key else {
            let fut = self.inner.call(req);
            #[allow(clippy::redundant_async_block)]
            return Box::pin(async move { fut.await });
        };

        let coordinator = Arc::clone(&self.coordinator);
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        Box::pin(async move {
            match coordinator.join(key).await {
                SingleflightHandle::Leader { complete } => {
                    let result = inner.call(req).await;
                    let sf_result: SingleflightResult = match &result {
                        Ok(resp) => match resp {
                            LlmResponse::Chat(r) => Ok(CachedResponse::Chat(r.clone())),
                            LlmResponse::Embed(r) => Ok(CachedResponse::Embed(r.clone())),
                            _ => Err(Arc::new(HiLlmError::InternalError {
                                message: "singleflight: non-cacheable response variant in leader"
                                    .into(),
                            })),
                        },
                        Err(e) => Err(Arc::new(e.to_singleflight_error())),
                    };
                    complete(sf_result);
                    result
                }
                SingleflightHandle::Follower { mut recv } => {
                    drop(inner);
                    match recv.recv().await {
                        Ok(Ok(cached)) => {
                            record_cache_state(CacheState::ExactHit);
                            cached.into_llm_response()
                        }
                        Ok(Err(arc_err)) => Err(Arc::try_unwrap(arc_err)
                            .unwrap_or_else(|arc| arc.to_singleflight_error())),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::debug!(
                                skipped = n,
                                "singleflight follower lagged; resubscribing"
                            );
                            let mut rx2 = recv.resubscribe();
                            match rx2.recv().await {
                                Ok(Ok(cached)) => {
                                    record_cache_state(CacheState::ExactHit);
                                    cached.into_llm_response()
                                }
                                Ok(Err(arc_err)) => Err(Arc::try_unwrap(arc_err)
                                    .unwrap_or_else(|arc| arc.to_singleflight_error())),
                                Err(_) => Err(HiLlmError::InternalError {
                                    message: "singleflight: follower lagged and retry also failed"
                                        .into(),
                                }),
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            Err(HiLlmError::InternalError {
                                message:
                                    "singleflight: leader closed channel without sending a result"
                                        .into(),
                            })
                        }
                    }
                }
            }
        })
    }
}
