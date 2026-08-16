use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::DashMap;
use tokio::sync::broadcast;
use tower::{Layer, Service};

use super::cache::{CachedResponse, record_cache_state};
use super::types::{LLMRequest, LLMRequestKind, LLMResponse};
use crate::client::BoxFuture;
use crate::error::{HiLLMError, HiLLMResult};
use crate::observability::usage::CacheState;

type InFlightMap = Arc<DashMap<u64, broadcast::Sender<SingleflightResult>>>;

pub type SingleflightResult = std::result::Result<CachedResponse, Arc<HiLLMError>>;

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

fn singleflight_key(req: &LLMRequest) -> Option<u64> {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let json = match &req.kind {
        LLMRequestKind::Chat(r) => serde_json::to_string(r).ok()?,
        LLMRequestKind::Embed(r) => serde_json::to_string(r).ok()?,
        _ => return None,
    };
    let mut hasher = DefaultHasher::new();
    json.hash(&mut hasher);
    Some(hasher.finish())
}

impl<C, S> Service<LLMRequest> for SingleflightService<C, S>
where
    C: SingleflightCoordinator,
    S: Service<LLMRequest, Response = LLMResponse, Error = HiLLMError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LLMResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LLMResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LLMRequest) -> Self::Future {
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
                            LLMResponse::Chat(r) => Ok(CachedResponse::Chat(r.clone())),
                            LLMResponse::Embed(r) => Ok(CachedResponse::Embed(r.clone())),
                            _ => Err(Arc::new(HiLLMError::InternalError {
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
                                Err(_) => Err(HiLLMError::InternalError {
                                    message: "singleflight: follower lagged and retry also failed"
                                        .into(),
                                }),
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            Err(HiLLMError::InternalError {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::types::{LLMRequest, LLMRequestKind, LLMResponse};
    use crate::types::{ChatCompletionRequest, ChatCompletionResponse, Message, Usage};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{Duration, sleep};
    use tower::ServiceExt;

    fn create_chat_request(content: &str) -> LLMRequest {
        LLMRequest {
            kind: LLMRequestKind::Chat(ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![Message::User(crate::types::UserMessage {
                    content: crate::types::MessageContent::Text(content.to_string()),
                    name: None,
                })],
                ..Default::default()
            }),
            tenant_id: None,
            idempotency_key: None,
        }
    }

    fn create_chat_response(content: &str) -> LLMResponse {
        LLMResponse::Chat(ChatCompletionResponse {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "test-model".to_string(),
            choices: vec![crate::types::Choice {
                index: 0,
                message: crate::types::AssistantMessage {
                    content: Some(crate::types::MessageContent::Text(content.to_string())),
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
        call_count: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl MockService {
        fn new(delay_ms: u64) -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
                delay: Duration::from_millis(delay_ms),
            }
        }

        fn get_call_count(&self) -> usize {
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
            let count = self.call_count.clone();
            let delay = self.delay;
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                sleep(delay).await;
                Ok(create_chat_response("test response"))
            })
        }
    }

    #[tokio::test]
    async fn singleflight_deduplicates_concurrent_requests() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let mock = MockService::new(100);
        let service = SingleflightService {
            coordinator: coordinator.clone(),
            inner: mock.clone(),
        };

        // 发送 5 个相同的并发请求
        let mut handles = vec![];
        for _ in 0..5 {
            let mut svc = service.clone();
            let req = create_chat_request("test");
            handles.push(tokio::spawn(async move {
                svc.ready().await.unwrap().call(req).await
            }));
        }

        // 等待所有请求完成
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Request should succeed");
        }

        // 验证底层服务只被调用一次
        assert_eq!(
            mock.get_call_count(),
            1,
            "Service should only be called once for identical requests"
        );
    }

    #[tokio::test]
    async fn singleflight_different_requests_not_deduplicated() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let mock = MockService::new(50);
        let service = SingleflightService {
            coordinator: coordinator.clone(),
            inner: mock.clone(),
        };

        // 发送 3 个不同的请求
        let mut handles = vec![];
        for i in 0..3 {
            let mut svc = service.clone();
            let req = create_chat_request(&format!("test {}", i));
            handles.push(tokio::spawn(async move {
                svc.ready().await.unwrap().call(req).await
            }));
        }

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Request should succeed");
        }

        // 验证底层服务被调用 3 次（每个不同的请求一次）
        assert_eq!(
            mock.get_call_count(),
            3,
            "Service should be called once per unique request"
        );
    }

    #[tokio::test]
    async fn singleflight_leader_cancellation() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let mock = MockService::new(200);
        let service = SingleflightService {
            coordinator: coordinator.clone(),
            inner: mock.clone(),
        };

        // 启动 leader 请求
        let mut leader_svc = service.clone();
        let leader_req = create_chat_request("test");
        let leader_handle =
            tokio::spawn(async move { leader_svc.ready().await.unwrap().call(leader_req).await });

        // 等待一小段时间让 leader 开始
        sleep(Duration::from_millis(50)).await;

        // 启动 follower 请求
        let mut follower_svc = service.clone();
        let follower_req = create_chat_request("test");
        let follower_handle =
            tokio::spawn(
                async move { follower_svc.ready().await.unwrap().call(follower_req).await },
            );

        // 取消 leader
        leader_handle.abort();

        // Follower 应该能够完成（可能收到错误）
        let follower_result = follower_handle.await.unwrap();
        // Follower 可能成功也可能失败，取决于 leader 取消的时机
        // 关键是 follower 不应该永远阻塞
        // 这里我们只验证 follower 确实完成了（无论成功还是失败）
        let _ = follower_result;
    }

    #[tokio::test]
    async fn singleflight_follower_receives_leader_result() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let mock = MockService::new(100);
        let service = SingleflightService {
            coordinator: coordinator.clone(),
            inner: mock.clone(),
        };

        // 启动 leader
        let mut leader_svc = service.clone();
        let leader_req = create_chat_request("test");
        let leader_handle =
            tokio::spawn(async move { leader_svc.ready().await.unwrap().call(leader_req).await });

        // 等待 leader 开始执行
        sleep(Duration::from_millis(20)).await;

        // 启动 follower
        let mut follower_svc = service.clone();
        let follower_req = create_chat_request("test");
        let follower_handle =
            tokio::spawn(
                async move { follower_svc.ready().await.unwrap().call(follower_req).await },
            );

        // 等待两个请求完成
        let leader_result = leader_handle.await.unwrap().unwrap();
        let follower_result = follower_handle.await.unwrap().unwrap();

        // 验证两个请求都成功（LLMResponse 是枚举类型，不是 Result）
        // 只要没有 panic 或返回错误，就说明成功了
        let _ = leader_result;
        let _ = follower_result;

        // 验证服务只被调用一次
        assert_eq!(mock.get_call_count(), 1);
    }

    #[tokio::test]
    async fn singleflight_error_propagation() {
        let coordinator = Arc::new(InMemorySingleflight::new());

        // 创建一个会失败的服务
        #[derive(Clone)]
        struct FailingService;

        impl Service<LLMRequest> for FailingService {
            type Response = LLMResponse;
            type Error = HiLLMError;
            type Future = BoxFuture<'static, HiLLMResult<LLMResponse>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, _req: LLMRequest) -> Self::Future {
                Box::pin(async move {
                    sleep(Duration::from_millis(50)).await;
                    Err(HiLLMError::InternalError {
                        message: "test error".to_string(),
                    })
                })
            }
        }

        let service = SingleflightService {
            coordinator: coordinator.clone(),
            inner: FailingService,
        };

        // 发送两个相同的请求
        let mut handles = vec![];
        for _ in 0..2 {
            let mut svc = service.clone();
            let req = create_chat_request("test");
            handles.push(tokio::spawn(async move {
                svc.ready().await.unwrap().call(req).await
            }));
        }

        // 验证两个请求都收到错误
        for handle in handles {
            let inner_result = handle.await.unwrap();
            assert!(inner_result.is_err());
            if let Err(e) = inner_result {
                assert!(e.to_string().contains("test error"));
            }
        }
    }

    #[tokio::test]
    async fn singleflight_key_generation() {
        let req1 = create_chat_request("test");
        let req2 = create_chat_request("test");
        let req3 = create_chat_request("different");

        let key1 = singleflight_key(&req1);
        let key2 = singleflight_key(&req2);
        let key3 = singleflight_key(&req3);

        // 相同请求应该生成相同的 key
        assert_eq!(key1, key2, "Identical requests should have the same key");

        // 不同请求应该生成不同的 key
        assert_ne!(key1, key3, "Different requests should have different keys");
        assert_ne!(key2, key3, "Different requests should have different keys");
    }

    #[tokio::test]
    async fn singleflight_non_cacheable_request_bypasses() {
        let coordinator = Arc::new(InMemorySingleflight::new());
        let mock = MockService::new(50);
        let service = SingleflightService {
            coordinator: coordinator.clone(),
            inner: mock.clone(),
        };

        // 创建一个非 cacheable 的请求（例如 Image 请求）
        let req = LLMRequest {
            kind: LLMRequestKind::ImageGenerate(Default::default()),
            tenant_id: None,
            idempotency_key: None,
        };

        // 发送两个相同的非 cacheable 请求
        let mut handles = vec![];
        for _ in 0..2 {
            let mut svc = service.clone();
            let req = req.clone();
            handles.push(tokio::spawn(async move {
                svc.ready().await.unwrap().call(req).await
            }));
        }

        for handle in handles {
            // 非 cacheable 请求会失败（因为 MockService 返回 Chat 响应）
            // 但这不是重点，重点是它们不会被 singleflight 去重
            let _ = handle.await;
        }

        // 验证服务被调用两次（没有去重）
        assert_eq!(
            mock.get_call_count(),
            2,
            "Non-cacheable requests should not be deduplicated"
        );
    }
}
