use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::Instant;

use dashmap::DashMap;
use futures_core::Stream;
use thiserror::Error;
use tower::Service;
use tower::discover::{Change, Discover};
use tower::limit::ConcurrencyLimit;
use tower::ready_cache::ReadyCache;

use super::route_classify::RouteClassifier;
use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::cost::completion_cost;
use crate::types::{Message, MessageContent};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Weight(u32);

impl Weight {
    pub const ZERO: Weight = Weight(0);
    pub const ONE: Weight = Weight(1);
    pub const MAX: Weight = Weight(u32::MAX);

    #[must_use]
    pub fn from_f64(f: f64) -> Self {
        if f.is_nan() || f < 0.0 {
            Self::ZERO
        } else if f.is_infinite() {
            Self::MAX
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let w = f.round().min(f64::from(u32::MAX)) as u32;
            Self(w)
        }
    }

    #[must_use]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl Default for Weight {
    fn default() -> Self {
        Self::ONE
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone)]
pub enum RoutingStrategy {
    RoundRobin,
    Fallback,
    LatencyBased,
    CostBased,
    WeightedRandom { weights: Vec<Weight> },
    Semantic(Arc<dyn RouteClassifier>),
}

impl std::fmt::Debug for RoutingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::Fallback => write!(f, "Fallback"),
            Self::LatencyBased => write!(f, "LatencyBased"),
            Self::CostBased => write!(f, "CostBased"),
            Self::WeightedRandom { weights } => f
                .debug_struct("WeightedRandom")
                .field("weights", weights)
                .finish(),
            Self::Semantic(_) => write!(f, "Semantic(…)"),
        }
    }
}

#[derive(Debug)]
struct DeploymentMetrics {
    latency_ema: f64,
    request_count: u64,
}

impl Default for DeploymentMetrics {
    fn default() -> Self {
        Self {
            latency_ema: 0.0,
            request_count: 0,
        }
    }
}

impl DeploymentMetrics {
    fn record_latency(&mut self, latency_secs: f64) {
        const ALPHA: f64 = 0.3;

        if self.request_count == 0 {
            self.latency_ema = latency_secs;
        } else {
            self.latency_ema = ALPHA * latency_secs + (1.0 - ALPHA) * self.latency_ema;
        }
        self.request_count += 1;
    }
}

pub struct RouterState {
    metrics: Arc<DashMap<usize, DeploymentMetrics>>,
}

impl RouterState {
    fn new() -> Self {
        Self {
            metrics: Arc::new(DashMap::new()),
        }
    }
}

impl Clone for RouterState {
    fn clone(&self) -> Self {
        Self {
            metrics: Arc::clone(&self.metrics),
        }
    }
}

pub struct Router<S> {
    deployments: Vec<S>,
    strategy: RoutingStrategy,
    counter: Arc<AtomicUsize>,
    state: RouterState,
    provider: String,
}

impl<S> Router<S> {
    pub fn new(
        deployments: Vec<S>,
        strategy: RoutingStrategy,
        provider: impl Into<String>,
    ) -> HiLlmResult<Self> {
        if deployments.is_empty() {
            return Err(HiLlmError::BadRequest {
                message: "Router requires at least one deployment".into(),
                status: 400,
            });
        }
        if let RoutingStrategy::WeightedRandom { ref weights } = strategy {
            if weights.len() != deployments.len() {
                return Err(HiLlmError::BadRequest {
                    message: format!(
                        "WeightedRandom: weights length ({}) must match deployments length ({})",
                        weights.len(),
                        deployments.len()
                    ),
                    status: 400,
                });
            }
            let total: u64 = weights.iter().map(|w| u64::from(w.as_u32())).sum();
            if total == 0 {
                return Err(HiLlmError::BadRequest {
                    message: "WeightedRandom: total weight must be positive".into(),
                    status: 400,
                });
            }
        }
        Ok(Self {
            deployments,
            strategy,
            counter: Arc::new(AtomicUsize::new(0)),
            state: RouterState::new(),
            provider: provider.into(),
        })
    }
}

impl<S: Clone> Clone for Router<S> {
    fn clone(&self) -> Self {
        Self {
            deployments: self.deployments.clone(),
            strategy: self.strategy.clone(),
            counter: Arc::clone(&self.counter),
            state: self.state.clone(),
            provider: self.provider.clone(),
        }
    }
}

impl<S> Service<LlmRequest> for Router<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        match &self.strategy {
            RoutingStrategy::RoundRobin => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.deployments.len();
                let mut svc = self.deployments[idx].clone();
                Box::pin(async move { svc.call(req).await })
            }
            RoutingStrategy::Fallback => {
                let deployments = self.deployments.clone();
                Box::pin(async move {
                    let mut last_err: Option<HiLlmError> = None;
                    for mut svc in deployments {
                        match svc.call(req.clone()).await {
                            Ok(resp) => return Ok(resp),
                            Err(e) if e.is_transient() => {
                                tracing::warn!(
                                    error = %e,
                                    "deployment failed with transient error; trying next deployment"
                                );
                                last_err = Some(e);
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    Err(last_err.unwrap_or(HiLlmError::ServerError {
                        message: "all deployments failed".into(),
                        status: 500,
                    }))
                })
            }
            RoutingStrategy::LatencyBased => {
                let state = self.state.clone();
                let n = self.deployments.len();

                let mut best_idx = 0;
                let mut best_ema = f64::MAX;
                for i in 0..n {
                    let ema = state.metrics.get(&i).map_or(0.0, |m| m.latency_ema);
                    if ema < best_ema {
                        best_ema = ema;
                        best_idx = i;
                    }
                }

                let mut svc = self.deployments[best_idx].clone();
                let idx = best_idx;

                Box::pin(async move {
                    let start = Instant::now();
                    let result = svc.call(req).await;
                    let latency = start.elapsed().as_secs_f64();

                    state
                        .metrics
                        .entry(idx)
                        .or_default()
                        .record_latency(latency);

                    result
                })
            }
            RoutingStrategy::CostBased => {
                let model = req.model().map(ToOwned::to_owned);
                let deployments = self.deployments.clone();
                let provider = self.provider.clone();

                Box::pin(async move {
                    let mut last_err: Option<HiLlmError> = None;
                    for mut svc in deployments {
                        match svc.call(req.clone()).await {
                            Ok(resp) => {
                                if let (Some(model), Some(usage)) = (&model, resp.usage())
                                    && let Some(cost) = completion_cost(
                                        &provider,
                                        model,
                                        usage.prompt_tokens,
                                        usage.completion_tokens,
                                    )
                                    .unwrap_or_default()
                                {
                                    tracing::debug!(
                                        model = %model,
                                        cost = cost,
                                        "cost-based routing: estimated cost"
                                    );
                                }
                                return Ok(resp);
                            }
                            Err(e) if e.is_transient() => {
                                last_err = Some(e);
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    Err(last_err.unwrap_or(HiLlmError::ServerError {
                        message: "all deployments failed".into(),
                        status: 500,
                    }))
                })
            }
            RoutingStrategy::WeightedRandom { weights } => {
                let idx = weighted_random_select(weights);
                let mut svc = self.deployments[idx].clone();
                Box::pin(async move { svc.call(req).await })
            }
            RoutingStrategy::Semantic(classifier) => {
                use super::route_classify::ClassifyContext;
                use std::collections::HashMap;

                let classifier = Arc::clone(classifier);
                let deployments = self.deployments.clone();
                let counter = Arc::clone(&self.counter);

                let n = deployments.len();
                let available_models: Vec<String> = (0..n).map(|i| i.to_string()).collect();

                let (prompt, system_prompt) = match &req.kind {
                    LlmRequestKind::Chat(r) => {
                        let prompt = r
                            .messages
                            .iter()
                            .rev()
                            .find_map(|m| {
                                if let crate::types::Message::User(u) = m {
                                    match &u.content {
                                        MessageContent::Text(t) => Some(t.clone()),
                                        MessageContent::Parts(_) => None,
                                    }
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        let system = r.messages.iter().find_map(|m| {
                            if let Message::System(s) = m {
                                s.content.as_text()
                            } else {
                                None
                            }
                        });
                        (prompt, system)
                    }
                    _ => (String::new(), None),
                };

                Box::pin(async move {
                    let meta: HashMap<String, String> = HashMap::new();
                    let ctx = ClassifyContext {
                        prompt: &prompt,
                        system_prompt: system_prompt.as_deref(),
                        metadata: &meta,
                        available_models: &available_models,
                    };

                    let idx = classifier
                        .classify(&ctx)
                        .await
                        .and_then(|model_str| model_str.parse::<usize>().ok())
                        .filter(|&i| i < n);

                    let idx = idx.unwrap_or_else(|| counter.fetch_add(1, Ordering::Relaxed) % n);

                    deployments[idx].clone().call(req).await
                })
            }
        }
    }
}

fn weighted_random_select(weights: &[Weight]) -> usize {
    let total: u64 = weights.iter().map(|w| u64::from(w.as_u32())).sum();
    if total == 0 {
        return 0;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let threshold = u64::from(nanos) % total;

    let mut cumulative: u64 = 0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += u64::from(w.as_u32());
        if threshold < cumulative {
            return i;
        }
    }
    weights.len() - 1
}

pub trait UpstreamDiscover: Discover<Key = String> + Unpin + Send {}

impl<D> UpstreamDiscover for D where D: Discover<Key = String> + Unpin + Send {}

#[derive(Debug, Error)]
pub enum RouterError {
    #[error("discovery error (code 2001): {source}")]
    Discover { source: tower::BoxError, code: u32 },
    #[error("no ready upstream available (code 2002)")]
    NoReadyUpstream { code: u32 },
}

impl RouterError {
    #[must_use]
    pub fn code(&self) -> u32 {
        match self {
            Self::Discover { code, .. } | Self::NoReadyUpstream { code } => *code,
        }
    }
}

impl From<RouterError> for HiLlmError {
    fn from(e: RouterError) -> Self {
        HiLlmError::ServerError {
            message: e.to_string(),
            status: 503,
        }
    }
}

pub struct StaticDiscover<S> {
    keys: std::collections::VecDeque<String>,
    services: std::collections::VecDeque<S>,
}

impl<S> StaticDiscover<S> {
    pub fn new(services: impl IntoIterator<Item = (String, S)>) -> Self {
        let (keys, services): (std::collections::VecDeque<_>, std::collections::VecDeque<_>) =
            services.into_iter().unzip();
        Self { keys, services }
    }
}

impl<S: Unpin> Unpin for StaticDiscover<S> {}

impl<S: Unpin> Stream for StaticDiscover<S> {
    type Item = std::result::Result<Change<String, S>, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match (self.keys.pop_front(), self.services.pop_front()) {
            (Some(key), Some(svc)) => Poll::Ready(Some(Ok(Change::Insert(key, svc)))),
            _ => Poll::Ready(None),
        }
    }
}

pub const DEFAULT_CONCURRENCY_LIMIT: usize = 256;

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub concurrency_limit: usize,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            concurrency_limit: DEFAULT_CONCURRENCY_LIMIT,
        }
    }
}

pub struct DynamicRouter<D>
where
    D: Discover<Key = String>,
    D::Service: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError>,
{
    discover: D,
    services: ReadyCache<String, ConcurrencyLimit<D::Service>, LlmRequest>,
    provider_configs: HashMap<String, ProviderConfig>,
    _marker: PhantomData<LlmRequest>,
}

impl<D> fmt::Debug for DynamicRouter<D>
where
    D: Discover<Key = String> + fmt::Debug,
    D::Service: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicRouter")
            .field("discover", &self.discover)
            .finish_non_exhaustive()
    }
}

impl<D> DynamicRouter<D>
where
    D: Discover<Key = String> + Unpin,
    D::Service:
        Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + Unpin + 'static,
    <D::Service as Service<LlmRequest>>::Future: Send + 'static,
    D::Error: Into<tower::BoxError>,
{
    pub fn new(discover: D) -> Self {
        Self {
            discover,
            services: ReadyCache::default(),
            provider_configs: HashMap::new(),
            _marker: PhantomData,
        }
    }

    pub fn with_provider_config(mut self, key: impl Into<String>, config: ProviderConfig) -> Self {
        self.provider_configs.insert(key.into(), config);
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.services.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }

    fn update_from_discover(
        &mut self,
        cx: &mut Context<'_>,
    ) -> std::result::Result<(), RouterError> {
        loop {
            match Pin::new(&mut self.discover).poll_discover(cx) {
                Poll::Pending => return Ok(()),
                Poll::Ready(None) => return Ok(()), // stream exhausted
                Poll::Ready(Some(Err(e))) => {
                    return Err(RouterError::Discover {
                        source: e.into(),
                        code: 2001,
                    });
                }
                Poll::Ready(Some(Ok(Change::Insert(key, svc)))) => {
                    let limit = self
                        .provider_configs
                        .get(&key)
                        .map_or(DEFAULT_CONCURRENCY_LIMIT, |c| c.concurrency_limit);
                    tracing::debug!(provider = %key, concurrency_limit = limit, "discovered new upstream");
                    self.services.push(key, ConcurrencyLimit::new(svc, limit));
                }
                Poll::Ready(Some(Ok(Change::Remove(key)))) => {
                    tracing::debug!(provider = %key, "upstream removed from discovery");
                    self.services.evict(&key);
                }
            }
        }
    }
}

impl<D> Service<LlmRequest> for DynamicRouter<D>
where
    D: Discover<Key = String> + Unpin + Send,
    D::Service:
        Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + Unpin + 'static,
    <D::Service as Service<LlmRequest>>::Future: Send + 'static,
    D::Error: Into<tower::BoxError>,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        if let Err(e) = self.update_from_discover(cx) {
            return Poll::Ready(Err(e.into()));
        }

        let _ = self.services.poll_pending(cx);

        if self.services.ready_len() > 0 {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        if self.services.ready_len() == 0 {
            return Box::pin(async { Err(RouterError::NoReadyUpstream { code: 2002 }.into()) });
        }
        let fut = self.services.call_ready_index(0, req);
        Box::pin(fut)
    }
}
