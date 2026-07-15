use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime};

use dashmap::DashMap;
use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::cost;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RateLimitConfig {
    pub rpm: Option<u32>,
    pub tpm: Option<u64>,
    pub window: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            rpm: None,
            tpm: None,
            window: Duration::from_secs(60),
        }
    }
}

struct ModelRateState {
    request_count: u64,
    token_count: u64,
    window_start: Instant,
}

impl ModelRateState {
    fn new() -> Self {
        Self {
            request_count: 0,
            token_count: 0,
            window_start: Instant::now(),
        }
    }

    fn maybe_reset(&mut self, window: Duration) {
        if self.window_start.elapsed() >= window {
            self.request_count = 0;
            self.token_count = 0;
            self.window_start = Instant::now();
        }
    }
}

pub struct ModelRateLimitLayer {
    config: RateLimitConfig,
    state: Arc<DashMap<String, ModelRateState>>,
}

impl ModelRateLimitLayer {
    #[must_use]
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            state: Arc::new(DashMap::new()),
        }
    }
}

impl<S> Layer<S> for ModelRateLimitLayer {
    type Service = ModelRateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ModelRateLimitService {
            inner,
            config: self.config.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

pub struct ModelRateLimitService<S> {
    inner: S,
    config: RateLimitConfig,
    state: Arc<DashMap<String, ModelRateState>>,
}

impl<S: Clone> Clone for ModelRateLimitService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

impl<S> Service<LlmRequest> for ModelRateLimitService<S>
where
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
        let model = req.model().unwrap_or("unknown").to_owned();
        let config = self.config.clone();
        let state = Arc::clone(&self.state);

        {
            let mut entry = state
                .entry(model.clone())
                .or_insert_with(ModelRateState::new);
            entry.maybe_reset(config.window);

            if let Some(rpm) = config.rpm
                && entry.request_count >= u64::from(rpm)
            {
                return Box::pin(async move {
                    Err(HiLlmError::RateLimited {
                        message: format!(
                            "model {model} exceeded {rpm} requests per {:.0}s window",
                            config.window.as_secs_f64()
                        ),
                        retry_after: Some(config.window),
                    })
                });
            }

            if let Some(tpm) = config.tpm
                && entry.token_count >= tpm
            {
                return Box::pin(async move {
                    Err(HiLlmError::RateLimited {
                        message: format!(
                            "model {model} exceeded {tpm} tokens per {:.0}s window",
                            config.window.as_secs_f64()
                        ),
                        retry_after: Some(config.window),
                    })
                });
            }

            entry.request_count += 1;
        }

        let fut = self.inner.call(req);

        Box::pin(async move {
            let resp = fut.await?;

            if let Some(usage) = resp.usage() {
                let total_tokens = usage.prompt_tokens + usage.completion_tokens;
                if let Some(mut entry) = state.get_mut(&model) {
                    entry.maybe_reset(config.window);
                    entry.token_count += total_tokens;
                }
            }

            Ok(resp)
        })
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CostRateLimitConfig {
    pub max_cost_per_minute: Option<f64>,

    pub max_cost_per_hour: Option<f64>,

    pub max_cost_per_day: Option<f64>,
}

#[derive(Debug)]
struct CostWindow {
    spend_mc: AtomicU64,
    window_start_secs: AtomicU64,
    window_secs: u64,
}

impl CostWindow {
    fn new(window: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            spend_mc: AtomicU64::new(0),
            window_start_secs: AtomicU64::new(now),
            window_secs: window.as_secs(),
        }
    }

    fn spend_cost(&self, now_secs: u64) -> f64 {
        let start = self.window_start_secs.load(Ordering::Relaxed);
        if now_secs.saturating_sub(start) >= self.window_secs {
            self.spend_mc.store(0, Ordering::Relaxed);
            self.window_start_secs.store(now_secs, Ordering::Relaxed);
        }
        let mc = self.spend_mc.load(Ordering::Relaxed);
        mc as f64 / 1_000_000.0
    }

    fn add(&self, cost: f64, now_secs: u64) {
        let _ = self.spend_cost(now_secs); // reset if expired
        if cost > 0.0 {
            let mc = (cost * 1_000_000.0).round() as u64;
            self.spend_mc.fetch_add(mc, Ordering::Relaxed);
        }
    }
}

#[derive(Debug)]
struct CostRateLimitState {
    per_minute: CostWindow,
    per_hour: CostWindow,
    per_day: CostWindow,
}

impl CostRateLimitState {
    fn new() -> Self {
        Self {
            per_minute: CostWindow::new(Duration::from_secs(60)),
            per_hour: CostWindow::new(Duration::from_secs(3600)),
            per_day: CostWindow::new(Duration::from_secs(86_400)),
        }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn check(&self, config: &CostRateLimitConfig) -> Option<HiLlmError> {
        let now = Self::now_secs();

        if let Some(limit) = config.max_cost_per_minute {
            let spend = self.per_minute.spend_cost(now);
            if spend >= limit {
                return Some(HiLlmError::RateLimited {
                    message: format!(
                        "cost rate limit exceeded: ${spend:.6} >= ${limit:.6} per minute"
                    ),
                    retry_after: Some(Duration::from_secs(60)),
                });
            }
        }

        if let Some(limit) = config.max_cost_per_hour {
            let spend = self.per_hour.spend_cost(now);
            if spend >= limit {
                return Some(HiLlmError::RateLimited {
                    message: format!(
                        "cost rate limit exceeded: ${spend:.6} >= ${limit:.6} per hour"
                    ),
                    retry_after: Some(Duration::from_secs(3600)),
                });
            }
        }

        if let Some(limit) = config.max_cost_per_day {
            let spend = self.per_day.spend_cost(now);
            if spend >= limit {
                return Some(HiLlmError::RateLimited {
                    message: format!(
                        "cost rate limit exceeded: ${spend:.6} >= ${limit:.6} per day"
                    ),
                    retry_after: Some(Duration::from_secs(86_400)),
                });
            }
        }

        None
    }

    fn record(&self, cost: f64) {
        let now = Self::now_secs();
        self.per_minute.add(cost, now);
        self.per_hour.add(cost, now);
        self.per_day.add(cost, now);
    }
}

pub struct CostRateLimitLayer {
    config: CostRateLimitConfig,
    state: Arc<CostRateLimitState>,
    provider: String,
}

impl CostRateLimitLayer {
    #[must_use]
    pub fn new(config: CostRateLimitConfig, provider: impl Into<String>) -> Self {
        Self {
            config,
            state: Arc::new(CostRateLimitState::new()),
            provider: provider.into(),
        }
    }
}

impl<S> Layer<S> for CostRateLimitLayer {
    type Service = CostRateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CostRateLimitService {
            inner,
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            provider: self.provider.clone(),
        }
    }
}

pub struct CostRateLimitService<S> {
    inner: S,
    config: CostRateLimitConfig,
    state: Arc<CostRateLimitState>,
    provider: String,
}

impl<S: Clone> Clone for CostRateLimitService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            provider: self.provider.clone(),
        }
    }
}

impl<S> Service<LlmRequest> for CostRateLimitService<S>
where
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
        let model = req.model().unwrap_or("unknown").to_owned();
        let config = self.config.clone();
        let state = Arc::clone(&self.state);
        let provider = self.provider.clone();

        if let Some(err) = state.check(&config) {
            return Box::pin(async move { Err(err) });
        }

        let fut = self.inner.call(req);

        Box::pin(async move {
            let resp = fut.await?;

            if let Some(usage) = resp.usage()
                && let Some(cost) = cost::completion_cost(
                    &provider,
                    &model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )
                .unwrap_or_default()
            {
                state.record(cost);
            }

            Ok(resp)
        })
    }
}
