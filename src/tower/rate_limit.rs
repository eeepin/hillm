use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime};

use dashmap::DashMap;
use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLLMError, HiLLMResult};
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
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLLMError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
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
                    Err(HiLLMError::RateLimited {
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
                    Err(HiLLMError::RateLimited {
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

    fn check(&self, config: &CostRateLimitConfig) -> Option<HiLLMError> {
        let now = Self::now_secs();

        if let Some(limit) = config.max_cost_per_minute {
            let spend = self.per_minute.spend_cost(now);
            if spend >= limit {
                return Some(HiLLMError::RateLimited {
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
                return Some(HiLLMError::RateLimited {
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
                return Some(HiLLMError::RateLimited {
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
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLLMError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rate_limit_config_creation() {
        let config = RateLimitConfig::default();
        assert_eq!(config.rpm, None);
        assert_eq!(config.tpm, None);
        assert_eq!(config.window, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn rate_limit_config_with_limits() {
        let config = RateLimitConfig {
            rpm: Some(100),
            tpm: Some(10000),
            window: Duration::from_secs(60),
        };
        assert_eq!(config.rpm, Some(100));
        assert_eq!(config.tpm, Some(10000));
    }

    #[tokio::test]
    async fn model_rate_state_creation() {
        let state = ModelRateState::new();
        assert_eq!(state.request_count, 0);
        assert_eq!(state.token_count, 0);
    }

    #[tokio::test]
    async fn model_rate_state_reset_after_window() {
        let mut state = ModelRateState::new();
        state.request_count = 10;
        state.token_count = 100;

        // Simulate window expiry by backdating window_start
        state.window_start = Instant::now() - Duration::from_secs(120);

        // Reset should clear counts
        state.maybe_reset(Duration::from_secs(60));
        assert_eq!(state.request_count, 0);
        assert_eq!(state.token_count, 0);
    }

    #[tokio::test]
    async fn model_rate_state_no_reset_within_window() {
        let mut state = ModelRateState::new();
        state.request_count = 10;
        state.token_count = 100;

        // Window hasn't expired yet
        state.maybe_reset(Duration::from_secs(60));

        // Counts should remain
        assert_eq!(state.request_count, 10);
        assert_eq!(state.token_count, 100);
    }

    #[tokio::test]
    async fn cost_rate_limit_config_creation() {
        let config = CostRateLimitConfig::default();
        assert_eq!(config.max_cost_per_minute, None);
        assert_eq!(config.max_cost_per_hour, None);
        assert_eq!(config.max_cost_per_day, None);
    }

    #[tokio::test]
    async fn cost_rate_limit_config_with_limits() {
        let config = CostRateLimitConfig {
            max_cost_per_minute: Some(1.0),
            max_cost_per_hour: Some(50.0),
            max_cost_per_day: Some(1000.0),
        };
        assert_eq!(config.max_cost_per_minute, Some(1.0));
        assert_eq!(config.max_cost_per_hour, Some(50.0));
        assert_eq!(config.max_cost_per_day, Some(1000.0));
    }

    #[tokio::test]
    async fn cost_window_creation() {
        let window = CostWindow::new(Duration::from_secs(60));
        let now = CostRateLimitState::now_secs();
        let spend = window.spend_cost(now);
        assert_eq!(spend, 0.0);
    }

    #[tokio::test]
    async fn cost_window_add_and_track() {
        let window = CostWindow::new(Duration::from_secs(60));
        let now = CostRateLimitState::now_secs();

        window.add(1.5, now);
        let spend = window.spend_cost(now);
        assert!((spend - 1.5).abs() < 0.01);

        window.add(2.5, now);
        let spend = window.spend_cost(now);
        assert!((spend - 4.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn cost_window_reset_after_expiry() {
        let window = CostWindow::new(Duration::from_secs(60));
        let now = CostRateLimitState::now_secs();

        window.add(10.0, now);
        let spend = window.spend_cost(now);
        assert!((spend - 10.0).abs() < 0.01);

        // Simulate window expiry
        let future_now = now + 120;
        let spend = window.spend_cost(future_now);
        assert_eq!(spend, 0.0, "Window should reset after expiry");
    }

    #[tokio::test]
    async fn cost_rate_limit_state_creation() {
        let state = CostRateLimitState::new();
        let now = CostRateLimitState::now_secs();

        assert_eq!(state.per_minute.spend_cost(now), 0.0);
        assert_eq!(state.per_hour.spend_cost(now), 0.0);
        assert_eq!(state.per_day.spend_cost(now), 0.0);
    }

    #[tokio::test]
    async fn cost_rate_limit_state_check_under_limit() {
        let state = CostRateLimitState::new();
        let config = CostRateLimitConfig {
            max_cost_per_minute: Some(10.0),
            max_cost_per_hour: Some(100.0),
            max_cost_per_day: Some(1000.0),
        };

        let result = state.check(&config);
        assert!(result.is_none(), "Should allow when under all limits");
    }

    #[tokio::test]
    async fn cost_rate_limit_state_check_over_minute_limit() {
        let state = CostRateLimitState::new();
        let config = CostRateLimitConfig {
            max_cost_per_minute: Some(1.0),
            max_cost_per_hour: Some(100.0),
            max_cost_per_day: Some(1000.0),
        };

        let now = CostRateLimitState::now_secs();
        state.per_minute.add(2.0, now);

        let result = state.check(&config);
        assert!(result.is_some(), "Should reject when over minute limit");
        if let Some(err) = result {
            assert!(matches!(err, HiLLMError::RateLimited { .. }));
        }
    }
}
