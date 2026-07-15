use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
}

#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub unhealthy_threshold: u32,
    pub healthy_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            unhealthy_threshold: 3,
            healthy_threshold: 2,
        }
    }
}

pub trait HealthChecker: Send + Sync + 'static {
    fn check(
        &self,
        upstream: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = HealthStatus> + Send + 'static>>;
}

#[derive(Debug, Clone)]
pub struct HttpProbeHealthChecker {
    client: reqwest::Client,
    probe_urls: std::collections::HashMap<String, String>,
}

impl HttpProbeHealthChecker {
    pub fn new(
        timeout: Duration,
        probe_urls: impl IntoIterator<Item = (String, String)>,
    ) -> HiLlmResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| HiLlmError::BadRequest {
                message: format!("failed to build HTTP client for health checker: {e}"),
                status: 500,
            })?;
        Ok(Self {
            client,
            probe_urls: probe_urls.into_iter().collect(),
        })
    }
}

impl HealthChecker for HttpProbeHealthChecker {
    fn check(
        &self,
        upstream: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = HealthStatus> + Send + 'static>> {
        let url = self.probe_urls.get(&upstream).cloned().unwrap_or(upstream);
        let client = self.client.clone();

        Box::pin(async move {
            let result = client.get(&url).send().await;
            match result {
                Ok(resp) if resp.status().is_success() || resp.status().is_redirection() => {
                    HealthStatus::Healthy
                }
                Ok(resp) => {
                    tracing::debug!(
                        upstream = %url,
                        status = resp.status().as_u16(),
                        "health probe returned non-success status"
                    );
                    HealthStatus::Unhealthy
                }
                Err(e) => {
                    tracing::debug!(
                        upstream = %url,
                        error = %e,
                        "health probe failed"
                    );
                    HealthStatus::Unhealthy
                }
            }
        })
    }
}

#[derive(Debug)]
struct ProviderHealthState {
    healthy: AtomicBool,
    consecutive_failures: AtomicU32,
    consecutive_successes: AtomicU32,
}

impl ProviderHealthState {
    fn new(initially_healthy: bool) -> Arc<Self> {
        Arc::new(Self {
            healthy: AtomicBool::new(initially_healthy),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
        })
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn record(&self, status: HealthStatus, config: &HealthCheckConfig) {
        match status {
            HealthStatus::Healthy => {
                self.consecutive_failures.store(0, Ordering::Release);
                let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;
                if successes >= config.healthy_threshold {
                    let was_unhealthy = !self.healthy.load(Ordering::Acquire);
                    self.healthy.store(true, Ordering::Release);
                    if was_unhealthy {
                        tracing::info!(
                            consecutive_successes = successes,
                            "health probe: upstream marked healthy"
                        );
                    }
                }
            }
            HealthStatus::Unhealthy => {
                self.consecutive_successes.store(0, Ordering::Release);
                let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
                if failures >= config.unhealthy_threshold {
                    let was_healthy = self.healthy.load(Ordering::Acquire);
                    self.healthy.store(false, Ordering::Release);
                    if was_healthy {
                        tracing::warn!(
                            consecutive_failures = failures,
                            "health probe: upstream marked unhealthy"
                        );
                    }
                }
            }
        }
    }
}

async fn run_provider_health_probe<C: HealthChecker>(
    checker: Arc<C>,
    upstream: String,
    state: Arc<ProviderHealthState>,
    config: HealthCheckConfig,
) {
    loop {
        tokio::time::sleep(config.interval).await;

        if Arc::strong_count(&state) <= 1 {
            break;
        }

        let status = checker.check(upstream.clone()).await;
        state.record(status, &config);
    }
}

pub struct PerProviderHealthCheck<S> {
    inner: S,
    state: Arc<ProviderHealthState>,
}

impl<S: Clone> Clone for PerProviderHealthCheck<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

impl<S> PerProviderHealthCheck<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    pub fn new<C: HealthChecker>(
        inner: S,
        checker: Arc<C>,
        upstream: String,
        config: HealthCheckConfig,
    ) -> Self {
        let state = ProviderHealthState::new(true);
        let probe_state = Arc::clone(&state);
        let probe_checker = Arc::clone(&checker);

        tokio::spawn(async move {
            run_provider_health_probe(probe_checker, upstream, probe_state, config).await;
        });

        Self { inner, state }
    }

    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.state.is_healthy()
    }
}

impl<S> Service<LlmRequest> for PerProviderHealthCheck<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        if !self.state.is_healthy() {
            return Poll::Ready(Err(HiLlmError::ServiceUnavailable {
                message: "provider is unhealthy (health check failed)".into(),
                status: 503,
            }));
        }
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        if !self.state.is_healthy() {
            return Box::pin(async {
                Err(HiLlmError::ServiceUnavailable {
                    message: "provider is unhealthy (health check failed)".into(),
                    status: 503,
                })
            });
        }
        let fut = self.inner.call(req);
        Box::pin(fut)
    }
}

pub struct HealthCheckLayer {
    interval: Duration,
}

impl HealthCheckLayer {
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        Self { interval }
    }
}

impl<S> Layer<S> for HealthCheckLayer
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Service = HealthCheckService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        let healthy = Arc::new(AtomicBool::new(true));

        let probe_svc = inner.clone();
        let probe_healthy = Arc::clone(&healthy);
        let interval = self.interval;

        tokio::spawn(async move {
            run_health_probe(probe_svc, probe_healthy, interval).await;
        });

        HealthCheckService { inner, healthy }
    }
}

async fn run_health_probe<S>(mut svc: S, healthy: Arc<AtomicBool>, interval: Duration)
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + 'static,
    S::Future: Send + 'static,
{
    loop {
        tokio::time::sleep(interval).await;

        if Arc::strong_count(&healthy) <= 1 {
            break;
        }

        let result = svc.call(LlmRequest::ListModels()).await;
        let is_healthy = result.is_ok();
        healthy.store(is_healthy, Ordering::Release);

        if !is_healthy {
            tracing::warn!("health check failed; marking service as unhealthy");
        }
    }
}

pub struct HealthCheckService<S> {
    inner: S,
    healthy: Arc<AtomicBool>,
}

impl<S: Clone> Clone for HealthCheckService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            healthy: Arc::clone(&self.healthy),
        }
    }
}

impl<S> HealthCheckService<S> {
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

impl<S> Service<LlmRequest> for HealthCheckService<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        if !self.healthy.load(Ordering::Acquire) {
            return Poll::Ready(Err(HiLlmError::ServiceUnavailable {
                message: "service is unhealthy (health check failed)".into(),
                status: 503,
            }));
        }
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        if !self.healthy.load(Ordering::Acquire) {
            return Box::pin(async {
                Err(HiLlmError::ServiceUnavailable {
                    message: "service is unhealthy (health check failed)".into(),
                    status: 503,
                })
            });
        }
        let fut = self.inner.call(req);
        Box::pin(fut)
    }
}
