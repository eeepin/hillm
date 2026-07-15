use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};

struct CooldownState {
    cooldown_start: Option<Instant>,
}

pub struct CooldownLayer {
    duration: Duration,
}

impl CooldownLayer {
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

impl<S> Layer<S> for CooldownLayer {
    type Service = CooldownService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CooldownService {
            inner,
            duration: self.duration,
            state: Arc::new(RwLock::new(CooldownState {
                cooldown_start: None,
            })),
        }
    }
}

pub struct CooldownService<S> {
    inner: S,
    duration: Duration,
    state: Arc<RwLock<CooldownState>>,
}

impl<S: Clone> Clone for CooldownService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            duration: self.duration,
            state: Arc::clone(&self.state),
        }
    }
}

impl<S> Service<LlmRequest> for CooldownService<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let state = Arc::clone(&self.state);
        let duration = self.duration;
        let mut inner = self.inner.clone();

        Box::pin(async move {
            {
                let read = state.read().await;
                if let Some(start) = read.cooldown_start {
                    if start.elapsed() < duration {
                        return Err(HiLlmError::ServiceUnavailable {
                            message: format!(
                                "service is cooling down for {:.0}s after a transient error",
                                duration.as_secs_f64()
                            ),
                            status: 503,
                        });
                    }
                    drop(read);
                    let mut write = state.write().await;
                    if let Some(s) = write.cooldown_start
                        && s.elapsed() >= duration
                    {
                        write.cooldown_start = None;
                    }
                }
            }

            match inner.call(req).await {
                Ok(resp) => Ok(resp),
                Err(e) if e.is_transient() => {
                    // Enter cooldown.
                    let mut write = state.write().await;
                    write.cooldown_start = Some(Instant::now());
                    Err(e)
                }
                Err(e) => Err(e),
            }
        })
    }
}
