use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::tenant::TenantId;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsageEvent {
    pub tenant_id: Option<TenantId>,
    pub request_id: String,
    pub model: String,
    pub provider: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: Decimal,
    pub cache_state: CacheState,
    pub effective_model: Option<String>,
    pub finish_reason: Option<String>,
    pub outcome: UsageEventOutcome,
    pub latency_ms: u64,
    pub metadata: HashMap<String, String>,
    pub received_at: SystemTime,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheState {
    Miss,
    ExactHit,
    SemanticHit,
    StaleHit,
    #[default]
    Bypass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageEventOutcome {
    Success,
    Error,
    Cancelled,
    TimedOut,
}

pub trait UsageSink: Send + Sync + 'static {
    fn emit(&self, event: UsageEvent) -> impl Future<Output = Result<(), UsageSinkError>> + Send;
}

pub trait UsageSinkErased: Send + Sync + 'static {
    fn emit_erased<'a>(
        &'a self,
        event: UsageEvent,
    ) -> Pin<Box<dyn Future<Output = Result<(), UsageSinkError>> + Send + 'a>>;
}

impl<T: UsageSink> UsageSinkErased for T {
    fn emit_erased<'a>(
        &'a self,
        event: UsageEvent,
    ) -> Pin<Box<dyn Future<Output = Result<(), UsageSinkError>> + Send + 'a>> {
        Box::pin(self.emit(event))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UsageSinkError {
    #[error("Usage sink backend error: {0}")]
    Backend(String),
}

#[derive(Clone, Debug, Default)]
pub struct LoggingUsageSink;

impl UsageSink for LoggingUsageSink {
    #[cfg_attr(not(feature = "tracing"), allow(unused_variables))]
    async fn emit(&self, event: UsageEvent) -> Result<(), UsageSinkError> {
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "gen_ai.usage",
            tenant_id = event.tenant_id.as_ref().map(|t| t.as_ref()),
            request_id = %event.request_id,
            model = %event.model,
            effective_model = event.effective_model.as_deref(),
            provider = %event.provider,
            prompt_tokens = event.prompt_tokens,
            completion_tokens = event.completion_tokens,
            cost_usd = %event.cost_usd,
            cache_state = ?event.cache_state,
            outcome = ?event.outcome,
            latency_ms = event.latency_ms,
            "usage_event"
        );
        Ok(())
    }
}

pub struct MultiUsageSink {
    sinks: Vec<Arc<dyn UsageSinkErased>>,
}

impl MultiUsageSink {
    #[must_use]
    pub fn from_sinks<S: UsageSink>(sinks: Vec<Arc<S>>) -> Self {
        Self {
            sinks: sinks
                .into_iter()
                .map(|s| s as Arc<dyn UsageSinkErased>)
                .collect(),
        }
    }

    #[must_use]
    pub fn from_erased(sinks: Vec<Arc<dyn UsageSinkErased>>) -> Self {
        Self { sinks }
    }

    #[must_use]
    pub fn empty() -> Self {
        Self { sinks: Vec::new() }
    }

    pub fn push<S: UsageSink>(&mut self, sink: Arc<S>) {
        self.sinks.push(sink as Arc<dyn UsageSinkErased>);
    }

    pub fn push_erased(&mut self, sink: Arc<dyn UsageSinkErased>) {
        self.sinks.push(sink);
    }
}

impl UsageSink for MultiUsageSink {
    async fn emit(&self, event: UsageEvent) -> Result<(), UsageSinkError> {
        for sink in &self.sinks {
            if let Err(_err) = sink.emit_erased(event.clone()).await {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    target: "gen_ai.usage",
                    error = %_err,
                    "usage sink emit failed"
                );
            }
        }
        Ok(())
    }
}
