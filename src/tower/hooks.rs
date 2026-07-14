use std::cell::Cell;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::task::{Context, Poll};
use std::time::Instant;

use futures_util::FutureExt as _;
use tower::Layer;
use tower::Service;

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::observability::usage::{CacheState, UsageEvent, UsageEventOutcome, UsageSinkErased};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

// Hook trait
/// Callback trait
pub trait LlmHook: Send + Sync + 'static {
    /// Called before the request
    fn on_request(
        &self,
        _req: &LlmRequest,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }

    /// Called after the response
    fn on_response(
        &self,
        _req: &LlmRequest,
        _resp: &LlmResponse,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }

    /// Called when errors occur
    fn on_error(
        &self,
        _req: &LlmRequest,
        _err: &HiLlmError,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}
