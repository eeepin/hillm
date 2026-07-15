use std::time::Duration;

use thiserror::Error;

use crate::error::HiLlmError;

#[derive(Debug, Error)]
#[error("circuit open for provider {provider}: retry after {retry_after:?}")]
pub struct CircuitOpenError {
    pub provider: String,
    pub retry_after: Duration,
}

#[derive(Debug, Error)]
#[error("all {attempts} hedged attempts exhausted")]
pub struct HedgeExhaustedError {
    pub attempts: u32,
}

impl From<CircuitOpenError> for HiLlmError {
    fn from(e: CircuitOpenError) -> Self {
        // TODO(1.A): replace with a dedicated CircuitOpen variant.
        Self::ServiceUnavailable {
            message: e.to_string(),
            status: 503,
        }
    }
}

impl From<HedgeExhaustedError> for HiLlmError {
    fn from(_e: HedgeExhaustedError) -> Self {
        // TODO(1.A): replace with a dedicated HedgeExhausted variant.
        Self::Timeout
    }
}
