use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ApiError {
    message: String,
    #[serde(default)]
    code: Option<String>,
}

/// Errors that may occur when using `hillm`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HiLlmError {
    #[error("authentication failed: {message}")]
    Authentication { message: String, status: u16 },

    #[error("rate limited: {message}")]
    RateLimited {
        message: String,
        retry_after: Option<Duration>,
    },

    #[error("bad request: {message}")]
    BadRequest { message: String, status: u16 },

    #[error("context window exceeded: {message}")]
    ContextWindowExceeded { message: String },

    #[error("content policy violation: {message}")]
    ContentPolicy { message: String },

    #[error("not found: {message}")]
    NotFound { message: String },

    #[error("server error: {message}")]
    ServerError { message: String, status: u16 },

    #[error("service unavailable: {message}")]
    ServiceUnavailable { message: String, status: u16 },

    #[error("request timeout")]
    Timeout,

    #[error(transparent)]
    Network(#[from] reqwest::Error),

    #[error("streaming error: {message}")]
    Streaming { message: String },

    #[error("provider {provider} does not support {endpoint}")]
    EndpointNotSupported { endpoint: String, provider: String },

    #[error("invalid header {name:?}: {reason}")]
    InvalidHeader { name: String, reason: String },

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("budget exceeded: {message}")]
    BudgetExceeded {
        message: String,
        model: Option<String>,
    },

    #[error("hook rejected: {message}")]
    HookRejected { message: String },

    #[error("internal error: {message}")]
    InternalError { message: String },

    #[error("outbound request to {url} forbidden: {reason}")]
    OutboundForbidden { url: String, reason: String },

    #[error("idempotency conflict: key '{key}' was already used with a different request body")]
    IdempotencyConflict { key: String },

    #[error(
        "idempotency key '{key}' is currently in-flight; retry after the first request completes"
    )]
    IdempotencyInFlight { key: String },
}

impl HiLlmError {
    #[must_use]
    pub fn status_code(&self) -> u16 {
        match self {
            Self::Authentication { status, .. } => *status,
            Self::RateLimited { .. } => 429,
            Self::BadRequest { status, .. } => *status,
            Self::ContextWindowExceeded { .. } => 400,
            Self::ContentPolicy { .. } => 400,
            Self::NotFound { .. } => 404,
            Self::ServerError { status, .. } => *status,
            Self::ServiceUnavailable { status, .. } => *status,
            Self::Timeout => 408,
            Self::Network(_) => 0,
            Self::Streaming { .. } => 0,
            Self::EndpointNotSupported { .. } => 400,
            Self::InvalidHeader { .. } => 400,
            Self::Serialization(_) => 0,
            Self::BudgetExceeded { .. } => 0,
            Self::HookRejected { .. } => 0,
            Self::InternalError { .. } => 0,
            Self::OutboundForbidden { .. } => 0,
            Self::IdempotencyConflict { .. } => 409,
            Self::IdempotencyInFlight { .. } => 409,
        }
    }

    #[must_use]
    pub fn is_transient(&self) -> bool {
        match self {
            Self::RateLimited { .. }
            | Self::ServiceUnavailable { .. }
            | Self::Timeout
            | Self::ServerError { .. } => true,
            Self::Network(_) => true,
            _ => false,
        }
    }

    #[must_use]
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::Authentication { .. } => "Authentication",
            Self::RateLimited { .. } => "RateLimited",
            Self::BadRequest { .. } => "BadRequest",
            Self::ContextWindowExceeded { .. } => "ContextWindowExceeded",
            Self::ContentPolicy { .. } => "ContentPolicy",
            Self::NotFound { .. } => "NotFound",
            Self::ServerError { .. } => "ServerError",
            Self::ServiceUnavailable { .. } => "ServiceUnavailable",
            Self::Timeout => "Timeout",
            Self::Network(_) => "Network",
            Self::Streaming { .. } => "Streaming",
            Self::EndpointNotSupported { .. } => "EndpointNotSupported",
            Self::InvalidHeader { .. } => "InvalidHeader",
            Self::Serialization(_) => "Serialization",
            Self::BudgetExceeded { .. } => "BudgetExceeded",
            Self::HookRejected { .. } => "HookRejected",
            Self::InternalError { .. } => "InternalError",
            Self::OutboundForbidden { .. } => "OutboundForbidden",
            Self::IdempotencyConflict { .. } => "IdempotencyConflict",
            Self::IdempotencyInFlight { .. } => "IdempotencyInFlight",
        }
    }

    #[cfg(feature = "tower")]
    #[allow(dead_code)]
    pub(crate) fn to_singleflight_error(&self) -> Self {
        match self {
            Self::Authentication { message, status } => Self::Authentication {
                message: message.clone(),
                status: *status,
            },
            Self::RateLimited {
                message,
                retry_after,
            } => Self::RateLimited {
                message: message.clone(),
                retry_after: *retry_after,
            },
            Self::BadRequest { message, status } => Self::BadRequest {
                message: message.clone(),
                status: *status,
            },
            Self::ContextWindowExceeded { message } => Self::ContextWindowExceeded {
                message: message.clone(),
            },
            Self::ContentPolicy { message } => Self::ContentPolicy {
                message: message.clone(),
            },
            Self::NotFound { message } => Self::NotFound {
                message: message.clone(),
            },
            Self::ServerError { message, status } => Self::ServerError {
                message: message.clone(),
                status: *status,
            },
            Self::ServiceUnavailable { message, status } => Self::ServiceUnavailable {
                message: message.clone(),
                status: *status,
            },
            Self::Timeout => Self::Timeout,
            Self::Network(e) => Self::InternalError {
                message: e.to_string(),
            },
            Self::Streaming { message } => Self::Streaming {
                message: message.clone(),
            },
            Self::EndpointNotSupported { endpoint, provider } => Self::EndpointNotSupported {
                endpoint: endpoint.clone(),
                provider: provider.clone(),
            },
            Self::InvalidHeader { name, reason } => Self::InvalidHeader {
                name: name.clone(),
                reason: reason.clone(),
            },
            Self::Serialization(e) => Self::InternalError {
                message: e.to_string(),
            },
            Self::BudgetExceeded { message, model } => Self::BudgetExceeded {
                message: message.clone(),
                model: model.clone(),
            },
            Self::HookRejected { message } => Self::HookRejected {
                message: message.clone(),
            },
            Self::InternalError { message } => Self::InternalError {
                message: message.clone(),
            },
            Self::OutboundForbidden { url, reason } => Self::OutboundForbidden {
                url: url.clone(),
                reason: reason.clone(),
            },
            Self::IdempotencyConflict { key } => Self::IdempotencyConflict { key: key.clone() },
            Self::IdempotencyInFlight { key } => Self::IdempotencyInFlight { key: key.clone() },
        }
    }

    pub(crate) fn from_status(status: u16, body: &str, retry_after: Option<Duration>) -> Self {
        let parsed = serde_json::from_str::<ErrorResponse>(body).ok();
        let code = parsed.as_ref().and_then(|r| r.error.code.clone());
        let message = parsed
            .map(|r| r.error.message)
            .unwrap_or_else(|| body.to_string());

        match status {
            401 | 403 => Self::Authentication { message, status },
            429 => Self::RateLimited {
                message,
                retry_after,
            },
            400 | 422 => {
                if code.as_deref() == Some("context_length_exceeded") {
                    Self::ContextWindowExceeded { message }
                } else if code.as_deref() == Some("content_policy_violation")
                    || code.as_deref() == Some("content_filter")
                {
                    Self::ContentPolicy { message }
                } else if message.contains("context_length_exceeded")
                    || message.contains("context window")
                    || message.contains("maximum context length")
                {
                    Self::ContextWindowExceeded { message }
                } else if message.contains("content_policy") || message.contains("content_filter") {
                    Self::ContentPolicy { message }
                } else {
                    Self::BadRequest { message, status }
                }
            }
            404 => Self::NotFound { message },
            405 | 413 => Self::BadRequest { message, status },
            408 => Self::Timeout,
            500 => Self::ServerError { message, status },
            502..=504 => Self::ServiceUnavailable { message, status },
            400..=499 => Self::BadRequest { message, status },
            _ => Self::ServerError { message, status },
        }
    }
}

pub type HiLlmResult<T> = std::result::Result<T, HiLlmError>;
