//! API request and response traits for type-safe LLM API interactions.

use serde::{Serialize, de::DeserializeOwned};

use crate::error::HiLLMResult;

/// Trait for API requests that can be sent to LLM providers.
///
/// This trait defines the common interface for all request types across
/// different API protocols (OpenAI Chat, OpenAI Responses, Anthropic Messages, etc.).
pub trait APIRequest: Serialize + Send + Sync {
    /// The response type this request expects.
    type Response: APIResponse;

    /// The stream event type for streaming responses.
    type StreamEvent: Send + Sync;

    /// Returns the model name for this request.
    fn model(&self) -> &str;

    /// Returns whether this request should be streamed.
    ///
    /// Default implementation returns `false`.
    fn stream(&self) -> bool {
        false
    }
}

/// Trait for API responses from LLM providers.
///
/// This trait defines the common interface for all response types across
/// different API protocols.
pub trait APIResponse: DeserializeOwned + Send + Sync {
    /// Validates the response after deserialization.
    ///
    /// This can be used to check for provider-specific error conditions
    /// or to normalize response data.
    ///
    /// Default implementation returns `Ok(())`.
    fn validate(&self) -> HiLLMResult<()> {
        Ok(())
    }
}
