//! Base provider trait for LLM service identity and authentication.

use std::borrow::Cow;

use crate::error::HiLlmResult;
use crate::provider::APIType;

/// Base trait for LLM providers, defining identity, authentication, and model matching.
///
/// This trait focuses on provider-level concerns: who are you, how do I authenticate,
/// and which models do you support? Protocol-specific encoding/decoding is handled
/// by `APITypeCodec` implementations.
pub trait BaseProvider: Send + Sync {
    /// Returns the provider's unique name (e.g., "openai", "anthropic").
    fn name(&self) -> &str;

    /// Returns the base URL for API requests (e.g., "https://api.openai.com/v1").
    fn base_url(&self) -> &str;

    /// Returns the authentication header for the given API key.
    ///
    /// Returns `None` if no authentication is required or if the provider
    /// uses a different authentication mechanism (e.g., AWS SigV4).
    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)>;

    /// Returns additional static headers required by this provider.
    ///
    /// Default implementation returns an empty slice.
    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// Returns dynamic headers based on the request body.
    ///
    /// This is used for headers that depend on request content, such as
    /// Anthropic's `anthropic-beta` header which varies based on features used.
    ///
    /// Default implementation returns an empty vector.
    fn dynamic_headers(&self, _body: &serde_json::Value) -> Vec<(String, String)> {
        vec![]
    }

    /// Checks if this provider supports the given model.
    fn matches_model(&self, model: &str) -> bool;

    /// Returns the list of API types supported by this provider.
    fn available_api_types(&self) -> Vec<APIType>;

    /// Returns the environment variable name for the API key, if any.
    ///
    /// Default implementation returns `None`.
    fn env_var(&self) -> Option<&'static str> {
        None
    }

    /// Validates the provider configuration.
    ///
    /// This is called during provider construction to ensure all required
    /// configuration is present (e.g., AWS credentials for Bedrock).
    ///
    /// Default implementation returns `Ok(())`.
    fn validate(&self) -> HiLlmResult<()> {
        Ok(())
    }
}
