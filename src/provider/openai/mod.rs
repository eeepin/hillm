use super::{Provider, registry_get};
use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::ApiType;
use crate::provider::codec::ApiTypeCodec;
use std::borrow::Cow;
use std::collections::HashMap;

pub mod chat_completions_codec;
pub mod responses_codec;

pub use chat_completions_codec::OpenAIChatCompletionsCodec;
pub use responses_codec::OpenAIResponsesCodec;

pub(crate) struct OpenAIProvider {
    /// The API type this instance is bound to. Fixed at creation time.
    api_type: ApiType,
}

impl OpenAIProvider {
    /// Creates a provider instance bound to `api_type`, failing with
    /// [`HiLlmError::ApiTypeUnsupported`] if OpenAI does not support it.
    pub(crate) fn with_api_type(api_type: ApiType) -> HiLlmResult<Self> {
        let available = [ApiType::OpenAIChatCompletions, ApiType::OpenAIResponses];
        if !available.contains(&api_type) {
            return Err(HiLlmError::ApiTypeUnsupported {
                api_type: api_type.to_string(),
                provider: "openai".to_string(),
            });
        }
        Ok(Self { api_type })
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self {
            api_type: ApiType::OpenAIChatCompletions,
        }
    }
}

impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn base_url(&self) -> &str {
        "https://api.openai.com/v1"
    }

    fn env_vars(&self) -> HashMap<&str, &str> {
        [("api_key", "OPENAI_API_KEY")].into_iter().collect()
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        Some((
            Cow::Borrowed("Authorization"),
            Cow::Owned(format!("Bearer {api_key}")),
        ))
    }

    fn matches_model(&self, model: &str) -> bool {
        registry_get().is_some_and(|reg| {
            reg.get("openai")
                .is_some_and(|p| p.models.contains_key(model))
        })
    }

    fn available_api_types(&self) -> Vec<ApiType> {
        vec![ApiType::OpenAIChatCompletions, ApiType::OpenAIResponses]
    }

    fn api_type(&self) -> ApiType {
        self.api_type
    }

    fn codec_for(&self, api_type: ApiType) -> Option<Box<dyn ApiTypeCodec>> {
        // Only return a codec if the requested API type matches the bound type
        if api_type != self.api_type {
            return None;
        }
        match api_type {
            ApiType::OpenAIChatCompletions => Some(Box::new(OpenAIChatCompletionsCodec)),
            ApiType::OpenAIResponses => Some(Box::new(OpenAIResponsesCodec)),
            _ => None,
        }
    }
}
