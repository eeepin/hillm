use super::{Provider, registry_get};
use crate::error::{HiLLMError, HiLLMResult};
use crate::provider::APIType;
use crate::provider::codec::APITypeCodec;
use std::borrow::Cow;
use std::collections::HashMap;

pub mod chat_completions_codec;
pub mod responses_codec;

pub use chat_completions_codec::OpenAIChatCompletionsCodec;
pub use responses_codec::OpenAIResponsesCodec;

pub(crate) struct OpenAIProvider {
    /// The API type this instance is bound to. Fixed at creation time.
    api_type: APIType,
}

impl OpenAIProvider {
    /// Creates a provider instance bound to `api_type`, failing with
    /// [`HiLLMError::APITypeUnsupported`] if OpenAI does not support it.
    pub(crate) fn with_api_type(api_type: APIType) -> HiLLMResult<Self> {
        let available = [APIType::OpenAIChatCompletions, APIType::OpenAIResponses];
        if !available.contains(&api_type) {
            return Err(HiLLMError::APITypeUnsupported {
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
            api_type: APIType::OpenAIChatCompletions,
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

    fn available_api_types(&self) -> Vec<APIType> {
        vec![APIType::OpenAIChatCompletions, APIType::OpenAIResponses]
    }

    fn api_type(&self) -> APIType {
        self.api_type
    }

    fn codec_for(&self, api_type: APIType) -> Option<Box<dyn APITypeCodec>> {
        // Only return a codec if the requested API type matches the bound type
        if api_type != self.api_type {
            return None;
        }
        match api_type {
            APIType::OpenAIChatCompletions => Some(Box::new(OpenAIChatCompletionsCodec)),
            APIType::OpenAIResponses => Some(Box::new(OpenAIResponsesCodec)),
            _ => None,
        }
    }
}
