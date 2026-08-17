use super::Provider;
use super::api_type::APIType;
use std::borrow::Cow;

/// An OpenAI-compatible provider bound to an explicit API type.
///
/// Compatibility with the OpenAI wire format is not a protocol of its own:
/// every instance declares which API type it speaks (e.g.
/// [`APIType::OpenAIChatCompletions`]) and that choice is fixed for the
/// lifetime of the instance.
#[allow(dead_code)]
pub(crate) struct OpenAiCompatibleProvider {
    pub name: String,
    pub base_url: String,
    pub env_var: Option<&'static str>,
    pub models: Vec<String>,
    /// The API type this instance speaks. Fixed at creation time.
    pub api_type: APIType,
}

impl Provider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn env_var(&self) -> Option<&str> {
        self.env_var
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        Some((
            Cow::Borrowed("Authorization"),
            Cow::Owned(format!("Bearer {api_key}")),
        ))
    }

    fn matches_model(&self, model: &str) -> bool {
        // A custom base URL provider is expected to serve whatever model it
        // is given; an empty model list means "accept any model".
        self.models.is_empty() || self.models.iter().any(|model_name| model == model_name)
    }

    fn available_api_types(&self) -> Vec<APIType> {
        vec![self.api_type]
    }

    fn api_type(&self) -> APIType {
        self.api_type
    }

    fn codec_for(&self, api_type: APIType) -> Option<Box<dyn super::codec::APITypeCodec>> {
        if api_type != self.api_type {
            return None;
        }
        super::codec_for_api_type(api_type)
    }
}
