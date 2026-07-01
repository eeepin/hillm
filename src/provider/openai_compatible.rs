use super::common::{Provider, registry};
use std::borrow::Cow;

pub(crate) struct OpenAiCompatibleProvider {
    pub name: String,
    pub base_url: String,
    pub env_var: Option<&'static str>,
    pub model_prefixes: Vec<String>,
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

    async fn matches_model(&self, model: &str) -> bool {
        if let Ok(reg) = registry().await {
            reg.get(self.name())
                .is_some_and(|p| p.models.contains_key(model))
        } else {
            false
        }
    }
}
