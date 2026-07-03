use super::{AuthType, Provider, ProviderConfig};
use crate::error::HiLlmResult;
use std::borrow::Cow;

pub(crate) struct ConfigDrivenProvider {
    config: &'static ProviderConfig,
}

impl ConfigDrivenProvider {
    #[must_use]
    pub(crate) fn new(config: &'static ProviderConfig) -> Self {
        Self { config }
    }
}

impl Provider for ConfigDrivenProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or("")
    }

    fn env_var(&self) -> Option<&str> {
        self.config.auth.as_ref().and_then(|a| a.env_var.as_deref())
    }

    fn transform_request(&self, body: &mut serde_json::Value) -> HiLlmResult<()> {
        if let Some(mappings) = &self.config.param_mappings
            && let Some(obj) = body.as_object_mut()
        {
            for (from, to) in mappings {
                if let Some(val) = obj.remove(from.as_str()) {
                    obj.insert(to.clone(), val);
                }
            }
        }
        Ok(())
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        let auth_type = self
            .config
            .auth
            .as_ref()
            .map(|a| &a.auth_type)
            .unwrap_or(&AuthType::Bearer);

        match auth_type {
            AuthType::None => None,
            AuthType::ApiKey => Some((Cow::Borrowed("x-api-key"), Cow::Borrowed(api_key))),
            AuthType::Bearer | AuthType::Unknown => Some((
                Cow::Borrowed("Authorization"),
                Cow::Owned(format!("Bearer {api_key}")),
            )),
        }
    }

    async fn matches_model(&self, model: &str) -> bool {
        self.config
            .models
            .iter()
            .any(|model_name| model == model_name)
    }
}
