use super::{APIType, AuthType, Provider, ProviderConfig};
use crate::error::HiLLMResult;
use std::borrow::Cow;
use std::collections::HashMap;

pub(crate) struct ConfigDrivenProvider {
    config: ProviderConfig,
}

impl ConfigDrivenProvider {
    #[must_use]
    pub(crate) fn new(config: ProviderConfig) -> Self {
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

    fn env_vars(&self) -> HashMap<&str, &str> {
        self.config
            .auth
            .as_ref()
            .map(|a| {
                a.env_vars
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn transform_request(&self, body: &mut serde_json::Value) -> HiLLMResult<()> {
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

    fn matches_model(&self, model: &str) -> bool {
        self.config
            .models
            .iter()
            .any(|model_name| model == model_name)
    }

    fn available_api_types(&self) -> Vec<APIType> {
        self.config.effective_api_types()
    }

    fn api_type(&self) -> APIType {
        self.config.effective_default_api_type()
    }

    fn codec_for(&self, api_type: APIType) -> Option<Box<dyn super::codec::APITypeCodec>> {
        if !self.config.effective_api_types().contains(&api_type) {
            return None;
        }
        super::codec_for_api_type(api_type)
    }
}
