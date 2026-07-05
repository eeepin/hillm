use super::Provider;
use crate::error::{HiLlmError, HiLlmResult};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::RwLock;

static CUSTOM_PROVIDERS: RwLock<Vec<CustomProviderConfig>> = RwLock::new(Vec::new());

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderConfig {
    pub name: String,
    pub base_url: String,
    pub auth_header: AuthHeaderFormat,
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum AuthHeaderFormat {
    #[default]
    Bearer,
    ApiKey(String),
    None,
}

pub fn register_custom_provider(config: CustomProviderConfig) -> HiLlmResult<()> {
    validate_config(&config)?;
    crate::provider::validate_outbound_url_sync(&config.base_url)?;
    let mut providers = CUSTOM_PROVIDERS
        .write()
        .map_err(|e| HiLlmError::ServerError {
            message: format!("custom provider registry lock poisoned: {e}"),
            status: 500,
        })?;
    if let Some(existing) = providers.iter_mut().find(|p| p.name == config.name) {
        *existing = config;
    } else {
        providers.push(config);
    }

    Ok(())
}

pub fn unregister_custom_provider(name: &str) -> HiLlmResult<bool> {
    let mut providers = CUSTOM_PROVIDERS
        .write()
        .map_err(|e| HiLlmError::ServerError {
            message: format!("custom provider registry lock poisoned: {e}"),
            status: 500,
        })?;

    let before = providers.len();
    providers.retain(|p| p.name != name);
    Ok(providers.len() < before)
}

pub(crate) fn detect_custom_provider(name: &str) -> Option<Box<dyn Provider>> {
    let providers = CUSTOM_PROVIDERS.read().ok()?;

    for cfg in providers.iter() {
        let matches = cfg.name == name;

        if matches {
            return Some(Box::new(CustomProvider {
                config: cfg.clone(),
            }));
        }
    }

    None
}

#[cfg(test)]
pub(crate) fn clear_custom_providers() {
    if let Ok(mut providers) = CUSTOM_PROVIDERS.write() {
        providers.clear();
    }
}

fn validate_config(config: &CustomProviderConfig) -> HiLlmResult<()> {
    if config.name.trim().is_empty() {
        return Err(HiLlmError::BadRequest {
            message: "custom provider name must not be empty or whitespace-only".into(),
            status: 400,
        });
    }
    if config.base_url.trim().is_empty() {
        return Err(HiLlmError::BadRequest {
            message: "custom provider base_url must not be empty or whitespace-only".into(),
            status: 400,
        });
    }
    if config.models.is_empty() {
        return Err(HiLlmError::BadRequest {
            message: "custom provider must have at least one model".into(),
            status: 400,
        });
    }
    for model_name in &config.models {
        if model_name.is_empty() {
            return Err(HiLlmError::BadRequest {
                message: "custom provider's model name must not be empty".into(),
                status: 400,
            });
        }
    }
    Ok(())
}

// Provider implementation

pub(crate) struct CustomProvider {
    config: CustomProviderConfig,
}

impl Provider for CustomProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn base_url(&self) -> &str {
        &self.config.base_url
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        match &self.config.auth_header {
            AuthHeaderFormat::Bearer => Some((
                Cow::Borrowed("Authorization"),
                Cow::Owned(format!("Bearer {api_key}")),
            )),
            AuthHeaderFormat::ApiKey(header_name) => {
                Some((Cow::Owned(header_name.clone()), Cow::Borrowed(api_key)))
            }
            AuthHeaderFormat::None => None,
        }
    }

    fn matches_model(&self, model: &str) -> bool {
        self.config
            .models
            .iter()
            .any(|model_name| model == model_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutex to serialize tests that share the global custom-provider registry.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire the test lock and clear the registry.
    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_custom_providers();
        guard
    }

    #[test]
    fn register_and_detect_by_model_prefix() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "my-provider".into(),
            base_url: "https://api.my-provider.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["my-".into(), "my-provider/".into()],
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider = detect_custom_provider("my-model-7b");
        assert!(
            provider.is_some(),
            "should detect custom provider by prefix 'my-'"
        );
        let provider = provider.expect("custom provider should be found");
        assert_eq!(provider.name(), "my-provider");
        assert_eq!(provider.base_url(), "https://api.my-provider.com/v1");

        // Also detect via slash-prefix routing.
        let provider2 = detect_custom_provider("my-provider/llama-70b");
        assert!(
            provider2.is_some(),
            "should detect custom provider by slash prefix"
        );

        // Non-matching model should not detect.
        let none = detect_custom_provider("gpt-4");
        assert!(none.is_none(), "should not match unrelated model");
    }

    #[test]
    fn unregister_removes_provider() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "ephemeral".into(),
            base_url: "https://api.ephemeral.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["eph-".into()],
        };

        register_custom_provider(config).expect("registration should succeed");
        assert!(detect_custom_provider("eph-model").is_some());

        let removed = unregister_custom_provider("ephemeral").expect("unregister should succeed");
        assert!(removed, "should return true when provider was found");

        assert!(
            detect_custom_provider("eph-model").is_none(),
            "should no longer detect after unregister"
        );

        // Unregistering again returns false.
        let removed_again =
            unregister_custom_provider("ephemeral").expect("unregister should succeed");
        assert!(
            !removed_again,
            "should return false when provider not found"
        );
    }

    #[test]
    fn custom_provider_with_api_key_auth() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "secure-provider".into(),
            base_url: "https://api.secure.com/v1".into(),
            auth_header: AuthHeaderFormat::ApiKey("X-Custom-Auth".into()),
            models: vec!["secure/".into()],
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider = detect_custom_provider("secure/model-1").expect("should detect provider");
        let (header_name, header_value) = provider
            .auth_header("my-secret-key")
            .expect("should return auth header");
        assert_eq!(header_name.as_ref(), "X-Custom-Auth");
        assert_eq!(header_value.as_ref(), "my-secret-key");
    }

    #[test]
    fn custom_provider_with_no_auth() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "local-provider".into(),
            base_url: "http://localhost:8080/v1".into(),
            auth_header: AuthHeaderFormat::None,
            models: vec!["local/".into()],
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider = detect_custom_provider("local/model").expect("should detect provider");
        assert!(
            provider.auth_header("unused").is_none(),
            "no-auth provider should return None"
        );
    }

    #[test]
    fn custom_provider_bearer_auth() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "bearer-provider".into(),
            base_url: "https://api.bearer.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["bearer/".into()],
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider = detect_custom_provider("bearer/model").expect("should detect provider");
        let (header_name, header_value) = provider
            .auth_header("my-token")
            .expect("should return auth header");
        assert_eq!(header_name.as_ref(), "Authorization");
        assert_eq!(header_value.as_ref(), "Bearer my-token");
    }

    #[test]
    fn register_replaces_existing_provider() {
        let _guard = setup();

        let config1 = CustomProviderConfig {
            name: "updatable".into(),
            base_url: "https://old.example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["upd/".into()],
        };
        register_custom_provider(config1).expect("first registration should succeed");

        let config2 = CustomProviderConfig {
            name: "updatable".into(),
            base_url: "https://new.example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["upd/".into()],
        };
        register_custom_provider(config2).expect("second registration should succeed");

        let provider = detect_custom_provider("upd/model").expect("should detect provider");
        assert_eq!(
            provider.base_url(),
            "https://new.example.com/v1",
            "should use the updated config"
        );
    }

    #[test]
    fn validation_rejects_empty_name() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: String::new(),
            base_url: "https://example.com".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["x/".into()],
        };
        let result = register_custom_provider(config);
        assert!(result.is_err(), "should reject empty name");
    }

    #[test]
    fn validation_rejects_empty_base_url() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "valid-name".into(),
            base_url: String::new(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["x/".into()],
        };
        let result = register_custom_provider(config);
        assert!(result.is_err(), "should reject empty base_url");
    }

    #[test]
    fn validation_rejects_no_prefixes() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "valid-name".into(),
            base_url: "https://example.com".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec![],
        };
        let result = register_custom_provider(config);
        assert!(result.is_err(), "should reject empty models");
    }

    #[test]
    fn config_serde_round_trip() {
        let config = CustomProviderConfig {
            name: "serde-test".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::ApiKey("X-Api-Key".into()),
            models: vec!["serde/".into()],
        };

        let json = serde_json::to_string(&config).expect("should serialize");
        let parsed: CustomProviderConfig = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(parsed.name, "serde-test");
        assert_eq!(parsed.base_url, "https://example.com/v1");
        assert_eq!(parsed.models, vec!["serde/"]);
    }
}
