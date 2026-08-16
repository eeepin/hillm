use super::Provider;
use super::api_type::APIType;
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
    /// API types this provider supports. Empty defaults to `[OpenAIChatCompletions]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_api_types: Vec<APIType>,
    /// The default API type to use when creating a provider instance.
    /// Must be one of `available_api_types` if both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_api_type: Option<APIType>,
}

impl CustomProviderConfig {
    /// Returns the effective available API types, falling back to
    /// `[OpenAIChatCompletions]` if empty.
    #[must_use]
    pub fn effective_api_types(&self) -> Vec<APIType> {
        if self.available_api_types.is_empty() {
            vec![APIType::OpenAIChatCompletions]
        } else {
            self.available_api_types.clone()
        }
    }

    /// Returns the effective default API type, falling back to the first
    /// available type.
    #[must_use]
    pub fn effective_default_api_type(&self) -> APIType {
        self.default_api_type
            .unwrap_or_else(|| self.effective_api_types()[0])
    }

    /// Validates the API type configuration: `default_api_type`, when set,
    /// must be one of the (effective) available API types.
    pub fn validate_api_types(&self) -> HiLlmResult<()> {
        if let Some(default) = self.default_api_type {
            let available = self.effective_api_types();
            if !available.contains(&default) {
                return Err(HiLlmError::BadRequest {
                    message: format!(
                        "custom provider '{}': default_api_type '{default}' is not in \
                         available_api_types {:?}",
                        self.name, available
                    ),
                    status: 400,
                });
            }
        }
        Ok(())
    }
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
    config.validate_api_types()?;
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

/// API type filter for custom provider detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApiTypeFilter {
    /// No explicit API type constraint; fall back to the provider's
    /// configured default API type. This preserves the legacy behavior of
    /// `detect_custom_provider(name, model)`.
    Any,
    /// Only match providers that declare support for this API type.
    Exact(APIType),
}

/// Detects a registered custom provider by explicit name first, then by
/// model match, honoring the API type filter.
///
/// Returns:
/// - `Ok(Some(provider))` when exactly one provider matches;
/// - `Ok(None)` when nothing matches;
/// - `Err(HiLlmError::AmbiguousProvider)` when a non-empty `model` matches
///   several providers and no provider name was given — callers must not
///   silently pick one by registration order.
pub(crate) fn detect_custom_provider(
    name: &str,
    model: &str,
    filter: ApiTypeFilter,
) -> Result<Option<Box<dyn Provider>>, HiLlmError> {
    let providers = CUSTOM_PROVIDERS
        .read()
        .map_err(|e| HiLlmError::ServerError {
            message: format!("custom provider registry lock poisoned: {e}"),
            status: 500,
        })?;

    let supports = |cfg: &CustomProviderConfig| match filter {
        ApiTypeFilter::Any => true,
        ApiTypeFilter::Exact(api_type) => cfg.effective_api_types().contains(&api_type),
    };

    // First, try to match by provider name (exact match). Names are unique
    // in the registry (registration replaces), so this cannot be ambiguous.
    for cfg in providers.iter() {
        if cfg.name == name && supports(cfg) {
            return Ok(Some(Box::new(CustomProvider::from_config(
                cfg.clone(),
                filter,
            ))));
        }
    }

    // If no match by name, try to match by model.
    if !model.is_empty() {
        let mut candidates: Vec<&CustomProviderConfig> = Vec::new();
        for cfg in providers.iter() {
            if supports(cfg) && cfg.models.iter().any(|model_name| model == model_name) {
                candidates.push(cfg);
            }
        }
        match candidates.len() {
            0 => {}
            1 => {
                return Ok(Some(Box::new(CustomProvider::from_config(
                    candidates[0].clone(),
                    filter,
                ))));
            }
            _ => {
                let mut names: Vec<String> =
                    candidates.iter().map(|cfg| cfg.name.clone()).collect();
                names.sort();
                return Err(HiLlmError::AmbiguousProvider {
                    model: model.to_string(),
                    candidates: names,
                });
            }
        }
    }

    Ok(None)
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
    /// The API type this instance is bound to. Fixed at creation time.
    api_type: APIType,
}

impl CustomProvider {
    fn from_config(config: CustomProviderConfig, filter: ApiTypeFilter) -> Self {
        let api_type = match filter {
            ApiTypeFilter::Exact(api_type) => api_type,
            ApiTypeFilter::Any => config.effective_default_api_type(),
        };
        Self { config, api_type }
    }
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

    fn available_api_types(&self) -> Vec<APIType> {
        self.config.effective_api_types()
    }

    fn api_type(&self) -> APIType {
        self.api_type
    }

    fn codec_for(
        &self,
        api_type: APIType,
    ) -> Option<Box<dyn crate::provider::codec::APITypeCodec>> {
        if !self.config.effective_api_types().contains(&api_type) {
            return None;
        }
        crate::provider::codec_for_api_type(api_type)
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

    /// Unwrap the detect result for assertions.
    fn detect_ok(name: &str, model: &str, filter: ApiTypeFilter) -> Option<Box<dyn Provider>> {
        detect_custom_provider(name, model, filter).unwrap()
    }

    #[test]
    fn register_and_detect_by_model_name() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "my-provider".into(),
            base_url: "https://api.my-provider.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["my-model-7b".into(), "my-provider-llama-70b".into()],
            available_api_types: vec![],
            default_api_type: None,
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider = detect_ok("", "my-model-7b", ApiTypeFilter::Any);
        assert!(
            provider.is_some(),
            "should detect custom provider by model name 'my-model-7b'"
        );
        let provider = provider.expect("custom provider should be found");
        assert_eq!(provider.name(), "my-provider");
        assert_eq!(provider.base_url(), "https://api.my-provider.com/v1");

        // Also detect via second model name.
        let provider2 = detect_ok("", "my-provider-llama-70b", ApiTypeFilter::Any);
        assert!(
            provider2.is_some(),
            "should detect custom provider by model name 'my-provider-llama-70b'"
        );

        // Non-matching model should not detect.
        let none = detect_ok("", "gpt-4", ApiTypeFilter::Any);
        assert!(none.is_none(), "should not match unrelated model");
    }

    #[test]
    fn unregister_removes_provider() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "ephemeral".into(),
            base_url: "https://api.ephemeral.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["eph-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };

        register_custom_provider(config).expect("registration should succeed");
        assert!(detect_ok("", "eph-model", ApiTypeFilter::Any).is_some());

        let removed = unregister_custom_provider("ephemeral").expect("unregister should succeed");
        assert!(removed, "should return true when provider was found");

        assert!(
            detect_ok("", "eph-model", ApiTypeFilter::Any).is_none(),
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
            models: vec!["secure-model-1".into()],
            available_api_types: vec![],
            default_api_type: None,
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider =
            detect_ok("", "secure-model-1", ApiTypeFilter::Any).expect("should detect provider");
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
            models: vec!["local-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider =
            detect_ok("", "local-model", ApiTypeFilter::Any).expect("should detect provider");
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
            models: vec!["bearer-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };

        register_custom_provider(config).expect("registration should succeed");

        let provider =
            detect_ok("", "bearer-model", ApiTypeFilter::Any).expect("should detect provider");
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
            models: vec!["upd-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };
        register_custom_provider(config1).expect("first registration should succeed");

        let config2 = CustomProviderConfig {
            name: "updatable".into(),
            base_url: "https://new.example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["upd-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };
        register_custom_provider(config2).expect("second registration should succeed");

        let provider =
            detect_ok("", "upd-model", ApiTypeFilter::Any).expect("should detect provider");
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
            models: vec!["model-a".into()],
            available_api_types: vec![],
            default_api_type: None,
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
            models: vec!["model-b".into()],
            available_api_types: vec![],
            default_api_type: None,
        };
        let result = register_custom_provider(config);
        assert!(result.is_err(), "should reject empty base_url");
    }

    #[test]
    fn validation_rejects_no_models() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "valid-name".into(),
            base_url: "https://example.com".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec![],
            available_api_types: vec![],
            default_api_type: None,
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
            models: vec!["serde-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };

        let json = serde_json::to_string(&config).expect("should serialize");
        let parsed: CustomProviderConfig = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(parsed.name, "serde-test");
        assert_eq!(parsed.base_url, "https://example.com/v1");
        assert_eq!(parsed.models, vec!["serde-model"]);
    }

    #[test]
    fn config_serde_round_trip_with_api_types() {
        let config = CustomProviderConfig {
            name: "serde-test".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["serde-model".into()],
            available_api_types: vec![APIType::AnthropicMessages],
            default_api_type: Some(APIType::AnthropicMessages),
        };

        let json = serde_json::to_string(&config).expect("should serialize");
        assert!(json.contains("anthropic_messages"));
        let parsed: CustomProviderConfig = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(parsed.available_api_types, vec![APIType::AnthropicMessages]);
        assert_eq!(parsed.default_api_type, Some(APIType::AnthropicMessages));
    }

    #[test]
    fn register_rejects_invalid_default_api_type() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "bad-default".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["model-a".into()],
            available_api_types: vec![APIType::OpenAIChatCompletions],
            default_api_type: Some(APIType::AnthropicMessages),
        };
        let result = register_custom_provider(config);
        assert!(result.is_err(), "should reject default not in available");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("default_api_type"));
    }

    #[test]
    fn detect_respects_api_type_filter() {
        let _guard = setup();

        // A custom provider that only speaks the Anthropic Messages protocol.
        let config = CustomProviderConfig {
            name: "anthropic-ish".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["claude-like".into()],
            available_api_types: vec![APIType::AnthropicMessages],
            default_api_type: None,
        };
        register_custom_provider(config).expect("registration should succeed");

        // Exact filter for an unsupported API type must not match.
        assert!(
            detect_ok(
                "anthropic-ish",
                "",
                ApiTypeFilter::Exact(APIType::OpenAIChatCompletions)
            )
            .is_none(),
            "provider without OpenAI Chat support must not match an Exact Chat filter"
        );

        // Exact filter for the supported API type matches and binds it.
        let provider = detect_ok(
            "anthropic-ish",
            "",
            ApiTypeFilter::Exact(APIType::AnthropicMessages),
        )
        .expect("should match supported api type");
        assert_eq!(provider.api_type(), APIType::AnthropicMessages);
        assert_eq!(
            provider.available_api_types(),
            vec![APIType::AnthropicMessages]
        );
    }

    #[test]
    fn detect_by_model_respects_api_type_filter() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "multi-model".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["mm-1".into()],
            available_api_types: vec![APIType::OpenAIChatCompletions],
            default_api_type: None,
        };
        register_custom_provider(config).expect("registration should succeed");

        // Model match alone is not enough when the API type is not supported.
        assert!(
            detect_ok("", "mm-1", ApiTypeFilter::Exact(APIType::AnthropicMessages)).is_none(),
            "model match must not bypass the api type filter"
        );

        // Supported API type + model match returns a bound instance.
        let provider = detect_ok(
            "",
            "mm-1",
            ApiTypeFilter::Exact(APIType::OpenAIChatCompletions),
        )
        .expect("should match");
        assert_eq!(provider.api_type(), APIType::OpenAIChatCompletions);
    }

    #[test]
    fn detect_by_model_is_ambiguous_when_multiple_providers_match() {
        let _guard = setup();

        let config_a = CustomProviderConfig {
            name: "amb-a".into(),
            base_url: "https://a.example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["shared-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };
        let config_b = CustomProviderConfig {
            name: "amb-b".into(),
            base_url: "https://b.example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["shared-model".into()],
            available_api_types: vec![],
            default_api_type: None,
        };
        register_custom_provider(config_a).expect("registration should succeed");
        register_custom_provider(config_b).expect("registration should succeed");

        let result = detect_custom_provider("", "shared-model", ApiTypeFilter::Any);
        match result {
            Err(HiLlmError::AmbiguousProvider { model, candidates }) => {
                assert_eq!(model, "shared-model");
                assert_eq!(candidates, vec!["amb-a".to_string(), "amb-b".to_string()]);
            }
            Err(other) => panic!("expected AmbiguousProvider, got error: {other}"),
            Ok(Some(_)) => panic!("expected AmbiguousProvider, got a provider"),
            Ok(None) => panic!("expected AmbiguousProvider, got None"),
        }

        // An explicit provider name resolves the ambiguity.
        let provider = detect_ok("amb-a", "", ApiTypeFilter::Any).expect("name match");
        assert_eq!(provider.name(), "amb-a");

        // An API type filter that only one provider satisfies also resolves it.
        assert!(
            detect_ok(
                "",
                "shared-model",
                ApiTypeFilter::Exact(APIType::AnthropicMessages)
            )
            .is_none(),
            "no provider supports AnthropicMessages, so nothing may match"
        );
    }

    #[test]
    fn any_filter_binds_configured_default_api_type() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "defaulted".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["d-model".into()],
            available_api_types: vec![APIType::OpenAIResponses, APIType::OpenAIChatCompletions],
            default_api_type: Some(APIType::OpenAIResponses),
        };
        register_custom_provider(config).expect("registration should succeed");

        let provider =
            detect_ok("defaulted", "", ApiTypeFilter::Any).expect("should detect by name");
        assert_eq!(provider.api_type(), APIType::OpenAIResponses);
    }

    #[test]
    fn custom_provider_codec_matches_api_type() {
        let _guard = setup();

        let config = CustomProviderConfig {
            name: "codec-check".into(),
            base_url: "https://example.com/v1".into(),
            auth_header: AuthHeaderFormat::Bearer,
            models: vec!["cc-model".into()],
            available_api_types: vec![APIType::OpenAIChatCompletions],
            default_api_type: None,
        };
        register_custom_provider(config).expect("registration should succeed");

        let provider = detect_ok("codec-check", "", ApiTypeFilter::Any).expect("should detect");
        let codec = provider
            .codec_for(APIType::OpenAIChatCompletions)
            .expect("codec for supported api type");
        assert_eq!(codec.api_type(), APIType::OpenAIChatCompletions);
        assert_eq!(codec.endpoint_path(), "/chat/completions");
        assert!(
            provider.codec_for(APIType::AnthropicMessages).is_none(),
            "codec for unsupported api type must be None"
        );
    }
}
