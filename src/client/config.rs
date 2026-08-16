use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;

use crate::auth::CredentialProvider;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::error::{HiLLMError, HiLLMResult};
use crate::http::transport::TransportConfig;
use crate::provider::APIType;
use crate::provider::outbound_policy::OutboundPolicyValidator;
#[cfg(feature = "tower")]
use crate::tower::{BudgetConfig, CacheConfig, CacheStore, LlmHook, RateLimitConfig};

#[derive(Clone)]
pub struct ClientConfig {
    pub api_key: SecretString,
    pub base_url: Option<String>,
    /// Explicit API type selection for the provider instance.
    ///
    /// When `None`, the provider's effective default API type is used.
    pub api_type: Option<APIType>,
    pub timeout: Duration,
    pub max_retries: u32,
    pub(crate) extra_headers: Vec<(String, String)>,
    pub credential_provider: Option<Arc<dyn CredentialProvider>>,
    pub load_env: bool,
    pub transport: TransportConfig,
    /// Per-client outbound policy validator.
    ///
    /// When `Some`, this validator is used for URL and DNS checks instead
    /// of the process-global policy. Enables per-tenant isolation in
    /// server-side deployments.
    pub outbound_policy: Option<Arc<OutboundPolicyValidator>>,
    #[cfg(feature = "tower")]
    pub cache_config: Option<CacheConfig>,
    #[cfg(feature = "tower")]
    pub cache_store: Option<Arc<dyn CacheStore>>,
    #[cfg(feature = "tower")]
    pub budget_config: Option<BudgetConfig>,
    #[cfg(feature = "tower")]
    pub hooks: Vec<Arc<dyn LlmHook>>,
    #[cfg(feature = "tower")]
    pub cooldown_duration: Option<Duration>,
    #[cfg(feature = "tower")]
    pub rate_limit_config: Option<RateLimitConfig>,
    #[cfg(feature = "tower")]
    pub health_check_interval: Option<Duration>,
    #[cfg(feature = "tower")]
    pub enable_cost_tracking: bool,
    #[cfg(feature = "tower")]
    pub enable_tracing: bool,
}

impl ClientConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: None,
            api_type: None,
            timeout: Duration::from_secs(60),
            max_retries: 3,
            extra_headers: Vec::new(),
            credential_provider: None,
            load_env: true,
            transport: TransportConfig::default(),
            outbound_policy: None,
            #[cfg(feature = "tower")]
            cache_config: None,
            #[cfg(feature = "tower")]
            cache_store: None,
            #[cfg(feature = "tower")]
            budget_config: None,
            #[cfg(feature = "tower")]
            hooks: Vec::new(),
            #[cfg(feature = "tower")]
            cooldown_duration: None,
            #[cfg(feature = "tower")]
            rate_limit_config: None,
            #[cfg(feature = "tower")]
            health_check_interval: None,
            #[cfg(feature = "tower")]
            enable_cost_tracking: false,
            #[cfg(feature = "tower")]
            enable_tracing: false,
        }
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.extra_headers
    }
}

impl std::fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let redacted_headers: Vec<(&str, &str)> = self
            .extra_headers
            .iter()
            .map(|(k, _v)| (k.as_str(), "[redacted]"))
            .collect();
        let mut dbg = f.debug_struct("ClientConfig");
        dbg.field("api_key", &"[redacted]")
            .field("base_url", &self.base_url)
            .field("api_type", &self.api_type)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("extra_headers", &redacted_headers)
            .field("load_env", &self.load_env)
            .field(
                "credential_provider",
                &self.credential_provider.as_ref().map(|_| "[configured]"),
            );

        #[cfg(feature = "tower")]
        {
            dbg.field("cache_config", &self.cache_config)
                .field(
                    "cache_store",
                    &self.cache_store.as_ref().map(|_| "[configured]"),
                )
                .field("budget_config", &self.budget_config)
                .field("hooks_count", &self.hooks.len())
                .field("cooldown_duration", &self.cooldown_duration)
                .field("rate_limit_config", &self.rate_limit_config)
                .field("health_check_interval", &self.health_check_interval)
                .field("enable_cost_tracking", &self.enable_cost_tracking)
                .field("enable_tracing", &self.enable_tracing);
        }

        dbg.finish()
    }
}

#[must_use]
pub struct ClientConfigBuilder {
    pub(crate) config: ClientConfig,
}

impl ClientConfigBuilder {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: ClientConfig::new(api_key),
        }
    }

    pub fn from_env() -> Self {
        Self {
            config: ClientConfig::new(""),
        }
    }

    pub fn load_env(mut self, enabled: bool) -> Self {
        self.config.load_env = enabled;
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        self.config.base_url = Some(url.trim_end_matches('/').to_string());
        self
    }

    /// Explicitly select the API type the provider instance should speak.
    ///
    /// When combined with a custom `base_url`, this declares which protocol
    /// the endpoint implements. When omitted, the provider's effective
    /// default API type is used.
    pub fn api_type(mut self, api_type: APIType) -> Self {
        self.config.api_type = Some(api_type);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    pub fn credential_provider(mut self, provider: Arc<dyn CredentialProvider>) -> Self {
        self.config.credential_provider = Some(provider);
        self
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> HiLLMResult<Self> {
        let key = key.into();
        let value = value.into();

        reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
            HiLLMError::InvalidHeader {
                name: key.clone(),
                reason: e.to_string(),
            }
        })?;

        reqwest::header::HeaderValue::from_str(&value).map_err(|e| HiLLMError::InvalidHeader {
            name: key.clone(),
            reason: e.to_string(),
        })?;

        self.config.extra_headers.push((key, value));
        Ok(self)
    }

    pub fn transport(mut self, config: TransportConfig) -> Self {
        self.config.transport = config;
        self
    }

    /// Set a per-client outbound policy validator.
    ///
    /// When set, this validator replaces the process-global policy for URL
    /// and DNS checks made by this client. Useful for multi-tenant servers
    /// that need different policies per client (e.g. some tenants may be
    /// restricted to an allowlist, others to `DenyPrivate`).
    pub fn outbound_policy(mut self, validator: Arc<OutboundPolicyValidator>) -> Self {
        self.config.outbound_policy = Some(validator);
        self
    }

    #[cfg(feature = "tower")]
    pub fn cache(mut self, config: CacheConfig) -> Self {
        self.config.cache_config = Some(config);
        self
    }

    #[cfg(feature = "tower")]
    pub fn cache_store(mut self, store: Arc<dyn CacheStore>) -> Self {
        self.config.cache_store = Some(store);
        self
    }

    #[cfg(feature = "tower")]
    pub fn budget(mut self, config: BudgetConfig) -> Self {
        self.config.budget_config = Some(config);
        self
    }

    #[cfg(feature = "tower")]
    pub fn hook(mut self, hook: Arc<dyn LlmHook>) -> Self {
        self.config.hooks.push(hook);
        self
    }

    #[cfg(feature = "tower")]
    pub fn hooks(mut self, hooks: Vec<Arc<dyn LlmHook>>) -> Self {
        self.config.hooks = hooks;
        self
    }

    #[cfg(feature = "tower")]
    pub fn cooldown(mut self, duration: Duration) -> Self {
        self.config.cooldown_duration = Some(duration);
        self
    }

    #[cfg(feature = "tower")]
    pub fn rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.config.rate_limit_config = Some(config);
        self
    }

    #[cfg(feature = "tower")]
    pub fn health_check(mut self, interval: Duration) -> Self {
        self.config.health_check_interval = Some(interval);
        self
    }

    #[cfg(feature = "tower")]
    pub fn cost_tracking(mut self, enabled: bool) -> Self {
        self.config.enable_cost_tracking = enabled;
        self
    }

    #[cfg(feature = "tower")]
    pub fn tracing(mut self, enabled: bool) -> Self {
        self.config.enable_tracing = enabled;
        self
    }

    #[must_use]
    pub fn build(self) -> ClientConfig {
        self.config
    }
}
