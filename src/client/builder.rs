use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;

use crate::auth::CredentialProvider;
#[cfg(any(feature = "default-http", feature = "wasm-http"))]
use crate::error::HiLlmResult;
use crate::http::transport::TransportConfig;
#[cfg(feature = "tower")]
use crate::tower::{BudgetConfig, CacheConfig, CacheStore, LlmHook, RateLimitConfig};

pub struct NoApiKey;
pub struct WithApiKey;
pub struct NoProvider;
pub struct WithProvider;

#[must_use = "call .build() to construct the client"]
pub struct ClientBuilder<K = NoApiKey, P = NoProvider> {
    api_key: SecretString,
    provider_name: Option<String>,
    base_url: Option<String>,
    timeout: Duration,
    max_retries: u32,
    transport: TransportConfig,
    load_env: bool,
    credential_provider: Option<Arc<dyn CredentialProvider>>,
    #[cfg(feature = "tower")]
    cache_config: Option<CacheConfig>,
    #[cfg(feature = "tower")]
    cache_store: Option<Arc<dyn CacheStore>>,
    #[cfg(feature = "tower")]
    budget_config: Option<BudgetConfig>,
    #[cfg(feature = "tower")]
    hooks: Vec<Arc<dyn LlmHook>>,
    #[cfg(feature = "tower")]
    cooldown_duration: Option<Duration>,
    #[cfg(feature = "tower")]
    rate_limit_config: Option<RateLimitConfig>,
    #[cfg(feature = "tower")]
    health_check_interval: Option<Duration>,
    #[cfg(feature = "tower")]
    enable_cost_tracking: bool,
    #[cfg(feature = "tower")]
    enable_tracing: bool,
    _key_state: std::marker::PhantomData<K>,
    _provider_state: std::marker::PhantomData<P>,
}

impl ClientBuilder<NoApiKey, NoProvider> {
    pub fn new() -> Self {
        Self {
            api_key: SecretString::from(String::new()),
            provider_name: None,
            base_url: None,
            timeout: Duration::from_secs(60),
            max_retries: 3,
            transport: TransportConfig::default(),
            load_env: false,
            credential_provider: None,
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
            _key_state: std::marker::PhantomData,
            _provider_state: std::marker::PhantomData,
        }
    }
}

impl Default for ClientBuilder<NoApiKey, NoProvider> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, P> ClientBuilder<K, P> {
    pub fn api_key(self, key: impl Into<String>) -> ClientBuilder<WithApiKey, P> {
        ClientBuilder {
            api_key: SecretString::from(key.into()),
            provider_name: self.provider_name,
            base_url: self.base_url,
            timeout: self.timeout,
            max_retries: self.max_retries,
            transport: self.transport,
            load_env: self.load_env,
            credential_provider: self.credential_provider,
            #[cfg(feature = "tower")]
            cache_config: self.cache_config,
            #[cfg(feature = "tower")]
            cache_store: self.cache_store,
            #[cfg(feature = "tower")]
            budget_config: self.budget_config,
            #[cfg(feature = "tower")]
            hooks: self.hooks,
            #[cfg(feature = "tower")]
            cooldown_duration: self.cooldown_duration,
            #[cfg(feature = "tower")]
            rate_limit_config: self.rate_limit_config,
            #[cfg(feature = "tower")]
            health_check_interval: self.health_check_interval,
            #[cfg(feature = "tower")]
            enable_cost_tracking: self.enable_cost_tracking,
            #[cfg(feature = "tower")]
            enable_tracing: self.enable_tracing,
            _key_state: std::marker::PhantomData,
            _provider_state: self._provider_state,
        }
    }

    pub fn provider(self, provider_name: impl Into<String>) -> ClientBuilder<K, WithProvider> {
        ClientBuilder {
            api_key: self.api_key,
            provider_name: Some(provider_name.into()),
            base_url: self.base_url,
            timeout: self.timeout,
            max_retries: self.max_retries,
            transport: self.transport,
            load_env: self.load_env,
            credential_provider: self.credential_provider,
            #[cfg(feature = "tower")]
            cache_config: self.cache_config,
            #[cfg(feature = "tower")]
            cache_store: self.cache_store,
            #[cfg(feature = "tower")]
            budget_config: self.budget_config,
            #[cfg(feature = "tower")]
            hooks: self.hooks,
            #[cfg(feature = "tower")]
            cooldown_duration: self.cooldown_duration,
            #[cfg(feature = "tower")]
            rate_limit_config: self.rate_limit_config,
            #[cfg(feature = "tower")]
            health_check_interval: self.health_check_interval,
            #[cfg(feature = "tower")]
            enable_cost_tracking: self.enable_cost_tracking,
            #[cfg(feature = "tower")]
            enable_tracing: self.enable_tracing,
            _key_state: self._key_state,
            _provider_state: std::marker::PhantomData,
        }
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        self.base_url = Some(url.trim_end_matches('/').to_string());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn transport(mut self, config: TransportConfig) -> Self {
        self.transport = config;
        self
    }

    pub fn load_env(mut self, enabled: bool) -> Self {
        self.load_env = enabled;
        self
    }

    pub fn credential_provider(mut self, provider: Arc<dyn CredentialProvider>) -> Self {
        self.credential_provider = Some(provider);
        self
    }

    #[cfg(feature = "tower")]
    pub fn cache(mut self, config: CacheConfig) -> Self {
        self.cache_config = Some(config);
        self
    }

    #[cfg(feature = "tower")]
    pub fn cache_store(mut self, store: Arc<dyn CacheStore>) -> Self {
        self.cache_store = Some(store);
        self
    }

    #[cfg(feature = "tower")]
    pub fn budget(mut self, config: BudgetConfig) -> Self {
        self.budget_config = Some(config);
        self
    }

    #[cfg(feature = "tower")]
    pub fn hook(mut self, hook: Arc<dyn LlmHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    #[cfg(feature = "tower")]
    pub fn hooks(mut self, hooks: Vec<Arc<dyn LlmHook>>) -> Self {
        self.hooks = hooks;
        self
    }

    #[cfg(feature = "tower")]
    pub fn cooldown(mut self, duration: Duration) -> Self {
        self.cooldown_duration = Some(duration);
        self
    }

    #[cfg(feature = "tower")]
    pub fn rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit_config = Some(config);
        self
    }

    #[cfg(feature = "tower")]
    pub fn health_check(mut self, interval: Duration) -> Self {
        self.health_check_interval = Some(interval);
        self
    }

    #[cfg(feature = "tower")]
    pub fn cost_tracking(mut self, enabled: bool) -> Self {
        self.enable_cost_tracking = enabled;
        self
    }

    #[cfg(feature = "tower")]
    pub fn tracing(mut self, enabled: bool) -> Self {
        self.enable_tracing = enabled;
        self
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ClientBuilder<WithApiKey, WithProvider> {
    pub fn build(self) -> HiLlmResult<super::DefaultClient> {
        use super::config::ClientConfig;

        let config = ClientConfig {
            api_key: self.api_key,
            base_url: self.base_url,
            timeout: self.timeout,
            max_retries: self.max_retries,
            extra_headers: Vec::new(),
            credential_provider: self.credential_provider,
            load_env: self.load_env,
            transport: self.transport,
            #[cfg(feature = "tower")]
            cache_config: self.cache_config,
            #[cfg(feature = "tower")]
            cache_store: self.cache_store,
            #[cfg(feature = "tower")]
            budget_config: self.budget_config,
            #[cfg(feature = "tower")]
            hooks: self.hooks,
            #[cfg(feature = "tower")]
            cooldown_duration: self.cooldown_duration,
            #[cfg(feature = "tower")]
            rate_limit_config: self.rate_limit_config,
            #[cfg(feature = "tower")]
            health_check_interval: self.health_check_interval,
            #[cfg(feature = "tower")]
            enable_cost_tracking: self.enable_cost_tracking,
            #[cfg(feature = "tower")]
            enable_tracing: self.enable_tracing,
        };

        super::DefaultClient::new(config, self.provider_name)
    }
}
