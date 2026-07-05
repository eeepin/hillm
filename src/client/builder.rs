use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;

use crate::auth::CredentialProvider;

use crate::error::HiLlmResult;
use crate::http::transport::TransportConfig;

pub struct NoApiKey;
pub struct WithApiKey;
pub struct NoProvider;
pub struct WithProvider;

pub struct ClientBuilder<K = NoApiKey, P = NoProvider> {
    api_key: SecretString,
    provider_name: Option<String>,
    base_url: Option<String>,
    timeout: Duration,
    max_retries: u32,
    transport: TransportConfig,
    load_env: bool,
    credential_provider: Option<Arc<dyn CredentialProvider>>,
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
}

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
        };

        super::DefaultClient::new(config, self.provider_name)
    }
}
