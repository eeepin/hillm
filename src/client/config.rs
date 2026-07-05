use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;

use crate::auth::CredentialProvider;
use crate::error::{HiLlmError, HiLlmResult};
use crate::http::transport::TransportConfig;

#[derive(Clone)]
pub struct ClientConfig {
    pub api_key: SecretString,
    pub base_url: Option<String>,
    pub timeout: Duration,
    pub max_retries: u32,
    pub(crate) extra_headers: Vec<(String, String)>,
    pub credential_provider: Option<Arc<dyn CredentialProvider>>,
    pub load_env: bool,
    pub transport: TransportConfig,
}

impl ClientConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: None,
            timeout: Duration::from_secs(60),
            max_retries: 3,
            extra_headers: Vec::new(),
            credential_provider: None,
            load_env: true,
            transport: TransportConfig::default(),
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
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("extra_headers", &redacted_headers)
            .field("load_env", &self.load_env)
            .field(
                "credential_provider",
                &self.credential_provider.as_ref().map(|_| "[configured]"),
            );
        dbg.finish()
    }
}

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

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> HiLlmResult<Self> {
        let key = key.into();
        let value = value.into();

        reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| {
            HiLlmError::InvalidHeader {
                name: key.clone(),
                reason: e.to_string(),
            }
        })?;

        reqwest::header::HeaderValue::from_str(&value).map_err(|e| HiLlmError::InvalidHeader {
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

    pub fn build(self) -> ClientConfig {
        self.config
    }
}
