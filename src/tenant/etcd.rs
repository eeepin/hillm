use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use etcd_client::{Client, ConnectOptions};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use super::resolver::{KeyResolver, KeyResolverError, ResolvedKey};

#[derive(Clone, Debug)]
pub struct EtcdKeyResolverConfig {
    pub endpoints: Vec<String>,
    pub prefix: String,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl Default for EtcdKeyResolverConfig {
    fn default() -> Self {
        Self {
            endpoints: vec!["http://localhost:2379".into()],
            prefix: "liter-llm/keys".into(),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(2),
            username: None,
            password: None,
        }
    }
}

#[derive(Clone)]
pub struct EtcdKeyResolver {
    client: Arc<Mutex<Client>>,
    prefix: String,
}

impl EtcdKeyResolver {
    pub async fn connect(config: EtcdKeyResolverConfig) -> Result<Self, KeyResolverError> {
        let mut options = ConnectOptions::new()
            .with_connect_timeout(config.connect_timeout)
            .with_timeout(config.request_timeout);
        if let (Some(username), Some(password)) =
            (config.username.as_deref(), config.password.as_deref())
        {
            options = options.with_user(username, password);
        }
        let client = Client::connect(config.endpoints, Some(options))
            .await
            .map_err(|e| KeyResolverError::Backend(format!("etcd connect failed: {e}")))?;
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            prefix: config.prefix,
        })
    }

    pub fn hash_api_key(api_key: &str) -> String {
        let digest = Sha256::digest(api_key.as_bytes());
        hex::encode(digest)
    }

    pub fn key_path(prefix: &str, api_key_hash: &str) -> String {
        format!("{}/{}", prefix.trim_end_matches('/'), api_key_hash)
    }
}

impl KeyResolver for EtcdKeyResolver {
    fn resolve(
        &self,
        api_key: String,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedKey, KeyResolverError>> + Send + 'static>> {
        let client = Arc::clone(&self.client);
        let etcd_key = Self::key_path(&self.prefix, &Self::hash_api_key(&api_key));
        Box::pin(async move {
            let mut guard = client.lock().await;
            let response = guard
                .get(etcd_key.as_bytes(), None)
                .await
                .map_err(|e| KeyResolverError::Backend(format!("etcd get failed: {e}")))?;
            let kv = response.kvs().first().ok_or(KeyResolverError::NotFound)?;
            let resolved: ResolvedKey = serde_json::from_slice(kv.value()).map_err(|e| {
                KeyResolverError::Backend(format!("invalid resolved key json: {e}"))
            })?;
            if !resolved.active {
                return Err(KeyResolverError::Inactive);
            }
            Ok(resolved)
        })
    }
}
