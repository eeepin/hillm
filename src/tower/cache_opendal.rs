use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opendal::Operator;
use serde::{Deserialize, Serialize};

use super::cache::{CacheStore, CachedResponse};
use crate::error::{HiLlmError, HiLlmResult};

#[derive(Serialize, Deserialize)]
struct StoredEntry {
    request_body: String,
    response: CachedResponse,
    expires_at: u64,
}

pub struct OpenDalCacheStore {
    operator: Operator,
    prefix: String,
    ttl: Duration,
}

impl OpenDalCacheStore {
    pub fn new(operator: Operator, prefix: impl Into<String>, ttl: Duration) -> Self {
        Self {
            operator,
            prefix: prefix.into(),
            ttl,
        }
    }

    pub fn from_config(
        scheme: &str,
        config: HashMap<String, String>,
        prefix: impl Into<String>,
        ttl: Duration,
    ) -> HiLlmResult<Self> {
        let operator =
            Operator::via_iter(scheme, config).map_err(|e| HiLlmError::InternalError {
                message: format!("failed to build OpenDAL operator for '{scheme}': {e}"),
            })?;
        Ok(Self::new(operator, prefix, ttl))
    }

    fn key_path(&self, key: u64) -> String {
        format!("{}{key}", self.prefix)
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl CacheStore for OpenDalCacheStore {
    fn get(
        &self,
        key: u64,
        request_body: &str,
    ) -> Pin<Box<dyn Future<Output = Option<CachedResponse>> + Send + '_>> {
        let path = self.key_path(key);
        let request_body = request_body.to_owned();
        Box::pin(async move {
            let bytes = match self.operator.read(&path).await {
                Ok(b) => b,
                Err(_) => return None,
            };
            let entry: StoredEntry = match serde_json::from_slice(bytes.to_bytes().as_ref()) {
                Ok(e) => e,
                Err(_) => return None,
            };
            if Self::now_secs() > entry.expires_at {
                let _ = self.operator.delete(&path).await;
                return None;
            }
            if entry.request_body != request_body {
                return None;
            }
            Some(entry.response)
        })
    }

    fn put(
        &self,
        key: u64,
        request_body: String,
        response: CachedResponse,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let path = self.key_path(key);
        let entry = StoredEntry {
            request_body,
            response,
            expires_at: Self::now_secs() + self.ttl.as_secs(),
        };
        Box::pin(async move {
            let bytes = match serde_json::to_vec(&entry) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("OpenDAL cache: failed to serialize entry: {e}");
                    return;
                }
            };
            if let Err(e) = self.operator.write(&path, bytes).await {
                tracing::warn!("OpenDAL cache: failed to write {path}: {e}");
            }
        })
    }

    fn remove(&self, key: u64) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let path = self.key_path(key);
        Box::pin(async move {
            if let Err(e) = self.operator.delete(&path).await {
                tracing::warn!("OpenDAL cache: failed to delete {path}: {e}");
            }
        })
    }
}
