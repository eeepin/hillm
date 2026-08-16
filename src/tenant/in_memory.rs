use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dashmap::DashMap;

use super::resolver::{KeyResolver, KeyResolverError, ResolvedKey};

/// Thread-safe in-memory [`KeyResolver`] backed by a [`DashMap`].
pub struct InMemoryKeyResolver {
    keys: Arc<DashMap<String, ResolvedKey>>,
}

impl InMemoryKeyResolver {
    /// Create an empty resolver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keys: Arc::new(DashMap::new()),
        }
    }

    /// Create a resolver pre-populated with the given entries.
    #[must_use]
    pub fn with_entries(entries: impl IntoIterator<Item = (String, ResolvedKey)>) -> Self {
        let keys = DashMap::new();
        for (k, v) in entries {
            keys.insert(k, v);
        }
        Self {
            keys: Arc::new(keys),
        }
    }

    /// Insert or replace a key record.
    pub fn insert(&self, api_key: impl Into<String>, resolved: ResolvedKey) {
        self.keys.insert(api_key.into(), resolved);
    }

    /// Remove a key record.  Returns the removed record if it existed.
    pub fn remove(&self, api_key: &str) -> Option<ResolvedKey> {
        self.keys.remove(api_key).map(|(_, v)| v)
    }
}

impl Default for InMemoryKeyResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyResolver for InMemoryKeyResolver {
    fn resolve(
        &self,
        api_key: String,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedKey, KeyResolverError>> + Send + 'static>> {
        // Clone the Arc-backed DashMap so the future owns it and is 'static.
        let keys = self.keys.clone();
        Box::pin(async move {
            match keys.get(&api_key) {
                None => Err(KeyResolverError::NotFound),
                Some(entry) => {
                    if !entry.active {
                        Err(KeyResolverError::Inactive)
                    } else {
                        Ok(entry.clone())
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenant::context::TenantId;

    fn create_test_resolved_key(tenant: &str, active: bool) -> ResolvedKey {
        ResolvedKey {
            tenant_id: TenantId::from(tenant),
            allowed_models: vec![],
            monthly_budget: None,
            currency: None,
            metadata: std::collections::HashMap::new(),
            active,
        }
    }

    #[tokio::test]
    async fn in_memory_key_resolver_creation() {
        let resolver = InMemoryKeyResolver::new();
        assert_eq!(resolver.keys.len(), 0);
    }

    #[tokio::test]
    async fn in_memory_key_resolver_default() {
        let resolver = InMemoryKeyResolver::default();
        assert_eq!(resolver.keys.len(), 0);
    }

    #[tokio::test]
    async fn in_memory_key_resolver_with_entries() {
        let entries = vec![
            ("key1".to_string(), create_test_resolved_key("tenant1", true)),
            ("key2".to_string(), create_test_resolved_key("tenant2", true)),
        ];
        let resolver = InMemoryKeyResolver::with_entries(entries);
        assert_eq!(resolver.keys.len(), 2);
    }

    #[tokio::test]
    async fn in_memory_key_resolver_insert() {
        let resolver = InMemoryKeyResolver::new();
        let resolved = create_test_resolved_key("tenant1", true);
        resolver.insert("key1", resolved.clone());
        assert_eq!(resolver.keys.len(), 1);
    }

    #[tokio::test]
    async fn in_memory_key_resolver_remove() {
        let resolver = InMemoryKeyResolver::new();
        let resolved = create_test_resolved_key("tenant1", true);
        resolver.insert("key1", resolved);
        assert_eq!(resolver.keys.len(), 1);

        let removed = resolver.remove("key1");
        assert!(removed.is_some());
        assert_eq!(resolver.keys.len(), 0);
    }

    #[tokio::test]
    async fn in_memory_key_resolver_remove_nonexistent() {
        let resolver = InMemoryKeyResolver::new();
        let removed = resolver.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[tokio::test]
    async fn in_memory_key_resolver_resolve_success() {
        let resolver = InMemoryKeyResolver::new();
        let resolved = create_test_resolved_key("tenant1", true);
        resolver.insert("key1", resolved.clone());

        let result = resolver.resolve("key1".to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tenant_id, TenantId::from("tenant1"));
    }

    #[tokio::test]
    async fn in_memory_key_resolver_resolve_not_found() {
        let resolver = InMemoryKeyResolver::new();
        let result = resolver.resolve("nonexistent".to_string()).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), KeyResolverError::NotFound));
    }

    #[tokio::test]
    async fn in_memory_key_resolver_resolve_inactive() {
        let resolver = InMemoryKeyResolver::new();
        let resolved = create_test_resolved_key("tenant1", false);
        resolver.insert("key1", resolved);

        let result = resolver.resolve("key1".to_string()).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), KeyResolverError::Inactive));
    }
}
