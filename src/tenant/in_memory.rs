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
            (
                "key1".to_string(),
                create_test_resolved_key("tenant1", true),
            ),
            (
                "key2".to_string(),
                create_test_resolved_key("tenant2", true),
            ),
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

    // -----------------------------------------------------------------------
    // Concurrency tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn concurrent_insert_and_resolve() {
        let resolver = Arc::new(InMemoryKeyResolver::new());
        let n = 100;

        // Spawn N tasks that each insert a unique key.
        let mut insert_handles = Vec::new();
        for i in 0..n {
            let r = Arc::clone(&resolver);
            insert_handles.push(tokio::spawn(async move {
                let key = format!("key-{i}");
                let resolved = create_test_resolved_key(&format!("tenant-{i}"), true);
                r.insert(key.clone(), resolved);
                key
            }));
        }

        // Wait for all inserts.
        let keys: Vec<String> = futures_util::future::join_all(insert_handles)
            .await
            .into_iter()
            .map(|h| h.unwrap())
            .collect();

        // Resolve all keys; each must succeed with the right tenant.
        for key in keys {
            let r = Arc::clone(&resolver);
            let k = key.clone();
            let result = tokio::spawn(async move { r.resolve(k).await })
                .await
                .unwrap();
            let resolved = result.expect("resolve should succeed");
            let expected_tenant = key.strip_prefix("key-").unwrap();
            assert_eq!(
                resolved.tenant_id,
                TenantId::from(format!("tenant-{expected_tenant}"))
            );
        }

        assert_eq!(resolver.keys.len(), n as usize);
    }

    #[tokio::test]
    async fn concurrent_insert_and_remove_same_key() {
        let resolver = Arc::new(InMemoryKeyResolver::new());
        let key = "shared-key".to_string();

        // Concurrently insert, then remove the same key.
        // Should not panic; result should be either Ok or NotFound.
        let r1 = Arc::clone(&resolver);
        let r2 = Arc::clone(&resolver);
        let k1 = key.clone();
        let k2 = key.clone();

        let (ins, rem) = tokio::join!(
            tokio::spawn(async move {
                let resolved = create_test_resolved_key("t1", true);
                r1.insert(k1, resolved);
            }),
            tokio::spawn(async move { r2.remove(&k2) }),
        );

        ins.unwrap();
        let remove_result = rem.unwrap();
        // remove may or may not find the key depending on timing — both are valid.
        let _ = remove_result;
    }

    #[tokio::test]
    async fn insert_replaces_existing_key() {
        let resolver = InMemoryKeyResolver::new();
        resolver.insert("key1", create_test_resolved_key("tenant-A", true));
        resolver.insert("key1", create_test_resolved_key("tenant-B", true));

        let result = resolver.resolve("key1".to_string()).await.unwrap();
        assert_eq!(
            result.tenant_id,
            TenantId::from("tenant-B"),
            "insert should replace existing key"
        );
        assert_eq!(
            resolver.keys.len(),
            1,
            "replaced key should not increase count"
        );
    }

    #[tokio::test]
    async fn remove_returns_the_removed_value() {
        let resolver = InMemoryKeyResolver::new();
        let resolved = create_test_resolved_key("tenant-42", true);
        resolver.insert("key1", resolved.clone());

        let removed = resolver.remove("key1").expect("should remove existing key");
        assert_eq!(removed.tenant_id, resolved.tenant_id);
    }

    #[tokio::test]
    async fn resolve_works_after_resolver_clone() {
        // The resolver's internal Arc<DashMap> should keep the data alive
        // even if the original resolver is dropped.
        let resolver = InMemoryKeyResolver::new();
        resolver.insert("key1", create_test_resolved_key("t1", true));
        let future = resolver.resolve("key1".to_string());
        drop(resolver);
        let result = future.await;
        assert!(
            result.is_ok(),
            "future should work after original resolver dropped"
        );
    }

    #[tokio::test]
    async fn with_entries_duplicate_keys_last_wins() {
        // If with_entries is given duplicate keys, DashMap semantics apply:
        // later inserts overwrite earlier ones.
        let entries = vec![
            (
                "key".to_string(),
                create_test_resolved_key("tenant-A", true),
            ),
            (
                "key".to_string(),
                create_test_resolved_key("tenant-B", true),
            ),
        ];
        let resolver = InMemoryKeyResolver::with_entries(entries);
        // DashMap iter may have only one entry per key; count should be 1.
        assert_eq!(resolver.keys.len(), 1);
    }
}
