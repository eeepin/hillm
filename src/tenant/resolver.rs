use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use super::context::TenantId;

/// Resolved virtual-key record returned by [`KeyResolver::resolve`].
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ResolvedKey {
    /// Tenant id.
    pub tenant_id: TenantId,
    /// Model names that this key allowed to access.  Empty means all allowed.
    pub allowed_models: Vec<String>,
    /// Optional per-period spending cap.
    pub monthly_budget: Option<rust_decimal::Decimal>,
    /// ISO-4217 currency code for `monthly_budget`, e.g. `"CNY"`.
    pub currency: Option<String>,
    /// Arbitrary key-value metadata (e.g. `"tier"`, `"label"`).
    pub metadata: HashMap<String, String>,
    /// Whether the key is currently active.
    pub active: bool,
}

/// Errors returned by [`KeyResolver::resolve`].
#[derive(Debug, thiserror::Error)]
pub enum KeyResolverError {
    #[error("api key not found")]
    NotFound,
    #[error("api key is inactive")]
    Inactive,
    #[error("key resolver backend error: {0}")]
    Backend(String),
}

/// Resolves a raw API token to a [`ResolvedKey`].
pub trait KeyResolver: Send + Sync + 'static {
    fn resolve(
        &self,
        api_key: String,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedKey, KeyResolverError>> + Send + 'static>>;
}
