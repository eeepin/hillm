use std::collections::HashMap;

/// Tenant identifier.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TenantId(pub String);

impl AsRef<str> for TenantId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Id and attributes associated with a single request.
#[derive(Clone, Debug)]
pub struct TenantContext {
    pub tenant_id: TenantId,
    pub user_id: Option<String>,
    pub attributes: HashMap<String, String>,
}

impl TenantContext {
    pub fn new(tenant_id: impl Into<TenantId>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            user_id: None,
            attributes: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}
