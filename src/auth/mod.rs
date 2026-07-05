use std::sync::Arc;

use secrecy::SecretString;

use crate::client::BoxFuture;
use crate::error::HiLlmResult;

pub trait CredentialProvider: Send + Sync {
    fn resolve(&self) -> BoxFuture<'_, HiLlmResult<Credential>>;
}

impl CredentialProvider for Arc<dyn CredentialProvider> {
    fn resolve(&self) -> BoxFuture<'_, HiLlmResult<Credential>> {
        (**self).resolve()
    }
}

#[derive(Debug, Clone)]
pub enum Credential {
    BearerToken(SecretString),
    AwsCredentials {
        access_key_id: SecretString,
        secret_access_key: SecretString,
        session_token: Option<SecretString>,
    },
}

pub struct StaticTokenProvider {
    token: SecretString,
}

impl StaticTokenProvider {
    pub fn new(token: SecretString) -> Self {
        Self { token }
    }
}

impl CredentialProvider for StaticTokenProvider {
    fn resolve(&self) -> BoxFuture<'_, HiLlmResult<Credential>> {
        let token = self.token.clone();
        Box::pin(async move { Ok(Credential::BearerToken(token)) })
    }
}
