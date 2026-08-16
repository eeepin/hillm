use crate::client::{str_pair, Client, ModelClient};
use crate::error::HiLLMResult;
use crate::http;
use crate::types::model::ModelsListResponse;

use super::super::BoxFuture;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ModelClient for Client {
    fn list_models(&self) -> BoxFuture<'_, HiLLMResult<ModelsListResponse>> {
        Box::pin(async move {
            let url = self.provider.build_url(self.provider.models_path(), "");
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let mut raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            self.provider.transform_response(&mut raw)?;
            serde_json::from_value::<ModelsListResponse>(raw).map_err(crate::error::HiLLMError::from)
        })
    }
}
