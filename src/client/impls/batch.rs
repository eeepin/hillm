use std::time::Duration;

use crate::client::{
    BatchClient, BatchRetriever, BatchWaitError, BoxFuture, Client, WaitForBatchConfig,
};
use crate::error::{HiLlmError, HiLlmResult};
use crate::http;
use crate::types::batch::{
    BatchListQuery, BatchListResponse, BatchObject, BatchStatus, CreateBatchRequest,
};

use super::super::str_pair;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl BatchClient for Client {
    fn create_batch(&self, req: CreateBatchRequest) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        Box::pin(async move {
            let url = self.provider.build_url(self.provider.batches_path(), "");
            let body_bytes = bytes::Bytes::from(serde_json::to_vec(&req)?);
            let body_json = serde_json::to_value(&req)?;

            let auth_header = self.resolve_auth_header().await?;
            let all_headers = self.all_headers("POST", &url, &body_json, &body_bytes);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let raw = http::request::post_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn retrieve_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        let batch_id = batch_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}",
                self.provider.build_url(self.provider.batches_path(), ""),
                batch_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchObject>(raw).map_err(HiLlmError::from)
        })
    }

    fn list_batches(
        &self,
        query: Option<BatchListQuery>,
    ) -> BoxFuture<'_, HiLlmResult<BatchListResponse>> {
        Box::pin(async move {
            let base_url = self.provider.build_url(self.provider.batches_path(), "");
            let url = if let Some(ref q) = query {
                let mut params = Vec::new();
                if let Some(limit) = q.limit {
                    params.push(format!("limit={limit}"));
                }
                if let Some(ref after) = q.after {
                    params.push(format!("after={after}"));
                }
                if params.is_empty() {
                    base_url
                } else {
                    format!("{base_url}?{}", params.join("&"))
                }
            } else {
                base_url
            };
            let auth_header = self.resolve_auth_header().await?;
            let auth = auth_header.as_ref().map(str_pair);
            let all_headers = self.all_headers("GET", &url, &serde_json::Value::Null, &[]);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let raw = http::request::get_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchListResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn cancel_batch(&self, batch_id: &str) -> BoxFuture<'_, HiLlmResult<BatchObject>> {
        let batch_id = batch_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}/cancel",
                self.provider.build_url(self.provider.batches_path(), ""),
                batch_id
            );
            let auth_header = self.resolve_auth_header().await?;
            let body_json = serde_json::Value::Null;
            let body_bytes = bytes::Bytes::new();
            let all_headers = self.all_headers("POST", &url, &body_json, &body_bytes);
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let raw = http::request::post_json_raw(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
            )
            .await?;
            serde_json::from_value::<BatchObject>(raw).map_err(HiLlmError::from)
        })
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl BatchRetriever for Client {
    async fn fetch_batch_for_polling(&self, batch_id: &str) -> HiLlmResult<BatchObject> {
        self.retrieve_batch(batch_id).await
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
pub async fn wait_for_batch_impl<R: BatchRetriever>(
    retriever: &R,
    batch_id: &str,
    config: WaitForBatchConfig,
) -> std::result::Result<BatchObject, BatchWaitError> {
    #[cfg(not(target_arch = "wasm32"))]
    let started = tokio::time::Instant::now();
    #[cfg(target_arch = "wasm32")]
    let started = web_time::Instant::now();
    let mut interval_secs = config.initial_interval_secs;

    loop {
        let batch = retriever.fetch_batch_for_polling(batch_id).await?;

        match batch.status {
            BatchStatus::Completed => return Ok(batch),
            BatchStatus::Failed | BatchStatus::Expired | BatchStatus::Cancelled => {
                return Err(BatchWaitError::Failed {
                    status: batch.status,
                });
            }
            BatchStatus::Validating
            | BatchStatus::InProgress
            | BatchStatus::Finalizing
            | BatchStatus::Cancelling => {
                if let Some(timeout_secs) = config.timeout_secs {
                    let timeout = Duration::from_secs_f64(timeout_secs);
                    if started.elapsed() >= timeout {
                        return Err(BatchWaitError::Timeout { timeout_secs });
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                tokio::time::sleep(Duration::from_secs_f64(interval_secs)).await;
                #[cfg(target_arch = "wasm32")]
                gloo_timers::future::sleep(Duration::from_secs_f64(interval_secs)).await;
                let next = (interval_secs as f32 * config.backoff_multiplier)
                    .min(config.max_interval_secs as f32) as f64;
                interval_secs = next;
            }
        }
    }
}

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl Client {
    pub async fn wait_for_batch(
        &self,
        batch_id: &str,
        config: WaitForBatchConfig,
    ) -> std::result::Result<BatchObject, BatchWaitError> {
        wait_for_batch_impl(self, batch_id, config).await
    }
}
