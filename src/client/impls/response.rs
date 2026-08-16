use crate::client::{BoxFuture, BoxStream, Client, ResponseClient};
use crate::error::{HiLLMError, HiLLMResult};
use crate::http;
use crate::provider;
use crate::types::response::{CreateResponseRequest, ResponseObject, ResponsesStreamEvent};

use super::super::str_pair;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ResponseClient for Client {
    fn create_response(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLLMResult<ResponseObject>> {
        Box::pin(async move {
            // Force non-streaming for the non-stream call.
            let mut req = req;
            req.stream = Some(false);

            // Prefer the Responses codec path when the provider supports it.
            if let Some(codec) = self.provider.codec_for(provider::APIType::OpenAIResponses) {
                let endpoint_path = codec.endpoint_path();
                let url = self.provider.build_url(endpoint_path, "");
                let body_json = serde_json::to_value(&req)?;
                let body_bytes = codec.encode_request(&body_json)?;

                let auth_header = self
                    .resolve_auth_header_for_provider(self.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    self.provider.as_ref(),
                    "POST",
                    &url,
                    &body_json,
                    &body_bytes,
                );
                let extra: Vec<(&str, &str)> = all_headers
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_str()))
                    .collect();
                let auth = auth_header.as_ref().map(str_pair);

                let raw_bytes = http::request::post_json_raw(
                    &self.http_client,
                    &url,
                    auth,
                    &extra,
                    body_bytes,
                    self.config.max_retries,
                )
                .await?;

                let raw_bytes_vec = serde_json::to_vec(&raw_bytes)?;
                let response_value = codec.decode_response(&raw_bytes_vec)?;
                return serde_json::from_value::<ResponseObject>(response_value)
                    .map_err(HiLLMError::from);
            }

            // Legacy path for providers without a Responses codec.
            let url = self.provider.build_url(self.provider.responses_path(), "");
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
            serde_json::from_value::<ResponseObject>(raw).map_err(HiLLMError::from)
        })
    }

    fn create_response_stream(
        &self,
        req: CreateResponseRequest,
    ) -> BoxFuture<'_, HiLLMResult<BoxStream<'static, HiLLMResult<ResponsesStreamEvent>>>> {
        Box::pin(async move {
            // Streaming Responses requires the OpenAI Responses codec; fail
            // before sending when the provider does not support it.
            let codec = self
                .provider
                .codec_for(provider::APIType::OpenAIResponses)
                .ok_or_else(|| HiLLMError::EndpointNotSupported {
                    endpoint: "responses".to_string(),
                    provider: self.provider.name().to_string(),
                })?;

            let mut req = req;
            req.stream = Some(true);

            let endpoint_path = codec.endpoint_path();
            let url = self.provider.build_stream_url(endpoint_path, &req.model);
            let body_json = serde_json::to_value(&req)?;
            let body_bytes = codec.encode_request(&body_json)?;

            let auth_header = self
                .resolve_auth_header_for_provider(self.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                self.provider.as_ref(),
                "POST",
                &url,
                &body_json,
                &body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let parse_event = move |data: &str| {
                codec
                    .parse_stream_event(data)?
                    .map(serde_json::from_value::<ResponsesStreamEvent>)
                    .transpose()
                    .map_err(HiLLMError::from)
            };
            let stream = http::stream::post_typed_stream(
                &self.http_client,
                &url,
                auth,
                &extra,
                body_bytes,
                self.config.max_retries,
                parse_event,
            )
            .await?;
            Ok(stream)
        })
    }

    fn retrieve_response(&self, response_id: &str) -> BoxFuture<'_, HiLLMResult<ResponseObject>> {
        let response_id = response_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}",
                self.provider.build_url(self.provider.responses_path(), ""),
                response_id
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
            serde_json::from_value::<ResponseObject>(raw).map_err(HiLLMError::from)
        })
    }

    fn cancel_response(&self, response_id: &str) -> BoxFuture<'_, HiLLMResult<ResponseObject>> {
        let response_id = response_id.to_owned();
        Box::pin(async move {
            let url = format!(
                "{}/{}/cancel",
                self.provider.build_url(self.provider.responses_path(), ""),
                response_id
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
            serde_json::from_value::<ResponseObject>(raw).map_err(HiLLMError::from)
        })
    }
}
