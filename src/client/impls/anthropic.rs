use crate::client::{AnthropicMessagesClient, BoxFuture, BoxStream, Client};
use crate::error::{HiLlmError, HiLlmResult};
use crate::http;
use crate::provider;
use crate::types::anthropic::{
    AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicStreamEvent,
};

use super::super::str_pair;

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl AnthropicMessagesClient for Client {
    fn create_message(
        &self,
        req: AnthropicMessagesRequest,
    ) -> BoxFuture<'_, HiLlmResult<AnthropicMessagesResponse>> {
        Box::pin(async move {
            // Native Messages calls require an instance bound to the
            // Anthropic Messages API type; fail before sending otherwise.
            let codec = self
                .provider
                .codec_for(provider::APIType::AnthropicMessages)
                .ok_or_else(|| HiLlmError::EndpointNotSupported {
                    endpoint: "messages".to_string(),
                    provider: self.provider.name().to_string(),
                })?;

            // Force non-streaming for the non-stream call.
            let mut req = req;
            req.stream = Some(false);

            let endpoint_path = codec.endpoint_path();
            let url = self.provider.build_url(endpoint_path, &req.model);
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
            serde_json::from_value::<AnthropicMessagesResponse>(response_value)
                .map_err(HiLlmError::from)
        })
    }

    fn create_message_stream(
        &self,
        req: AnthropicMessagesRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<AnthropicStreamEvent>>>> {
        Box::pin(async move {
            let codec = self
                .provider
                .codec_for(provider::APIType::AnthropicMessages)
                .ok_or_else(|| HiLlmError::EndpointNotSupported {
                    endpoint: "messages".to_string(),
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
                    .map(serde_json::from_value::<AnthropicStreamEvent>)
                    .transpose()
                    .map_err(HiLlmError::from)
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
}
