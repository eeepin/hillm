use std::sync::Arc;

use crate::client::str_pair;
use crate::client::{BoxFuture, BoxStream, ChatCompletionClient, Client};
use crate::error::{HiLlmError, HiLlmResult};
use crate::http;
use crate::provider;
use crate::provider::endpoint::{Endpoint, EndpointRequest, EndpointResponse};
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::types::raw::{RawExchange, RawStreamExchange};

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ChatCompletionClient for Client {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>> {
        Box::pin(async move {
            // Try new endpoint codec path first
            if let Some(codec) = self.provider.codec_for_endpoint(Endpoint::ChatCompletion) {
                // Use the codec's endpoint, not the requested endpoint
                // This allows compat adapters to redirect to different endpoints
                let endpoint_path = codec.endpoint().default_path();
                let url = self.provider.build_url(endpoint_path, &req.model);

                // Ensure stream is explicitly set to false for non-streaming requests
                let mut req_with_stream = req.clone();
                req_with_stream.stream = Some(false);

                let endpoint_req = EndpointRequest::ChatCompletion(req_with_stream.clone());
                let body_bytes = codec.encode(&endpoint_req)?;

                let auth_header = self
                    .resolve_auth_header_for_provider(self.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    self.provider.as_ref(),
                    "POST",
                    &url,
                    &serde_json::to_value(&req_with_stream)?,
                    &body_bytes,
                );
                let extra: Vec<(&str, &str)> = all_headers
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_str()))
                    .collect();

                let auth = auth_header.as_ref().map(str_pair);
                let raw_value = http::request::post_json_raw(
                    &self.http_client,
                    &url,
                    auth,
                    &extra,
                    body_bytes,
                    self.config.max_retries,
                )
                .await?;

                let raw_bytes = serde_json::to_vec(&raw_value)?;
                let endpoint_resp = codec.decode(&raw_bytes)?;
                match endpoint_resp {
                    EndpointResponse::ChatCompletion(resp) => Ok(resp),
                    _ => Err(HiLlmError::InternalError {
                        message: "Unexpected response type from ChatCompletion codec".into(),
                    }),
                }
            } else {
                // Fall back to legacy path
                let prepared = self.prepare_request(
                    &req,
                    |p| p.chat_completions_path(),
                    &req.model,
                    Some(false),
                )?;

                let auth_header = self
                    .resolve_auth_header_for_provider(prepared.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    prepared.provider.as_ref(),
                    "POST",
                    &prepared.url,
                    &prepared.body_json,
                    &prepared.body_bytes,
                );
                let extra: Vec<(&str, &str)> = all_headers
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_str()))
                    .collect();

                let auth = auth_header.as_ref().map(str_pair);
                let mut raw = http::request::post_json_raw(
                    &self.http_client,
                    &prepared.url,
                    auth,
                    &extra,
                    prepared.body_bytes,
                    self.config.max_retries,
                )
                .await?;
                prepared.provider.transform_response(&mut raw)?;
                serde_json::from_value::<ChatCompletionResponse>(raw).map_err(HiLlmError::from)
            }
        })
    }

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>> {
        Box::pin(async move {
            // Try new endpoint codec path first
            if let Some(codec) = self.provider.codec_for_endpoint(Endpoint::ChatCompletion) {
                // Use the codec's endpoint, not the requested endpoint
                // This allows compat adapters to redirect to different endpoints
                let endpoint_path = codec.endpoint().default_path();
                let url = self.provider.build_stream_url(endpoint_path, &req.model);

                // Ensure stream is explicitly set to true for streaming requests
                let mut req_with_stream = req.clone();
                req_with_stream.stream = Some(true);

                let endpoint_req = EndpointRequest::ChatCompletion(req_with_stream.clone());
                let body_bytes = codec.encode(&endpoint_req)?;

                let auth_header = self
                    .resolve_auth_header_for_provider(self.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    self.provider.as_ref(),
                    "POST",
                    &url,
                    &serde_json::to_value(&req_with_stream)?,
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
                        .map(|event| match event {
                            crate::provider::endpoint::EndpointStreamEvent::ChatCompletion(chunk) => {
                                Ok(chunk)
                            }
                            _ => Err(HiLlmError::InternalError {
                                message: "Unexpected stream event type".into(),
                            }),
                        })
                        .transpose()
                };
                let stream = http::stream::post_stream(
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
            } else {
                // Fall back to legacy path
                let prepared = self.prepare_request(
                    &req,
                    |p| p.chat_completions_path(),
                    &req.model,
                    Some(true),
                )?;

                let url = prepared
                    .provider
                    .build_stream_url(prepared.provider.chat_completions_path(), &req.model);

                let auth_header = self
                    .resolve_auth_header_for_provider(prepared.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    prepared.provider.as_ref(),
                    "POST",
                    &url,
                    &prepared.body_json,
                    &prepared.body_bytes,
                );
                let extra: Vec<(&str, &str)> = all_headers
                    .iter()
                    .map(|(n, v)| (n.as_str(), v.as_str()))
                    .collect();
                let auth = auth_header.as_ref().map(str_pair);

                match prepared.provider.stream_format() {
                    provider::StreamFormat::Sse => {
                        let provider = Arc::clone(&prepared.provider);
                        let parse_event = move |data: &str| provider.parse_stream_event(data);
                        let stream = http::stream::post_stream(
                            &self.http_client,
                            &url,
                            auth,
                            &extra,
                            prepared.body_bytes,
                            self.config.max_retries,
                            parse_event,
                        )
                        .await?;
                        Ok(stream)
                    }
                    provider::StreamFormat::AwsEventStream => {
                        let stream = http::eventstream::post_eventstream(
                            &self.http_client,
                            &url,
                            auth,
                            &extra,
                            prepared.body_bytes,
                            self.config.max_retries,
                            provider::bedrock::parse_bedrock_stream_event,
                        )
                        .await?;
                        Ok(stream)
                    }
                }
            }
        })
    }

    fn chat_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<RawExchange<ChatCompletionResponse>>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.chat_completions_path(), &req.model, Some(false))?;
            let raw_request = prepared.body_json.clone();

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &prepared.url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();

            let auth = auth_header.as_ref().map(str_pair);
            let mut raw = http::request::post_json_raw(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await?;

            let raw_response = Some(raw.clone());
            prepared.provider.transform_response(&mut raw)?;
            let data =
                serde_json::from_value::<ChatCompletionResponse>(raw).map_err(HiLlmError::from)?;

            Ok(RawExchange {
                data,
                raw_request,
                raw_response,
            })
        })
    }

    fn chat_stream_raw(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<
        '_,
        HiLlmResult<RawStreamExchange<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>>,
    > {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.chat_completions_path(), &req.model, Some(true))?;
            let raw_request = prepared.body_json.clone();
            let url = prepared
                .provider
                .build_stream_url(prepared.provider.chat_completions_path(), &req.model);

            let auth_header = self
                .resolve_auth_header_for_provider(prepared.provider.as_ref())
                .await?;
            let all_headers = self.all_headers_for_provider(
                prepared.provider.as_ref(),
                "POST",
                &url,
                &prepared.body_json,
                &prepared.body_bytes,
            );
            let extra: Vec<(&str, &str)> = all_headers
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_str()))
                .collect();
            let auth = auth_header.as_ref().map(str_pair);

            let stream = match prepared.provider.stream_format() {
                provider::StreamFormat::Sse => {
                    let provider = Arc::clone(&prepared.provider);
                    let parse_event = move |data: &str| provider.parse_stream_event(data);
                    http::stream::post_stream(
                        &self.http_client,
                        &url,
                        auth,
                        &extra,
                        prepared.body_bytes,
                        self.config.max_retries,
                        parse_event,
                    )
                    .await?
                }
                provider::StreamFormat::AwsEventStream => {
                    http::eventstream::post_eventstream(
                        &self.http_client,
                        &url,
                        auth,
                        &extra,
                        prepared.body_bytes,
                        self.config.max_retries,
                        provider::bedrock::parse_bedrock_stream_event,
                    )
                    .await?
                }
            };

            Ok(RawStreamExchange {
                stream,
                raw_request,
            })
        })
    }
}
