use std::sync::Arc;

use crate::client::str_pair;
use crate::client::{BoxFuture, BoxStream, ChatCompletionClient, Client};
use crate::error::{HiLLMError, HiLLMResult};
use crate::http;
use crate::provider;
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::types::raw::{RawExchange, RawStreamExchange};

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ChatCompletionClient for Client {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLLMResult<ChatCompletionResponse>> {
        Box::pin(async move {
            // Try codec path first
            if let Some(codec) = self
                .provider
                .codec_for(provider::APIType::OpenAIChatCompletions)
            {
                let endpoint_path = codec.endpoint_path();
                let url = self.provider.build_url(endpoint_path, &req.model);

                let mut body = serde_json::to_value(&req)?;
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("model".into(), serde_json::Value::String(req.model.clone()));
                    obj.insert("stream".into(), serde_json::Value::Bool(false));
                }

                let body_bytes = codec.encode_request(&body)?;

                let auth_header = self
                    .resolve_auth_header_for_provider(self.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    self.provider.as_ref(),
                    "POST",
                    &url,
                    &body,
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
                serde_json::from_value::<ChatCompletionResponse>(response_value)
                    .map_err(HiLLMError::from)
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
                serde_json::from_value::<ChatCompletionResponse>(raw).map_err(HiLLMError::from)
            }
        })
    }

    fn chat_stream(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLLMResult<BoxStream<'static, HiLLMResult<ChatCompletionChunk>>>> {
        Box::pin(async move {
            // Try codec path first
            if let Some(codec) = self
                .provider
                .codec_for(provider::APIType::OpenAIChatCompletions)
            {
                let endpoint_path = codec.endpoint_path();
                let url = self.provider.build_stream_url(endpoint_path, &req.model);

                let mut body = serde_json::to_value(&req)?;
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("model".into(), serde_json::Value::String(req.model.clone()));
                    obj.insert("stream".into(), serde_json::Value::Bool(true));
                }
                let body_bytes = codec.encode_request(&body)?;

                let auth_header = self
                    .resolve_auth_header_for_provider(self.provider.as_ref())
                    .await?;
                let all_headers = self.all_headers_for_provider(
                    self.provider.as_ref(),
                    "POST",
                    &url,
                    &body,
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
                        .map(serde_json::from_value::<ChatCompletionChunk>)
                        .transpose()
                        .map_err(HiLLMError::from)
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
                    provider::StreamFormat::SSE => {
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
    ) -> BoxFuture<'_, HiLLMResult<RawExchange<ChatCompletionResponse>>> {
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
                serde_json::from_value::<ChatCompletionResponse>(raw).map_err(HiLLMError::from)?;

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
        HiLLMResult<RawStreamExchange<BoxStream<'static, HiLLMResult<ChatCompletionChunk>>>>,
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
                provider::StreamFormat::SSE => {
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
