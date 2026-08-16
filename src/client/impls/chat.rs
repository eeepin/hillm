use std::sync::Arc;

use crate::client::str_pair;
use crate::client::{BoxFuture, BoxStream, Client, ChatCompletionClient};
use crate::error::{HiLlmError, HiLlmResult};
use crate::http;
use crate::provider;
use crate::types::audio::{CreateSpeechRequest, CreateTranscriptionRequest, TranscriptionResponse};
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::types::embedding::{EmbeddingRequest, EmbeddingResponse};
use crate::types::image::{CreateImageRequest, ImagesResponse};
use crate::types::model::ModelsListResponse;
use crate::types::moderation::{ModerationRequest, ModerationResponse};
use crate::types::ocr::{OcrRequest, OcrResponse};
use crate::types::rerank::{RerankRequest, RerankResponse};
use crate::types::search::{SearchRequest, SearchResponse};

#[cfg(any(feature = "default-http", feature = "wasm-http"))]
impl ChatCompletionClient for Client {
    fn chat(
        &self,
        req: ChatCompletionRequest,
    ) -> BoxFuture<'_, HiLlmResult<ChatCompletionResponse>> {
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
                    .map_err(HiLlmError::from)
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
                        .map_err(HiLlmError::from)
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

    fn embed(&self, req: EmbeddingRequest) -> BoxFuture<'_, HiLlmResult<EmbeddingResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.embeddings_path(), &req.model, None)?;

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
            serde_json::from_value::<EmbeddingResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn list_models(&self) -> BoxFuture<'_, HiLlmResult<ModelsListResponse>> {
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
            serde_json::from_value::<ModelsListResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn image_generate(
        &self,
        req: CreateImageRequest,
    ) -> BoxFuture<'_, HiLlmResult<ImagesResponse>> {
        Box::pin(async move {
            let model = req.model.as_deref().unwrap_or_default();
            let prepared =
                self.prepare_request(&req, |p| p.image_generations_path(), model, None)?;

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
            serde_json::from_value::<ImagesResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn speech(&self, req: CreateSpeechRequest) -> BoxFuture<'_, HiLlmResult<bytes::Bytes>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.audio_speech_path(), &req.model, None)?;

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
            http::request::post_binary(
                &self.http_client,
                &prepared.url,
                auth,
                &extra,
                prepared.body_bytes,
                self.config.max_retries,
            )
            .await
        })
    }

    fn transcribe(
        &self,
        req: CreateTranscriptionRequest,
    ) -> BoxFuture<'_, HiLlmResult<TranscriptionResponse>> {
        Box::pin(async move {
            let prepared =
                self.prepare_request(&req, |p| p.audio_transcriptions_path(), &req.model, None)?;

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
            serde_json::from_value::<TranscriptionResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn moderate(&self, req: ModerationRequest) -> BoxFuture<'_, HiLlmResult<ModerationResponse>> {
        Box::pin(async move {
            let model = req.model.as_deref().unwrap_or_default();
            let prepared = self.prepare_request(&req, |p| p.moderations_path(), model, None)?;

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
            serde_json::from_value::<ModerationResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn rerank(&self, req: RerankRequest) -> BoxFuture<'_, HiLlmResult<RerankResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.rerank_path(), &req.model, None)?;

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
            serde_json::from_value::<RerankResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn search(&self, req: SearchRequest) -> BoxFuture<'_, HiLlmResult<SearchResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.search_path(), &req.model, None)?;

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
            serde_json::from_value::<SearchResponse>(raw).map_err(HiLlmError::from)
        })
    }

    fn ocr(&self, req: OcrRequest) -> BoxFuture<'_, HiLlmResult<OcrResponse>> {
        Box::pin(async move {
            let prepared = self.prepare_request(&req, |p| p.ocr_path(), &req.model, None)?;

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
            serde_json::from_value::<OcrResponse>(raw).map_err(HiLlmError::from)
        })
    }
}
