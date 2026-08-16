//! Explicit compatibility adapter between OpenAI Chat Completions and
//! Anthropic Messages.
//!
//! This module is the *explicit* compatibility layer required by the API-type
//! routing design: the Anthropic provider's native protocol is
//! [`APIType::AnthropicMessages`]. Callers that still want to use the OpenAI
//! Chat Completions shapes ([`LlmClient::chat`](crate::client::LlmClient))
//! against Anthropic must opt in by creating a provider instance through
//! [`AnthropicChatCompatProvider`] (e.g. by explicitly selecting
//! [`APIType::OpenAIChatCompletions`] for the `anthropic` provider). The
//! adapter is never applied implicitly by [`AnthropicProvider`].
//!
//! # Information loss
//!
//! The Chat Completions surface cannot express every Anthropic concept. The
//! adapter documents what is lost:
//!
//! Request (Chat → Anthropic):
//! - `n`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
//!   `stream_options`, `parallel_tool_calls`, `service_tier` and `user` are
//!   dropped (no Anthropic equivalent).
//! - `response_format` is approximated by prepending a system instruction;
//!   strict schema enforcement is lost.
//! - `reasoning_effort` is approximated by a `thinking` budget
//!   (low → 1024, medium → 4096, high → 16384 tokens).
//!
//! Response (Anthropic → Chat):
//! - Only `text` and `tool_use` content blocks survive; `thinking` blocks and
//!   their signatures are dropped.
//! - Anthropic stop reasons are mapped to the nearest OpenAI finish reason
//!   (`end_turn`/`stop_sequence` → `stop`, `tool_use` → `tool_calls`,
//!   `max_tokens` → `length`, `content_filtered`/`refusal` →
//!   `content_filter`); anything else becomes `stop`.
//! - Cache usage is folded into `prompt_tokens`
//!   (`input + cache_creation + cache_read`) and reported through
//!   `prompt_tokens_details` (`cached_tokens` = cache reads,
//!   `cache_write_tokens` = cache creation).
//!
//! Streaming (Anthropic events → Chat chunks):
//! - `ping`, `content_block_stop` and thinking deltas produce no chunks.
//! - Usage is reported partially: prompt tokens on `message_start`,
//!   completion tokens on `message_delta`.

#[path = "compat_convert.rs"]
mod compat_convert;

use bytes::Bytes;
use serde_json::Value;

use crate::error::HiLlmResult;
use crate::provider::APIType;
use crate::provider::anthropic::AnthropicProvider;
use crate::provider::codec::APITypeCodec;
use compat_convert::*;

pub(crate) const DEFAULT_MAX_TOKENS: u64 = 4096;
pub(crate) const HOSTED_TOOL_TYPES: &[&str] = &[
    "computer_20241022",
    "computer_use_20250124",
    "web_search_20250305",
    "code_execution_20250522",
];

/// Codec that lets an OpenAI Chat Completions call flow over the Anthropic
/// Messages protocol.
///
/// Requests are converted Chat → Anthropic before sending, responses and
/// stream events are converted Anthropic → Chat after receiving. See the
/// module docs for what is lost in each direction.
pub(crate) struct AnthropicChatCompatCodec;

impl APITypeCodec for AnthropicChatCompatCodec {
    fn api_type(&self) -> APIType {
        // The adapter exists so callers can keep using the Chat Completions
        // shapes; the codec therefore reports the Chat protocol.
        APIType::OpenAIChatCompletions
    }

    fn endpoint_path(&self) -> &str {
        // On the wire this is still the Anthropic Messages endpoint.
        "/messages"
    }

    fn encode_request(&self, request: &Value) -> HiLlmResult<Bytes> {
        let mut body = request.clone();
        convert_chat_request_to_anthropic(&mut body)?;
        Ok(Bytes::from(serde_json::to_vec(&body)?))
    }

    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<Value> {
        let mut value: Value = serde_json::from_slice(bytes)?;
        convert_anthropic_response_to_chat(&mut value)?;
        Ok(value)
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<Value>> {
        match parse_anthropic_event_as_chat_chunk(data)? {
            Some(chunk) => Ok(Some(serde_json::to_value(chunk)?)),
            None => Ok(None),
        }
    }
}

/// Anthropic provider instance bound to the Chat Completions shapes via the
/// explicit compatibility adapter.
///
/// Create one through
/// [`create_provider("anthropic", APIType::OpenAIChatCompletions)`](crate::provider::create_provider).
/// For the native Messages protocol, use a separate instance created with
/// [`APIType::AnthropicMessages`].
pub(crate) struct AnthropicChatCompatProvider {
    inner: AnthropicProvider,
}

impl AnthropicChatCompatProvider {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            inner: AnthropicProvider::new(),
        }
    }
}

impl crate::provider::Provider for AnthropicChatCompatProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn base_url(&self) -> &str {
        self.inner.base_url()
    }

    fn auth_header<'a>(
        &'a self,
        api_key: &'a str,
    ) -> Option<(std::borrow::Cow<'static, str>, std::borrow::Cow<'a, str>)> {
        self.inner.auth_header(api_key)
    }

    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        self.inner.extra_headers()
    }

    fn dynamic_headers(&self, body: &Value) -> Vec<(String, String)> {
        self.inner.dynamic_headers(body)
    }

    fn matches_model(&self, model: &str) -> bool {
        self.inner.matches_model(model)
    }

    fn env_var(&self) -> Option<&str> {
        self.inner.env_var()
    }

    fn available_api_types(&self) -> Vec<APIType> {
        // This instance speaks Chat Completions (translated to the Anthropic
        // wire protocol). Use a native Messages instance for
        // AnthropicMessages.
        vec![APIType::OpenAIChatCompletions]
    }

    fn api_type(&self) -> APIType {
        APIType::OpenAIChatCompletions
    }

    fn codec_for(&self, api_type: APIType) -> Option<Box<dyn APITypeCodec>> {
        match api_type {
            APIType::OpenAIChatCompletions => Some(Box::new(AnthropicChatCompatCodec)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::error::HiLlmError;
    use crate::types::FinishReason;

    fn chat_request(messages: Value) -> Value {
        json!({
            "model": "claude-sonnet-4-5",
            "messages": messages
        })
    }

    #[test]
    fn codec_reports_chat_api_type_and_messages_endpoint() {
        let codec = AnthropicChatCompatCodec;
        assert_eq!(codec.api_type(), APIType::OpenAIChatCompletions);
        assert_eq!(codec.endpoint_path(), "/messages");
    }

    #[test]
    fn request_extracts_system_messages() {
        let mut body = chat_request(json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"}
        ]));
        convert_chat_request_to_anthropic(&mut body).unwrap();

        assert_eq!(body["system"][0]["text"], "You are terse.");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn request_defaults_max_tokens() {
        let mut body = chat_request(json!([{"role": "user", "content": "Hi"}]));
        convert_chat_request_to_anthropic(&mut body).unwrap();
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn request_maps_stop_to_stop_sequences() {
        let mut body = chat_request(json!([{"role": "user", "content": "Hi"}]));
        body["stop"] = json!(["END"]);
        convert_chat_request_to_anthropic(&mut body).unwrap();
        assert_eq!(body["stop_sequences"], json!(["END"]));
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn request_rejects_empty_messages() {
        let mut body = chat_request(json!([]));
        let result = convert_chat_request_to_anthropic(&mut body);
        assert!(result.is_err());
    }

    #[test]
    fn request_converts_tools_and_tool_choice() {
        let mut body = chat_request(json!([{"role": "user", "content": "Use the tool"}]));
        body["tools"] = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }]);
        body["tool_choice"] = json!({"type": "function", "function": {"name": "get_weather"}});
        convert_chat_request_to_anthropic(&mut body).unwrap();

        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(
            body["tool_choice"],
            json!({"type": "tool", "name": "get_weather"})
        );
    }

    #[test]
    fn request_tool_choice_none_drops_tools() {
        let mut body = chat_request(json!([{"role": "user", "content": "Hi"}]));
        body["tools"] = json!([{
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}}
        }]);
        body["tool_choice"] = json!("none");
        convert_chat_request_to_anthropic(&mut body).unwrap();

        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn request_drops_unsupported_fields() {
        let mut body = chat_request(json!([{"role": "user", "content": "Hi"}]));
        body["n"] = json!(2);
        body["presence_penalty"] = json!(0.5);
        body["user"] = json!("u-1");
        body["stream_options"] = json!({"include_usage": true});
        convert_chat_request_to_anthropic(&mut body).unwrap();

        for key in ["n", "presence_penalty", "user", "stream_options"] {
            assert!(body.get(key).is_none(), "{key} should be dropped");
        }
    }

    #[test]
    fn request_maps_reasoning_effort_to_thinking() {
        let mut body = chat_request(json!([{"role": "user", "content": "Hi"}]));
        body["reasoning_effort"] = json!("high");
        convert_chat_request_to_anthropic(&mut body).unwrap();

        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 16384);
        // max_tokens must exceed the thinking budget
        assert!(body["max_tokens"].as_u64().unwrap() > 16384);
    }

    #[test]
    fn request_converts_tool_result_messages() {
        let mut body = chat_request(json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ]));
        convert_chat_request_to_anthropic(&mut body).unwrap();

        let messages = body["messages"].as_array().unwrap();
        // assistant tool_use block
        let assistant = &messages[1];
        assert_eq!(assistant["content"][0]["type"], "tool_use");
        assert_eq!(assistant["content"][0]["name"], "get_weather");
        // tool result becomes a user message
        let tool_result = &messages[2];
        assert_eq!(tool_result["role"], "user");
        assert_eq!(tool_result["content"][0]["type"], "tool_result");
        assert_eq!(tool_result["content"][0]["tool_use_id"], "call_1");
    }

    #[test]
    fn response_converts_text_and_usage() {
        let mut body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2
            }
        });
        convert_anthropic_response_to_chat(&mut body).unwrap();

        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 15); // 10 + 3 + 2
        assert_eq!(body["usage"]["completion_tokens"], 5);
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
        assert_eq!(
            body["usage"]["prompt_tokens_details"]["cache_write_tokens"],
            2
        );
    }

    #[test]
    fn response_converts_tool_use_and_stop_reasons() {
        let mut body = json!({
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "Paris"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        convert_anthropic_response_to_chat(&mut body).unwrap();

        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let tool_call = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tool_call["function"]["name"], "get_weather");
        assert_eq!(tool_call["function"]["arguments"], "{\"city\":\"Paris\"}");
    }

    #[test]
    fn response_leaves_error_bodies_untouched() {
        let mut body = json!({"type": "error", "error": {"message": "boom"}});
        let original = body.clone();
        convert_anthropic_response_to_chat(&mut body).unwrap();
        assert_eq!(body, original);
    }

    #[test]
    fn stream_message_start_yields_role_chunk_with_usage() {
        let event = json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "claude-sonnet-4-5",
                "role": "assistant",
                "usage": {"input_tokens": 7, "cache_read_input_tokens": 1}
            }
        });
        let chunk = parse_anthropic_event_as_chat_chunk(&event.to_string())
            .unwrap()
            .unwrap();
        assert_eq!(chunk.id, "msg_1");
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
        assert_eq!(chunk.usage.as_ref().unwrap().prompt_tokens, 8);
    }

    #[test]
    fn stream_text_delta_yields_content_chunk() {
        let event = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"}
        });
        let chunk = parse_anthropic_event_as_chat_chunk(&event.to_string())
            .unwrap()
            .unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn stream_thinking_delta_is_dropped() {
        let event = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"}
        });
        assert!(
            parse_anthropic_event_as_chat_chunk(&event.to_string())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn stream_message_delta_yields_finish_reason() {
        let event = json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 12}
        });
        let chunk = parse_anthropic_event_as_chat_chunk(&event.to_string())
            .unwrap()
            .unwrap();
        assert_eq!(chunk.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(chunk.usage.as_ref().unwrap().completion_tokens, 12);
    }

    #[test]
    fn stream_error_event_returns_streaming_error() {
        let event = json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "overloaded"}
        });
        let result = parse_anthropic_event_as_chat_chunk(&event.to_string());
        assert!(matches!(result, Err(HiLlmError::Streaming { .. })));
    }

    #[test]
    fn stream_tool_events_map_to_tool_call_chunks() {
        let start = json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_weather"}
        });
        let chunk = parse_anthropic_event_as_chat_chunk(&start.to_string())
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.id.as_deref(), Some("tu_1"));
        assert_eq!(
            tc.function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );

        let delta = json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}
        });
        let chunk = parse_anthropic_event_as_chat_chunk(&delta.to_string())
            .unwrap()
            .unwrap();
        let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(
            tc.function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"city\":")
        );
    }

    #[test]
    fn compat_provider_is_bound_to_chat_api_type() {
        use crate::provider::Provider;

        let provider = AnthropicChatCompatProvider::new();
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.api_type(), APIType::OpenAIChatCompletions);
        assert_eq!(
            provider.available_api_types(),
            vec![APIType::OpenAIChatCompletions]
        );
        assert!(provider.codec_for(APIType::OpenAIChatCompletions).is_some());
        assert!(provider.codec_for(APIType::AnthropicMessages).is_none());
        // The base URL honors ANTHROPIC_BASE_URL when set.
        let expected_base = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.trim_end_matches('/').to_string())
            .unwrap_or_else(|| "https://api.anthropic.com/v1".to_owned());
        assert_eq!(provider.base_url(), expected_base);
        // Keeps Anthropic auth and version headers.
        let (name, value) = provider.auth_header("sk-key").unwrap();
        assert_eq!(name.as_ref(), "x-api-key");
        assert_eq!(value.as_ref(), "sk-key");
        assert!(
            provider
                .extra_headers()
                .iter()
                .any(|(k, v)| *k == "anthropic-version" && *v == "2023-06-01")
        );
    }

    #[test]
    fn codec_round_trips_request_and_response() {
        let codec = AnthropicChatCompatCodec;
        let request = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"}
            ],
            "stream": false
        });
        let bytes = codec.encode_request(&request).unwrap();
        let wire: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(wire["system"][0]["text"], "Be brief.");
        assert_eq!(wire["messages"][0]["role"], "user");

        let response = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let response_bytes = serde_json::to_vec(&response).unwrap();
        let chat_response = codec.decode_response(&response_bytes).unwrap();
        assert_eq!(chat_response["object"], "chat.completion");
        assert_eq!(chat_response["choices"][0]["message"]["content"], "Hi");
    }
}
