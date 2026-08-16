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

use bytes::Bytes;
use serde_json::{Value, json};

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::APIType;
use crate::provider::anthropic::AnthropicProvider;
use crate::provider::codec::APITypeCodec;
use crate::provider::unix_timestamp_secs;
use crate::types::{
    ChatCompletionChunk, FinishReason, PromptTokensDetails, StreamChoice, StreamDelta,
    StreamFunctionCall, StreamToolCall, ToolType, Usage,
};

const DEFAULT_MAX_TOKENS: u64 = 4096;
const HOSTED_TOOL_TYPES: &[&str] = &[
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

// ============ Request conversion: OpenAI Chat -> Anthropic Messages ============

/// Converts an OpenAI Chat Completions request body in place into an
/// Anthropic Messages request body.
///
/// See the module docs for the information this conversion drops.
pub(crate) fn convert_chat_request_to_anthropic(body: &mut Value) -> HiLlmResult<()> {
    let messages = body
        .as_object_mut()
        .and_then(|o| o.remove("messages"))
        .and_then(|v| match v {
            Value::Array(arr) => Some(arr),
            _ => None,
        })
        .unwrap_or_default();

    if messages.is_empty() {
        return Err(HiLlmError::BadRequest {
            message: "messages array must not be empty".to_owned(),
            status: 400,
        });
    }

    let mut system_blocks: Vec<Value> = Vec::new();
    let mut non_system_messages: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        match role {
            "system" | "developer" => match msg.get("content") {
                Some(Value::String(s)) if !s.is_empty() => {
                    let mut block = json!({"type": "text", "text": s});
                    if let Some(cc) = msg.get("cache_control") {
                        block["cache_control"] = cc.clone();
                    }
                    system_blocks.push(block);
                }
                Some(Value::Array(parts)) => {
                    for part in parts {
                        system_blocks.push(part.clone());
                    }
                }
                _ => {}
            },
            _ => non_system_messages.push(msg),
        }
    }

    if !system_blocks.is_empty() {
        body["system"] = json!(system_blocks);
    }

    let converted_messages: Vec<Value> = non_system_messages
        .into_iter()
        .map(convert_message_to_anthropic)
        .collect();

    let merged_messages = merge_consecutive_same_role(converted_messages);

    body["messages"] = json!(merged_messages);

    if body.get("max_tokens").is_none() {
        if let Some(mct) = body.get("max_completion_tokens").cloned() {
            body["max_tokens"] = mct;
        } else {
            body["max_tokens"] = json!(DEFAULT_MAX_TOKENS);
        }
    }
    if let Some(obj) = body.as_object_mut() {
        obj.remove("max_completion_tokens");
    }

    if let Some(stop) = body.as_object_mut().and_then(|o| o.remove("stop")) {
        let stop_sequences = match stop {
            Value::String(s) => json!([s]),
            arr @ Value::Array(_) => arr,
            _ => json!([]),
        };
        body["stop_sequences"] = stop_sequences;
    }

    if let Some(tool_choice) = body.as_object_mut().and_then(|o| o.remove("tool_choice")) {
        match convert_tool_choice(&tool_choice) {
            Some(tc) => {
                body["tool_choice"] = tc;
            }
            None => {
                // "none" has no Anthropic equivalent; drop the tools too so
                // the model cannot call them.
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("tools");
                }
            }
        }
    }

    if let Some(tools) = body.as_object_mut().and_then(|o| o.remove("tools"))
        && let Some(tools_array) = tools.as_array()
    {
        let anthropic_tools: Vec<Value> = tools_array
            .iter()
            .map(|tool| {
                let tool_type = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if is_hosted_tool_type(tool_type) {
                    tool.clone()
                } else {
                    convert_tool_to_anthropic(tool)
                }
            })
            .collect();
        body["tools"] = json!(anthropic_tools);
    }

    let reasoning_effort = body
        .as_object_mut()
        .and_then(|o| o.remove("reasoning_effort"))
        .and_then(|v| v.as_str().map(String::from))
        .or_else(|| {
            body.pointer("/extra_body/reasoning_effort")
                .and_then(|v| v.as_str().map(String::from))
        });

    if let Some(effort) = reasoning_effort {
        let budget_tokens: u64 = match effort.as_str() {
            "low" => 1024,
            "medium" => 4096,
            "high" => 16384,
            _ => 4096,
        };
        body["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": budget_tokens
        });

        let min_max_tokens = budget_tokens + 1;
        let current_max = body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        if current_max < min_max_tokens {
            body["max_tokens"] = json!(min_max_tokens);
        }
    }

    if let Some(response_format) = body
        .as_object_mut()
        .and_then(|o| o.remove("response_format"))
    {
        let rf_type = response_format
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("");
        match rf_type {
            "json_object" => {
                let instruction = json!({"type": "text", "text": "Respond with valid JSON only. Do not include any text outside the JSON object."});
                prepend_system_block(body, instruction);
            }
            "json_schema" => {
                if let Some(schema_def) = response_format.get("json_schema") {
                    let schema_name = schema_def
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("output");
                    let schema = schema_def.get("schema").cloned().unwrap_or(json!({}));
                    let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_default();
                    let instruction_text = format!(
                        "Respond with valid JSON matching the following schema named '{schema_name}':\n```json\n{schema_str}\n```\nDo not include any text outside the JSON object."
                    );
                    let instruction = json!({"type": "text", "text": instruction_text});
                    prepend_system_block(body, instruction);
                }
            }
            _ => {}
        }
    }

    // Fields without an Anthropic equivalent are dropped (documented in the
    // module docs). A `stream: false` flag is dropped too — Anthropic treats
    // the field as streaming-or-not and defaults to non-streaming; only an
    // explicit `stream: true` is forwarded so that `chat_stream()` over the
    // adapter produces a real SSE stream (whose events the codec converts
    // back to Chat chunks).
    if let Some(obj) = body.as_object_mut() {
        for key in &[
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "stream_options",
            "parallel_tool_calls",
            "service_tier",
            "user",
            "reasoning_effort",
            "extra_body",
        ] {
            obj.remove(*key);
        }
        if obj.get("stream") == Some(&json!(false)) {
            obj.remove("stream");
        }
    }

    Ok(())
}

fn prepend_system_block(body: &mut Value, instruction: Value) {
    if let Some(system) = body.get_mut("system").and_then(|s| s.as_array_mut()) {
        system.insert(0, instruction);
    } else {
        body["system"] = json!([instruction]);
    }
}

// ============ Response conversion: Anthropic Messages -> OpenAI Chat ============

/// Converts an Anthropic Messages response body in place into an OpenAI Chat
/// Completions response body. Bodies without a `stop_reason` (e.g. error
/// payloads) are left untouched.
pub(crate) fn convert_anthropic_response_to_chat(body: &mut Value) -> HiLlmResult<()> {
    if body.get("stop_reason").is_none() {
        return Ok(());
    }

    let id = body.get("id").cloned().unwrap_or(json!(""));
    let model = body.get("model").cloned().unwrap_or(json!(""));

    let content_blocks = body.get("content").and_then(|v| v.as_array()).cloned();

    // Note: thinking blocks are intentionally dropped here — Chat Completions
    // has no place for them.
    let text_content: Option<String> = content_blocks.as_ref().map(|blocks| {
        blocks
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("")
    });

    let tool_calls: Option<Vec<Value>> = content_blocks.as_ref().map(|blocks| {
        blocks
            .iter()
            .filter(|b| {
                matches!(
                    b.get("type").and_then(|t| t.as_str()),
                    Some("tool_use") | Some("server_tool_use")
                )
            })
            .map(|b| {
                let arguments =
                    serde_json::to_string(b.get("input").unwrap_or(&json!({}))).unwrap_or_default();
                json!({
                    "id": b.get("id").cloned().unwrap_or(json!("")),
                    "type": "function",
                    "function": {
                        "name": b.get("name").cloned().unwrap_or(json!("")),
                        "arguments": arguments
                    }
                })
            })
            .collect()
    });

    let stop_reason = body
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("end_turn");
    let finish_reason = map_stop_reason(stop_reason);

    let input_tokens = body
        .pointer("/usage/input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation_tokens = body
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read_tokens = body
        .pointer("/usage/cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = body
        .pointer("/usage/output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let prompt_tokens = input_tokens + cache_creation_tokens + cache_read_tokens;

    let has_tool_calls = tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
    let message_content = if has_tool_calls && text_content.as_deref().unwrap_or("").is_empty() {
        Value::Null
    } else {
        json!(text_content)
    };

    let mut message = json!({
        "role": "assistant",
        "content": message_content
    });

    if let (Some(tc), true) = (tool_calls, has_tool_calls) {
        message["tool_calls"] = json!(tc);
    }

    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens
    });
    if cache_read_tokens > 0 || cache_creation_tokens > 0 {
        usage["prompt_tokens_details"] = json!({
            "cached_tokens": cache_read_tokens,
            "cache_write_tokens": cache_creation_tokens
        });
    }

    *body = json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_timestamp_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": usage
    });

    Ok(())
}

// ============ Stream conversion: Anthropic events -> OpenAI Chat chunks ============

/// Parses one Anthropic SSE event payload and returns the equivalent Chat
/// Completions chunk, if the event carries one.
pub(crate) fn parse_anthropic_event_as_chat_chunk(
    event_data: &str,
) -> HiLlmResult<Option<ChatCompletionChunk>> {
    let event: Value = serde_json::from_str(event_data).map_err(|e| HiLlmError::Streaming {
        message: format!("failed to parse Anthropic SSE event: {e}"),
    })?;

    let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");

    match event_type {
        "message_start" => {
            let msg = &event["message"];
            let id = msg
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let model = msg
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();

            let input_tokens = msg
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cache_creation = msg
                .pointer("/usage/cache_creation_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cache_read = msg
                .pointer("/usage/cache_read_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let prompt_tokens = input_tokens + cache_creation + cache_read;

            let usage = if prompt_tokens > 0 {
                Some(Usage {
                    prompt_tokens,
                    completion_tokens: 0,
                    total_tokens: prompt_tokens,
                    prompt_tokens_details: Some(PromptTokensDetails {
                        cached_tokens: cache_read,
                        cache_write_tokens: Some(cache_creation),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            } else {
                None
            };

            Ok(Some(ChatCompletionChunk {
                id,
                object: "chat.completion.chunk".to_owned(),
                created: unix_timestamp_secs(),
                model,
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: Some("assistant".to_owned()),
                        ..Default::default()
                    },
                    finish_reason: None,
                }],
                usage,
                system_fingerprint: None,
                service_tier: None,
            }))
        }

        "content_block_start" => {
            let block = &event["content_block"];
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let anthropic_index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

            if block_type == "tool_use" || block_type == "server_tool_use" {
                let tool_id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                let tool_name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();

                return Ok(Some(make_empty_chunk_with_tool_start(
                    anthropic_index,
                    tool_id,
                    tool_name,
                )));
            }
            Ok(None)
        }

        "content_block_delta" => {
            let delta = &event["delta"];
            let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

            match delta_type {
                "text_delta" => {
                    let text = delta.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    Ok(Some(make_text_chunk("", "", text)))
                }
                // Thinking deltas are dropped: Chat chunks cannot carry them.
                "thinking_delta" => Ok(None),
                "input_json_delta" => {
                    let partial_json = delta
                        .get("partial_json")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    Ok(Some(make_tool_arguments_delta(index, partial_json)))
                }
                _ => Ok(None),
            }
        }

        "message_delta" => {
            let stop_reason = event.pointer("/delta/stop_reason").and_then(|v| v.as_str());
            let finish = stop_reason.map(map_stop_reason_to_enum);
            let output_tokens = event
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64());

            let usage = output_tokens.map(|ct| Usage {
                prompt_tokens: 0,
                completion_tokens: ct,
                total_tokens: ct,
                ..Default::default()
            });

            Ok(Some(ChatCompletionChunk {
                id: String::new(),
                object: "chat.completion.chunk".to_owned(),
                created: unix_timestamp_secs(),
                model: String::new(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta::default(),
                    finish_reason: finish,
                }],
                usage,
                system_fingerprint: None,
                service_tier: None,
            }))
        }

        "message_stop" | "content_block_stop" | "ping" => Ok(None),

        "error" => {
            let message = event
                .pointer("/error/message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown Anthropic streaming error");
            Err(HiLlmError::Streaming {
                message: message.to_owned(),
            })
        }
        _ => Ok(None),
    }
}

// ============ Helper functions ============

fn convert_image_url_to_anthropic_source(url: &str) -> Value {
    if url.starts_with("data:")
        && let Some((header, data)) = url.split_once(',')
    {
        let media_type = header
            .trim_start_matches("data:")
            .trim_end_matches(";base64");
        return json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data
            }
        });
    }
    json!({
        "type": "image",
        "source": {"type": "url", "url": url}
    })
}

fn sanitize_tool_call_id(id: &str) -> std::borrow::Cow<'_, str> {
    if id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        std::borrow::Cow::Borrowed(id)
    } else {
        std::borrow::Cow::Owned(
            id.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect(),
        )
    }
}

fn merge_consecutive_same_role(messages: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if let Some(last) = merged.last_mut() {
            let last_role = last.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if last_role == role {
                let incoming_content = match msg.get("content") {
                    Some(Value::Array(arr)) => arr.clone(),
                    Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
                    Some(other) => vec![json!({"type": "text", "text": other.to_string()})],
                    None => vec![],
                };

                if let Some(Value::Array(existing)) = last.get_mut("content") {
                    existing.extend(incoming_content);
                } else {
                    let existing_content = match last.get("content") {
                        Some(Value::String(s)) => {
                            vec![json!({"type": "text", "text": s.clone()})]
                        }
                        Some(Value::Array(arr)) => arr.clone(),
                        Some(other) => vec![json!({"type": "text", "text": other.to_string()})],
                        None => vec![],
                    };
                    let mut combined = existing_content;
                    combined.extend(incoming_content);
                    last["content"] = json!(combined);
                }
                continue;
            }
        }

        merged.push(msg);
    }

    merged
}

fn convert_message_to_anthropic(msg: Value) -> Value {
    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

    match role {
        "user" => {
            let content = convert_user_content_to_anthropic(msg.get("content"));
            let mut user_msg = json!({"role": "user", "content": content});
            if let Some(cc) = msg.get("cache_control")
                && let Some(blocks) = user_msg.get_mut("content").and_then(|c| c.as_array_mut())
                && let Some(last) = blocks.last_mut()
            {
                last["cache_control"] = cc.clone();
            }
            user_msg
        }
        "assistant" => {
            let mut blocks: Vec<Value> = Vec::new();

            if let Some(text) = msg.get("content").and_then(|c| c.as_str())
                && !text.is_empty()
            {
                let mut block = json!({"type": "text", "text": text});
                if let Some(cc) = msg.get("cache_control") {
                    block["cache_control"] = cc.clone();
                }
                blocks.push(block);
            }

            if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                for tc in tool_calls {
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = tc
                        .pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let arguments_str = tc
                        .pointer("/function/arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let input: Value =
                        serde_json::from_str(arguments_str).unwrap_or_else(|_| json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": input
                    }));
                }
            }

            if blocks.is_empty() {
                blocks.push(json!({"type": "text", "text": ""}));
            }

            json!({"role": "assistant", "content": blocks})
        }
        "tool" => {
            let raw_id = msg
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let tool_use_id = sanitize_tool_call_id(raw_id);

            let result_content = match msg.get("content") {
                Some(Value::Array(arr)) => arr
                    .iter()
                    .map(|part| {
                        let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                        match part_type {
                            "image_url" => {
                                let url = part
                                    .pointer("/image_url/url")
                                    .and_then(|u| u.as_str())
                                    .unwrap_or("");
                                convert_image_url_to_anthropic_source(url)
                            }
                            _ => {
                                let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                json!({"type": "text", "text": text})
                            }
                        }
                    })
                    .collect::<Vec<_>>(),
                Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
                _ => vec![json!({"type": "text", "text": ""})],
            };

            let mut tool_result_block = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_content
            });
            if let Some(cc) = msg.get("cache_control") {
                tool_result_block["cache_control"] = cc.clone();
            }

            json!({
                "role": "user",
                "content": [tool_result_block]
            })
        }
        "function" => {
            let name = msg.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let sanitized_name = sanitize_tool_call_id(name);
            let content_text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": sanitized_name,
                    "content": [{"type": "text", "text": content_text}]
                }]
            })
        }
        _ => msg,
    }
}

fn convert_user_content_to_anthropic(content: Option<&Value>) -> Value {
    match content {
        None => json!([]),
        Some(Value::String(s)) => json!([{"type": "text", "text": s}]),
        Some(Value::Array(parts)) => {
            let blocks: Vec<Value> = parts
                .iter()
                .filter_map(|part| {
                    let part_type = part.get("type").and_then(|t| t.as_str())?;
                    match part_type {
                        "text" => {
                            let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                            let mut block = json!({"type": "text", "text": text});
                            if let Some(cc) = part.get("cache_control") {
                                block["cache_control"] = cc.clone();
                            }
                            Some(block)
                        }
                        "image_url" => {
                            let url = part.pointer("/image_url/url").and_then(|u| u.as_str())?;
                            let mut block = convert_image_url_to_anthropic_source(url);
                            if let Some(cc) = part.get("cache_control") {
                                block["cache_control"] = cc.clone();
                            }
                            Some(block)
                        }
                        "document" => {
                            let data = part.pointer("/document/data").and_then(|d| d.as_str())?;
                            let media_type = part
                                .pointer("/document/media_type")
                                .and_then(|m| m.as_str())
                                .unwrap_or("application/pdf");
                            let mut block = json!({
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            });
                            if let Some(cc) = part.get("cache_control") {
                                block["cache_control"] = cc.clone();
                            }
                            Some(block)
                        }
                        _ => {
                            let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                            if text.is_empty() {
                                None
                            } else {
                                Some(json!({"type": "text", "text": text}))
                            }
                        }
                    }
                })
                .collect();
            json!(blocks)
        }
        Some(other) => json!([{"type": "text", "text": other.to_string()}]),
    }
}

fn convert_tool_choice(tool_choice: &Value) -> Option<Value> {
    match tool_choice {
        Value::String(s) => match s.as_str() {
            "none" => None,
            "required" => Some(json!({"type": "any"})),
            _ => Some(json!({"type": "auto"})),
        },
        Value::Object(_) => {
            // {"type": "function", "function": {"name": "X"}} → {"type": "tool", "name": "X"}
            let name = tool_choice
                .pointer("/function/name")
                .and_then(|v| v.as_str());
            if let Some(name) = name {
                Some(json!({"type": "tool", "name": name}))
            } else {
                Some(json!({"type": "auto"}))
            }
        }
        _ => Some(json!({"type": "auto"})),
    }
}

fn convert_tool_to_anthropic(tool: &Value) -> Value {
    let function = tool.get("function");
    let name = function
        .and_then(|f| f.get("name"))
        .cloned()
        .unwrap_or(json!(""));
    let description = function.and_then(|f| f.get("description")).cloned();
    let mut parameters = function
        .and_then(|f| f.get("parameters"))
        .cloned()
        .unwrap_or(json!({"type": "object", "properties": {}}));

    // Normalize input_schema.type to "object" — Anthropic rejects other values.
    if parameters.get("type").and_then(|t| t.as_str()) != Some("object") {
        parameters["type"] = json!("object");
    }

    let mut tool_def = json!({
        "name": name,
        "input_schema": parameters
    });

    if let Some(desc) = description {
        tool_def["description"] = desc;
    }

    // Propagate cache_control if present on the tool definition.
    if let Some(cc) = tool.get("cache_control") {
        tool_def["cache_control"] = cc.clone();
    } else if let Some(cc) = function.and_then(|f| f.get("cache_control")) {
        tool_def["cache_control"] = cc.clone();
    }

    tool_def
}

fn is_hosted_tool_type(tool_type: &str) -> bool {
    HOSTED_TOOL_TYPES.contains(&tool_type)
}

fn map_stop_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "end_turn" | "stop_sequence" => "stop",
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        "content_filtered" | "refusal" => "content_filter",
        _ => "stop",
    }
}

fn map_stop_reason_to_enum(stop_reason: &str) -> FinishReason {
    match stop_reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolCalls,
        "max_tokens" => FinishReason::Length,
        "content_filtered" | "refusal" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn make_text_chunk(id: &str, model: &str, text: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: unix_timestamp_secs(),
        model: model.to_owned(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                content: Some(text.to_owned()),
                ..Default::default()
            },
            finish_reason: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    }
}

fn make_empty_chunk_with_tool_start(
    tool_index: u32,
    tool_id: String,
    tool_name: String,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: String::new(),
        object: "chat.completion.chunk".to_owned(),
        created: unix_timestamp_secs(),
        model: String::new(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                tool_calls: Some(vec![StreamToolCall {
                    index: tool_index,
                    id: Some(tool_id),
                    call_type: Some(ToolType::Function),
                    function: Some(StreamFunctionCall {
                        name: Some(tool_name),
                        arguments: None,
                    }),
                }]),
                ..Default::default()
            },
            finish_reason: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    }
}

fn make_tool_arguments_delta(tool_index: u32, partial_json: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: String::new(),
        object: "chat.completion.chunk".to_owned(),
        created: unix_timestamp_secs(),
        model: String::new(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                tool_calls: Some(vec![StreamToolCall {
                    index: tool_index,
                    id: None,
                    call_type: None,
                    function: Some(StreamFunctionCall {
                        name: None,
                        arguments: Some(partial_json.to_owned()),
                    }),
                }]),
                ..Default::default()
            },
            finish_reason: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
