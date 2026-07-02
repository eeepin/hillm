use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::common::{Provider, registry, unix_timestamp_secs};
use crate::types::{
    ChatCompletionChunk, FinishReason, StreamChoice, StreamDelta, StreamFunctionCall,
    StreamToolCall,
};
use serde_json::{Value, json};
use std::borrow::Cow;

static ANTHROPIC_EXTRA_HEADERS: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];
const DEFAULT_MAX_TOKENS: u64 = 4096;
const HOSTED_TOOL_TYPES: &[&str] = &[
    "computer_20241022",
    "computer_use_20250124",
    "web_search_20250305",
    "code_execution_20250522",
];

const BETA_COMPUTER_USE: &str = "computer-use-2025-01-24";
const BETA_WEB_SEARCH: &str = "web-search-2025-03-05";
const BETA_CODE_EXECUTION: &str = "code-execution-2025-05-22";
const BETA_THINKING: &str = "thinking-2025-04-14";
const BETA_PROMPT_CACHING: &str = "prompt-caching-2024-07-31";
const BETA_PDFS: &str = "pdfs-2024-09-25";

/// Anthropic provider
pub struct AnthropicProvider;

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn base_url(&self) -> &str {
        "https://api.anthropic.com/v1"
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        Some((Cow::Borrowed("x-api-key"), Cow::Borrowed(api_key)))
    }

    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        ANTHROPIC_EXTRA_HEADERS
    }

    fn dynamic_headers(&self, body: &serde_json::Value) -> Vec<(String, String)> {
        let mut betas: Vec<&str> = Vec::new();

        // Check for extended thinking.
        if body.get("thinking").is_some() {
            betas.push(BETA_THINKING);
        }

        // Check for hosted tools in the tools array.
        if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
            for tool in tools {
                let tool_type = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match tool_type {
                    "computer_20241022" | "computer_use_20250124"
                        if !betas.contains(&BETA_COMPUTER_USE) =>
                    {
                        betas.push(BETA_COMPUTER_USE);
                    }
                    "web_search_20250305" if !betas.contains(&BETA_WEB_SEARCH) => {
                        betas.push(BETA_WEB_SEARCH);
                    }
                    "code_execution_20250522" if !betas.contains(&BETA_CODE_EXECUTION) => {
                        betas.push(BETA_CODE_EXECUTION);
                    }
                    _ => {}
                }
            }
        }

        // Check for prompt caching: any `cache_control` field anywhere in the body.
        if body_contains_cache_control(body) && !betas.contains(&BETA_PROMPT_CACHING) {
            betas.push(BETA_PROMPT_CACHING);
        }

        // Check for PDF/document content blocks.
        if body_contains_document_block(body) && !betas.contains(&BETA_PDFS) {
            betas.push(BETA_PDFS);
        }

        if betas.is_empty() {
            vec![]
        } else {
            vec![("anthropic-beta".to_owned(), betas.join(","))]
        }
    }

    async fn matches_model(&self, model: &str) -> bool {
        registry()
            .await
            .unwrap_or_default()
            .get("anthropic")
            .is_some_and(|p| p.models.contains_key(model))
    }

    /// Anthropic uses `/messages` instead of `/chat/completions`.
    fn chat_completions_path(&self) -> &str {
        "/messages"
    }

    /// Transform an OpenAI-format request body into Anthropic Messages API format.
    fn transform_request(&self, body: &mut Value) -> HiLlmResult<()> {
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
        body.as_object_mut()
            .map(|o| o.remove("max_completion_tokens"));

        if let Some(stop) = body.as_object_mut().and_then(|o| o.remove("stop")) {
            let stop_sequences = match stop {
                Value::String(s) => json!([s]),
                arr @ Value::Array(_) => arr,
                _ => json!([]),
            };
            body["stop_sequences"] = stop_sequences;
        }

        if let Some(tool_choice) = body.as_object_mut().and_then(|o| o.remove("tool_choice")) {
            let anthropic_tool_choice = convert_tool_choice(&tool_choice);
            match anthropic_tool_choice {
                Some(tc) => {
                    body["tool_choice"] = tc;
                }
                None => {
                    body.as_object_mut().map(|o| o.remove("tools"));
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
                    if let Some(system) = body.get_mut("system").and_then(|s| s.as_array_mut()) {
                        system.insert(0, instruction);
                    } else {
                        body["system"] = json!([instruction]);
                    }
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
                        if let Some(system) = body.get_mut("system").and_then(|s| s.as_array_mut())
                        {
                            system.insert(0, instruction);
                        } else {
                            body["system"] = json!([instruction]);
                        }
                    }
                }
                _ => {}
            }
        }

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
        }

        Ok(())
    }

    fn transform_response(&self, body: &mut Value) -> HiLlmResult<()> {
        if body.get("stop_reason").is_none() {
            return Ok(());
        }

        let id = body.get("id").cloned().unwrap_or(json!(""));
        let model = body.get("model").cloned().unwrap_or(json!(""));

        let content_blocks = body.get("content").and_then(|v| v.as_array()).cloned();

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
                    let arguments = serde_json::to_string(b.get("input").unwrap_or(&json!({})))
                        .unwrap_or_default();
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
        let message_content = if has_tool_calls && text_content.as_deref().unwrap_or("").is_empty()
        {
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
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens
            }
        });

        Ok(())
    }

    fn parse_stream_event(&self, event_data: &str) -> HiLlmResult<Option<ChatCompletionChunk>> {
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
                    Some(crate::types::Usage {
                        prompt_tokens,
                        completion_tokens: 0,
                        total_tokens: prompt_tokens,
                        prompt_tokens_details: None,
                        completion_tokens_details: None,
                        cost: None,
                        cost_details: None,
                        is_byok: None,
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
                            content: None,
                            tool_calls: None,
                            refusal: None,
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
                let anthropic_index =
                    event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

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
                let finish_reason = stop_reason.map(map_stop_reason);
                let output_tokens = event
                    .pointer("/usage/output_tokens")
                    .and_then(|v| v.as_u64());

                let finish = finish_reason.map(|fr| match fr {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::Length,
                    "tool_calls" => FinishReason::ToolCalls,
                    _ => FinishReason::Other,
                });

                let usage = output_tokens.map(|ct| crate::types::Usage {
                    prompt_tokens: 0,
                    completion_tokens: ct,
                    total_tokens: ct,
                    prompt_tokens_details: None,
                    completion_tokens_details: None,
                    cost: None,
                    cost_details: None,
                    is_byok: None,
                });

                Ok(Some(ChatCompletionChunk {
                    id: String::new(),
                    object: "chat.completion.chunk".to_owned(),
                    created: unix_timestamp_secs(),
                    model: String::new(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                            refusal: None,
                        },
                        finish_reason: finish,
                    }],
                    usage,
                    system_fingerprint: None,
                    service_tier: None,
                }))
            }

            "message_stop" => Ok(None),
            "content_block_stop" | "ping" => Ok(None),
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
}

// Helper Functions

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

fn sanitize_tool_call_id(id: &str) -> Cow<'_, str> {
    if id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        Cow::Borrowed(id)
    } else {
        Cow::Owned(
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
                        Some(Value::String(s)) => vec![json!({"type": "text", "text": s.clone()})],
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

            let has_tool_use = blocks
                .iter()
                .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"));
            if blocks.is_empty() {
                blocks.push(json!({"type": "text", "text": ""}));
            } else if !has_tool_use {
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

fn body_contains_cache_control(body: &Value) -> bool {
    match body {
        Value::Object(map) => {
            if map.contains_key("cache_control") {
                return true;
            }
            map.values().any(body_contains_cache_control)
        }
        Value::Array(arr) => arr.iter().any(body_contains_cache_control),
        _ => false,
    }
}

fn body_contains_document_block(body: &Value) -> bool {
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if let Some(content) = msg.get("content").and_then(|c| c.as_array()) {
                for part in content {
                    if part.get("type").and_then(|t| t.as_str()) == Some("document") {
                        return true;
                    }
                }
            }
        }
    }
    false
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

fn make_text_chunk(id: &str, model: &str, text: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: unix_timestamp_secs(),
        model: model.to_owned(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                role: None,
                content: Some(text.to_owned()),
                tool_calls: None,
                refusal: None,
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
                role: None,
                content: None,
                tool_calls: Some(vec![StreamToolCall {
                    index: tool_index,
                    id: Some(tool_id),
                    call_type: Some(crate::types::ToolType::Function),
                    function: Some(StreamFunctionCall {
                        name: Some(tool_name),
                        arguments: None,
                    }),
                }]),
                refusal: None,
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
                role: None,
                content: None,
                tool_calls: Some(vec![StreamToolCall {
                    index: tool_index,
                    id: None,
                    call_type: None,
                    function: Some(StreamFunctionCall {
                        name: None,
                        arguments: Some(partial_json.to_owned()),
                    }),
                }]),
                refusal: None,
            },
            finish_reason: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    }
}

// Tests
