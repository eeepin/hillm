use std::borrow::Cow;

#[cfg(feature = "bedrock")]
use crate::error::HiLlmError;
use crate::error::HiLlmResult;
use crate::provider::{Provider, StreamFormat, registry_get};
use crate::types::ChatCompletionChunk;

/// Default AWS region for Bedrock when none is specified.
const DEFAULT_REGION: &str = "us-east-1";

fn reasoning_effort_to_budget_tokens(effort: &str) -> u64 {
    match effort {
        "low" => 1024,
        "medium" => 4096,
        "high" => 16384,
        _ => 4096, // default to medium
    }
}

fn format_from_media_type(media_type: &str) -> &str {
    media_type.split('/').nth(1).unwrap_or("pdf")
}

fn dns_suffix_for_region(region: &str) -> &'static str {
    if region.starts_with("eusc-") {
        "amazonaws.eu"
    } else if region.starts_with("cn-") {
        "amazonaws.com.cn"
    } else {
        "amazonaws.com"
    }
}

fn percent_encode_model(model: &str) -> String {
    let mut encoded = String::with_capacity(model.len());
    for byte in model.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            other => {
                encoded.push('%');
                let hi = char::from_digit(u32::from(other >> 4), 16).unwrap_or('0');
                let lo = char::from_digit(u32::from(other & 0xf), 16).unwrap_or('0');
                encoded.push(hi.to_ascii_uppercase());
                encoded.push(lo.to_ascii_uppercase());
            }
        }
    }
    encoded
}

pub struct BedrockProvider {
    region: String,
    base_url: String,
    cross_region_prefix: Option<String>,
}

impl BedrockProvider {
    #[must_use]
    pub fn new(region: impl Into<String>) -> Self {
        let region = region.into();
        let custom_base_url = std::env::var("BEDROCK_BASE_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.trim_end_matches('/').to_string());
        let base_url = custom_base_url.clone().unwrap_or_else(|| {
            let dns_suffix = dns_suffix_for_region(&region);
            format!("https://bedrock-runtime.{region}.{dns_suffix}")
        });
        let cross_region_prefix = if custom_base_url.is_some() {
            None
        } else {
            std::env::var("BEDROCK_CROSS_REGION")
                .ok()
                .filter(|v| !v.is_empty())
                .map(|v| format!("{v}."))
        };
        Self {
            region,
            base_url,
            cross_region_prefix,
        }
    }

    #[must_use]
    pub fn from_env() -> Self {
        let region = std::env::var("AWS_DEFAULT_REGION")
            .or_else(|_| std::env::var("AWS_REGION"))
            .unwrap_or_else(|_| DEFAULT_REGION.to_owned());
        Self::new(region)
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn region(&self) -> &str {
        &self.region
    }
}

impl Provider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_header<'a>(&'a self, _api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        None
    }

    fn matches_model(&self, model: &str) -> bool {
        registry_get().is_some_and(|reg| {
            reg.get("amazon-bedrock")
                .is_some_and(|p| p.models.contains_key(model))
        })
    }

    fn validate(&self) -> HiLlmResult<()> {
        #[cfg(feature = "bedrock")]
        {
            if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
                return Err(HiLlmError::BadRequest {
                    message: "AWS Bedrock requires AWS credentials. \
                              Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and optionally \
                              AWS_SESSION_TOKEN) in the environment."
                        .into(),
                    status: 400,
                });
            }
        }
        Ok(())
    }

    fn stream_format(&self) -> StreamFormat {
        StreamFormat::AwsEventStream
    }

    fn build_url(&self, endpoint_path: &str, model: &str) -> String {
        let base = self.base_url();
        let effective_model = self.apply_cross_region_prefix(model);
        let encoded_model = percent_encode_model(&effective_model);
        if endpoint_path.contains("chat/completions") {
            format!("{base}/model/{encoded_model}/converse")
        } else if endpoint_path.contains("embeddings") {
            format!("{base}/model/{encoded_model}/invoke")
        } else {
            format!("{base}{endpoint_path}")
        }
    }

    fn build_stream_url(&self, endpoint_path: &str, model: &str) -> String {
        let base = self.base_url();
        let effective_model = self.apply_cross_region_prefix(model);
        let encoded_model = percent_encode_model(&effective_model);
        if endpoint_path.contains("chat/completions") {
            format!("{base}/model/{encoded_model}/converse-stream")
        } else {
            // Non-chat streaming falls back to the regular URL.
            self.build_url(endpoint_path, model)
        }
    }

    fn transform_request(&self, body: &mut serde_json::Value) -> HiLlmResult<()> {
        use serde_json::json;

        let messages = body
            .as_object_mut()
            .and_then(|o| o.remove("messages"))
            .and_then(|v| match v {
                serde_json::Value::Array(arr) => Some(arr),
                _ => None,
            })
            .unwrap_or_default();

        let mut system_parts = vec![];
        let mut converse_messages = vec![];

        for msg in &messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let content = msg.get("content");

            match role {
                "system" | "developer" => {
                    if let Some(text) = content.and_then(|c| c.as_str()) {
                        system_parts.push(json!({"text": text}));
                    } else if let Some(array) = content.and_then(|c| c.as_array()) {
                        for part in array {
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                system_parts.push(json!({"text": text}));
                            }
                        }
                    }
                }
                "user" => {
                    let parts = if let Some(text) = content.and_then(|c| c.as_str()) {
                        vec![json!({"text": text})]
                    } else if let Some(array) = content.and_then(|c| c.as_array()) {
                        array
                            .iter()
                            .filter_map(|part| {
                                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                match part_type {
                                    "text" => {
                                        let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                        Some(json!({"text": text}))
                                    }
                                    "image_url" => {
                                        let url = part.pointer("/image_url/url").and_then(|u| u.as_str()).unwrap_or("");
                                        if let Some(data_part) = url.strip_prefix("data:") {
                                            let mut iter = data_part.splitn(2, ';');
                                            let media_type = iter.next().unwrap_or("image/jpeg");
                                            let b64 = iter.next().and_then(|s| s.strip_prefix("base64,")).unwrap_or("");
                                            Some(json!({
                                                "image": {
                                                    "format": media_type.split('/').nth(1).unwrap_or("jpeg"),
                                                    "source": {"bytes": b64}
                                                }
                                            }))
                                        } else {
                                            Some(json!({"text": url}))
                                        }
                                    }
                                    "document" => {
                                        let data =
                                            part.pointer("/document/data").and_then(|d| d.as_str()).unwrap_or("");
                                        let media_type = part
                                            .pointer("/document/media_type")
                                            .and_then(|m| m.as_str())
                                            .unwrap_or("application/pdf");
                                        let format = format_from_media_type(media_type);
                                        Some(json!({
                                            "document": {
                                                "name": "doc",
                                                "format": format,
                                                "source": {"bytes": data}
                                            }
                                        }))
                                    }
                                    _ => None,
                                }
                            })
                            .collect()
                    } else {
                        vec![json!({"text": ""})]
                    };
                    converse_messages.push(json!({"role": "user", "content": parts}));
                }
                "assistant" => {
                    let mut parts = vec![];
                    if let Some(text) = content.and_then(|c| c.as_str())
                        && !text.is_empty()
                    {
                        parts.push(json!({"text": text}));
                    }
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tool_calls {
                            let input: serde_json::Value = tc
                                .pointer("/function/arguments")
                                .and_then(|a| a.as_str())
                                .and_then(|s| serde_json::from_str(s).ok())
                                .unwrap_or_else(|| json!({}));
                            parts.push(json!({
                                "toolUse": {
                                    "toolUseId": tc.get("id"),
                                    "name": tc.pointer("/function/name"),
                                    "input": input
                                }
                            }));
                        }
                    }
                    if parts.is_empty() {
                        parts.push(json!({"text": ""}));
                    }
                    converse_messages.push(json!({"role": "assistant", "content": parts}));
                }
                "tool" => {
                    let tool_call_id = msg
                        .get("tool_call_id")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let result_text = content.and_then(|c| c.as_str()).unwrap_or("");
                    let is_error = msg
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let status = if is_error { "error" } else { "success" };
                    converse_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": tool_call_id,
                                "content": [{"text": result_text}],
                                "status": status
                            }
                        }]
                    }));
                }
                _ => {}
            }
        }

        let mut inference_config = json!({});
        if let Some(max_tokens) = body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
        {
            inference_config["maxTokens"] = max_tokens.clone();
        }
        if let Some(temp) = body.get("temperature") {
            inference_config["temperature"] = temp.clone();
        }
        if let Some(top_p) = body.get("top_p") {
            inference_config["topP"] = top_p.clone();
        }
        if let Some(stop) = body.get("stop") {
            let sequences = if let Some(s) = stop.as_str() {
                vec![json!(s)]
            } else {
                stop.as_array().cloned().unwrap_or_default()
            };
            inference_config["stopSequences"] = json!(sequences);
        }

        let tool_config = body.get("tools").and_then(|tools| {
            tools.as_array().map(|arr| {
                let bedrock_tools: Vec<serde_json::Value> = arr
                    .iter()
                    .map(|t| {
                        let parameters = t
                            .pointer("/function/parameters")
                            .cloned()
                            .unwrap_or_else(|| json!({"type": "object"}));
                        json!({
                            "toolSpec": {
                                "name": t.pointer("/function/name"),
                                "description": t.pointer("/function/description"),
                                "inputSchema": {"json": parameters}
                            }
                        })
                    })
                    .collect();
                json!({"tools": bedrock_tools})
            })
        });

        let mut additional_model_fields: Option<serde_json::Value> = None;
        if let Some(effort) = body.get("reasoning_effort").and_then(|e| e.as_str()) {
            let budget_tokens = reasoning_effort_to_budget_tokens(effort);
            additional_model_fields = Some(json!({
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
            }));
        }

        if let Some(response_format) = body.get("response_format") {
            let rf_type = response_format
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("");
            match rf_type {
                "json_schema" => {
                    let schema = response_format
                        .get("json_schema")
                        .and_then(|js| js.get("schema"));
                    let schema_str = schema
                        .map(|s| serde_json::to_string_pretty(s).unwrap_or_default())
                        .unwrap_or_default();
                    let instruction = if schema_str.is_empty() {
                        "You MUST respond with valid JSON only. No other text.".to_owned()
                    } else {
                        format!(
                            "You MUST respond with valid JSON only that conforms to this schema:\n```json\n{schema_str}\n```\nNo other text outside the JSON."
                        )
                    };
                    system_parts.push(json!({"text": instruction}));
                }
                "json_object" => {
                    system_parts.push(
                        json!({"text": "You MUST respond with valid JSON only. No other text."}),
                    );
                }
                _ => {}
            }
        }

        let guardrail_config = body
            .get("extra_body")
            .and_then(|eb| eb.get("guardrailConfig"))
            .cloned();

        let mut new_body = json!({
            "messages": converse_messages,
        });
        if !system_parts.is_empty() {
            new_body["system"] = json!(system_parts);
        }
        if let Some(obj) = inference_config.as_object()
            && !obj.is_empty()
        {
            new_body["inferenceConfig"] = inference_config;
        }
        if let Some(tc) = tool_config {
            new_body["toolConfig"] = tc;
        }
        if let Some(amf) = additional_model_fields {
            new_body["additionalModelRequestFields"] = amf;
        }
        if let Some(gc) = guardrail_config {
            new_body["guardrailConfig"] = gc;
        }

        *body = new_body;
        Ok(())
    }

    fn transform_response(&self, body: &mut serde_json::Value) -> HiLlmResult<()> {
        use serde_json::json;

        let stop_reason = body
            .get("stopReason")
            .and_then(|s| s.as_str())
            .unwrap_or("end_turn");
        let usage = body.get("usage").cloned();

        let content_blocks = body
            .pointer("/output/message/content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        let text: String = content_blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("");

        let tool_calls: Vec<serde_json::Value> = content_blocks
            .iter()
            .filter_map(|b| {
                b.get("toolUse").map(|tu| {
                    let arguments = serde_json::to_string(tu.get("input").unwrap_or(&json!({})))
                        .unwrap_or_default();
                    json!({
                        "id": tu.get("toolUseId"),
                        "type": "function",
                        "function": {
                            "name": tu.get("name"),
                            "arguments": arguments
                        }
                    })
                })
            })
            .collect();

        let finish_reason = match stop_reason {
            "end_turn" => "stop",
            "tool_use" => "tool_calls",
            "max_tokens" => "length",
            "stop_sequence" => "stop",
            "content_filtered" | "guardrail_intervened" => "content_filter",
            _ => "stop",
        };

        let input_tokens = usage
            .as_ref()
            .and_then(|u| u.get("inputTokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .as_ref()
            .and_then(|u| u.get("outputTokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let response_id = body
            .get("requestId")
            .or_else(|| body.get("conversationId"))
            .cloned()
            .unwrap_or_else(|| json!("bedrock-resp"));

        let content_value: serde_json::Value = if text.is_empty() {
            json!(null)
        } else {
            json!(text)
        };

        let mut message = json!({"role": "assistant", "content": content_value});
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }

        *body = json!({
            "id": response_id,
            "object": "chat.completion",
            "created": super::unix_timestamp_secs(),
            "model": "",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        });

        Ok(())
    }

    fn signing_headers(&self, method: &str, url: &str, body: &[u8]) -> Vec<(String, String)> {
        #[cfg(feature = "bedrock")]
        {
            sigv4_sign(method, url, body, &self.region).unwrap_or_default()
        }

        #[cfg(not(feature = "bedrock"))]
        {
            let _ = (method, url, body);
            vec![]
        }
    }
}

pub(crate) fn parse_bedrock_stream_event(
    event_type: &str,
    payload: &str,
) -> HiLlmResult<Option<ChatCompletionChunk>> {
    use crate::error::HiLlmError;
    use serde_json::json;

    let v: serde_json::Value =
        serde_json::from_str(payload).map_err(|e| HiLlmError::Streaming {
            message: format!("Bedrock stream event parse error: {e}"),
        })?;

    let chunk_from_json = |chunk_json: serde_json::Value| -> HiLlmResult<ChatCompletionChunk> {
        serde_json::from_value(chunk_json).map_err(|e| HiLlmError::Streaming {
            message: format!("Bedrock chunk deserialization error: {e}"),
        })
    };

    match event_type {
        "messageStart" => {
            let role = v
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("assistant");
            chunk_from_json(json!({
                "id": "bedrock-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {"role": role},
                    "finish_reason": null
                }]
            }))
            .map(Some)
        }
        "contentBlockStart" => {
            let index = v
                .get("contentBlockIndex")
                .and_then(|i| i.as_u64())
                .unwrap_or(0);
            if let Some(tool_use) = v.pointer("/start/toolUse") {
                let tool_use_id = tool_use
                    .get("toolUseId")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                let name = tool_use.get("name").and_then(|n| n.as_str()).unwrap_or("");
                chunk_from_json(json!({
                    "id": "bedrock-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": tool_use_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""}
                            }]
                        },
                        "finish_reason": null
                    }]
                }))
                .map(Some)
            } else {
                Ok(None)
            }
        }
        "contentBlockDelta" => {
            let index = v
                .get("contentBlockIndex")
                .and_then(|i| i.as_u64())
                .unwrap_or(0);

            if let Some(text) = v.pointer("/delta/text").and_then(|t| t.as_str()) {
                return chunk_from_json(json!({
                    "id": "bedrock-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null
                    }]
                }))
                .map(Some);
            }

            if let Some(input_json) = v.pointer("/delta/toolUse/input").and_then(|i| i.as_str()) {
                return chunk_from_json(json!({
                    "id": "bedrock-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {"arguments": input_json}
                            }]
                        },
                        "finish_reason": null
                    }]
                }))
                .map(Some);
            }

            #[cfg(feature = "tracing")]
            tracing::debug!(
                content_block_index = index,
                "Bedrock contentBlockDelta with unrecognized delta shape; skipping"
            );

            Ok(None)
        }
        "contentBlockStop" => Ok(None),
        "messageStop" => {
            let stop_reason = v
                .get("stopReason")
                .and_then(|s| s.as_str())
                .unwrap_or("end_turn");
            let finish_reason = match stop_reason {
                "end_turn" => "stop",
                "tool_use" => "tool_calls",
                "max_tokens" => "length",
                "stop_sequence" => "stop",
                "content_filtered" | "guardrail_intervened" => "content_filter",
                _ => "stop",
            };
            chunk_from_json(json!({
                "id": "bedrock-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }]
            }))
            .map(Some)
        }
        "metadata" => {
            let input_tokens = v
                .pointer("/usage/inputTokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0);
            let output_tokens = v
                .pointer("/usage/outputTokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0);
            chunk_from_json(json!({
                "id": "bedrock-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }))
            .map(Some)
        }
        _ => Ok(None),
    }
}

impl BedrockProvider {
    fn apply_cross_region_prefix(&self, model: &str) -> String {
        match &self.cross_region_prefix {
            Some(prefix) => {
                if model.starts_with(prefix.as_str()) {
                    model.to_owned()
                } else {
                    format!("{prefix}{model}")
                }
            }
            None => model.to_owned(),
        }
    }
}

#[cfg(feature = "bedrock")]
fn sigv4_sign(
    method: &str,
    url: &str,
    body: &[u8],
    region: &str,
) -> HiLlmResult<Vec<(String, String)>> {
    use aws_credential_types::Credentials;
    use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
    use aws_sigv4::sign::v4::SigningParams;

    let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| HiLlmError::BadRequest {
        message: "AWS_ACCESS_KEY_ID environment variable is required for Bedrock requests".into(),
        status: 400,
    })?;
    let secret_key =
        std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| HiLlmError::BadRequest {
            message: "AWS_SECRET_ACCESS_KEY environment variable is required for Bedrock requests"
                .into(),
            status: 400,
        })?;
    let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

    let credentials = Credentials::new(access_key, secret_key, session_token, None, "env");

    let identity = credentials.into();

    let signing_settings = SigningSettings::default();
    let now = std::time::SystemTime::now();

    let params = SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name("bedrock")
        .time(now)
        .settings(signing_settings)
        .build()
        .map_err(|e| HiLlmError::BadRequest {
            message: format!("failed to build SigV4 signing params: {e}"),
            status: 400,
        })?;

    let signable = SignableRequest::new(
        method,
        url,
        std::iter::empty::<(&str, &str)>(),
        SignableBody::Bytes(body),
    )
    .map_err(|e| HiLlmError::BadRequest {
        message: format!("failed to create signable request: {e}"),
        status: 400,
    })?;

    let signing_output = sign(signable, &params.into()).map_err(|e| HiLlmError::BadRequest {
        message: format!("SigV4 signing failed: {e}"),
        status: 400,
    })?;

    let instructions = signing_output.output();
    let signed_headers: Vec<(String, String)> = instructions
        .headers()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect();

    Ok(signed_headers)
}
