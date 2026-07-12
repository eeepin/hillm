use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use super::{ContentPart, RealtimeEvent, RealtimeTranslator, ResponseStatus};
use crate::error::{HiLlmError, HiLlmResult};

#[derive(Debug, Clone, Default)]
pub struct OpenAiRealtimeTranslator;

impl OpenAiRealtimeTranslator {
    pub fn new() -> Self {
        Self
    }
}

// Helper

fn get_str<'a>(obj: &'a Value, key: &str) -> HiLlmResult<&'a str> {
    obj.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| HiLlmError::BadRequest {
            message: format!("Realtime event missing required field '{key}'"),
            status: 400,
        })
}

fn get_str_opt<'a>(obj: &'a Value, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(|v| v.as_str())
}

fn get_u32(obj: &Value, key: &str) -> Option<u32> {
    obj.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
}

fn parse_content_parts(raw: &Value) -> Vec<ContentPart> {
    let Some(arr) = raw.as_array() else {
        return vec![];
    };
    arr.iter()
        .filter_map(|item| {
            let kind = item.get("type").and_then(|v| v.as_str())?;
            match kind {
                "text" | "input_text" => {
                    let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    Some(ContentPart::text(text))
                }
                "audio" | "input_audio" => {
                    let base64 = item.get("audio").and_then(|v| v.as_str()).unwrap_or("");
                    Some(ContentPart::audio(base64))
                }
                "image_url" => {
                    let url = item
                        .get("image_url")
                        .and_then(|u| u.get("url"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    Some(ContentPart::image_ref(url))
                }
                _ => None,
            }
        })
        .collect()
}

fn parse_reset_at_ms(obj: &Value) -> i64 {
    if let Some(ts) = obj.get("reset_at").and_then(|v| v.as_f64()) {
        return (ts * 1_000.0) as i64;
    }
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    now_ms + 60_000
}

impl RealtimeTranslator for OpenAiRealtimeTranslator {
    fn provider(&self) -> &'static str {
        "openai"
    }

    fn translate_inbound(&self, raw: Value) -> HiLlmResult<RealtimeEvent> {
        let event_type = get_str(&raw, "type")?;
        let event = match event_type {
            "session.created" => {
                let session = raw.get("session").unwrap_or(&Value::Null);
                RealtimeEvent::SessionCreated {
                    session_id: get_str_opt(session, "id").unwrap_or("").into(),
                    model: get_str_opt(session, "model").unwrap_or("").into(),
                }
            }
            "session.updated" => {
                let session = raw.get("session").unwrap_or(&Value::Null);
                RealtimeEvent::SessionUpdated {
                    session_id: get_str_opt(session, "id").unwrap_or("").into(),
                    instructions: get_str_opt(session, "instructions").map(str::to_owned),
                }
            }
            "conversation.item.created" | "conversation.item.added" => {
                let item = raw.get("item").unwrap_or(&Value::Null);
                let content = item
                    .get("content")
                    .map(parse_content_parts)
                    .unwrap_or_default();
                RealtimeEvent::ConversationItemCreated {
                    item_id: get_str_opt(item, "id").unwrap_or("").into(),
                    role: get_str_opt(item, "role").unwrap_or("").into(),
                    content,
                }
            }
            "conversation.item.deleted" => RealtimeEvent::ConversationItemDeleted {
                item_id: raw
                    .get("item_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.created" => {
                let response = raw.get("response").unwrap_or(&Value::Null);
                RealtimeEvent::ResponseCreated {
                    response_id: get_str_opt(response, "id").unwrap_or("").into(),
                }
            }
            "response.done" => {
                let response = raw.get("response").unwrap_or(&Value::Null);
                let status_str = get_str_opt(response, "status").unwrap_or("completed");
                let status = match status_str {
                    "cancelled" => ResponseStatus::Cancelled,
                    "failed" => ResponseStatus::Failed,
                    "incomplete" => ResponseStatus::Incomplete,
                    _ => ResponseStatus::Completed,
                };
                RealtimeEvent::ResponseDone {
                    response_id: get_str_opt(response, "id").unwrap_or("").into(),
                    status,
                }
            }
            "response.text.delta" => RealtimeEvent::ResponseTextDelta {
                response_id: raw
                    .get("response_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                delta: raw
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.text.done" => RealtimeEvent::ResponseTextDone {
                response_id: raw
                    .get("response_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                text: raw
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.audio.delta" => RealtimeEvent::ResponseAudioDelta {
                response_id: raw
                    .get("response_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                delta_base64: raw
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.audio.done" => RealtimeEvent::ResponseAudioDone {
                response_id: raw
                    .get("response_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.audio_transcript.delta" => RealtimeEvent::ResponseAudioTranscriptDelta {
                response_id: raw
                    .get("response_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                delta: raw
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.audio_transcript.done" => RealtimeEvent::ResponseAudioTranscriptDone {
                response_id: raw
                    .get("response_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                transcript: raw
                    .get("transcript")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "response.function_call_arguments.delta" => {
                RealtimeEvent::ResponseFunctionCallArgumentsDelta {
                    response_id: raw
                        .get("response_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    call_id: raw
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    delta: raw
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                }
            }
            "response.function_call_arguments.done" => {
                RealtimeEvent::ResponseFunctionCallArgumentsDone {
                    response_id: raw
                        .get("response_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    call_id: raw
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    name: raw
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                    arguments: raw
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .into(),
                }
            }
            "input_audio_buffer.append" => RealtimeEvent::InputAudioBufferAppend {
                audio_base64: raw
                    .get("audio")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "input_audio_buffer.commit" => RealtimeEvent::InputAudioBufferCommit,
            "input_audio_buffer.clear" => RealtimeEvent::InputAudioBufferClear,
            "input_audio_buffer.speech_started" => RealtimeEvent::InputAudioBufferSpeechStarted {
                item_id: raw
                    .get("item_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
            },
            "input_audio_buffer.speech_stopped" => RealtimeEvent::InputAudioBufferSpeechStopped {
                item_id: raw
                    .get("item_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .into(),
                audio_end_ms: get_u32(&raw, "audio_end_ms").unwrap_or(0),
            },
            "rate_limits.updated" => {
                let limits = raw.get("rate_limits").and_then(|v| v.as_array());
                let mut remaining_requests = None;
                let mut remaining_tokens = None;
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);
                let mut reset_at_unix_ms = now_ms + 60_000;

                if let Some(limits) = limits {
                    for limit in limits {
                        let name = limit.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        match name {
                            "requests" => {
                                remaining_requests = get_u32(limit, "remaining");
                                reset_at_unix_ms = parse_reset_at_ms(limit);
                            }
                            "tokens" => {
                                remaining_tokens = get_u32(limit, "remaining");
                            }
                            _ => {}
                        }
                    }
                }
                RealtimeEvent::RateLimitsUpdated {
                    remaining_requests,
                    remaining_tokens,
                    reset_at_unix_ms,
                }
            }
            "error" => {
                let err = raw.get("error").unwrap_or(&raw);
                RealtimeEvent::Error {
                    code: get_str_opt(err, "code").unwrap_or("unknown").into(),
                    message: get_str_opt(err, "message").unwrap_or("").into(),
                    event_id: get_str_opt(&raw, "event_id").map(str::to_owned),
                }
            }
            other => RealtimeEvent::Raw {
                event_type: other.into(),
                payload: raw,
            },
        };

        Ok(event)
    }

    fn translate_outbound(&self, event: &RealtimeEvent) -> HiLlmResult<serde_json::Value> {
        use serde_json::json;

        let value = match event {
            RealtimeEvent::SessionCreated { session_id, model } => json!({
                "type": "session.created",
                "session": { "id": session_id, "model": model }
            }),
            RealtimeEvent::SessionUpdated {
                session_id,
                instructions,
            } => {
                let mut session = serde_json::Map::new();
                session.insert("id".into(), json!(session_id));
                if let Some(instr) = instructions {
                    session.insert("instructions".into(), json!(instr));
                }
                json!({ "type": "session.updated", "session": session })
            }
            RealtimeEvent::ConversationItemCreated {
                item_id,
                role,
                content,
            } => {
                let content_json: Vec<_> = content
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => json!({"type": "text", "text": text}),
                        ContentPart::Audio { base64 } => {
                            json!({"type": "audio", "audio": base64})
                        }
                        ContentPart::ImageRef { url } => {
                            json!({"type": "image_url", "image_url": {"url": url}})
                        }
                    })
                    .collect();
                json!({
                    "type": "conversation.item.created",
                    "item": { "id": item_id, "role": role, "content": content_json }
                })
            }
            RealtimeEvent::ConversationItemDeleted { item_id } => {
                json!({ "type": "conversation.item.deleted", "item_id": item_id })
            }
            RealtimeEvent::ResponseCreated { response_id } => {
                json!({ "type": "response.created", "response": { "id": response_id } })
            }
            RealtimeEvent::ResponseDone {
                response_id,
                status,
            } => {
                let status_str = match status {
                    ResponseStatus::Completed => "completed",
                    ResponseStatus::Cancelled => "cancelled",
                    ResponseStatus::Failed => "failed",
                    ResponseStatus::Incomplete => "incomplete",
                };
                json!({
                    "type": "response.done",
                    "response": { "id": response_id, "status": status_str }
                })
            }
            RealtimeEvent::ResponseTextDelta { response_id, delta } => {
                json!({ "type": "response.text.delta", "response_id": response_id, "delta": delta })
            }
            RealtimeEvent::ResponseTextDone { response_id, text } => {
                json!({ "type": "response.text.done", "response_id": response_id, "text": text })
            }
            RealtimeEvent::ResponseAudioDelta {
                response_id,
                delta_base64,
            } => {
                json!({
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": delta_base64
                })
            }
            RealtimeEvent::ResponseAudioDone { response_id } => {
                json!({ "type": "response.audio.done", "response_id": response_id })
            }
            RealtimeEvent::ResponseAudioTranscriptDelta { response_id, delta } => {
                json!({
                    "type": "response.audio_transcript.delta",
                    "response_id": response_id,
                    "delta": delta
                })
            }
            RealtimeEvent::ResponseAudioTranscriptDone {
                response_id,
                transcript,
            } => {
                json!({
                    "type": "response.audio_transcript.done",
                    "response_id": response_id,
                    "transcript": transcript
                })
            }
            RealtimeEvent::ResponseFunctionCallArgumentsDelta {
                response_id,
                call_id,
                delta,
            } => {
                json!({
                    "type": "response.function_call_arguments.delta",
                    "response_id": response_id,
                    "call_id": call_id,
                    "delta": delta
                })
            }
            RealtimeEvent::ResponseFunctionCallArgumentsDone {
                response_id,
                call_id,
                name,
                arguments,
            } => {
                json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments
                })
            }
            RealtimeEvent::InputAudioBufferAppend { audio_base64 } => {
                json!({ "type": "input_audio_buffer.append", "audio": audio_base64 })
            }
            RealtimeEvent::InputAudioBufferCommit => {
                json!({ "type": "input_audio_buffer.commit" })
            }
            RealtimeEvent::InputAudioBufferClear => {
                json!({ "type": "input_audio_buffer.clear" })
            }
            RealtimeEvent::InputAudioBufferSpeechStarted { item_id } => {
                json!({ "type": "input_audio_buffer.speech_started", "item_id": item_id })
            }
            RealtimeEvent::InputAudioBufferSpeechStopped {
                item_id,
                audio_end_ms,
            } => {
                json!({
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": item_id,
                    "audio_end_ms": audio_end_ms
                })
            }
            RealtimeEvent::RateLimitsUpdated {
                remaining_requests,
                remaining_tokens,
                reset_at_unix_ms,
            } => {
                let reset_ts = *reset_at_unix_ms as f64 / 1_000.0;
                let mut limits = vec![];
                if let Some(r) = remaining_requests {
                    limits.push(json!({"name": "requests", "remaining": r, "reset_at": reset_ts}));
                }
                if let Some(t) = remaining_tokens {
                    limits.push(json!({"name": "tokens", "remaining": t, "reset_at": reset_ts}));
                }
                json!({ "type": "rate_limits.updated", "rate_limits": limits })
            }
            RealtimeEvent::Error {
                code,
                message,
                event_id,
            } => {
                let mut obj = json!({
                    "type": "error",
                    "error": { "code": code, "message": message }
                });
                if let Some(eid) = event_id {
                    obj["event_id"] = json!(eid);
                }
                obj
            }
            RealtimeEvent::Raw {
                event_type,
                payload,
            } => {
                // Forward raw events as-is, but normalise the type field.
                let mut out = payload.clone();
                if let Some(obj) = out.as_object_mut() {
                    obj.insert("type".into(), json!(event_type));
                }
                out
            }
        };

        Ok(value)
    }
}
