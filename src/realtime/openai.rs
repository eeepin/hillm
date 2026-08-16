use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use super::{ContentPart, RealtimeEvent, RealtimeTranslator, ResponseStatus};
use crate::error::{HiLLMError, HiLLMResult};

#[derive(Debug, Clone, Default)]
pub struct OpenAiRealtimeTranslator;

impl OpenAiRealtimeTranslator {
    pub fn new() -> Self {
        Self
    }
}

// Helper

fn get_str<'a>(obj: &'a Value, key: &str) -> HiLLMResult<&'a str> {
    obj.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| HiLLMError::BadRequest {
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

    fn translate_inbound(&self, raw: Value) -> HiLLMResult<RealtimeEvent> {
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

    fn translate_outbound(&self, event: &RealtimeEvent) -> HiLLMResult<serde_json::Value> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::{ContentPart, RealtimeEnvelope, ResponseStatus};
    use serde_json::json;

    fn translator() -> OpenAiRealtimeTranslator {
        OpenAiRealtimeTranslator::new()
    }

    // -----------------------------------------------------------------------
    // Provider identity
    // -----------------------------------------------------------------------

    #[test]
    fn provider_is_openai() {
        assert_eq!(translator().provider(), "openai");
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_missing_type_field_errors() {
        let raw = json!({"session": {"id": "s1"}});
        let result = translator().translate_inbound(raw);
        assert!(result.is_err(), "missing 'type' should error");
    }

    #[test]
    fn inbound_unknown_type_becomes_raw() {
        let raw = json!({"type": "some.unknown.event", "data": 42});
        let event = translator().translate_inbound(raw.clone()).unwrap();
        match event {
            RealtimeEvent::Raw {
                event_type,
                payload,
            } => {
                assert_eq!(event_type, "some.unknown.event");
                assert_eq!(payload["data"], 42);
            }
            other => panic!("expected Raw, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Session events
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_session_created() {
        let raw = json!({
            "type": "session.created",
            "session": {"id": "sess_1", "model": "gpt-4o-realtime"}
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::SessionCreated { session_id, model } => {
                assert_eq!(session_id, "sess_1");
                assert_eq!(model, "gpt-4o-realtime");
            }
            other => panic!("expected SessionCreated, got {other:?}"),
        }
    }

    #[test]
    fn inbound_session_created_missing_session() {
        let raw = json!({"type": "session.created"});
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::SessionCreated { session_id, model } => {
                assert_eq!(session_id, "");
                assert_eq!(model, "");
            }
            other => panic!("expected SessionCreated with empty fields, got {other:?}"),
        }
    }

    #[test]
    fn inbound_session_updated() {
        let raw = json!({
            "type": "session.updated",
            "session": {"id": "sess_1", "instructions": "be helpful"}
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::SessionUpdated {
                session_id,
                instructions,
            } => {
                assert_eq!(session_id, "sess_1");
                assert_eq!(instructions.as_deref(), Some("be helpful"));
            }
            other => panic!("expected SessionUpdated, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Conversation items
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_conversation_item_created_with_text() {
        let raw = json!({
            "type": "conversation.item.created",
            "item": {
                "id": "item_1",
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::ConversationItemCreated {
                item_id,
                role,
                content,
            } => {
                assert_eq!(item_id, "item_1");
                assert_eq!(role, "user");
                assert_eq!(content.len(), 1);
                assert!(matches!(&content[0], ContentPart::Text { text } if text == "hello"));
            }
            other => panic!("expected ConversationItemCreated, got {other:?}"),
        }
    }

    #[test]
    fn inbound_conversation_item_added_alias() {
        // "conversation.item.added" is an alias for "conversation.item.created"
        let raw = json!({
            "type": "conversation.item.added",
            "item": {"id": "i2", "role": "assistant", "content": []}
        });
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::ConversationItemCreated { item_id, .. } if item_id == "i2"
        ));
    }

    #[test]
    fn inbound_conversation_item_deleted() {
        let raw = json!({"type": "conversation.item.deleted", "item_id": "item_42"});
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::ConversationItemDeleted { item_id } if item_id == "item_42"
        ));
    }

    // -----------------------------------------------------------------------
    // Response lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_response_created() {
        let raw = json!({"type": "response.created", "response": {"id": "r1"}});
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::ResponseCreated { response_id } if response_id == "r1"
        ));
    }

    #[test]
    fn inbound_response_done_all_statuses() {
        for (status_str, expected) in [
            ("completed", ResponseStatus::Completed),
            ("cancelled", ResponseStatus::Cancelled),
            ("failed", ResponseStatus::Failed),
            ("incomplete", ResponseStatus::Incomplete),
            ("anything_else", ResponseStatus::Completed), // unknown defaults to Completed
        ] {
            let raw = json!({
                "type": "response.done",
                "response": {"id": "r1", "status": status_str}
            });
            let event = translator().translate_inbound(raw).unwrap();
            match event {
                RealtimeEvent::ResponseDone {
                    response_id,
                    status,
                } => {
                    assert_eq!(response_id, "r1");
                    assert_eq!(status, expected);
                }
                other => panic!("expected ResponseDone for {status_str}, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Delta / done pairs
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_response_text_delta() {
        let raw = json!({
            "type": "response.text.delta",
            "response_id": "r1",
            "delta": "hi"
        });
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::ResponseTextDelta { response_id, delta }
                if response_id == "r1" && delta == "hi"
        ));
    }

    #[test]
    fn inbound_response_audio_delta() {
        let raw = json!({
            "type": "response.audio.delta",
            "response_id": "r1",
            "delta": "YWJj"
        });
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::ResponseAudioDelta { response_id, delta_base64 }
                if response_id == "r1" && delta_base64 == "YWJj"
        ));
    }

    #[test]
    fn inbound_response_audio_transcript_done() {
        let raw = json!({
            "type": "response.audio_transcript.done",
            "response_id": "r1",
            "transcript": "hello world"
        });
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::ResponseAudioTranscriptDone { transcript, .. }
                if transcript == "hello world"
        ));
    }

    #[test]
    fn inbound_response_function_call_arguments_done() {
        let raw = json!({
            "type": "response.function_call_arguments.done",
            "response_id": "r1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"SF\"}"
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::ResponseFunctionCallArgumentsDone {
                response_id,
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(response_id, "r1");
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, "{\"city\":\"SF\"}");
            }
            other => panic!("expected ResponseFunctionCallArgumentsDone, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Input audio buffer events
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_audio_buffer_append() {
        let raw = json!({"type": "input_audio_buffer.append", "audio": "base64data"});
        let event = translator().translate_inbound(raw).unwrap();
        assert!(matches!(
            event,
            RealtimeEvent::InputAudioBufferAppend { audio_base64 } if audio_base64 == "base64data"
        ));
    }

    #[test]
    fn inbound_audio_buffer_commit_and_clear() {
        let commit = translator()
            .translate_inbound(json!({"type": "input_audio_buffer.commit"}))
            .unwrap();
        assert!(matches!(commit, RealtimeEvent::InputAudioBufferCommit));

        let clear = translator()
            .translate_inbound(json!({"type": "input_audio_buffer.clear"}))
            .unwrap();
        assert!(matches!(clear, RealtimeEvent::InputAudioBufferClear));
    }

    #[test]
    fn inbound_speech_started_and_stopped() {
        let started = translator()
            .translate_inbound(json!({
                "type": "input_audio_buffer.speech_started",
                "item_id": "i1"
            }))
            .unwrap();
        assert!(matches!(
            started,
            RealtimeEvent::InputAudioBufferSpeechStarted { item_id } if item_id == "i1"
        ));

        let stopped = translator()
            .translate_inbound(json!({
                "type": "input_audio_buffer.speech_stopped",
                "item_id": "i1",
                "audio_end_ms": 500
            }))
            .unwrap();
        match stopped {
            RealtimeEvent::InputAudioBufferSpeechStopped {
                item_id,
                audio_end_ms,
            } => {
                assert_eq!(item_id, "i1");
                assert_eq!(audio_end_ms, 500);
            }
            other => panic!("expected SpeechStopped, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Rate limits
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_rate_limits_updated() {
        let raw = json!({
            "type": "rate_limits.updated",
            "rate_limits": [
                {"name": "requests", "remaining": 100, "reset_at": 1700000.0},
                {"name": "tokens", "remaining": 50000}
            ]
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::RateLimitsUpdated {
                remaining_requests,
                remaining_tokens,
                reset_at_unix_ms,
            } => {
                assert_eq!(remaining_requests, Some(100));
                assert_eq!(remaining_tokens, Some(50000));
                assert_eq!(reset_at_unix_ms, 1_700_000_000); // 1700000.0 * 1000
            }
            other => panic!("expected RateLimitsUpdated, got {other:?}"),
        }
    }

    #[test]
    fn inbound_rate_limits_unknown_name_ignored() {
        let raw = json!({
            "type": "rate_limits.updated",
            "rate_limits": [{"name": "custom", "remaining": 99}]
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::RateLimitsUpdated {
                remaining_requests,
                remaining_tokens,
                ..
            } => {
                assert!(remaining_requests.is_none());
                assert!(remaining_tokens.is_none());
            }
            other => panic!("expected RateLimitsUpdated, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Error events
    // -----------------------------------------------------------------------

    #[test]
    fn inbound_error_with_sub_object() {
        let raw = json!({
            "type": "error",
            "error": {"code": "rate_limit", "message": "slow down"},
            "event_id": "evt_1"
        });
        let event = translator().translate_inbound(raw).unwrap();
        match event {
            RealtimeEvent::Error {
                code,
                message,
                event_id,
            } => {
                assert_eq!(code, "rate_limit");
                assert_eq!(message, "slow down");
                assert_eq!(event_id.as_deref(), Some("evt_1"));
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Outbound translation
    // -----------------------------------------------------------------------

    #[test]
    fn outbound_session_created() {
        let event = RealtimeEvent::SessionCreated {
            session_id: "s1".into(),
            model: "gpt-4o".into(),
        };
        let value = translator().translate_outbound(&event).unwrap();
        assert_eq!(value["type"], "session.created");
        assert_eq!(value["session"]["id"], "s1");
        assert_eq!(value["session"]["model"], "gpt-4o");
    }

    #[test]
    fn outbound_conversation_item_with_content() {
        let event = RealtimeEvent::ConversationItemCreated {
            item_id: "i1".into(),
            role: "user".into(),
            content: vec![
                ContentPart::text("hello"),
                ContentPart::audio("YWJj"),
                ContentPart::image_ref("https://img"),
            ],
        };
        let value = translator().translate_outbound(&event).unwrap();
        assert_eq!(value["type"], "conversation.item.created");
        assert_eq!(value["item"]["id"], "i1");
        let content = value["item"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "audio");
        assert_eq!(content[2]["type"], "image_url");
        assert_eq!(content[2]["image_url"]["url"], "https://img");
    }

    #[test]
    fn outbound_error_with_event_id() {
        let event = RealtimeEvent::Error {
            code: "bad".into(),
            message: "oops".into(),
            event_id: Some("evt_5".into()),
        };
        let value = translator().translate_outbound(&event).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["code"], "bad");
        assert_eq!(value["event_id"], "evt_5");
    }

    #[test]
    fn outbound_raw_event_preserves_payload() {
        let payload = json!({"type": "old_type", "data": [1, 2, 3]});
        let event = RealtimeEvent::Raw {
            event_type: "new_type".into(),
            payload,
        };
        let value = translator().translate_outbound(&event).unwrap();
        assert_eq!(value["type"], "new_type", "type should be normalized");
        assert_eq!(value["data"], json!([1, 2, 3]), "other fields preserved");
    }

    #[test]
    fn outbound_rate_limits_updated() {
        let event = RealtimeEvent::RateLimitsUpdated {
            remaining_requests: Some(50),
            remaining_tokens: Some(1000),
            reset_at_unix_ms: 1_700_000_000,
        };
        let value = translator().translate_outbound(&event).unwrap();
        assert_eq!(value["type"], "rate_limits.updated");
        let limits = value["rate_limits"].as_array().unwrap();
        assert_eq!(limits.len(), 2);
        assert_eq!(limits[0]["name"], "requests");
        assert_eq!(limits[0]["remaining"], 50);
    }

    // -----------------------------------------------------------------------
    // Round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_simple_events() {
        let t = translator();
        let events = [
            RealtimeEvent::InputAudioBufferCommit,
            RealtimeEvent::InputAudioBufferClear,
            RealtimeEvent::ResponseCreated {
                response_id: "r1".into(),
            },
            RealtimeEvent::ConversationItemDeleted {
                item_id: "i1".into(),
            },
        ];
        for event in events {
            let outbound = t.translate_outbound(&event).unwrap();
            let inbound = t.translate_inbound(outbound).unwrap();
            assert_eq!(inbound, event, "round trip failed for {event:?}");
        }
    }

    #[test]
    fn round_trip_response_done() {
        let t = translator();
        let event = RealtimeEvent::ResponseDone {
            response_id: "r1".into(),
            status: ResponseStatus::Completed,
        };
        let outbound = t.translate_outbound(&event).unwrap();
        let inbound = t.translate_inbound(outbound).unwrap();
        assert_eq!(inbound, event);
    }

    #[test]
    fn envelope_round_trip() {
        let t = translator();
        let raw = json!({
            "type": "response.created",
            "response": {"id": "r1"}
        });
        let event = t.translate_inbound(raw).unwrap();
        let env = RealtimeEnvelope::with_id("evt_1", event.clone());
        let env_json = serde_json::to_value(&env).unwrap();
        assert_eq!(env_json["event_id"], "evt_1");
        let back: RealtimeEnvelope = serde_json::from_value(env_json).unwrap();
        assert_eq!(back.event_id, env.event_id);
        assert_eq!(back.event, env.event);
    }

    // -----------------------------------------------------------------------
    // Edge cases in content parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parse_content_parts_non_array_returns_empty() {
        let raw = json!("not an array");
        let parts = parse_content_parts(&raw);
        assert!(parts.is_empty());
    }

    #[test]
    fn parse_content_parts_unknown_type_filtered() {
        let raw = json!([{"type": "unknown_kind", "data": 123}]);
        let parts = parse_content_parts(&raw);
        assert!(parts.is_empty());
    }

    #[test]
    fn parse_content_parts_input_text_and_input_audio_aliases() {
        let raw = json!([
            {"type": "input_text", "text": "hi"},
            {"type": "input_audio", "audio": "base64"}
        ]);
        let parts = parse_content_parts(&raw);
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "hi"));
        assert!(matches!(&parts[1], ContentPart::Audio { base64 } if base64 == "base64"));
    }
}
