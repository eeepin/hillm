use super::common::{Provider, registry};
use crate::error::HiLlmResult;
use std::borrow::Cow;

pub(crate) struct OpenAiProvider;

impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn base_url(&self) -> &str {
        "https://api.openai.com/v1"
    }

    fn env_var(&self) -> Option<&str> {
        Some("OPENAI_API_KEY")
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        Some((
            Cow::Borrowed("Authorization"),
            Cow::Owned(format!("Bearer {api_key}")),
        ))
    }

    async fn matches_model(&self, model: &str) -> bool {
        registry()
            .await
            .unwrap_or_default()
            .get("openai")
            .is_some_and(|p| p.models.contains_key(model))
    }

    fn transform_response(&self, body: &mut serde_json::Value) -> HiLlmResult<()> {
        transform_openai_audio_response(body)
    }
}

pub(crate) fn transform_openai_audio_response(body: &mut serde_json::Value) -> HiLlmResult<()> {
    use serde_json::json;

    let choices = match body.get_mut("choices").and_then(|c| c.as_array_mut()) {
        Some(c) => c,
        None => return Ok(()),
    };

    for choice in choices {
        let audio = match choice.pointer("/message/audio") {
            Some(a) if !a.is_null() => a.clone(),
            _ => continue,
        };

        let data = audio
            .get("data")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let format = audio
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("wav")
            .to_owned();
        let transcript = audio
            .get("transcript")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();

        let mut parts: Vec<serde_json::Value> = vec![];
        if !transcript.is_empty() {
            parts.push(json!({"type": "text", "text": transcript}));
        }
        if !data.is_empty() {
            parts.push(json!({
                "type": "output_audio",
                "audio": {"data": data, "format": format}
            }));
        }

        if !parts.is_empty()
            && let Some(message) = choice.get_mut("message")
        {
            message["content"] = json!(parts);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn transform_response_audio_message_hoisted() {
        let mut body = json!({
            "id": "chatcmpl-audio-123",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o-audio-preview",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "audio": {
                        "id": "audio-abc",
                        "data": "aGVsbG8=",
                        "transcript": "hello",
                        "format": "wav",
                        "expires_at": 9999999999u64
                    }
                },
                "finish_reason": "stop"
            }]
        });

        transform_openai_audio_response(&mut body).expect("transform must succeed");

        let content = body
            .pointer("/choices/0/message/content")
            .expect("content must be present");
        assert!(
            content.is_array(),
            "content must be a parts array, got: {content}"
        );
        let parts = content.as_array().expect("array");
        // transcript emitted as text part, audio data as output_audio part.
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "hello");
        assert_eq!(parts[1]["type"], "output_audio");
        assert_eq!(parts[1]["audio"]["data"], "aGVsbG8=");
        assert_eq!(parts[1]["audio"]["format"], "wav");
    }

    #[test]
    fn transform_response_audio_no_op_when_no_audio_field() {
        let mut body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "plain text"
                }
            }]
        });
        let original = body.clone();
        transform_openai_audio_response(&mut body).expect("transform must succeed");
        assert_eq!(
            body, original,
            "body must be unchanged when no audio field is present"
        );
    }
}
