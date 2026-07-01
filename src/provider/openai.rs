use super::common::{Provider, registry};
use crate::error::{HiLlmError, HiLlmResult};
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
        if let Ok(reg) = registry().await {
            reg.get("openai")
                .is_some_and(|p| p.models.contains_key(model))
        } else {
            false
        }
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
