use serde::{Deserialize, Serialize};

use crate::error::HiLLMResult;

pub mod openai;
pub use openai::OpenAiRealtimeTranslator;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Audio { base64: String },
    ImageRef { url: String },
}

impl ContentPart {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text {
            text: content.into(),
        }
    }
    pub fn audio(base64: impl Into<String>) -> Self {
        Self::Audio {
            base64: base64.into(),
        }
    }
    pub fn image_ref(url: impl Into<String>) -> Self {
        Self::ImageRef { url: url.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Completed,
    Cancelled,
    Failed,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeEvent {
    SessionCreated {
        session_id: String,
        model: String,
    },
    SessionUpdated {
        session_id: String,
        instructions: Option<String>,
    },
    ConversationItemCreated {
        item_id: String,
        role: String,
        content: Vec<ContentPart>,
    },
    ConversationItemDeleted {
        item_id: String,
    },
    ResponseCreated {
        response_id: String,
    },
    ResponseDone {
        response_id: String,
        status: ResponseStatus,
    },
    ResponseTextDelta {
        response_id: String,
        delta: String,
    },
    ResponseTextDone {
        response_id: String,
        text: String,
    },
    ResponseAudioDelta {
        response_id: String,
        delta_base64: String,
    },
    ResponseAudioDone {
        response_id: String,
    },
    ResponseAudioTranscriptDelta {
        response_id: String,
        delta: String,
    },
    ResponseAudioTranscriptDone {
        response_id: String,
        transcript: String,
    },
    ResponseFunctionCallArgumentsDelta {
        response_id: String,
        call_id: String,
        delta: String,
    },
    ResponseFunctionCallArgumentsDone {
        response_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    InputAudioBufferAppend {
        audio_base64: String,
    },
    InputAudioBufferCommit,
    InputAudioBufferClear,
    InputAudioBufferSpeechStarted {
        item_id: String,
    },
    InputAudioBufferSpeechStopped {
        item_id: String,
        audio_end_ms: u32,
    },
    RateLimitsUpdated {
        remaining_requests: Option<u32>,
        remaining_tokens: Option<u32>,
        reset_at_unix_ms: i64,
    },
    Error {
        code: String,
        message: String,
        event_id: Option<String>,
    },
    Raw {
        event_type: String,
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeEnvelope {
    pub event_id: Option<String>,
    pub event: RealtimeEvent,
}

impl RealtimeEnvelope {
    pub fn new(event: RealtimeEvent) -> Self {
        Self {
            event_id: None,
            event,
        }
    }

    pub fn with_id(event_id: impl Into<String>, event: RealtimeEvent) -> Self {
        Self {
            event_id: Some(event_id.into()),
            event,
        }
    }
}

pub trait RealtimeTranslator: Send + Sync + 'static {
    fn translate_inbound(&self, raw: serde_json::Value) -> HiLLMResult<RealtimeEvent>;

    fn translate_outbound(&self, event: &RealtimeEvent) -> HiLLMResult<serde_json::Value>;

    fn provider(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_part_constructors() {
        assert!(matches!(
            ContentPart::text("hello"),
            ContentPart::Text { text } if text == "hello"
        ));
        assert!(matches!(
            ContentPart::audio("YWJj"),
            ContentPart::Audio { base64 } if base64 == "YWJj"
        ));
        assert!(matches!(
            ContentPart::image_ref("https://img"),
            ContentPart::ImageRef { url } if url == "https://img"
        ));
    }

    #[test]
    fn content_part_serde_round_trip_text() {
        let part = ContentPart::text("hi");
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hi");
        let back: ContentPart = serde_json::from_value(json).unwrap();
        assert_eq!(back, part);
    }

    #[test]
    fn content_part_serde_round_trip_audio() {
        let part = ContentPart::audio("YWJj");
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "audio");
        let back: ContentPart = serde_json::from_value(json).unwrap();
        assert_eq!(back, part);
    }

    #[test]
    fn content_part_serde_round_trip_image_ref() {
        let part = ContentPart::image_ref("https://img");
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "image_ref");
        let back: ContentPart = serde_json::from_value(json).unwrap();
        assert_eq!(back, part);
    }

    #[test]
    fn response_status_serde_variants() {
        for (status, expected_str) in [
            (ResponseStatus::Completed, "completed"),
            (ResponseStatus::Cancelled, "cancelled"),
            (ResponseStatus::Failed, "failed"),
            (ResponseStatus::Incomplete, "incomplete"),
        ] {
            let json = serde_json::to_value(status).unwrap();
            assert_eq!(json, expected_str);
            let back: ResponseStatus = serde_json::from_value(json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn envelope_new_has_no_id() {
        let env = RealtimeEnvelope::new(RealtimeEvent::InputAudioBufferCommit);
        assert!(env.event_id.is_none());
        assert!(matches!(env.event, RealtimeEvent::InputAudioBufferCommit));
    }

    #[test]
    fn envelope_with_id_has_id() {
        let env = RealtimeEnvelope::with_id("evt_1", RealtimeEvent::InputAudioBufferClear);
        assert_eq!(env.event_id.as_deref(), Some("evt_1"));
    }

    #[test]
    fn realtime_event_serde_round_trip_session_created() {
        let event = RealtimeEvent::SessionCreated {
            session_id: "s1".into(),
            model: "gpt-4o-realtime".into(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "session_created");
        let back: RealtimeEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn realtime_event_serde_round_trip_error() {
        let event = RealtimeEvent::Error {
            code: "rate_limit".into(),
            message: "too many".into(),
            event_id: Some("evt_2".into()),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "error");
        let back: RealtimeEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn realtime_event_serde_round_trip_raw() {
        let payload = serde_json::json!({"foo": "bar"});
        let event = RealtimeEvent::Raw {
            event_type: "custom.event".into(),
            payload: payload.clone(),
        };
        let json = serde_json::to_value(&event).unwrap();
        let back: RealtimeEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }
}
