use serde::{Deserialize, Serialize};

use crate::error::HiLlmResult;

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
    fn translate_inbound(&self, raw: serde_json::Value) -> HiLlmResult<RealtimeEvent>;

    fn translate_outbound(&self, event: &RealtimeEvent) -> HiLlmResult<serde_json::Value>;

    fn provider(&self) -> &'static str;
}
