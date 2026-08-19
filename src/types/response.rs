use serde::{Deserialize, Serialize};

use crate::types::{ApiRequest, ApiResponse};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateResponseRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Whether to stream the response via SSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    #[serde(flatten)]
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponseOutputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutputItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(flatten)]
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

/// Native streaming events of the OpenAI Responses API.
///
/// Variants follow the wire protocol's `type` field (e.g.
/// `response.output_text.delta`). Payloads that differ across event types are
/// kept as [`serde_json::Value`] so no native field is lost. Unknown event
/// types deserialize as [`ResponsesStreamEvent::Unknown`] and their payload is
/// dropped; match on the concrete variants you depend on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated { response: serde_json::Value },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress { response: serde_json::Value },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: u64,
        item: serde_json::Value,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: u64,
        item: serde_json::Value,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: serde_json::Value,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: serde_json::Value,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: u64,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: u64,
        arguments: String,
    },
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: serde_json::Value },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response: serde_json::Value },
    #[serde(rename = "response.failed")]
    ResponseFailed { response: serde_json::Value },
    #[serde(rename = "error")]
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        param: Option<String>,
    },
    /// An event whose `type` is not recognized by this version.
    #[serde(other)]
    Unknown,
}

impl ApiRequest for CreateResponseRequest {
    type Response = ResponseObject;
    type StreamEvent = ResponsesStreamEvent;

    fn model(&self) -> &str {
        &self.model
    }

    fn stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

impl ApiResponse for ResponseObject {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_stream_event_deserializes_text_delta() {
        let json = r#"{
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello"
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            ResponsesStreamEvent::OutputTextDelta { item_id, delta, .. } => {
                assert_eq!(item_id, "msg_1");
                assert_eq!(delta, "Hello");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn responses_stream_event_deserializes_completed_with_native_response() {
        let json = r#"{
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": []
            }
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            ResponsesStreamEvent::ResponseCompleted { response } => {
                assert_eq!(response["id"], "resp_1");
                assert_eq!(response["status"], "completed");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn responses_stream_event_unknown_type_is_preserved_as_unknown() {
        let json = r#"{"type": "response.brand_new.event", "foo": 1}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event, ResponsesStreamEvent::Unknown);
    }

    #[test]
    fn create_response_request_stream_flag_defaults_absent() {
        let req = CreateResponseRequest {
            model: "gpt-4o".into(),
            input: serde_json::json!("hi"),
            ..Default::default()
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("stream").is_none());
        assert!(!req.stream());
    }

    #[test]
    fn create_response_request_serializes_stream_true() {
        let req = CreateResponseRequest {
            model: "gpt-4o".into(),
            input: serde_json::json!("hi"),
            stream: Some(true),
            ..Default::default()
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
        assert!(req.stream());
    }
}
