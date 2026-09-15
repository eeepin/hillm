//! Anthropic Messages API endpoint codec implementation.

use bytes::Bytes;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::endpoint::{
    Endpoint, EndpointCodec, EndpointRequest, EndpointResponse, EndpointStreamEvent,
};
use crate::types::anthropic::{AnthropicMessagesResponse, AnthropicStreamEvent};

/// Codec for Anthropic Messages API (`/messages`).
///
/// Supports both streaming and non-streaming responses.
/// Uses native Anthropic Messages API types.
///
/// # Note on Transformation
///
/// This codec works with native Anthropic types. Transformation between
/// OpenAI Chat format and Anthropic Messages format is handled separately
/// and will be implemented in Phase 3.
#[allow(dead_code)] // Will be used in Phase 3
pub struct AnthropicMessagesCodec;

impl EndpointCodec for AnthropicMessagesCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::AnthropicMessages
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::AnthropicMessages(req) => {
                Ok(Bytes::from(serde_json::to_vec(req)?))
            }
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "AnthropicMessagesCodec received invalid request type: expected AnthropicMessages, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: AnthropicMessagesResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::AnthropicMessages(response))
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<EndpointStreamEvent>> {
        let event: AnthropicStreamEvent = serde_json::from_str(data).map_err(|e| {
            HiLlmError::Streaming {
                message: format!("Failed to parse AnthropicStreamEvent: {e}"),
            }
        })?;

        Ok(Some(EndpointStreamEvent::AnthropicMessages(event)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::anthropic::AnthropicMessagesRequest;

    #[test]
    fn test_anthropic_messages_codec_endpoint() {
        let codec = AnthropicMessagesCodec;
        assert_eq!(codec.endpoint(), Endpoint::AnthropicMessages);
    }

    #[test]
    fn test_anthropic_messages_codec_encode() {
        let codec = AnthropicMessagesCodec;
        let request = EndpointRequest::AnthropicMessages(AnthropicMessagesRequest {
            model: "claude-3-opus-20240229".to_string(),
            messages: vec![],
            max_tokens: 1024,
            ..Default::default()
        });

        let result = codec.encode(&request);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "claude-3-opus-20240229");
        assert_eq!(json["max_tokens"], 1024);
    }

    #[test]
    fn test_anthropic_messages_codec_decode() {
        let codec = AnthropicMessagesCodec;
        let json = r#"{
            "id": "msg_123",
            "model": "claude-3-opus-20240229",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let result = codec.decode(json.as_bytes());
        assert!(result.is_ok());
        if let EndpointResponse::AnthropicMessages(response) = result.unwrap() {
            assert_eq!(response.id, "msg_123");
            assert_eq!(response.model, "claude-3-opus-20240229");
        } else {
            panic!("Expected AnthropicMessages response");
        }
    }

    #[test]
    fn test_anthropic_messages_codec_parse_stream_event() {
        let codec = AnthropicMessagesCodec;
        let event_json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "model": "claude-3-opus-20240229",
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }"#;

        let result = codec.parse_stream_event(event_json);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_codec_invalid_request_type() {
        let codec = AnthropicMessagesCodec;
        let request = EndpointRequest::ChatCompletion(
            crate::types::chat::ChatCompletionRequest::default(),
        );

        let result = codec.encode(&request);
        assert!(result.is_err());
        if let Err(HiLlmError::InternalError { message }) = result {
            assert!(message.contains("invalid request type"));
        } else {
            panic!("Expected InternalError");
        }
    }
}
