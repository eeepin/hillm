//! AWS Bedrock Converse API endpoint codec implementation.

use bytes::Bytes;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::endpoint::{
    Endpoint, EndpointCodec, EndpointRequest, EndpointResponse, EndpointStreamEvent,
};
use crate::types::bedrock::{BedrockConverseResponse, BedrockStreamEvent};

/// Codec for AWS Bedrock Converse API (`/converse`).
///
/// Supports both streaming and non-streaming responses.
/// Uses native Bedrock Converse API types.
///
/// # Note on Transformation
///
/// This codec works with native Bedrock types. Transformation between
/// OpenAI Chat format and Bedrock Converse format is handled separately
/// and will be implemented in Phase 3.
///
/// # Note on Streaming
///
/// Bedrock uses AWS EventStream format (binary) for streaming, not SSE.
/// The `parse_stream_event` method handles JSON representation of events,
/// but actual EventStream parsing is done at the transport layer.
#[allow(dead_code)] // Will be used in Phase 3
pub struct BedrockConverseCodec;

impl EndpointCodec for BedrockConverseCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::BedrockConverse
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::BedrockConverse(req) => {
                Ok(Bytes::from(serde_json::to_vec(req)?))
            }
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "BedrockConverseCodec received invalid request type: expected BedrockConverse, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: BedrockConverseResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::BedrockConverse(response))
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<EndpointStreamEvent>> {
        let event: BedrockStreamEvent = serde_json::from_str(data).map_err(|e| {
            HiLlmError::Streaming {
                message: format!("Failed to parse BedrockStreamEvent: {e}"),
            }
        })?;

        Ok(Some(EndpointStreamEvent::BedrockConverse(event)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::bedrock::BedrockConverseRequest;

    #[test]
    fn test_bedrock_converse_codec_endpoint() {
        let codec = BedrockConverseCodec;
        assert_eq!(codec.endpoint(), Endpoint::BedrockConverse);
    }

    #[test]
    fn test_bedrock_converse_codec_encode() {
        let codec = BedrockConverseCodec;
        let request = EndpointRequest::BedrockConverse(BedrockConverseRequest {
            messages: vec![],
            ..Default::default()
        });

        let result = codec.encode(&request);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json.get("messages").is_some());
    }

    #[test]
    fn test_bedrock_converse_codec_decode() {
        let codec = BedrockConverseCodec;
        let json = r#"{
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": {"text": "Hello"}}]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 15
            }
        }"#;

        let result = codec.decode(json.as_bytes());
        if let Err(e) = &result {
            eprintln!("Decode error: {:?}", e);
        }
        assert!(result.is_ok());
        if let EndpointResponse::BedrockConverse(response) = result.unwrap() {
            assert_eq!(response.stop_reason, crate::types::bedrock::BedrockStopReason::EndTurn);
            assert!(response.usage.is_some());
        } else {
            panic!("Expected BedrockConverse response");
        }
    }

    #[test]
    fn test_bedrock_converse_codec_parse_stream_event() {
        let codec = BedrockConverseCodec;
        let event_json = r#"{
            "event_type": "messageStart",
            "role": "assistant"
        }"#;

        let result = codec.parse_stream_event(event_json);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_codec_invalid_request_type() {
        let codec = BedrockConverseCodec;
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
