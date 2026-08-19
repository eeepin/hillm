//! OpenAI Responses API codec implementation.

use bytes::Bytes;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::APIType;
use crate::provider::codec::APITypeCodec;

/// Codec for OpenAI Responses API.
#[allow(dead_code)]
pub struct OpenAIResponsesCodec;

impl APITypeCodec for OpenAIResponsesCodec {
    fn api_type(&self) -> APIType {
        APIType::OpenAIResponses
    }

    fn endpoint_path(&self) -> &str {
        "/responses"
    }

    fn encode_request(&self, request: &serde_json::Value) -> HiLlmResult<Bytes> {
        Ok(Bytes::from(serde_json::to_vec(request)?))
    }

    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<serde_json::Value> {
        Ok(serde_json::from_slice(bytes)?)
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<serde_json::Value>> {
        // The Responses API uses SSE with event types inside the JSON payload.
        // Unlike Chat Completions, it has no "[DONE]" sentinel: if a "[DONE]"
        // marker ever appears here it is a protocol violation, not end-of-stream.
        if data == "[DONE]" {
            return Err(HiLlmError::Streaming {
                message: "unexpected '[DONE]' sentinel in OpenAI Responses stream".into(),
            });
        }
        // Event types include: response.created, response.output_text.delta, etc.
        serde_json::from_str(data)
            .map(Some)
            .map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse Responses API event: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_responses_codec_api_type() {
        let codec = OpenAIResponsesCodec;
        assert_eq!(codec.api_type(), APIType::OpenAIResponses);
    }

    #[test]
    fn test_openai_responses_codec_endpoint_path() {
        let codec = OpenAIResponsesCodec;
        assert_eq!(codec.endpoint_path(), "/responses");
    }

    #[test]
    fn test_openai_responses_codec_encode_request() {
        let codec = OpenAIResponsesCodec;
        let request = serde_json::json!({
            "model": "gpt-4",
            "input": "Hello"
        });

        let result = codec.encode_request(&request);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "gpt-4");
    }

    #[test]
    fn test_openai_responses_codec_parse_stream_event() {
        let codec = OpenAIResponsesCodec;
        let event_json = r#"{
            "type": "response.output_text.delta",
            "delta": "Hello"
        }"#;

        let result = codec.parse_stream_event(event_json).unwrap();
        assert!(result.is_some());
        let event = result.unwrap();
        assert_eq!(event["type"], "response.output_text.delta");
        assert_eq!(event["delta"], "Hello");
    }

    #[test]
    fn test_openai_responses_codec_rejects_done_sentinel() {
        let codec = OpenAIResponsesCodec;
        let result = codec.parse_stream_event("[DONE]");
        assert!(
            result.is_err(),
            "[DONE] belongs to Chat Completions, not the Responses API"
        );
    }
}
