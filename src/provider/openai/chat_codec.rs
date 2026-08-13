//! OpenAI Chat Completions API codec implementation.

use bytes::Bytes;
use serde_json;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::APIType;
use crate::provider::codec::APITypeCodec;
use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk};

/// Codec for OpenAI Chat Completions API.
pub struct OpenAIChatCodec;

impl APITypeCodec for OpenAIChatCodec {
    type Request = ChatCompletionRequest;
    type Response = ChatCompletionResponse;
    type StreamEvent = ChatCompletionChunk;

    fn api_type(&self) -> APIType {
        APIType::OpenAIChatCompletions
    }

    fn endpoint_path(&self) -> &str {
        "/chat/completions"
    }

    fn encode_request(&self, request: &Self::Request) -> HiLlmResult<Bytes> {
        serde_json::to_vec(request)
            .map(Bytes::from)
            .map_err(|e| HiLlmError::Serialization {
                message: format!("Failed to serialize ChatCompletionRequest: {}", e),
            })
    }

    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<Self::Response> {
        serde_json::from_slice(bytes).map_err(|e| HiLlmError::Serialization {
            message: format!("Failed to deserialize ChatCompletionResponse: {}", e),
        })
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<Self::StreamEvent>> {
        // OpenAI sends "[DONE]" to signal end of stream
        if data == "[DONE]" {
            return Ok(None);
        }

        serde_json::from_str(data)
            .map(Some)
            .map_err(|e| HiLlmError::Serialization {
                message: format!("Failed to parse ChatCompletionChunk: {}", e),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk};
    use crate::types::message::{Message, UserMessage, MessageContent};

    #[test]
    fn test_openai_chat_codec_api_type() {
        let codec = OpenAIChatCodec;
        assert_eq!(codec.api_type(), APIType::OpenAIChatCompletions);
    }

    #[test]
    fn test_openai_chat_codec_endpoint_path() {
        let codec = OpenAIChatCodec;
        assert_eq!(codec.endpoint_path(), "/chat/completions");
    }

    #[test]
    fn test_openai_chat_codec_encode_request() {
        let codec = OpenAIChatCodec;
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![Message::User(UserMessage {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            })],
            stream: Some(false),
            ..Default::default()
        };

        let result = codec.encode_request(&request);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "gpt-4");
    }

    #[test]
    fn test_openai_chat_codec_parse_stream_event_done() {
        let codec = OpenAIChatCodec;
        let result = codec.parse_stream_event("[DONE]").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_openai_chat_codec_parse_stream_event_chunk() {
        let codec = OpenAIChatCodec;
        let chunk_json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let result = codec.parse_stream_event(chunk_json).unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();
        assert_eq!(chunk.id, "chatcmpl-123");
        assert_eq!(chunk.choices.len(), 1);
    }
}
