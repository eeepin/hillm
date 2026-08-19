//! OpenAI Chat Completions API codec implementation.

use bytes::Bytes;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::ApiType;
use crate::provider::codec::ApiTypeCodec;

/// Codec for OpenAI Chat Completions API.
#[allow(dead_code)]
pub struct OpenAIChatCompletionsCodec;

impl ApiTypeCodec for OpenAIChatCompletionsCodec {
    fn api_type(&self) -> ApiType {
        ApiType::OpenAIChatCompletions
    }

    fn endpoint_path(&self) -> &str {
        "/chat/completions"
    }

    fn encode_request(&self, request: &serde_json::Value) -> HiLlmResult<Bytes> {
        Ok(Bytes::from(serde_json::to_vec(request)?))
    }

    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<serde_json::Value> {
        Ok(serde_json::from_slice(bytes)?)
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<serde_json::Value>> {
        // OpenAI sends "[DONE]" to signal end of stream
        if data == "[DONE]" {
            return Ok(None);
        }

        serde_json::from_str(data)
            .map(Some)
            .map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse ChatCompletionChunk: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_chat_completions_codec_api_type() {
        let codec = OpenAIChatCompletionsCodec;
        assert_eq!(codec.api_type(), ApiType::OpenAIChatCompletions);
    }

    #[test]
    fn test_openai_chat_completions_codec_endpoint_path() {
        let codec = OpenAIChatCompletionsCodec;
        assert_eq!(codec.endpoint_path(), "/chat/completions");
    }

    #[test]
    fn test_openai_chat_completions_codec_encode_request() {
        let codec = OpenAIChatCompletionsCodec;
        let request = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = codec.encode_request(&request);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "gpt-4");
    }

    #[test]
    fn test_openai_chat_completions_codec_parse_stream_event_done() {
        let codec = OpenAIChatCompletionsCodec;
        let result = codec.parse_stream_event("[DONE]").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_openai_chat_completions_codec_parse_stream_event_chunk() {
        let codec = OpenAIChatCompletionsCodec;
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
        assert_eq!(chunk["id"], "chatcmpl-123");
        assert_eq!(chunk["choices"].as_array().unwrap().len(), 1);
    }
}
