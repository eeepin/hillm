//! Anthropic Messages API codec implementation.

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::ApiType;
use crate::provider::codec::ApiTypeCodec;
use bytes::Bytes;

/// Codec for Anthropic Messages API.
#[allow(dead_code)]
pub struct AnthropicMessagesCodec;

impl ApiTypeCodec for AnthropicMessagesCodec {
    fn api_type(&self) -> ApiType {
        ApiType::AnthropicMessages
    }

    fn endpoint_path(&self) -> &str {
        "/messages"
    }

    fn encode_request(&self, request: &serde_json::Value) -> HiLlmResult<Bytes> {
        Ok(Bytes::from(serde_json::to_vec(request)?))
    }

    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<serde_json::Value> {
        Ok(serde_json::from_slice(bytes)?)
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<serde_json::Value>> {
        // Anthropic doesn't use [DONE] sentinel
        serde_json::from_str(data)
            .map(Some)
            .map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse AnthropicStreamEvent: {}", e),
            })
    }
}
