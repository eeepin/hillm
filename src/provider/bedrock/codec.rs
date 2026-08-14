//! AWS Bedrock Converse API codec implementation.

use bytes::Bytes;
use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::APIType;
use crate::provider::codec::APITypeCodec;

/// Codec for AWS Bedrock Converse API.
#[allow(dead_code)]
pub struct BedrockConverseCodec;

impl APITypeCodec for BedrockConverseCodec {
    fn api_type(&self) -> APIType {
        APIType::BedrockConverse
    }

    fn endpoint_path(&self) -> &str {
        "/converse"
    }

    fn encode_request(&self, request: &serde_json::Value) -> HiLlmResult<Bytes> {
        Ok(Bytes::from(serde_json::to_vec(request)?))
    }

    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<serde_json::Value> {
        Ok(serde_json::from_slice(bytes)?)
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<serde_json::Value>> {
        // Bedrock uses AWS EventStream format, not SSE
        // This is a placeholder - actual implementation would handle EventStream
        serde_json::from_str(data)
            .map(Some)
            .map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse BedrockStreamEvent: {}", e),
            })
    }
}
