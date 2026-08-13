//! API type codec trait for protocol-specific encoding and decoding.

use bytes::Bytes;

use crate::error::HiLlmResult;
use crate::provider::APIType;
use crate::types::{APIRequest, APIResponse};

/// Codec trait for protocol-specific request/response encoding and decoding.
///
/// Each APIType has its own codec implementation that handles:
/// - Request serialization
/// - Response deserialization
/// - Stream event parsing
/// - Endpoint path construction
/// - URL building (including stream URLs)
/// - Signing headers (for providers like AWS Bedrock)
pub trait APITypeCodec: Send + Sync {
    /// The request type this codec handles.
    type Request: APIRequest;

    /// The response type this codec produces.
    type Response: APIResponse;

    /// The stream event type for streaming responses.
    type StreamEvent;

    /// Returns the API type this codec implements.
    fn api_type(&self) -> APIType;

    /// Returns the endpoint path for this API type (e.g., "/chat/completions").
    fn endpoint_path(&self) -> &str;

    /// Builds the full URL for a request.
    ///
    /// Default implementation concatenates base_url and endpoint_path.
    fn build_url(&self, base_url: &str, _model: &str) -> String {
        format!("{}{}", base_url, self.endpoint_path())
    }

    /// Builds the URL for streaming requests.
    ///
    /// Some providers use different endpoints or query parameters for streaming.
    /// Default implementation delegates to `build_url`.
    fn build_stream_url(&self, base_url: &str, model: &str) -> String {
        self.build_url(base_url, model)
    }

    /// Encodes a request into bytes for transmission.
    fn encode_request(&self, request: &Self::Request) -> HiLlmResult<Bytes>;

    /// Decodes a response from bytes.
    fn decode_response(&self, bytes: &[u8]) -> HiLlmResult<Self::Response>;

    /// Parses a stream event from SSE data.
    ///
    /// Returns `Ok(None)` if the event should be skipped (e.g., OpenAI's `[DONE]` marker).
    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<Self::StreamEvent>>;

    /// Returns signing headers for the request, if any.
    ///
    /// This is used for providers that require request signing (e.g., AWS SigV4).
    /// Default implementation returns an empty vector.
    fn signing_headers(
        &self,
        _method: &str,
        _url: &str,
        _body: &[u8],
    ) -> HiLlmResult<Vec<(String, String)>> {
        Ok(vec![])
    }
}
