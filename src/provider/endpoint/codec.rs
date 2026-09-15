//! Unified endpoint codec trait for protocol-specific encoding and decoding.
//!
//! This module provides the [`EndpointCodec`] trait, which is the **minimal** abstraction
//! for encoding requests and decoding responses for a specific endpoint. The codec is
//! deliberately kept small and focused on serialization/deserialization only:
//!
//! - **URL building** is handled by the transport layer (not the codec)
//! - **Signing headers** (e.g., AWS SigV4) are provider-specific (not codec-specific)
//! - **Request/response transforms** are provider-specific (not codec-specific)
//!
//! This separation of concerns makes the codec trait minimal and easy to implement,
//! while keeping provider-specific logic in the provider implementations.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::Endpoint;
use crate::error::{HiLlmError, HiLlmResult};

// Re-export all request/response types for convenience
use crate::types::{
    audio::{CreateSpeechRequest, CreateTranscriptionRequest, TranscriptionResponse},
    batch::{BatchListQuery, BatchListResponse, BatchObject, CreateBatchRequest},
    chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    embedding::{EmbeddingRequest, EmbeddingResponse},
    file::{CreateFileRequest, DeleteResponse, FileListQuery, FileListResponse, FileObject},
    image::{CreateImageRequest, ImagesResponse},
    model::ModelsListResponse,
    moderation::{ModerationRequest, ModerationResponse},
    ocr::{OcrRequest, OcrResponse},
    rerank::{RerankRequest, RerankResponse},
    response::{CreateResponseRequest, ResponseObject, ResponsesStreamEvent},
    search::{SearchRequest, SearchResponse},
};

/// Unified request type for all endpoints.
///
/// This enum wraps all possible request types, enabling the [`EndpointCodec`] trait
/// to work with a single request type while maintaining type safety through pattern
/// matching.
///
/// # Design Rationale
///
/// Using an enum instead of generics or trait objects keeps the codec trait minimal
/// and object-safe. Adding a new endpoint requires adding a variant here and implementing
/// the codec for it — no other changes needed.
///
/// # Note on Size
///
/// Some variants are large (e.g., ChatCompletionRequest is ~440 bytes), but this is
/// acceptable because:
/// 1. Requests are short-lived (created, sent, dropped)
/// 2. The alternative (Box-ing all variants) adds indirection overhead
/// 3. The size will be optimized when we implement proper codecs in Phase 2.3+
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum EndpointRequest {
    // === Core LLM APIs ===
    /// OpenAI Chat Completions request
    ChatCompletion(ChatCompletionRequest),
    /// OpenAI Responses API request
    Response(CreateResponseRequest),
    /// Anthropic Messages API request
    AnthropicMessages(AnthropicMessagesRequest),
    /// AWS Bedrock Converse API request
    BedrockConverse(BedrockConverseRequest),

    // === Text Processing APIs ===
    /// Embeddings API request
    Embedding(EmbeddingRequest),
    /// Reranking API request
    Rerank(RerankRequest),

    // === Multimedia APIs ===
    /// Image generation API request
    ImageGeneration(CreateImageRequest),
    /// Text-to-speech API request
    AudioSpeech(CreateSpeechRequest),
    /// Speech-to-text API request
    AudioTranscription(CreateTranscriptionRequest),

    // === Content Moderation APIs ===
    /// Moderation API request
    Moderation(ModerationRequest),
    /// OCR API request
    Ocr(OcrRequest),

    // === Search APIs ===
    /// Search API request
    Search(SearchRequest),

    // === Resource Management APIs ===
    /// Files API create request
    CreateFile(CreateFileRequest),
    /// Files API list query
    ListFiles(Option<FileListQuery>),
    /// Batches API create request
    CreateBatch(CreateBatchRequest),
    /// Batches API list query
    ListBatches(Option<BatchListQuery>),
}

impl EndpointRequest {
    /// Returns the endpoint this request is for.
    pub fn endpoint(&self) -> Endpoint {
        match self {
            Self::ChatCompletion(_) => Endpoint::ChatCompletion,
            Self::Response(_) => Endpoint::Response,
            Self::AnthropicMessages(_) => Endpoint::AnthropicMessages,
            Self::BedrockConverse(_) => Endpoint::BedrockConverse,
            Self::Embedding(_) => Endpoint::Embedding,
            Self::Rerank(_) => Endpoint::Rerank,
            Self::ImageGeneration(_) => Endpoint::ImageGeneration,
            Self::AudioSpeech(_) => Endpoint::AudioSpeech,
            Self::AudioTranscription(_) => Endpoint::AudioTranscription,
            Self::Moderation(_) => Endpoint::Moderation,
            Self::Ocr(_) => Endpoint::Ocr,
            Self::Search(_) => Endpoint::Search,
            Self::CreateFile(_) | Self::ListFiles(_) => Endpoint::Files,
            Self::CreateBatch(_) | Self::ListBatches(_) => Endpoint::Batches,
        }
    }

    /// Returns the model name if this request requires one.
    pub fn model(&self) -> Option<&str> {
        match self {
            Self::ChatCompletion(r) => Some(&r.model),
            Self::Response(r) => Some(&r.model),
            Self::AnthropicMessages(r) => Some(&r.model),
            Self::BedrockConverse(r) => Some(&r.model_id),
            Self::Embedding(r) => Some(&r.model),
            Self::ImageGeneration(r) => r.model.as_deref(),
            Self::AudioSpeech(r) => Some(&r.model),
            Self::AudioTranscription(r) => Some(&r.model),
            Self::Moderation(r) => r.model.as_deref(),
            Self::Ocr(r) => Some(&r.model),
            Self::Search(r) => Some(&r.model),
            Self::Rerank(r) => Some(&r.model),
            Self::CreateFile(_)
            | Self::ListFiles(_)
            | Self::CreateBatch(_)
            | Self::ListBatches(_) => None,
        }
    }

    /// Returns whether this request is for a streaming response.
    pub fn is_stream(&self) -> bool {
        match self {
            Self::ChatCompletion(r) => r.stream.unwrap_or(false),
            Self::Response(r) => r.stream.unwrap_or(false),
            Self::AnthropicMessages(r) => r.stream.unwrap_or(false),
            Self::BedrockConverse(r) => r.stream.unwrap_or(false),
            _ => false,
        }
    }
}

/// Unified response type for all endpoints.
///
/// This enum wraps all possible response types, enabling type-safe response handling
/// while keeping the codec trait minimal.
#[derive(Debug, Clone)]
pub enum EndpointResponse {
    // === Core LLM APIs ===
    /// OpenAI Chat Completions response
    ChatCompletion(ChatCompletionResponse),
    /// OpenAI Responses API response
    Response(ResponseObject),
    /// Anthropic Messages API response
    AnthropicMessages(AnthropicMessagesResponse),
    /// AWS Bedrock Converse API response
    BedrockConverse(BedrockConverseResponse),

    // === Text Processing APIs ===
    /// Embeddings API response
    Embedding(EmbeddingResponse),
    /// Reranking API response
    Rerank(RerankResponse),

    // === Multimedia APIs ===
    /// Image generation API response
    ImageGeneration(ImagesResponse),
    /// Text-to-speech API response (binary audio data)
    AudioSpeech(Bytes),
    /// Speech-to-text API response
    AudioTranscription(TranscriptionResponse),

    // === Content Moderation APIs ===
    /// Moderation API response
    Moderation(ModerationResponse),
    /// OCR API response
    Ocr(OcrResponse),

    // === Search APIs ===
    /// Search API response
    Search(SearchResponse),

    // === Resource Management APIs ===
    /// Models list API response
    Models(ModelsListResponse),
    /// Files API file object response
    File(FileObject),
    /// Files API list response
    FileList(FileListResponse),
    /// Files API delete response
    FileDelete(DeleteResponse),
    /// Files API binary content response
    FileContent(Bytes),
    /// Batches API batch object response
    Batch(BatchObject),
    /// Batches API list response
    BatchList(BatchListResponse),
}

/// Unified stream event type for streaming endpoints.
///
/// Only core LLM endpoints support streaming. This enum wraps the stream event
/// types for each streaming endpoint.
///
/// # Note on Size
///
/// Some variants are large, but this is acceptable because stream events are
/// processed immediately and not stored long-term.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum EndpointStreamEvent {
    /// OpenAI Chat Completions stream chunk
    ChatCompletion(ChatCompletionChunk),
    /// OpenAI Responses API stream event
    Response(ResponsesStreamEvent),
    /// Anthropic Messages API stream event
    AnthropicMessages(AnthropicStreamEvent),
    /// AWS Bedrock Converse API stream event
    BedrockConverse(BedrockStreamEvent),
}

/// Minimal trait for endpoint-specific encoding and decoding.
///
/// This trait is deliberately kept small and focused on **serialization/deserialization only**:
///
/// - **3 required methods**: `endpoint()`, `encode()`, `decode()`
/// - **1 optional method**: `parse_stream_event()` with a default error implementation
///
/// URL building, signing headers, and request/response transforms are **not** part of this
/// trait — they belong in the provider or transport layer.
///
/// # Implementing for a New Endpoint
///
/// 1. Add a variant to [`EndpointRequest`], [`EndpointResponse`], and (if streaming) [`EndpointStreamEvent`]
/// 2. Implement this trait for your codec struct
/// 3. That's it! The transport layer handles everything else.
///
/// # Example
///
/// ```rust,ignore
/// struct MyEndpointCodec;
///
/// impl EndpointCodec for MyEndpointCodec {
///     fn endpoint(&self) -> Endpoint {
///         Endpoint::MyEndpoint
///     }
///
///     fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
///         match request {
///             EndpointRequest::MyEndpoint(req) => {
///                 serde_json::to_vec(req).map(Into::into).map_err(Into::into)
///             }
///             _ => Err(HiLlmError::InternalError {
///                 message: "invalid request type for this codec".into(),
///             }),
///         }
///     }
///
///     fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
///         let response: MyResponse = serde_json::from_slice(bytes)?;
///         Ok(EndpointResponse::MyEndpoint(response))
///     }
/// }
/// ```
pub trait EndpointCodec: Send + Sync {
    /// Returns the endpoint this codec handles.
    fn endpoint(&self) -> Endpoint;

    /// Encodes a request into bytes for transmission.
    ///
    /// The codec should validate that the request matches its endpoint and return
    /// an error if not. This is a safety check to catch programming errors.
    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes>;

    /// Decodes response bytes into a typed response.
    ///
    /// The codec should deserialize the bytes into the appropriate response type
    /// for its endpoint.
    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse>;

    /// Parses a stream event from SSE data.
    ///
    /// Returns `Ok(None)` for sentinel values (e.g., OpenAI's `[DONE]`).
    /// Returns `Ok(Some(event))` for valid stream events.
    /// Returns `Err` for invalid data or non-streaming endpoints.
    ///
    /// The default implementation returns an error, indicating that this endpoint
    /// does not support streaming. Override this for streaming endpoints.
    fn parse_stream_event(&self, _data: &str) -> HiLlmResult<Option<EndpointStreamEvent>> {
        Err(HiLlmError::Streaming {
            message: format!(
                "{} endpoint does not support streaming",
                self.endpoint().display_name()
            ),
        })
    }
}

// Note: We need to import the Anthropic and Bedrock types for the enum variants.
// These are currently in the types module but may need to be re-exported or moved.
// For now, we'll use placeholder types that will be replaced with the actual types
// when we implement the codecs in Phase 2.3 and 2.4.

/// Placeholder for Anthropic Messages request (will be replaced with actual type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Placeholder for Anthropic Messages response (will be replaced with actual type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Placeholder for Anthropic stream event (will be replaced with actual type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicStreamEvent {
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Placeholder for Bedrock Converse request (will be replaced with actual type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockConverseRequest {
    pub model_id: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Placeholder for Bedrock Converse response (will be replaced with actual type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockConverseResponse {
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Placeholder for Bedrock stream event (will be replaced with actual type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockStreamEvent {
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_request_model() {
        let req = EndpointRequest::ChatCompletion(ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        });
        assert_eq!(req.model(), Some("gpt-4"));

        let req = EndpointRequest::ListFiles(None);
        assert_eq!(req.model(), None);
    }

    #[test]
    fn test_endpoint_request_is_stream() {
        let req = EndpointRequest::ChatCompletion(ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            stream: Some(true),
            ..Default::default()
        });
        assert!(req.is_stream());

        let req = EndpointRequest::Embedding(EmbeddingRequest {
            model: "text-embedding-3-small".to_string(),
            input: crate::types::embedding::EmbeddingInput::Single("test".to_string()),
            ..Default::default()
        });
        assert!(!req.is_stream());
    }

    #[test]
    fn test_endpoint_request_endpoint() {
        let req = EndpointRequest::ChatCompletion(ChatCompletionRequest::default());
        assert_eq!(req.endpoint(), Endpoint::ChatCompletion);

        let req = EndpointRequest::Embedding(EmbeddingRequest::default());
        assert_eq!(req.endpoint(), Endpoint::Embedding);
    }
}
