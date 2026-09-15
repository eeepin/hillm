//! OpenAI endpoint codec implementations.
//!
//! This module provides [`EndpointCodec`] implementations for all OpenAI endpoints:
//! - Chat Completions (with streaming)
//! - Responses API (with streaming)
//! - Embeddings
//! - Image Generation
//! - Audio Speech (TTS)
//! - Audio Transcription (STT)
//! - Moderation

use bytes::Bytes;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::endpoint::{
    Endpoint, EndpointCodec, EndpointRequest, EndpointResponse, EndpointStreamEvent,
};
use crate::types::{
    audio::TranscriptionResponse,
    chat::{ChatCompletionChunk, ChatCompletionResponse},
    embedding::EmbeddingResponse,
    image::ImagesResponse,
    moderation::ModerationResponse,
    response::{ResponseObject, ResponsesStreamEvent},
};

// ============================================================================
// Chat Completions Codec
// ============================================================================

/// Codec for OpenAI Chat Completions API (`/chat/completions`).
///
/// Supports both streaming and non-streaming responses.
/// Handles the `[DONE]` sentinel for stream termination.
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIChatCompletionCodec;

impl EndpointCodec for OpenAIChatCompletionCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::ChatCompletion
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::ChatCompletion(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIChatCompletionCodec received invalid request type: expected ChatCompletion, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: ChatCompletionResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::ChatCompletion(response))
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<EndpointStreamEvent>> {
        // OpenAI sends "[DONE]" to signal end of stream
        if data == "[DONE]" {
            return Ok(None);
        }

        let chunk: ChatCompletionChunk =
            serde_json::from_str(data).map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse ChatCompletionChunk: {e}"),
            })?;

        Ok(Some(EndpointStreamEvent::ChatCompletion(chunk)))
    }
}

// ============================================================================
// Responses API Codec
// ============================================================================

/// Codec for OpenAI Responses API (`/responses`).
///
/// Supports both streaming and non-streaming responses.
/// Uses native Responses API types (not Chat Completions).
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIResponseCodec;

impl EndpointCodec for OpenAIResponseCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::Response
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::Response(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIResponseCodec received invalid request type: expected Response, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: ResponseObject = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::Response(response))
    }

    fn parse_stream_event(&self, data: &str) -> HiLlmResult<Option<EndpointStreamEvent>> {
        // Responses API doesn't use "[DONE]" sentinel
        // Stream ends when connection closes
        let event: ResponsesStreamEvent =
            serde_json::from_str(data).map_err(|e| HiLlmError::Streaming {
                message: format!("Failed to parse ResponsesStreamEvent: {e}"),
            })?;

        Ok(Some(EndpointStreamEvent::Response(event)))
    }
}

// ============================================================================
// Embeddings Codec
// ============================================================================

/// Codec for OpenAI Embeddings API (`/embeddings`).
///
/// Non-streaming endpoint.
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIEmbeddingCodec;

impl EndpointCodec for OpenAIEmbeddingCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::Embedding
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::Embedding(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIEmbeddingCodec received invalid request type: expected Embedding, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: EmbeddingResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::Embedding(response))
    }
}

// ============================================================================
// Image Generation Codec
// ============================================================================

/// Codec for OpenAI Image Generation API (`/images/generations`).
///
/// Non-streaming endpoint.
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIImageGenerationCodec;

impl EndpointCodec for OpenAIImageGenerationCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::ImageGeneration
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::ImageGeneration(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIImageGenerationCodec received invalid request type: expected ImageGeneration, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: ImagesResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::ImageGeneration(response))
    }
}

// ============================================================================
// Audio Speech (TTS) Codec
// ============================================================================

/// Codec for OpenAI Text-to-Speech API (`/audio/speech`).
///
/// Non-streaming endpoint. Returns binary audio data.
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIAudioSpeechCodec;

impl EndpointCodec for OpenAIAudioSpeechCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::AudioSpeech
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::AudioSpeech(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIAudioSpeechCodec received invalid request type: expected AudioSpeech, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        // Audio speech returns binary audio data, not JSON
        Ok(EndpointResponse::AudioSpeech(Bytes::copy_from_slice(bytes)))
    }
}

// ============================================================================
// Audio Transcription (STT) Codec
// ============================================================================

/// Codec for OpenAI Speech-to-Text API (`/audio/transcriptions`).
///
/// Non-streaming endpoint.
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIAudioTranscriptionCodec;

impl EndpointCodec for OpenAIAudioTranscriptionCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::AudioTranscription
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::AudioTranscription(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIAudioTranscriptionCodec received invalid request type: expected AudioTranscription, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: TranscriptionResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::AudioTranscription(response))
    }
}

// ============================================================================
// Moderation Codec
// ============================================================================

/// Codec for OpenAI Moderation API (`/moderations`).
///
/// Non-streaming endpoint.
#[allow(dead_code)] // Will be used in Phase 3
pub struct OpenAIModerationCodec;

impl EndpointCodec for OpenAIModerationCodec {
    fn endpoint(&self) -> Endpoint {
        Endpoint::Moderation
    }

    fn encode(&self, request: &EndpointRequest) -> HiLlmResult<Bytes> {
        match request {
            EndpointRequest::Moderation(req) => Ok(Bytes::from(serde_json::to_vec(req)?)),
            _ => Err(HiLlmError::InternalError {
                message: format!(
                    "OpenAIModerationCodec received invalid request type: expected Moderation, got {:?}",
                    request.endpoint()
                ),
            }),
        }
    }

    fn decode(&self, bytes: &[u8]) -> HiLlmResult<EndpointResponse> {
        let response: ModerationResponse = serde_json::from_slice(bytes)?;
        Ok(EndpointResponse::Moderation(response))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{chat::ChatCompletionRequest, embedding::EmbeddingRequest};

    #[test]
    fn test_chat_completion_codec_endpoint() {
        let codec = OpenAIChatCompletionCodec;
        assert_eq!(codec.endpoint(), Endpoint::ChatCompletion);
    }

    #[test]
    fn test_chat_completion_codec_encode() {
        let codec = OpenAIChatCompletionCodec;
        let request = EndpointRequest::ChatCompletion(ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        });

        let result = codec.encode(&request);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "gpt-4");
    }

    #[test]
    fn test_chat_completion_codec_parse_stream_done() {
        let codec = OpenAIChatCompletionCodec;
        let result = codec.parse_stream_event("[DONE]").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_chat_completion_codec_parse_stream_chunk() {
        let codec = OpenAIChatCompletionCodec;
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
        if let Some(EndpointStreamEvent::ChatCompletion(chunk)) = result {
            assert_eq!(chunk.id, "chatcmpl-123");
            assert_eq!(chunk.choices.len(), 1);
        } else {
            panic!("Expected ChatCompletion stream event");
        }
    }

    #[test]
    fn test_response_codec_endpoint() {
        let codec = OpenAIResponseCodec;
        assert_eq!(codec.endpoint(), Endpoint::Response);
    }

    #[test]
    fn test_embedding_codec_endpoint() {
        let codec = OpenAIEmbeddingCodec;
        assert_eq!(codec.endpoint(), Endpoint::Embedding);
    }

    #[test]
    fn test_image_generation_codec_endpoint() {
        let codec = OpenAIImageGenerationCodec;
        assert_eq!(codec.endpoint(), Endpoint::ImageGeneration);
    }

    #[test]
    fn test_audio_speech_codec_endpoint() {
        let codec = OpenAIAudioSpeechCodec;
        assert_eq!(codec.endpoint(), Endpoint::AudioSpeech);
    }

    #[test]
    fn test_audio_transcription_codec_endpoint() {
        let codec = OpenAIAudioTranscriptionCodec;
        assert_eq!(codec.endpoint(), Endpoint::AudioTranscription);
    }

    #[test]
    fn test_moderation_codec_endpoint() {
        let codec = OpenAIModerationCodec;
        assert_eq!(codec.endpoint(), Endpoint::Moderation);
    }

    #[test]
    fn test_codec_invalid_request_type() {
        let codec = OpenAIChatCompletionCodec;
        let request = EndpointRequest::Embedding(EmbeddingRequest::default());

        let result = codec.encode(&request);
        assert!(result.is_err());
        if let Err(HiLlmError::InternalError { message }) = result {
            assert!(message.contains("invalid request type"));
        } else {
            panic!("Expected InternalError");
        }
    }
}
