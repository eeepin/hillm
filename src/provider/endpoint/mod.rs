//! Unified endpoint type system for all API endpoints.
//!
//! This module provides a comprehensive [`Endpoint`] enum that covers all supported
//! API endpoints across all providers, replacing the fragmented legacy system where
//! endpoint paths were scattered across Provider trait methods, ApiType enum, and
//! individual codec implementations.
//!
//! # Submodules
//!
//! - [`codec`] - The [`EndpointCodec`](codec::EndpointCodec) trait and unified request/response types

pub mod codec;

// Re-export codec types for convenience
pub use codec::{EndpointCodec, EndpointRequest, EndpointResponse, EndpointStreamEvent};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// All supported API endpoints.
///
/// This enum provides a unified type for all API endpoints, enabling:
/// - Consistent endpoint capability declaration via [`EndpointCapabilities`]
/// - Unified codec abstraction via `EndpointCodec`
/// - Single source of truth for default endpoint paths
///
/// # Categories
///
/// - **Core LLM APIs**: Chat completion, responses, and provider-specific message APIs
/// - **Text Processing**: Embeddings, reranking
/// - **Multimedia**: Image generation, audio speech/transcription
/// - **Content Moderation**: Moderation, OCR
/// - **Search**: Semantic search
/// - **Resource Management**: Models, files, batches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Endpoint {
    // === Core LLM APIs ===
    /// OpenAI Chat Completions API (`/chat/completions`)
    ChatCompletion,
    /// OpenAI Responses API (`/responses`)
    Response,
    /// Anthropic Messages API (`/messages`)
    AnthropicMessages,
    /// AWS Bedrock Converse API (`/converse`)
    BedrockConverse,

    // === Text Processing APIs ===
    /// Embeddings API (`/embeddings`)
    Embedding,
    /// Reranking API (`/rerank`)
    Rerank,

    // === Multimedia APIs ===
    /// Image generation API (`/images/generations`)
    ImageGeneration,
    /// Text-to-speech API (`/audio/speech`)
    AudioSpeech,
    /// Speech-to-text API (`/audio/transcriptions`)
    AudioTranscription,

    // === Content Moderation APIs ===
    /// Content moderation API (`/moderations`)
    Moderation,
    /// OCR API (`/ocr`)
    Ocr,

    // === Search APIs ===
    /// Semantic search API (`/search`)
    Search,

    // === Resource Management APIs ===
    /// Models list API (`/models`)
    Models,
    /// Files API (`/files`)
    Files,
    /// Batches API (`/batches`)
    Batches,
}

impl Endpoint {
    /// Returns the default endpoint path for this API.
    ///
    /// This is the standard path used by most providers. Individual providers
    /// can override this via `EndpointOverride::path` in their configuration.
    pub fn default_path(&self) -> &'static str {
        match self {
            Self::ChatCompletion => "/chat/completions",
            Self::Response => "/responses",
            Self::AnthropicMessages => "/messages",
            Self::BedrockConverse => "/converse",
            Self::Embedding => "/embeddings",
            Self::Rerank => "/rerank",
            Self::ImageGeneration => "/images/generations",
            Self::AudioSpeech => "/audio/speech",
            Self::AudioTranscription => "/audio/transcriptions",
            Self::Moderation => "/moderations",
            Self::Ocr => "/ocr",
            Self::Search => "/search",
            Self::Models => "/models",
            Self::Files => "/files",
            Self::Batches => "/batches",
        }
    }

    /// Returns whether this endpoint supports streaming responses.
    ///
    /// Only core LLM APIs support streaming. Other endpoints return non-streaming
    /// JSON or binary responses.
    pub fn supports_streaming(&self) -> bool {
        matches!(
            self,
            Self::ChatCompletion | Self::Response | Self::AnthropicMessages | Self::BedrockConverse
        )
    }

    /// Returns whether this endpoint requires a model parameter in the request.
    ///
    /// Resource management endpoints (models, files, batches) don't require a model.
    pub fn requires_model(&self) -> bool {
        !matches!(self, Self::Models | Self::Files | Self::Batches)
    }

    /// Returns the response category for this endpoint.
    ///
    /// This determines how the response should be decoded (JSON, binary, etc.).
    pub fn response_category(&self) -> ResponseCategory {
        match self {
            Self::AudioSpeech => ResponseCategory::Binary,
            Self::Files | Self::Batches => ResponseCategory::FileManagement,
            _ => ResponseCategory::Json,
        }
    }

    /// Returns a human-readable display name for this endpoint.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ChatCompletion => "Chat Completions",
            Self::Response => "Responses",
            Self::AnthropicMessages => "Anthropic Messages",
            Self::BedrockConverse => "Bedrock Converse",
            Self::Embedding => "Embeddings",
            Self::Rerank => "Rerank",
            Self::ImageGeneration => "Image Generation",
            Self::AudioSpeech => "Text-to-Speech",
            Self::AudioTranscription => "Speech-to-Text",
            Self::Moderation => "Moderation",
            Self::Ocr => "OCR",
            Self::Search => "Search",
            Self::Models => "Models",
            Self::Files => "Files",
            Self::Batches => "Batches",
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Response category for endpoint responses.
///
/// This enum categorizes endpoints by their response type, guiding the decoding
/// strategy in the HTTP transport layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseCategory {
    /// JSON response (most endpoints)
    Json,
    /// Binary response (e.g., audio speech)
    Binary,
    /// File management response (files, batches)
    FileManagement,
}

/// Declares which endpoints a provider supports.
///
/// This struct is used in [`ProviderConfig`] to declare the capabilities of a
/// provider. It enables the client to validate endpoint support before making
/// requests, providing clear error messages when an endpoint is not supported.
///
/// # Examples
///
/// ```rust
/// use hillm::provider::{Endpoint, EndpointCapabilities};
///
/// let capabilities = EndpointCapabilities::openai_default();
/// assert!(capabilities.supports(Endpoint::ChatCompletion));
/// assert!(capabilities.supports(Endpoint::Embedding));
/// assert!(!capabilities.supports(Endpoint::AnthropicMessages));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    /// List of supported endpoints.
    ///
    /// An empty list means no endpoints are supported (unusual but valid for
    /// placeholder configurations).
    #[serde(default)]
    pub supported: Vec<Endpoint>,
}

impl EndpointCapabilities {
    /// Creates a new empty capability set.
    pub fn new() -> Self {
        Self {
            supported: Vec::new(),
        }
    }

    /// Creates a capability set with the specified endpoints.
    pub fn with_endpoints(endpoints: Vec<Endpoint>) -> Self {
        Self {
            supported: endpoints,
        }
    }

    /// Returns whether this capability set includes the specified endpoint.
    pub fn supports(&self, endpoint: Endpoint) -> bool {
        self.supported.contains(&endpoint)
    }

    /// Adds an endpoint to the capability set.
    pub fn add(&mut self, endpoint: Endpoint) {
        if !self.supports(endpoint) {
            self.supported.push(endpoint);
        }
    }

    /// Removes an endpoint from the capability set.
    pub fn remove(&mut self, endpoint: Endpoint) {
        self.supported.retain(|&e| e != endpoint);
    }

    /// Returns the default capabilities for OpenAI providers.
    ///
    /// Includes: ChatCompletion, Response, Embedding, ImageGeneration,
    /// AudioSpeech, AudioTranscription, Moderation, Models, Files, Batches.
    pub fn openai_default() -> Self {
        Self {
            supported: vec![
                Endpoint::ChatCompletion,
                Endpoint::Response,
                Endpoint::Embedding,
                Endpoint::ImageGeneration,
                Endpoint::AudioSpeech,
                Endpoint::AudioTranscription,
                Endpoint::Moderation,
                Endpoint::Models,
                Endpoint::Files,
                Endpoint::Batches,
            ],
        }
    }

    /// Returns the default capabilities for Anthropic providers.
    ///
    /// Includes: AnthropicMessages.
    pub fn anthropic_default() -> Self {
        Self {
            supported: vec![Endpoint::AnthropicMessages],
        }
    }

    /// Returns the default capabilities for Bedrock providers.
    ///
    /// Includes: BedrockConverse.
    pub fn bedrock_default() -> Self {
        Self {
            supported: vec![Endpoint::BedrockConverse],
        }
    }
}

impl PartialEq for EndpointCapabilities {
    fn eq(&self, other: &Self) -> bool {
        // Compare as sets (order-independent)
        if self.supported.len() != other.supported.len() {
            return false;
        }
        self.supported.iter().all(|e| other.supported.contains(e))
    }
}

impl Eq for EndpointCapabilities {}

/// Endpoint-specific configuration overrides.
///
/// This struct allows providers to customize the behavior of specific endpoints,
/// such as using a non-standard path or adding extra headers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EndpointOverride {
    /// Custom endpoint path (overrides the default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Custom timeout for this endpoint (in milliseconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    /// Extra headers to include in requests to this endpoint.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_headers: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_serialization() {
        let endpoint = Endpoint::ChatCompletion;
        let json = serde_json::to_string(&endpoint).unwrap();
        assert_eq!(json, r#""chat_completion""#);

        let deserialized: Endpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, endpoint);
    }

    #[test]
    fn test_all_endpoints_have_paths() {
        let endpoints = vec![
            Endpoint::ChatCompletion,
            Endpoint::Response,
            Endpoint::AnthropicMessages,
            Endpoint::BedrockConverse,
            Endpoint::Embedding,
            Endpoint::Rerank,
            Endpoint::ImageGeneration,
            Endpoint::AudioSpeech,
            Endpoint::AudioTranscription,
            Endpoint::Moderation,
            Endpoint::Ocr,
            Endpoint::Search,
            Endpoint::Models,
            Endpoint::Files,
            Endpoint::Batches,
        ];

        for endpoint in endpoints {
            let path = endpoint.default_path();
            assert!(!path.is_empty(), "Endpoint {:?} has empty path", endpoint);
            assert!(path.starts_with('/'), "Path {} doesn't start with /", path);
        }
    }

    #[test]
    fn test_streaming_support() {
        assert!(Endpoint::ChatCompletion.supports_streaming());
        assert!(Endpoint::Response.supports_streaming());
        assert!(Endpoint::AnthropicMessages.supports_streaming());
        assert!(Endpoint::BedrockConverse.supports_streaming());
        assert!(!Endpoint::Embedding.supports_streaming());
        assert!(!Endpoint::ImageGeneration.supports_streaming());
        assert!(!Endpoint::Models.supports_streaming());
    }

    #[test]
    fn test_model_requirement() {
        assert!(Endpoint::ChatCompletion.requires_model());
        assert!(Endpoint::Embedding.requires_model());
        assert!(!Endpoint::Models.requires_model());
        assert!(!Endpoint::Files.requires_model());
        assert!(!Endpoint::Batches.requires_model());
    }

    #[test]
    fn test_response_category() {
        assert_eq!(
            Endpoint::ChatCompletion.response_category(),
            ResponseCategory::Json
        );
        assert_eq!(
            Endpoint::AudioSpeech.response_category(),
            ResponseCategory::Binary
        );
        assert_eq!(
            Endpoint::Files.response_category(),
            ResponseCategory::FileManagement
        );
    }

    #[test]
    fn test_endpoint_capabilities_supports() {
        let caps = EndpointCapabilities::openai_default();
        assert!(caps.supports(Endpoint::ChatCompletion));
        assert!(caps.supports(Endpoint::Embedding));
        assert!(!caps.supports(Endpoint::AnthropicMessages));
    }

    #[test]
    fn test_endpoint_capabilities_add_remove() {
        let mut caps = EndpointCapabilities::new();
        assert!(!caps.supports(Endpoint::ChatCompletion));

        caps.add(Endpoint::ChatCompletion);
        assert!(caps.supports(Endpoint::ChatCompletion));

        caps.add(Endpoint::ChatCompletion); // duplicate add
        assert_eq!(caps.supported.len(), 1);

        caps.remove(Endpoint::ChatCompletion);
        assert!(!caps.supports(Endpoint::ChatCompletion));
    }

    #[test]
    fn test_endpoint_capabilities_equality() {
        let caps1 = EndpointCapabilities::with_endpoints(vec![
            Endpoint::ChatCompletion,
            Endpoint::Embedding,
        ]);
        let caps2 = EndpointCapabilities::with_endpoints(vec![
            Endpoint::Embedding,
            Endpoint::ChatCompletion,
        ]);
        assert_eq!(caps1, caps2); // order-independent
    }

    #[test]
    fn test_endpoint_capabilities_serialization() {
        let caps = EndpointCapabilities::anthropic_default();
        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: EndpointCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(caps, deserialized);
    }

    #[test]
    fn test_endpoint_display() {
        assert_eq!(Endpoint::ChatCompletion.display_name(), "Chat Completions");
        assert_eq!(format!("{}", Endpoint::Embedding), "Embeddings");
    }

    #[test]
    fn test_endpoint_override_serialization() {
        let mut extra_headers = HashMap::new();
        extra_headers.insert("X-Custom".to_string(), "value".to_string());

        let override_config = EndpointOverride {
            path: Some("/custom/path".to_string()),
            timeout_ms: Some(5000),
            extra_headers,
        };

        let json = serde_json::to_string(&override_config).unwrap();
        let deserialized: EndpointOverride = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.path, override_config.path);
        assert_eq!(deserialized.timeout_ms, override_config.timeout_ms);
        assert_eq!(deserialized.extra_headers, override_config.extra_headers);
    }
}
