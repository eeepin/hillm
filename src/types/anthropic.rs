//! Anthropic Messages API native types.

use crate::types::{ApiRequest, ApiResponse};
use serde::{Deserialize, Serialize};

// ============ Request Types ============

/// Anthropic Messages API request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<AnthropicSystemBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
}

/// System message block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicSystemBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

/// A message in the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    pub content: Vec<AnthropicContentBlock>,
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// Content block variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Document {
        source: AnthropicDocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<AnthropicContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        thinking: String,
    },
}

/// Image source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// Document source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicDocumentSource {
    Base64 { media_type: String, data: String },
}

/// Cache control configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicCacheControl {
    Ephemeral,
}

/// Tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

/// Tool choice configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

/// Extended thinking configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicThinkingConfig {
    pub r#type: String,
    pub budget_tokens: u64,
}

/// Request metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// ============ Response Types ============

/// Anthropic Messages API response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    pub model: String,
    pub r#type: String,
    pub role: AnthropicRole,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub stop_reason: AnthropicStopReason,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Response content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
}

/// Stop reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    ContentFiltered,
}

/// Token usage.
///
/// `input_tokens` is absent from `message_delta` stream events (only
/// `output_tokens` is sent there), so both fields deserialize with defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

// ============ Stream Event Types ============

/// Anthropic stream event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: AnthropicDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: AnthropicUsage,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicStreamError,
    },
}

/// Message start event data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessageStart {
    pub id: String,
    pub model: String,
    pub r#type: String,
    pub role: AnthropicRole,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub stop_reason: Option<AnthropicStopReason>,
    pub usage: AnthropicUsage,
}

/// Content block start variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockStart {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
}

/// Delta variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
}

/// Message delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessageDelta {
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
}

/// Stream error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicStreamError {
    pub r#type: String,
    pub message: String,
}

// ============ Trait Implementations ============

impl ApiRequest for AnthropicMessagesRequest {
    type Response = AnthropicMessagesResponse;
    type StreamEvent = AnthropicStreamEvent;

    fn model(&self) -> &str {
        &self.model
    }

    fn stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

impl ApiResponse for AnthropicMessagesResponse {}
