//! AWS Bedrock Converse API native types.

use crate::types::{APIRequest, APIResponse};
use serde::{Deserialize, Serialize};

// ============ Request Types ============

/// Bedrock Converse API request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockConverseRequest {
    pub messages: Vec<BedrockMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<BedrockSystemBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<BedrockInferenceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<BedrockToolConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_model_request_fields: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guardrail_config: Option<serde_json::Value>,
}

/// A message in the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockMessage {
    pub role: BedrockRole,
    pub content: Vec<BedrockContentBlock>,
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockRole {
    User,
    Assistant,
}

/// Content block variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BedrockContentBlock {
    Text {
        text: String,
    },
    Image {
        format: String,
        source: BedrockImageSource,
    },
    Document {
        name: String,
        format: String,
        source: BedrockDocumentSource,
    },
    ToolUse {
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<BedrockContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
}

/// System block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BedrockSystemBlock {
    pub text: String,
}

/// Image source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BedrockImageSource {
    pub bytes: String, // base64
}

/// Document source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BedrockDocumentSource {
    pub bytes: String, // base64
}

/// Inference configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockInferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Tool configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BedrockToolConfig {
    pub tools: Vec<BedrockTool>,
}

/// Tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockTool {
    pub tool_spec: BedrockToolSpec,
}

/// Tool specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockToolSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: BedrockInputSchema,
}

/// Input schema wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BedrockInputSchema {
    pub json: serde_json::Value,
}

// ============ Response Types ============

/// Bedrock Converse API response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockConverseResponse {
    pub output: BedrockOutput,
    pub stop_reason: BedrockStopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<BedrockUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Output wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockOutput {
    pub message: BedrockMessage,
}

/// Stop reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BedrockStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    ContentFiltered,
    GuardrailIntervened,
}

/// Token usage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

// ============ Stream Event Types ============

/// Bedrock stream event (AWS EventStream format).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "camelCase")]
pub enum BedrockStreamEvent {
    MessageStart {
        role: BedrockRole,
    },
    ContentBlockStart {
        content_block_index: u32,
        start: BedrockContentBlockStart,
    },
    ContentBlockDelta {
        content_block_index: u32,
        delta: BedrockStreamDelta,
    },
    ContentBlockStop {
        content_block_index: u32,
    },
    MessageStop {
        stop_reason: BedrockStopReason,
    },
    Metadata {
        usage: BedrockUsage,
    },
}

/// Content block start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockContentBlockStart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<BedrockToolUseStart>,
}

/// Tool use start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockToolUseStart {
    pub tool_use_id: String,
    pub name: String,
}

/// Stream delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<BedrockToolUseDelta>,
}

/// Tool use delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockToolUseDelta {
    pub input: String,
}

// ============ Trait Implementations ============

impl APIRequest for BedrockConverseRequest {
    type Response = BedrockConverseResponse;
    type StreamEvent = BedrockStreamEvent;

    fn model(&self) -> &str {
        "" // Bedrock model is in URL, not body
    }

    fn stream(&self) -> bool {
        false // Determined by endpoint
    }
}

impl APIResponse for BedrockConverseResponse {}
