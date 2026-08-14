//! API type definitions for different LLM provider protocols.

use serde::{Deserialize, Serialize};

/// Represents the different API protocols supported by LLM providers.
///
/// Each variant corresponds to a distinct request/response format and endpoint structure.
/// Providers declare which API types they support, and clients select the appropriate
/// API type when making requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum APIType {
    /// OpenAI Chat Completions API (`/chat/completions`)
    #[serde(rename = "openai_chat_completions")]
    OpenAIChatCompletions,

    /// OpenAI Responses API (`/responses`)
    #[serde(rename = "openai_responses")]
    OpenAIResponses,

    /// Anthropic Messages API (`/messages`)
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,

    /// AWS Bedrock Converse API
    #[serde(rename = "bedrock_converse")]
    BedrockConverse,
}

impl APIType {
    /// Returns the default endpoint path for this API type.
    pub fn default_endpoint_path(&self) -> &'static str {
        match self {
            Self::OpenAIChatCompletions => "/chat/completions",
            Self::OpenAIResponses => "/responses",
            Self::AnthropicMessages => "/messages",
            Self::BedrockConverse => "/converse",
        }
    }

    /// Returns a human-readable name for this API type.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::OpenAIChatCompletions => "OpenAI Chat Completions",
            Self::OpenAIResponses => "OpenAI Responses",
            Self::AnthropicMessages => "Anthropic Messages",
            Self::BedrockConverse => "AWS Bedrock Converse",
        }
    }
}

impl std::fmt::Display for APIType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_type_serialization() {
        assert_eq!(
            serde_json::to_string(&APIType::OpenAIChatCompletions).unwrap(),
            "\"openai_chat_completions\""
        );
        assert_eq!(
            serde_json::to_string(&APIType::OpenAIResponses).unwrap(),
            "\"openai_responses\""
        );
        assert_eq!(
            serde_json::to_string(&APIType::AnthropicMessages).unwrap(),
            "\"anthropic_messages\""
        );
        assert_eq!(
            serde_json::to_string(&APIType::BedrockConverse).unwrap(),
            "\"bedrock_converse\""
        );
    }

    #[test]
    fn api_type_deserialization() {
        assert_eq!(
            serde_json::from_str::<APIType>("\"openai_chat_completions\"").unwrap(),
            APIType::OpenAIChatCompletions
        );
        assert_eq!(
            serde_json::from_str::<APIType>("\"openai_responses\"").unwrap(),
            APIType::OpenAIResponses
        );
        assert_eq!(
            serde_json::from_str::<APIType>("\"anthropic_messages\"").unwrap(),
            APIType::AnthropicMessages
        );
        assert_eq!(
            serde_json::from_str::<APIType>("\"bedrock_converse\"").unwrap(),
            APIType::BedrockConverse
        );
    }

    #[test]
    fn api_type_default_endpoints() {
        assert_eq!(
            APIType::OpenAIChatCompletions.default_endpoint_path(),
            "/chat/completions"
        );
        assert_eq!(APIType::OpenAIResponses.default_endpoint_path(), "/responses");
        assert_eq!(APIType::AnthropicMessages.default_endpoint_path(), "/messages");
        assert_eq!(APIType::BedrockConverse.default_endpoint_path(), "/converse");
    }

    #[test]
    fn api_type_display() {
        assert_eq!(APIType::OpenAIChatCompletions.to_string(), "OpenAI Chat Completions");
        assert_eq!(APIType::OpenAIResponses.to_string(), "OpenAI Responses");
        assert_eq!(APIType::AnthropicMessages.to_string(), "Anthropic Messages");
        assert_eq!(APIType::BedrockConverse.to_string(), "AWS Bedrock Converse");
    }
}
