use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::common::{
    AssistantMessage, Message, ResponseFormat, StopSequence, Tool, ToolChoice, ToolType, Usage,
};
use crate::cost;

/// Modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Audio,
    Image,
}

/// Finish Reason
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    #[default]
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
    #[serde(other)]
    Other,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        f.write_str(&s)
    }
}

/// Reasoning Effort
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
}

// Request

/// Chat completion request
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Value>,
}

/// Stream options
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

// Response

/// Chat completion response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

impl ChatCompletionResponse {
    pub async fn estimated_cost(&self, provider: &str) -> Result<Option<f64>, cost::CostError> {
        let Some(usage) = self.usage.as_ref() else {
            return Ok(None);
        };
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens);
        let provider_model = format!("{provider}/{}", self.model);
        cost::completion_cost_with_cache(
            &provider_model,
            usage.prompt_tokens,
            cached,
            usage.completion_tokens,
        ).await
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: Option<FinishReason>,
}

// Stream Chunk

/// Streamed chunk of response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

/// Stream choice
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<StreamToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// A streaming tool call being built incrementally.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamToolCall {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<ToolType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<StreamFunctionCall>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamFunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::common::PromptTokensDetails;

    fn make_response(model: &str, usage: Usage) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: model.into(),
            choices: vec![],
            usage: Some(usage),
            system_fingerprint: None,
            service_tier: None,
        }
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn estimated_cost_applies_cache_discount_when_prompt_tokens_details_present() {
        let resp = make_response(
            "claude-sonnet-4-5",
            Usage {
                prompt_tokens: 1_000,
                completion_tokens: 50,
                total_tokens: 1_050,
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens: 200,
                    audio_tokens: 0,
                }),
            },
        );
        let with_cache = resp.estimated_cost("anthropic").await.unwrap().expect("should price");
        let no_cache = make_response(
            "claude-sonnet-4-5",
            Usage {
                prompt_tokens: 1_000,
                completion_tokens: 50,
                total_tokens: 1_050,
                prompt_tokens_details: None,
            },
        )
        .estimated_cost("anthropic")
        .await
        .unwrap()
        .expect("should price");
        assert!(
            with_cache < no_cache,
            "cached cost ({with_cache}) must be cheaper than uncached ({no_cache})"
        );
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn estimated_cost_ignores_cached_tokens_when_no_pricing_difference() {
        let usage_with_cached = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 50,
            total_tokens: 1_050,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 500,
                audio_tokens: 0,
            }),
        };
        let usage_no_details = Usage {
            prompt_tokens: 1_000,
            completion_tokens: 50,
            total_tokens: 1_050,
            prompt_tokens_details: None,
        };
        let a = make_response("gpt-4", usage_with_cached)
            .estimated_cost("openai")
            .await
            .unwrap()
            .expect("cost estimation should succeed for known model");
        let b = make_response("gpt-4", usage_no_details)
            .estimated_cost("openai")
            .await
            .unwrap()
            .expect("cost estimation should succeed for known model");
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn modalities_serializes_when_present() {
        let req = ChatCompletionRequest {
            model: "gpt-4o-audio-preview".into(),
            modalities: Some(vec![Modality::Text, Modality::Audio]),
            ..Default::default()
        };
        let value = serde_json::to_value(&req).expect("must serialise");
        assert_eq!(value["modalities"], serde_json::json!(["text", "audio"]));
    }

    #[test]
    fn modalities_omitted_when_none() {
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            ..Default::default()
        };
        let value = serde_json::to_value(&req).expect("must serialise");
        assert!(
            value.get("modalities").is_none(),
            "modalities must be absent when None"
        );
    }

    #[test]
    fn usage_round_trips_prompt_tokens_details_via_serde() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 30, "audio_tokens": 0}
        }"#;
        let usage: Usage = serde_json::from_str(json).expect("valid OpenAI usage shape");
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens),
            Some(30)
        );
        let reser = serde_json::to_string(&usage).expect("serialization should not fail");
        assert!(reser.contains("\"cached_tokens\":30"));
    }
}
