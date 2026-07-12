use serde::{Deserialize, Serialize};

use super::Usage;
use crate::provider::cost;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingFormat {
    Float,
    Base64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EmbeddingFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl Default for EmbeddingInput {
    fn default() -> Self {
        Self::Single(String::new())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl EmbeddingResponse {
    pub fn estimated_cost(&self, provider: &str) -> Option<f64> {
        let usage = self.usage.as_ref()?;
        cost::completion_cost(
            provider,
            &self.model,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        .unwrap_or(None)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: u32,
}
