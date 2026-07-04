use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocument>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object { text: String },
}

impl Default for RerankDocument {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// Response from the rerank endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    pub id: Option<String>,
    pub results: Vec<RerankResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: u32,
    pub relevance_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankResultDocument>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResultDocument {
    pub text: String,
}
