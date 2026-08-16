use crate::error::HiLlmError;
use crate::error::HiLlmResult;
use crate::provider::anthropic::codec::AnthropicMessagesCodec;
use crate::provider::{APIType, Provider, codec::APITypeCodec, registry_get};
use std::borrow::Cow;

static ANTHROPIC_EXTRA_HEADERS: &[(&str, &str)] = &[("anthropic-version", "2023-06-01")];
const BETA_COMPUTER_USE: &str = "computer-use-2025-01-24";
const BETA_WEB_SEARCH: &str = "web-search-2025-03-05";
const BETA_CODE_EXECUTION: &str = "code-execution-2025-05-22";
const BETA_THINKING: &str = "thinking-2025-04-14";
const BETA_PROMPT_CACHING: &str = "prompt-caching-2024-07-31";
const BETA_PDFS: &str = "pdfs-2024-09-25";

pub mod codec;
pub mod compat;

/// Recursively check if any value in the JSON body contains a `cache_control` field.
fn body_contains_cache_control(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            if map.contains_key("cache_control") {
                return true;
            }
            map.values().any(body_contains_cache_control)
        }
        serde_json::Value::Array(arr) => arr.iter().any(body_contains_cache_control),
        _ => false,
    }
}

/// Recursively check if any value in the JSON body contains a document content block.
fn body_contains_document_block(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("document") {
                return true;
            }
            map.values().any(body_contains_document_block)
        }
        serde_json::Value::Array(arr) => arr.iter().any(body_contains_document_block),
        _ => false,
    }
}

/// Anthropic provider
pub struct AnthropicProvider {
    base_url: String,
}

/// Returns the Anthropic base URL, honoring the `ANTHROPIC_BASE_URL`
/// environment variable when it is set to a non-empty value.
pub(crate) fn anthropic_base_url() -> String {
    std::env::var("ANTHROPIC_BASE_URL")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "https://api.anthropic.com/v1".to_owned())
}

impl AnthropicProvider {
    /// Creates a provider instance, resolving `ANTHROPIC_BASE_URL`.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            base_url: anthropic_base_url(),
        }
    }

    /// Creates a provider instance bound to `api_type`, failing with
    /// [`HiLlmError::APITypeUnsupported`] unless it is
    /// [`APIType::AnthropicMessages`].
    pub(crate) fn with_api_type(api_type: APIType) -> HiLlmResult<Self> {
        if api_type != APIType::AnthropicMessages {
            return Err(HiLlmError::APITypeUnsupported {
                api_type: api_type.to_string(),
                provider: "anthropic".to_string(),
            });
        }
        Ok(Self::new())
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_header<'a>(&'a self, api_key: &'a str) -> Option<(Cow<'static, str>, Cow<'a, str>)> {
        Some((Cow::Borrowed("x-api-key"), Cow::Borrowed(api_key)))
    }

    fn extra_headers(&self) -> &'static [(&'static str, &'static str)] {
        ANTHROPIC_EXTRA_HEADERS
    }

    fn dynamic_headers(&self, body: &serde_json::Value) -> Vec<(String, String)> {
        let mut betas: Vec<&str> = Vec::new();

        // Check for extended thinking.
        if body.get("thinking").is_some() {
            betas.push(BETA_THINKING);
        }

        // Check for hosted tools in the tools array.
        if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
            for tool in tools {
                let tool_type = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match tool_type {
                    "computer_20241022" | "computer_use_20250124"
                        if !betas.contains(&BETA_COMPUTER_USE) =>
                    {
                        betas.push(BETA_COMPUTER_USE);
                    }
                    "web_search_20250305" if !betas.contains(&BETA_WEB_SEARCH) => {
                        betas.push(BETA_WEB_SEARCH);
                    }
                    "code_execution_20250522" if !betas.contains(&BETA_CODE_EXECUTION) => {
                        betas.push(BETA_CODE_EXECUTION);
                    }
                    _ => {}
                }
            }
        }

        // Check for prompt caching: any `cache_control` field anywhere in the body.
        if body_contains_cache_control(body) && !betas.contains(&BETA_PROMPT_CACHING) {
            betas.push(BETA_PROMPT_CACHING);
        }

        // Check for PDF/document content blocks.
        if body_contains_document_block(body) && !betas.contains(&BETA_PDFS) {
            betas.push(BETA_PDFS);
        }

        if betas.is_empty() {
            vec![]
        } else {
            vec![("anthropic-beta".to_owned(), betas.join(","))]
        }
    }

    fn matches_model(&self, model: &str) -> bool {
        registry_get().is_some_and(|reg| {
            reg.get("anthropic")
                .is_some_and(|p| p.models.contains_key(model))
        })
    }

    fn available_api_types(&self) -> Vec<APIType> {
        vec![APIType::AnthropicMessages]
    }

    fn api_type(&self) -> APIType {
        APIType::AnthropicMessages
    }

    fn codec_for(&self, api_type: APIType) -> Option<Box<dyn APITypeCodec>> {
        match api_type {
            APIType::AnthropicMessages => Some(Box::new(AnthropicMessagesCodec)),
            _ => None,
        }
    }

    fn env_var(&self) -> Option<&'static str> {
        Some("ANTHROPIC_API_KEY")
    }
}
