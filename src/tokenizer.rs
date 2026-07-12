use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

use crate::error::{HiLlmError, HiLlmResult};
use crate::types::{ChatCompletionRequest, ContentPart, Message, MessageContent};

static TOKENIZER_CACHE: OnceCell<ArcSwap<HashMap<String, Arc<Tokenizer>>>> = OnceCell::const_new();

async fn detect_tokenizer(model: &str) -> HiLlmResult<String> {
    // Try to find the model's tokenizer via HuggingFace Hub API
    if let Ok(api) = hf_hub::api::tokio::Api::new() {
        let repo = api.model(model.to_string());
        // tokenizer_config.json is a small metadata file; if it exists the model
        // either has its own tokenizer or references one via tokenizer_name.
        if let Ok(path) = repo.get("tokenizer_config.json").await {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    // Some models reference a separate tokenizer model
                    if let Some(name) = config.get("tokenizer_name").and_then(|v| v.as_str()) {
                        // Handle relative references: "author/model" → use as-is,
                        // "name_only" → prepend the model's author
                        if !name.contains('/') {
                            if let Some(author) = model.split('/').next() {
                                return Ok(format!("{author}/{name}"));
                            }
                        }
                        return Ok(name.to_string());
                    }
                    // Model has its own tokenizer files
                    return Ok(model.to_string());
                }
            }
        }
    }

    // Fallback for models not on HuggingFace Hub
    Ok(match model {
        m if m.starts_with("gpt-4")
            || m.starts_with("gpt-3.5")
            || m.starts_with("chatgpt")
            || m.starts_with("o1")
            || m.starts_with("o3")
            || m.starts_with("o4") =>
        {
            "Xenova/gpt-4o"
        }
        m if m.starts_with("claude") || m.starts_with("anthropic") => "Xenova/claude-tokenizer",
        m if m.starts_with("gemini") || m.starts_with("vertex_ai") => "google/gemma-2b",
        m if m.starts_with("mistral") || m.starts_with("codestral") => "mistralai/Mistral-7B-v0.1",
        m if m.starts_with("command") || m.starts_with("cohere") => {
            "Cohere/command-r-plus-tokenizer"
        }
        m if m.starts_with("llama") || m.starts_with("meta-llama") => "meta-llama/Meta-Llama-3-8B",
        _ => "Xenova/gpt-4o",
    }
    .to_string())
}

async fn get_tokenizer(model: &str) -> HiLlmResult<Arc<Tokenizer>> {
    let tokenizer_id = detect_tokenizer(model).await?;

    // Ensure cache is initialized (once)
    let cache = TOKENIZER_CACHE
        .get_or_init(|| async { ArcSwap::new(Arc::new(HashMap::new())) })
        .await;

    // Fast path: cache already has this tokenizer
    if let Some(tok) = cache.load().get(&tokenizer_id) {
        return Ok(Arc::clone(tok));
    }

    // Load from HuggingFace
    let tokenizer = Arc::new(
        Tokenizer::from_pretrained(&tokenizer_id, None).map_err(|e| HiLlmError::BadRequest {
            message: format!("Failed to load tokenizer '{tokenizer_id}': {e} from HuggingFace"),
            status: 400,
        })?,
    );

    // Atomically insert the new tokenizer into the cache map
    let mut new_map = HashMap::clone(&cache.load());
    new_map.insert(tokenizer_id, Arc::clone(&tokenizer));
    cache.store(Arc::new(new_map));

    Ok(tokenizer)
}

pub async fn count_tokens(model: &str, text: &str) -> HiLlmResult<usize> {
    let tokenizer = get_tokenizer(model).await?;
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| HiLlmError::BadRequest {
            message: format!("Tokenization failed: {e}"),
            status: 400,
        })?;
    Ok(encoding.get_ids().len())
}

fn content_part_text(part: &ContentPart) -> Option<&str> {
    match part {
        ContentPart::Text { text } => Some(text.as_str()),
        ContentPart::ImageUrl { .. }
        | ContentPart::Document { .. }
        | ContentPart::InputAudio { .. }
        | ContentPart::Refusal { .. }
        | ContentPart::OutputImage { .. }
        | ContentPart::OutputAudio { .. } => None,
    }
}

fn count_message_content_tokens(
    tokenizer: &Tokenizer,
    content: &MessageContent,
) -> HiLlmResult<usize> {
    let mut total = 0usize;
    match &content {
        MessageContent::Text(t) => total += encode(tokenizer, t)?,
        MessageContent::Parts(parts) => {
            for part in parts {
                if let Some(text) = content_part_text(part) {
                    total += encode(tokenizer, text)?;
                }
            }
        }
    }
    Ok(total)
}

fn encode(tokenizer: &Tokenizer, text: &str) -> HiLlmResult<usize> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| HiLlmError::BadRequest {
            message: format!("Tokenization failed: {e}"),
            status: 400,
        })?;
    Ok(encoding.get_ids().len())
}

/// Count tokens for a full [`ChatCompletionRequest`].
pub async fn count_request_tokens(model: &str, req: &ChatCompletionRequest) -> HiLlmResult<usize> {
    let tokenizer = get_tokenizer(model).await?;
    let mut total = 0usize;

    for msg in &req.messages {
        match msg {
            Message::System(m) => total += count_message_content_tokens(&tokenizer, &m.content)?,
            Message::User(m) => total += count_message_content_tokens(&tokenizer, &m.content)?,
            Message::Assistant(m) => {
                if m.content.is_none()
                    && let Some(ref calls) = m.tool_calls
                {
                    for call in calls {
                        total += encode(&tokenizer, call.function.arguments.as_str())?;
                    }
                } else {
                    total += count_message_content_tokens(
                        &tokenizer,
                        &m.content
                            .as_ref()
                            .unwrap_or(&MessageContent::Text(String::default())),
                    )?;
                }
            }
            Message::Tool(m) => total += count_message_content_tokens(&tokenizer, &m.content)?,
            Message::Developer(m) => total += count_message_content_tokens(&tokenizer, &m.content)?,
        }
    }

    // About 4 tokens of per-message overhead (for role, separators, and formatting metadata)
    total += req.messages.len() * 4;

    Ok(total)
}
