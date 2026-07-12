use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

use crate::error::{HiLlmError, HiLlmResult};

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
