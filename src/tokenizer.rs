fn detect_tokenizer(model: &str) -> &'static str {
    if model.starts_with("gpt-4")
        || model.starts_with("gpt-3.5")
        || model.starts_with("chatgpt")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
    {
        "Xenova/gpt-4o"
    } else if model.starts_with("claude") || model.starts_with("anthropic") {
        "Xenova/claude-tokenizer"
    } else if model.starts_with("gemini") || model.starts_with("vertex_ai") {
        "google/gemma-2b"
    } else if model.starts_with("mistral") || model.starts_with("codestral") {
        "mistralai/Mistral-7B-v0.1"
    } else if model.starts_with("command") || model.starts_with("cohere") {
        "Cohere/command-r-plus-tokenizer"
    } else if model.starts_with("llama") || model.starts_with("meta-llama") {
        "meta-llama/Meta-Llama-3-8B"
    } else {
        "Xenova/gpt-4o"
    }
}
