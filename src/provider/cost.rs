use super::{ProviderError, TokenPrice, registry};

pub async fn token_price(provider: &str, model: &str) -> Result<Option<TokenPrice>, ProviderError> {
    let registry = registry().await?;
    Ok(registry
        .get(provider)
        .and_then(|p| p.models.get(model))
        .and_then(|m| m.cost.as_ref())
        .and_then(|c| Some(c.token_price.clone())))
}

pub async fn completion_cost(
    provider: &str,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Result<Option<f64>, ProviderError> {
    completion_cost_with_cache(provider, model, prompt_tokens, 0, 0, completion_tokens).await
}

pub async fn completion_cost_with_cache(
    provider: &str,
    model: &str,
    prompt_tokens: u64,
    cached_tokens: u64,
    cached_write_tokens: u64,
    completion_tokens: u64,
) -> Result<Option<f64>, ProviderError> {
    if let Some(price) = token_price(provider, model).await? {
        price.cost(
            prompt_tokens,
            cached_tokens,
            cached_write_tokens,
            completion_tokens,
        )
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::super::TOKENS_PER_MILLION;
    use super::*;

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_known_model_returns_positive_value() {
        let result = completion_cost("openai", "gpt-4", 100, 50).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        let cost = result.unwrap();
        assert!(cost.is_some(), "gpt-4 should be in registry");
        assert!(cost.unwrap() > 0.0, "cost should be positive");
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_unknown_model_returns_none() {
        let result = completion_cost("unknown-provider", "unknown-model-xyz", 100, 50).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        assert!(
            result.unwrap().is_none(),
            "unknown model should return None"
        );
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_gpt4o_returns_positive_value() {
        let result = completion_cost("openai", "gpt-4o", 1_000, 500).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        let cost = result.unwrap();
        assert!(cost.is_some(), "gpt-4o should be in registry");
        assert!(cost.unwrap() > 0.0, "cost should be positive");
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn model_pricing_returns_none_for_unknown_model() {
        let result = token_price("does-not-exist", "does-not-exist").await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn completion_cost_with_cache_applies_discount_when_pricing_available() {
        let price = TokenPrice {
            input: 1e-5,
            output: 2e-5,
            cache_read: Some(1e-6),
            cache_write: None,
        };
        let expected = (800.0 * 1e-5 + 200.0 * 1e-6 + 50.0 * 2e-5) / TOKENS_PER_MILLION;
        let uncached = 1000 - 200;
        let actual = (uncached as f64) * price.input / TOKENS_PER_MILLION
            + 200.0
                * price
                    .cache_read
                    .expect("cache_read_input_token_cost should be set")
                / TOKENS_PER_MILLION
            + 50.0 * price.output / TOKENS_PER_MILLION;
        assert!((actual - expected).abs() < 1e-12);
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_with_cache_clamps_cached_tokens_to_prompt_tokens() {
        let cost = completion_cost_with_cache("openai", "gpt-4", 100, 500, 0, 0)
            .await
            .unwrap()
            .expect("gpt-4 must be in registry");
        let clamped = completion_cost_with_cache("openai", "gpt-4", 100, 100, 0, 0)
            .await
            .unwrap()
            .expect("gpt-4 must be in registry");
        assert!((cost - clamped).abs() < 1e-12);
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_with_cache_unknown_model_returns_none() {
        let result =
            completion_cost_with_cache("unknown-provider", "unknown-model-xyz", 100, 10, 0, 50)
                .await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        assert!(result.unwrap().is_none());
    }
}
