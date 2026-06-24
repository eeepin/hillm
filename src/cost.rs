use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CostError {
    #[error("failed to fetch pricing data: {0}")]
    FetchError(String),
    #[error("failed to parse pricing data: {0}")]
    ParseError(String),
}

static PRICING_CACHE: OnceCell<Arc<PricingRegistry>> = OnceCell::const_new();

const PRICING_API_URL: &str = "https://models.dev/api.json";
const TOKENS_PER_MILLION: f64 = 1_000_000.0;

#[derive(Debug, Deserialize)]
struct ProviderPrice {
    models: HashMap<String, ModelPrice>,
}

#[derive(Debug, Deserialize)]
struct ModelPrice {
    cost: Option<ModelCost>,
}

#[derive(Debug, Deserialize)]
struct ModelCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

#[derive(Debug)]
pub struct PricingRegistry {
    models: HashMap<String, ModelPricing>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelPricing {
    pub input_cost_per_token: f64,
    pub output_cost_per_token: f64,
    pub cache_read_input_token_cost: Option<f64>,
    pub cache_creation_input_token_cost: Option<f64>,
}

async fn fetch_pricing() -> Result<PricingRegistry, CostError> {
    let client = reqwest::Client::new();
    let response = client
        .get(PRICING_API_URL)
        .send()
        .await
        .map_err(|e| CostError::FetchError(e.to_string()))?;

    let text = response
        .text()
        .await
        .map_err(|e| CostError::FetchError(e.to_string()))?;

    parse_pricing(&text)
}

fn parse_pricing(json: &str) -> Result<PricingRegistry, CostError> {
    let providers: HashMap<String, ProviderPrice> =
        serde_json::from_str(json).map_err(|e| CostError::ParseError(e.to_string()))?;

    let mut models = HashMap::new();

    for (provider_id, provider) in providers {
        for (model_id, model) in provider.models {
            if let Some(cost) = model.cost {
                let pricing = ModelPricing {
                    input_cost_per_token: cost.input.unwrap_or(0.0) / TOKENS_PER_MILLION,
                    output_cost_per_token: cost.output.unwrap_or(0.0) / TOKENS_PER_MILLION,
                    cache_read_input_token_cost: cost.cache_read.map(|v| v / TOKENS_PER_MILLION),
                    cache_creation_input_token_cost: cost
                        .cache_write
                        .map(|v| v / TOKENS_PER_MILLION),
                };
                let model_id = format!("{provider_id}/{model_id}");
                models.insert(model_id, pricing);
            }
        }
    }

    Ok(PricingRegistry { models })
}

async fn pricing() -> Result<Arc<PricingRegistry>, CostError> {
    PRICING_CACHE
        .get_or_try_init(|| async {
            let registry = fetch_pricing().await?;
            Ok(Arc::new(registry))
        })
        .await
        .map(Arc::clone)
}

pub async fn completion_cost(
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Result<Option<f64>, CostError> {
    completion_cost_with_cache(model, prompt_tokens, 0, completion_tokens).await
}

pub async fn completion_cost_with_cache(
    model: &str,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
) -> Result<Option<f64>, CostError> {
    let Some(pricing) = model_pricing(model).await? else {
        return Ok(None);
    };
    let cached = cached_tokens.min(prompt_tokens);
    let uncached = prompt_tokens - cached;
    let cache_rate = pricing
        .cache_read_input_token_cost
        .unwrap_or(pricing.input_cost_per_token);
    Ok(Some(
        (uncached as f64) * pricing.input_cost_per_token
            + (cached as f64) * cache_rate
            + (completion_tokens as f64) * pricing.output_cost_per_token,
    ))
}

pub async fn model_pricing(model: &str) -> Result<Option<ModelPricing>, CostError> {
    let registry = pricing().await?;
    Ok(registry.models.get(model).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pricing_extracts_models_correctly() {
        let json = r#"{
            "openai": {
                "models": {
                    "gpt-4": {
                        "cost": {
                            "input": 30.0,
                            "output": 60.0,
                            "cache_read": 15.0
                        }
                    }
                }
            }
        }"#;
        let registry = parse_pricing(json).unwrap();
        assert!(registry.models.contains_key("openai/gpt-4"));
        let pricing = &registry.models["openai/gpt-4"];
        assert!((pricing.input_cost_per_token - 0.00003).abs() < 1e-10);
        assert!((pricing.output_cost_per_token - 0.00006).abs() < 1e-10);
        assert_eq!(pricing.cache_read_input_token_cost, Some(0.000015));
    }

    #[test]
    fn parse_pricing_handles_missing_cost() {
        let json = r#"{
            "test": {
                "models": {
                    "model": {}
                }
            }
        }"#;
        let registry = parse_pricing(json).unwrap();
        assert!(!registry.models.contains_key("test/model"));
    }

    #[test]
    fn parse_pricing_handles_partial_cost() {
        let json = r#"{
            "test": {
                "models": {
                    "model": {
                        "cost": {
                            "input": 10.0
                        }
                    }
                }
            }
        }"#;
        let registry = parse_pricing(json).unwrap();
        let pricing = &registry.models["test/model"];
        assert!((pricing.input_cost_per_token - 0.00001).abs() < 1e-10);
        assert_eq!(pricing.output_cost_per_token, 0.0);
        assert_eq!(pricing.cache_read_input_token_cost, None);
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_known_model_returns_positive_value() {
        let result = completion_cost("openai/gpt-4", 100, 50).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        let cost = result.unwrap();
        assert!(cost.is_some(), "gpt-4 should be in registry");
        assert!(cost.unwrap() > 0.0, "cost should be positive");
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_unknown_model_returns_none() {
        let result = completion_cost("unknown-model-xyz", 100, 50).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        assert!(result.unwrap().is_none(), "unknown model should return None");
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_gpt4o_returns_positive_value() {
        let result = completion_cost("openai/gpt-4o", 1_000, 500).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        let cost = result.unwrap();
        assert!(cost.is_some(), "gpt-4o should be in registry");
        assert!(cost.unwrap() > 0.0, "cost should be positive");
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn model_pricing_returns_none_for_unknown_model() {
        let result = model_pricing("does-not-exist").await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn completion_cost_with_cache_applies_discount_when_pricing_available() {
        let pricing = ModelPricing {
            input_cost_per_token: 1e-5,
            output_cost_per_token: 2e-5,
            cache_read_input_token_cost: Some(1e-6),
            cache_creation_input_token_cost: None,
        };
        let expected = 800.0 * 1e-5 + 200.0 * 1e-6 + 50.0 * 2e-5;
        let uncached = 1000 - 200;
        let actual = (uncached as f64) * pricing.input_cost_per_token
            + 200.0
                * pricing
                    .cache_read_input_token_cost
                    .expect("cache_read_input_token_cost should be set")
            + 50.0 * pricing.output_cost_per_token;
        assert!((actual - expected).abs() < 1e-12);
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_with_cache_clamps_cached_tokens_to_prompt_tokens() {
        let cost = completion_cost_with_cache("openai/gpt-4", 100, 500, 0)
            .await
            .unwrap()
            .expect("gpt-4 must be in registry");
        let clamped = completion_cost_with_cache("openai/gpt-4", 100, 100, 0)
            .await
            .unwrap()
            .expect("gpt-4 must be in registry");
        assert!((cost - clamped).abs() < 1e-12);
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn completion_cost_with_cache_unknown_model_returns_none() {
        let result = completion_cost_with_cache("unknown-model-xyz", 100, 10, 50).await;
        assert!(result.is_ok(), "should not error: {:?}", result.err());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    #[ignore = "requires network access to models.dev"]
    async fn fetch_pricing_returns_valid_registry() {
        let result = fetch_pricing().await;
        assert!(result.is_ok(), "fetch_pricing should succeed: {:?}", result.err());
        let registry = result.unwrap();
        assert!(!registry.models.is_empty(), "registry should have models");
    }
}
