use std::collections::HashMap;
use std::future::Future;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use regex::Regex;

use crate::client::LlmClient;
use crate::types::{
    ChatCompletionRequest, EmbeddingInput, EmbeddingRequest, Message, MessageContent,
    SystemMessage, UserMessage,
};

#[inline]
fn record_classify_duration(tier: &'static str, duration_secs: f64) {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::KeyValue;
        if let Some(meter) = super::metrics::global_meter() {
            meter
                .f64_histogram("gen_ai.route.classify.duration")
                .with_description("Semantic routing classifier latency per tier")
                .with_unit("s")
                .build()
                .record(
                    duration_secs,
                    &[KeyValue::new("route.classifier.tier", tier)],
                );
        }
    }
    let _ = (tier, duration_secs);
}

#[inline]
fn record_classify_hit(tier: &'static str) {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::KeyValue;
        if let Some(meter) = super::metrics::global_meter() {
            meter
                .u64_counter("gen_ai.route.classify.tier.hit")
                .with_description("Semantic routing classifier tier hits")
                .build()
                .add(1, &[KeyValue::new("route.classifier.tier", tier)]);
        }
    }
    let _ = tier;
}

pub struct ClassifyContext<'a> {
    pub prompt: &'a str,
    pub system_prompt: Option<&'a str>,
    pub metadata: &'a HashMap<String, String>,
    pub available_models: &'a [String],
}

pub trait RouteClassifier: Send + Sync + 'static {
    fn classify<'a>(
        &'a self,
        ctx: &'a ClassifyContext<'a>,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>>;

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}

pub struct KeywordClassifier {
    rules: Vec<(Regex, String)>,
}

impl KeywordClassifier {
    pub fn new(rules: Vec<(Regex, String)>) -> Self {
        Self { rules }
    }
}

impl RouteClassifier for KeywordClassifier {
    fn classify<'a>(
        &'a self,
        ctx: &'a ClassifyContext<'a>,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>> {
        let start = Instant::now();
        let result = self
            .rules
            .iter()
            .find(|(pattern, _)| pattern.is_match(ctx.prompt))
            .map(|(_, model)| model.clone());

        if result.is_some() {
            record_classify_duration("keyword", start.elapsed().as_secs_f64());
            record_classify_hit("keyword");
        }

        Box::pin(async move { result })
    }

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}

pub struct IntentPrototype {
    pub name: String,
    pub embedding: Vec<f64>,
    pub model: String,
}

pub struct EmbeddingSimilarityClassifier {
    client: Arc<dyn LlmClient>,
    embedding_model: String,
    prototypes: Vec<IntentPrototype>,
    threshold: f64,
}

impl EmbeddingSimilarityClassifier {
    pub fn new(
        client: Arc<dyn LlmClient>,
        embedding_model: impl Into<String>,
        prototypes: Vec<IntentPrototype>,
        threshold: f64,
    ) -> Self {
        Self {
            client,
            embedding_model: embedding_model.into(),
            prototypes,
            threshold,
        }
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

impl RouteClassifier for EmbeddingSimilarityClassifier {
    fn classify<'a>(
        &'a self,
        ctx: &'a ClassifyContext<'a>,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();

            let embed_req = EmbeddingRequest {
                model: self.embedding_model.clone(),
                input: EmbeddingInput::Single(ctx.prompt.to_owned()),
                encoding_format: None,
                dimensions: None,
                user: None,
            };

            let resp = match self.client.embed(embed_req).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(error = %e, "embedding classifier: embed request failed; deferring");
                    return None;
                }
            };

            let prompt_vec = match resp.data.into_iter().next() {
                Some(obj) => obj.embedding,
                None => {
                    tracing::warn!("embedding classifier: empty embedding response; deferring");
                    return None;
                }
            };

            let best = self
                .prototypes
                .iter()
                .map(|p| (cosine_similarity(&prompt_vec, &p.embedding), p))
                .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            match best {
                Some((score, prototype)) if score >= self.threshold => {
                    record_classify_duration("embedding", start.elapsed().as_secs_f64());
                    record_classify_hit("embedding");
                    tracing::debug!(
                        intent = %prototype.name,
                        model = %prototype.model,
                        score,
                        "embedding classifier: routed to intent prototype"
                    );
                    Some(prototype.model.clone())
                }
                Some((score, _)) => {
                    tracing::debug!(
                        score,
                        threshold = self.threshold,
                        "embedding classifier: best score below threshold; deferring"
                    );
                    None
                }
                None => None,
            }
        })
    }

    fn confidence_threshold(&self) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let t = self.threshold as f32;
        t
    }
}

pub struct LlmClassifier {
    client: Arc<dyn LlmClient>,
    model: String,
    system_prompt: String,
}

impl LlmClassifier {
    pub fn new(
        client: Arc<dyn LlmClient>,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            system_prompt: system_prompt.into(),
        }
    }

    fn build_routing_prompt(ctx: &ClassifyContext<'_>) -> String {
        let models = ctx.available_models.join(", ");
        format!(
            "Available models: [{models}]\n\
             User prompt: {prompt}\n\n\
             Respond with ONLY a JSON object in this exact format: {{\"model\": \"<model_id>\"}}\n\
             Choose the most appropriate model from the available models list.",
            models = models,
            prompt = ctx.prompt,
        )
    }

    fn parse_model_from_response(text: &str) -> Option<String> {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        if end < start {
            return None;
        }
        let json_str = &text[start..=end];

        let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
        value
            .get("model")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned)
    }
}

impl RouteClassifier for LlmClassifier {
    fn classify<'a>(
        &'a self,
        ctx: &'a ClassifyContext<'a>,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(async move {
            let start = Instant::now();

            let routing_prompt = Self::build_routing_prompt(ctx);
            let req = ChatCompletionRequest {
                model: self.model.clone(),
                messages: vec![
                    Message::System(SystemMessage {
                        content: MessageContent::Text(self.system_prompt.clone()),
                        name: None,
                    }),
                    Message::User(UserMessage {
                        content: MessageContent::Text(routing_prompt),
                        name: None,
                    }),
                ],
                ..Default::default()
            };

            let resp = match self.client.chat(req).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(error = %e, "llm classifier: chat call failed; deferring");
                    return None;
                }
            };

            let text = resp
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.text())?;

            let model_id = Self::parse_model_from_response(&text);

            if model_id.is_some() {
                record_classify_duration("llm", start.elapsed().as_secs_f64());
                record_classify_hit("llm");
                tracing::debug!(
                    model = ?model_id,
                    "llm classifier: parsed routing decision"
                );
            } else {
                tracing::warn!(
                    raw_response = %text,
                    "llm classifier: could not parse model from response; deferring"
                );
            }

            model_id.filter(|m| ctx.available_models.contains(m))
        })
    }

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}

pub struct CascadeClassifier {
    classifiers: Vec<Arc<dyn RouteClassifier>>,
}

impl CascadeClassifier {
    pub fn new(classifiers: Vec<Arc<dyn RouteClassifier>>) -> Self {
        Self { classifiers }
    }
}

impl RouteClassifier for CascadeClassifier {
    fn classify<'a>(
        &'a self,
        ctx: &'a ClassifyContext<'a>,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(async move {
            for classifier in &self.classifiers {
                if let Some(model) = classifier.classify(ctx).await {
                    return Some(model);
                }
            }
            None
        })
    }

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}

struct VerdictEntry {
    model: String,
    inserted_at: Instant,
}

pub struct ClassifierVerdictCache<C> {
    inner: C,
    ttl: Duration,
    cache: Arc<RwLock<HashMap<u64, VerdictEntry>>>,
}

impl<C: RouteClassifier> ClassifierVerdictCache<C> {
    pub const DEFAULT_TTL: Duration = Duration::from_secs(3600);

    pub fn new(inner: C) -> Self {
        Self::with_ttl(inner, Self::DEFAULT_TTL)
    }

    pub fn with_ttl(inner: C, ttl: Duration) -> Self {
        Self {
            inner,
            ttl,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn cache_key(ctx: &ClassifyContext<'_>) -> u64 {
        let mut h = DefaultHasher::new();
        ctx.prompt.hash(&mut h);
        ctx.system_prompt.hash(&mut h);
        h.finish()
    }

    fn get_cached(&self, key: u64) -> Option<String> {
        let cache = self.cache.read().ok()?;
        let entry = cache.get(&key)?;
        if entry.inserted_at.elapsed() > self.ttl {
            return None;
        }
        Some(entry.model.clone())
    }

    fn put_cached(&self, key: u64, model: String) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(
                key,
                VerdictEntry {
                    model,
                    inserted_at: Instant::now(),
                },
            );
        }
    }
}

impl<C: RouteClassifier> RouteClassifier for ClassifierVerdictCache<C> {
    fn classify<'a>(
        &'a self,
        ctx: &'a ClassifyContext<'a>,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(async move {
            let key = Self::cache_key(ctx);

            if let Some(model) = self.get_cached(key) {
                record_classify_hit("cache");
                tracing::debug!(%model, "classifier verdict cache hit");
                return Some(model);
            }

            let result = self.inner.classify(ctx).await;

            if let Some(ref model) = result {
                self.put_cached(key, model.clone());
            }

            result
        })
    }

    fn confidence_threshold(&self) -> f32 {
        self.inner.confidence_threshold()
    }
}
