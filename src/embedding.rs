use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::client::LlmClient;
use crate::error::HiLlmResult;
use crate::types::EmbeddingRequest;

pub trait EmbeddingProvider: Send + Sync + 'static {
    /// Embed `text` and return a dense float vector.
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<Vec<f32>>> + Send + 'a>>;

    /// The output dimensionality.
    fn dim(&self) -> usize;
}

pub struct SelfHostedEmbeddingProvider {
    client: Arc<dyn LlmClient>,
    model: String,
    dim: usize,
}

impl SelfHostedEmbeddingProvider {
    #[must_use]
    pub fn new(client: Arc<dyn LlmClient>, model: impl Into<String>, dim: usize) -> Self {
        Self {
            client,
            model: model.into(),
            dim,
        }
    }
}

impl EmbeddingProvider for SelfHostedEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<Vec<f32>>> + Send + 'a>> {
        let req = EmbeddingRequest {
            model: self.model.clone(),
            input: crate::types::EmbeddingInput::Single(text.to_owned()),
            encoding_format: None,
            dimensions: Some(self.dim as u32),
            user: None,
        };
        let client = Arc::clone(&self.client);
        Box::pin(async move {
            let resp = client.embed(req).await?;
            let vec: Vec<f32> = resp
                .data
                .into_iter()
                .next()
                .map(|obj| obj.embedding.into_iter().map(|x| x as f32).collect())
                .unwrap_or_default();
            Ok(vec)
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Clone)]
pub struct NoOpEmbeddingProvider {
    pub dim: usize,
}

impl EmbeddingProvider for NoOpEmbeddingProvider {
    fn embed<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<Vec<f32>>> + Send + 'a>> {
        Box::pin(std::future::ready(Ok(vec![0.0_f32; self.dim])))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
