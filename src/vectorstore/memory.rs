//! Suitable for small amount entries like ≤ 10 k.

use std::future::Future;
use std::pin::Pin;

use dashmap::DashMap;

use super::{VectorMatch, VectorMetadata, VectorStore};
use crate::error::{HiLlmError, HiLlmResult};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

struct Entry {
    vec: Vec<f32>,
    metadata: VectorMetadata,
}

pub struct InMemoryVectorStore {
    entries: DashMap<String, Entry>,
    dim: usize,
}

impl InMemoryVectorStore {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            entries: DashMap::new(),
            dim,
        }
    }
}

impl VectorStore for InMemoryVectorStore {
    fn search<'a>(
        &'a self,
        query_vec: &'a [f32],
        k: usize,
        threshold: f32,
    ) -> Pin<Box<dyn Future<Output = Vec<VectorMatch>> + Send + 'a>> {
        let mut matches: Vec<VectorMatch> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let sim = cosine_similarity(query_vec, &entry.vec);
                if sim >= threshold {
                    Some(VectorMatch {
                        id: entry.key().clone(),
                        similarity: sim,
                        metadata: entry.metadata.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending, then truncate to k.
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);

        Box::pin(std::future::ready(matches))
    }

    fn update<'a>(
        &'a self,
        id: String,
        vec: Vec<f32>,
        metadata: VectorMetadata,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + 'a>> {
        if vec.len() != self.dim {
            return Box::pin(std::future::ready(Err(HiLlmError::InternalError {
                message: format!(
                    "vector dimension mismatch: store expects {} but received {}",
                    self.dim,
                    vec.len()
                ),
            })));
        }
        self.entries.insert(id, Entry { vec, metadata });
        Box::pin(std::future::ready(Ok(())))
    }

    fn delete<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + 'a>> {
        self.entries.remove(id);
        Box::pin(std::future::ready(Ok(())))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
