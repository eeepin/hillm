use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use opendal::Operator;
use serde::{Deserialize, Serialize};

use super::{VectorMatch, VectorMetadata, VectorStore};
use crate::HiLlmResult;
use crate::error::HiLlmError;

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

#[derive(Serialize, Deserialize)]
struct StoredVector {
    vec: Vec<f32>,
    cache_key: u64,
    #[serde(default)]
    original_request_body: String,
    tenant_id: Option<String>,
    inserted_at_secs: u64,
    extra: HashMap<String, String>,
}

impl StoredVector {
    fn into_metadata(self) -> VectorMetadata {
        VectorMetadata {
            cache_key: self.cache_key,
            original_request_body: self.original_request_body,
            tenant_id: self.tenant_id,
            inserted_at: UNIX_EPOCH + Duration::from_secs(self.inserted_at_secs),
            extra: self.extra,
        }
    }
}

pub struct OpenDalVectorStore {
    operator: Operator,
    prefix: String,
    dim: usize,
}

impl OpenDalVectorStore {
    #[must_use]
    pub fn new(operator: Operator, prefix: impl Into<String>, dim: usize) -> Self {
        Self {
            operator,
            prefix: prefix.into(),
            dim,
        }
    }

    fn entry_path(&self, id: &str) -> String {
        format!("{}{}", self.prefix, id)
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl VectorStore for OpenDalVectorStore {
    fn search<'a>(
        &'a self,
        query_vec: &'a [f32],
        k: usize,
        threshold: f32,
    ) -> Pin<Box<dyn Future<Output = Vec<VectorMatch>> + Send + 'a>> {
        Box::pin(async move {
            let entries = match self.operator.list(&self.prefix).await {
                Ok(e) => e,
                Err(_) => return Vec::new(),
            };

            let mut matches = Vec::new();
            for entry in entries {
                let path = entry.path().to_owned();
                let bytes = match self.operator.read(&path).await {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let stored: StoredVector = match serde_json::from_slice(bytes.to_bytes().as_ref()) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let sim = cosine_similarity(query_vec, &stored.vec);
                if sim >= threshold {
                    let id = path.strip_prefix(&self.prefix).unwrap_or(&path).to_owned();
                    let metadata = stored.into_metadata();
                    matches.push(VectorMatch {
                        id,
                        similarity: sim,
                        metadata,
                    });
                }
            }

            matches.sort_by(|a, b| {
                b.similarity
                    .partial_cmp(&a.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            matches.truncate(k);
            matches
        })
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
        Box::pin(async move {
            let path = self.entry_path(&id);
            let stored = StoredVector {
                vec,
                cache_key: metadata.cache_key,
                original_request_body: metadata.original_request_body,
                tenant_id: metadata.tenant_id,
                inserted_at_secs: Self::now_secs(),
                extra: metadata.extra,
            };
            let bytes = serde_json::to_vec(&stored).map_err(|e| HiLlmError::InternalError {
                message: format!("vector store: serialization failed: {e}"),
            })?;
            self.operator
                .write(&path, bytes)
                .await
                .map(|_| ())
                .map_err(|e| HiLlmError::InternalError {
                    message: format!("vector store: write failed for '{path}': {e}"),
                })
        })
    }

    fn delete<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let path = self.entry_path(id);
            match self.operator.delete(&path).await {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == opendal::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(HiLlmError::InternalError {
                    message: format!("vector store: delete failed for '{path}': {e}"),
                }),
            }
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
