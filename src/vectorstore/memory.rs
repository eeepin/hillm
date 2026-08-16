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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::SystemTime;

    fn create_test_metadata() -> VectorMetadata {
        VectorMetadata {
            cache_key: 12345,
            original_request_body: "test request".to_string(),
            tenant_id: None,
            inserted_at: SystemTime::now(),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.001, "Opposite vectors should have similarity -1.0");
    }

    #[test]
    fn cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Different length vectors should return 0.0");
    }

    #[test]
    fn cosine_similarity_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Empty vectors should return 0.0");
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should return 0.0");
    }

    #[tokio::test]
    async fn in_memory_vector_store_creation() {
        let store = InMemoryVectorStore::new(3);
        assert_eq!(store.dim(), 3);
        assert_eq!(store.entries.len(), 0);
    }

    #[tokio::test]
    async fn in_memory_vector_store_update() {
        let store = InMemoryVectorStore::new(3);
        let vec = vec![1.0, 2.0, 3.0];
        let metadata = create_test_metadata();

        let result = store.update("key1".to_string(), vec.clone(), metadata.clone()).await;
        assert!(result.is_ok());
        assert_eq!(store.entries.len(), 1);
    }

    #[tokio::test]
    async fn in_memory_vector_store_update_dimension_mismatch() {
        let store = InMemoryVectorStore::new(3);
        let vec = vec![1.0, 2.0]; // Wrong dimension
        let metadata = create_test_metadata();

        let result = store.update("key1".to_string(), vec, metadata).await;
        assert!(result.is_err(), "Should reject vector with wrong dimension");
    }

    #[tokio::test]
    async fn in_memory_vector_store_delete() {
        let store = InMemoryVectorStore::new(3);
        let vec = vec![1.0, 2.0, 3.0];
        let metadata = create_test_metadata();

        store.update("key1".to_string(), vec, metadata).await.unwrap();
        assert_eq!(store.entries.len(), 1);

        let result = store.delete("key1").await;
        assert!(result.is_ok());
        assert_eq!(store.entries.len(), 0);
    }

    #[tokio::test]
    async fn in_memory_vector_store_delete_nonexistent() {
        let store = InMemoryVectorStore::new(3);
        let result = store.delete("nonexistent").await;
        assert!(result.is_ok(), "Deleting nonexistent key should succeed");
    }

    #[tokio::test]
    async fn in_memory_vector_store_search_empty() {
        let store = InMemoryVectorStore::new(3);
        let query = vec![1.0, 2.0, 3.0];

        let matches = store.search(&query, 10, 0.0).await;
        assert_eq!(matches.len(), 0, "Search on empty store should return no matches");
    }

    #[tokio::test]
    async fn in_memory_vector_store_search_exact_match() {
        let store = InMemoryVectorStore::new(3);
        let vec = vec![1.0, 2.0, 3.0];
        let metadata = create_test_metadata();

        store.update("key1".to_string(), vec.clone(), metadata).await.unwrap();

        let matches = store.search(&vec, 10, 0.99).await;
        assert_eq!(matches.len(), 1, "Should find exact match");
        assert_eq!(matches[0].id, "key1");
        assert!((matches[0].similarity - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn in_memory_vector_store_search_multiple_matches() {
        let store = InMemoryVectorStore::new(3);

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.9, 0.1, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];

        store.update("key1".to_string(), vec1, create_test_metadata()).await.unwrap();
        store.update("key2".to_string(), vec2, create_test_metadata()).await.unwrap();
        store.update("key3".to_string(), vec3, create_test_metadata()).await.unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let matches = store.search(&query, 10, 0.0).await;

        assert_eq!(matches.len(), 3, "Should find all matches");
        // Results should be sorted by similarity descending
        assert_eq!(matches[0].id, "key1", "Exact match should be first");
    }

    #[tokio::test]
    async fn in_memory_vector_store_search_with_threshold() {
        let store = InMemoryVectorStore::new(3);

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];

        store.update("key1".to_string(), vec1, create_test_metadata()).await.unwrap();
        store.update("key2".to_string(), vec2, create_test_metadata()).await.unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let matches = store.search(&query, 10, 0.5).await;

        assert_eq!(matches.len(), 1, "Should only find matches above threshold");
        assert_eq!(matches[0].id, "key1");
    }

    #[tokio::test]
    async fn in_memory_vector_store_search_with_limit() {
        let store = InMemoryVectorStore::new(3);

        for i in 0..5 {
            let vec = vec![1.0, i as f32 * 0.1, 0.0];
            store.update(format!("key{}", i), vec, create_test_metadata()).await.unwrap();
        }

        let query = vec![1.0, 0.0, 0.0];
        let matches = store.search(&query, 3, 0.0).await;

        assert_eq!(matches.len(), 3, "Should respect k limit");
    }
}
