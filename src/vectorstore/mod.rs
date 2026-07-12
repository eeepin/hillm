pub mod memory;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::SystemTime;

use crate::error::HiLlmResult;

#[derive(Debug, Clone)]
pub struct VectorMetadata {
    pub cache_key: u64,
    pub original_request_body: String,
    pub tenant_id: Option<String>,
    pub inserted_at: SystemTime,
    pub extra: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct VectorMatch {
    pub id: String,
    pub similarity: f32,
    pub metadata: VectorMetadata,
}

pub trait VectorStore: Send + Sync + 'static {
    fn search<'a>(
        &'a self,
        query_vec: &'a [f32],
        k: usize,
        threshold: f32,
    ) -> Pin<Box<dyn Future<Output = Vec<VectorMatch>> + Send + 'a>>;

    fn update<'a>(
        &'a self,
        id: String,
        vec: Vec<f32>,
        metadata: VectorMetadata,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + 'a>>;

    fn delete<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = HiLlmResult<()>> + Send + 'a>>;

    fn dim(&self) -> usize;
}
