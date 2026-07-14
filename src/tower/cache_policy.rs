use std::collections::HashMap;
use std::time::Duration;

pub struct CachePolicyContext<'a> {
    pub model: &'a str,
    pub tenant_id: Option<&'a str>,
    pub stream: bool,
    pub metadata: &'a HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CacheDecision {
    pub use_exact: bool,
    pub use_semantic: bool,
    pub use_streaming_replay: bool,
    pub bypass: bool,
    pub ttl_override: Option<Duration>,
    pub similarity_threshold: f32,
    pub stale_while_revalidate: Option<Duration>,
}

impl Default for CacheDecision {
    fn default() -> Self {
        Self {
            use_exact: true,
            use_semantic: false,
            use_streaming_replay: false,
            bypass: false,
            ttl_override: None,
            similarity_threshold: 0.95,
            stale_while_revalidate: None,
        }
    }
}

pub trait CachePolicy: Send + Sync + 'static {
    fn decide(&self, ctx: &CachePolicyContext<'_>) -> CacheDecision;
}

#[derive(Debug, Clone)]
pub struct StandardCachePolicy {
    pub exact_ttl: Duration,
    pub semantic_ttl: Option<Duration>,
    pub similarity_threshold: f32,
    pub bypass_on_no_store: bool,
}

impl Default for StandardCachePolicy {
    fn default() -> Self {
        Self {
            exact_ttl: Duration::from_secs(300),
            semantic_ttl: None,
            similarity_threshold: 0.95,
            bypass_on_no_store: true,
        }
    }
}

impl CachePolicy for StandardCachePolicy {
    fn decide(&self, ctx: &CachePolicyContext<'_>) -> CacheDecision {
        let bypass = self.bypass_on_no_store
            && ctx
                .metadata
                .get("cache")
                .is_some_and(|v| v.eq_ignore_ascii_case("no-store"));

        CacheDecision {
            use_exact: true,
            use_semantic: self.semantic_ttl.is_some(),
            use_streaming_replay: ctx.stream,
            bypass,
            ttl_override: if bypass { None } else { Some(self.exact_ttl) },
            similarity_threshold: self.similarity_threshold,
            stale_while_revalidate: None,
        }
    }
}
