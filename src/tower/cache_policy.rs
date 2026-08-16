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

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(
        model: &'a str,
        tenant_id: Option<&'a str>,
        stream: bool,
        metadata: &'a HashMap<String, String>,
    ) -> CachePolicyContext<'a> {
        CachePolicyContext {
            model,
            tenant_id,
            stream,
            metadata,
        }
    }

    #[test]
    fn default_policy_enables_exact_cache_only() {
        let policy = StandardCachePolicy::default();
        let md = HashMap::new();
        let decision = policy.decide(&ctx("gpt-4", None, false, &md));

        assert!(decision.use_exact);
        assert!(!decision.use_semantic, "no semantic_ttl by default");
        assert!(!decision.bypass);
        assert_eq!(decision.ttl_override, Some(Duration::from_secs(300)));
        assert!(!decision.use_streaming_replay);
    }

    #[test]
    fn bypass_on_no_store_metadata() {
        let policy = StandardCachePolicy::default();
        let mut md = HashMap::new();
        md.insert("cache".to_string(), "no-store".to_string());
        let decision = policy.decide(&ctx("gpt-4", None, false, &md));

        assert!(decision.bypass);
        assert!(
            decision.ttl_override.is_none(),
            "bypass clears TTL override"
        );
    }

    #[test]
    fn bypass_is_case_insensitive() {
        let policy = StandardCachePolicy::default();
        let mut md = HashMap::new();
        md.insert("cache".to_string(), "NO-STORE".to_string());
        let decision = policy.decide(&ctx("gpt-4", None, false, &md));
        assert!(decision.bypass, "case-insensitive match should bypass");
    }

    #[test]
    fn no_bypass_when_bypass_on_no_store_is_false() {
        let policy = StandardCachePolicy {
            bypass_on_no_store: false,
            ..Default::default()
        };
        let mut md = HashMap::new();
        md.insert("cache".to_string(), "no-store".to_string());
        let decision = policy.decide(&ctx("gpt-4", None, false, &md));
        assert!(!decision.bypass);
    }

    #[test]
    fn semantic_enabled_when_ttl_set() {
        let policy = StandardCachePolicy {
            semantic_ttl: Some(Duration::from_secs(60)),
            ..Default::default()
        };
        let md = HashMap::new();
        let decision = policy.decide(&ctx("gpt-4", None, false, &md));
        assert!(decision.use_semantic);
    }

    #[test]
    fn streaming_replay_flag_follows_stream() {
        let policy = StandardCachePolicy::default();
        let md = HashMap::new();
        let streaming = policy.decide(&ctx("gpt-4", None, true, &md));
        assert!(streaming.use_streaming_replay);

        let non_streaming = policy.decide(&ctx("gpt-4", None, false, &md));
        assert!(!non_streaming.use_streaming_replay);
    }

    #[test]
    fn default_decision_has_expected_values() {
        let d = CacheDecision::default();
        assert!(d.use_exact);
        assert!(!d.use_semantic);
        assert!(!d.use_streaming_replay);
        assert!(!d.bypass);
        assert!(d.ttl_override.is_none());
        assert!((d.similarity_threshold - 0.95).abs() < f32::EPSILON);
        assert!(d.stale_while_revalidate.is_none());
    }
}
