use std::sync::{Arc, OnceLock, RwLock};

use super::{Guardrail, GuardrailContext, GuardrailDecision, GuardrailStage};

pub struct GuardrailRegistry {
    guardrails: Vec<Arc<dyn Guardrail>>,
}

impl std::fmt::Debug for GuardrailRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.guardrails.iter().map(|g| g.name()).collect();
        f.debug_struct("GuardrailRegistry")
            .field("guardrails", &names)
            .finish()
    }
}

impl Default for GuardrailRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardrailRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            guardrails: Vec::new(),
        }
    }

    pub fn register(&mut self, guardrail: Arc<dyn Guardrail>) {
        self.guardrails.push(guardrail);
    }

    pub fn clear(&mut self) {
        self.guardrails.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Guardrail>> {
        self.guardrails.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.guardrails.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.guardrails.is_empty()
    }

    pub async fn run_stage(
        &self,
        stage: GuardrailStage,
        ctx: &GuardrailContext<'_>,
    ) -> GuardrailDecision {
        let mut last_mutation: Option<GuardrailDecision> = None;

        for guardrail in &self.guardrails {
            if !guardrail.supported_stages().contains(&stage) {
                continue;
            }

            let decision = guardrail.check(stage, ctx).await;
            match decision {
                GuardrailDecision::Allow => {}
                GuardrailDecision::Block { .. } => return decision,
                GuardrailDecision::Mutate { .. } => {
                    last_mutation = Some(decision);
                }
            }
        }

        last_mutation.unwrap_or(GuardrailDecision::Allow)
    }
}

static GLOBAL_REGISTRY: OnceLock<RwLock<GuardrailRegistry>> = OnceLock::new();

fn global_lock() -> &'static RwLock<GuardrailRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| RwLock::new(GuardrailRegistry::new()))
}

pub fn register(guardrail: Arc<dyn Guardrail>) {
    global_lock()
        .write()
        .expect("global guardrail registry lock poisoned")
        .register(guardrail);
}

pub fn clear() {
    global_lock()
        .write()
        .expect("global guardrail registry lock poisoned")
        .clear();
}

pub async fn run_stage(stage: GuardrailStage, ctx: &GuardrailContext<'_>) -> GuardrailDecision {
    let guardrails: Vec<Arc<dyn Guardrail>> = global_lock()
        .read()
        .expect("global guardrail registry lock poisoned")
        .guardrails
        .clone();

    let mut last_mutation: Option<GuardrailDecision> = None;

    for guardrail in &guardrails {
        if !guardrail.supported_stages().contains(&stage) {
            continue;
        }

        let decision = guardrail.check(stage, ctx).await;
        match decision {
            GuardrailDecision::Allow => {}
            GuardrailDecision::Block { .. } => return decision,
            GuardrailDecision::Mutate { .. } => {
                last_mutation = Some(decision);
            }
        }
    }

    last_mutation.unwrap_or(GuardrailDecision::Allow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockGuardrail {
        name: &'static str,
        stages: Vec<GuardrailStage>,
        call_count: Arc<AtomicUsize>,
        decision: GuardrailDecision,
    }

    impl MockGuardrail {
        fn new(
            name: &'static str,
            stages: Vec<GuardrailStage>,
            decision: GuardrailDecision,
        ) -> Self {
            Self {
                name,
                stages,
                call_count: Arc::new(AtomicUsize::new(0)),
                decision,
            }
        }

        fn get_call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl Guardrail for MockGuardrail {
        fn name(&self) -> &'static str {
            self.name
        }

        fn supported_stages(&self) -> &'static [GuardrailStage] {
            // Leak the vector to get a 'static slice
            Box::leak(self.stages.clone().into_boxed_slice())
        }

        fn check<'a>(
            &'a self,
            _stage: GuardrailStage,
            _ctx: &'a GuardrailContext<'a>,
        ) -> Pin<Box<dyn Future<Output = GuardrailDecision> + Send + 'a>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Box::pin(std::future::ready(self.decision.clone()))
        }
    }

    impl Clone for GuardrailDecision {
        fn clone(&self) -> Self {
            match self {
                Self::Allow => Self::Allow,
                Self::Block { reason, code } => Self::Block {
                    reason: reason.clone(),
                    code: *code,
                },
                Self::Mutate { new_payload } => Self::Mutate {
                    new_payload: new_payload.clone(),
                },
            }
        }
    }

    fn create_test_context() -> GuardrailContext<'static> {
        static REQUEST: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        static METADATA: std::sync::OnceLock<HashMap<String, String>> = std::sync::OnceLock::new();

        let request = REQUEST.get_or_init(|| serde_json::json!({"test": "request"}));
        let metadata = METADATA.get_or_init(HashMap::new);

        GuardrailContext {
            request,
            response: None,
            chunk: None,
            metadata,
        }
    }

    #[test]
    fn guardrail_registry_creation() {
        let registry = GuardrailRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn guardrail_registry_default() {
        let registry = GuardrailRegistry::default();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn guardrail_registry_register() {
        let mut registry = GuardrailRegistry::new();
        let guardrail = Arc::new(MockGuardrail::new(
            "test",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        registry.register(guardrail);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn guardrail_registry_clear() {
        let mut registry = GuardrailRegistry::new();
        let guardrail = Arc::new(MockGuardrail::new(
            "test",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        registry.register(guardrail);
        assert_eq!(registry.len(), 1);

        registry.clear();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn guardrail_registry_iter() {
        let mut registry = GuardrailRegistry::new();
        let g1 = Arc::new(MockGuardrail::new(
            "test1",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        let g2 = Arc::new(MockGuardrail::new(
            "test2",
            vec![GuardrailStage::Output],
            GuardrailDecision::Allow,
        ));
        registry.register(g1);
        registry.register(g2);

        let names: Vec<&str> = registry.iter().map(|g| g.name()).collect();
        assert_eq!(names, vec!["test1", "test2"]);
    }

    #[tokio::test]
    async fn guardrail_registry_run_stage_allow() {
        let mut registry = GuardrailRegistry::new();
        let guardrail = Arc::new(MockGuardrail::new(
            "test",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        registry.register(guardrail.clone());

        let ctx = create_test_context();
        let decision = registry.run_stage(GuardrailStage::Input, &ctx).await;

        assert!(decision.is_allow());
        assert_eq!(guardrail.get_call_count(), 1);
    }

    #[tokio::test]
    async fn guardrail_registry_run_stage_block() {
        let mut registry = GuardrailRegistry::new();
        let guardrail = Arc::new(MockGuardrail::new(
            "test",
            vec![GuardrailStage::Input],
            GuardrailDecision::Block {
                reason: "blocked".to_string(),
                code: 403,
            },
        ));
        registry.register(guardrail.clone());

        let ctx = create_test_context();
        let decision = registry.run_stage(GuardrailStage::Input, &ctx).await;

        assert!(decision.is_block());
        assert_eq!(guardrail.get_call_count(), 1);
    }

    #[tokio::test]
    async fn guardrail_registry_run_stage_mutate() {
        let mut registry = GuardrailRegistry::new();
        let mutated_payload = serde_json::json!({"mutated": true});
        let guardrail = Arc::new(MockGuardrail::new(
            "test",
            vec![GuardrailStage::Input],
            GuardrailDecision::Mutate {
                new_payload: mutated_payload.clone(),
            },
        ));
        registry.register(guardrail.clone());

        let ctx = create_test_context();
        let decision = registry.run_stage(GuardrailStage::Input, &ctx).await;

        match decision {
            GuardrailDecision::Mutate { new_payload } => {
                assert_eq!(new_payload, mutated_payload);
            }
            _ => panic!("Expected Mutate decision"),
        }
        assert_eq!(guardrail.get_call_count(), 1);
    }

    #[tokio::test]
    async fn guardrail_registry_run_stage_skips_unsupported() {
        let mut registry = GuardrailRegistry::new();
        let guardrail = Arc::new(MockGuardrail::new(
            "test",
            vec![GuardrailStage::Output], // Only supports Output, not Input
            GuardrailDecision::Allow,
        ));
        registry.register(guardrail.clone());

        let ctx = create_test_context();
        let decision = registry.run_stage(GuardrailStage::Input, &ctx).await;

        assert!(decision.is_allow());
        assert_eq!(guardrail.get_call_count(), 0); // Should not be called
    }

    #[tokio::test]
    async fn guardrail_registry_run_stage_multiple_guardrails() {
        let mut registry = GuardrailRegistry::new();
        let g1 = Arc::new(MockGuardrail::new(
            "test1",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        let g2 = Arc::new(MockGuardrail::new(
            "test2",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        registry.register(g1.clone());
        registry.register(g2.clone());

        let ctx = create_test_context();
        let decision = registry.run_stage(GuardrailStage::Input, &ctx).await;

        assert!(decision.is_allow());
        assert_eq!(g1.get_call_count(), 1);
        assert_eq!(g2.get_call_count(), 1);
    }

    #[tokio::test]
    async fn guardrail_registry_run_stage_block_short_circuits() {
        let mut registry = GuardrailRegistry::new();
        let g1 = Arc::new(MockGuardrail::new(
            "test1",
            vec![GuardrailStage::Input],
            GuardrailDecision::Block {
                reason: "blocked".to_string(),
                code: 403,
            },
        ));
        let g2 = Arc::new(MockGuardrail::new(
            "test2",
            vec![GuardrailStage::Input],
            GuardrailDecision::Allow,
        ));
        registry.register(g1.clone());
        registry.register(g2.clone());

        let ctx = create_test_context();
        let decision = registry.run_stage(GuardrailStage::Input, &ctx).await;

        assert!(decision.is_block());
        assert_eq!(g1.get_call_count(), 1);
        assert_eq!(g2.get_call_count(), 0); // Should not be called after block
    }

    #[test]
    fn guardrail_stage_variants() {
        let input = GuardrailStage::Input;
        let output = GuardrailStage::Output;
        let chunk = GuardrailStage::OutputChunk;

        assert_ne!(input, output);
        assert_ne!(input, chunk);
        assert_ne!(output, chunk);
    }

    #[test]
    fn guardrail_decision_is_block() {
        let allow = GuardrailDecision::Allow;
        let block = GuardrailDecision::Block {
            reason: "test".to_string(),
            code: 403,
        };
        let mutate = GuardrailDecision::Mutate {
            new_payload: serde_json::json!({}),
        };

        assert!(!allow.is_block());
        assert!(block.is_block());
        assert!(!mutate.is_block());
    }

    #[test]
    fn guardrail_decision_is_allow() {
        let allow = GuardrailDecision::Allow;
        let block = GuardrailDecision::Block {
            reason: "test".to_string(),
            code: 403,
        };
        let mutate = GuardrailDecision::Mutate {
            new_payload: serde_json::json!({}),
        };

        assert!(allow.is_allow());
        assert!(!block.is_allow());
        assert!(!mutate.is_allow());
    }
}
