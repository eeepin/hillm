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
