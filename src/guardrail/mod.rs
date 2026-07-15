use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub mod builtin;
#[cfg(feature = "guardrail")]
pub mod cel;
pub mod registry;

pub use registry::GuardrailRegistry;

pub trait Guardrail: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn supported_stages(&self) -> &'static [GuardrailStage];

    fn check<'a>(
        &'a self,
        stage: GuardrailStage,
        ctx: &'a GuardrailContext<'a>,
    ) -> Pin<Box<dyn Future<Output = GuardrailDecision> + Send + 'a>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuardrailStage {
    Input,
    Output,
    OutputChunk,
}

pub struct GuardrailContext<'a> {
    pub request: &'a serde_json::Value,
    pub response: Option<&'a serde_json::Value>,
    pub chunk: Option<&'a str>,
    pub metadata: &'a HashMap<String, String>,
}

#[derive(Debug)]
pub enum GuardrailDecision {
    Allow,
    Block { reason: String, code: u32 },
    Mutate { new_payload: serde_json::Value },
}

impl GuardrailDecision {
    #[must_use]
    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block { .. })
    }

    #[must_use]
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }
}
