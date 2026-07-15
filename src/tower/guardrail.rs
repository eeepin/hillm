use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll};

use tower::Layer;
use tower::Service;

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::guardrail::registry::GuardrailRegistry;
use crate::guardrail::{GuardrailContext, GuardrailDecision, GuardrailStage};

#[derive(Clone)]
pub struct GuardrailLayer {
    registry: Arc<GuardrailRegistry>,
    metadata: Arc<HashMap<String, String>>,
}

impl GuardrailLayer {
    #[must_use]
    pub fn new(registry: Arc<GuardrailRegistry>, metadata: HashMap<String, String>) -> Self {
        Self {
            registry,
            metadata: Arc::new(metadata),
        }
    }

    #[must_use]
    pub fn with_registry(registry: Arc<GuardrailRegistry>) -> Self {
        Self::new(registry, HashMap::new())
    }
}

impl<S> Layer<S> for GuardrailLayer {
    type Service = GuardrailService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GuardrailService {
            inner,
            registry: Arc::clone(&self.registry),
            metadata: Arc::clone(&self.metadata),
        }
    }
}

pub struct GuardrailService<S> {
    inner: S,
    registry: Arc<GuardrailRegistry>,
    metadata: Arc<HashMap<String, String>>,
}

impl<S: Clone> Clone for GuardrailService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            registry: Arc::clone(&self.registry),
            metadata: Arc::clone(&self.metadata),
        }
    }
}

impl<S> Service<LlmRequest> for GuardrailService<S>
where
    S: Service<LlmRequest, Response = LlmResponse, Error = HiLlmError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let registry = Arc::clone(&self.registry);
        let metadata = Arc::clone(&self.metadata);
        let inner_fut = self.inner.call(req.clone());

        Box::pin(async move {
            let request_json = match serde_json::to_value(&req) {
                Ok(v) => v,
                Err(e) => {
                    return Err(HiLlmError::InternalError {
                        message: format!("guardrail: failed to serialize request: {e}"),
                    });
                }
            };

            let input_ctx = GuardrailContext {
                request: &request_json,
                response: None,
                chunk: None,
                metadata: &metadata,
            };

            let input_decision = registry.run_stage(GuardrailStage::Input, &input_ctx).await;
            match input_decision {
                GuardrailDecision::Block { reason, code } => {
                    return Err(HiLlmError::HookRejected {
                        message: format!("guardrail blocked [code={code}]: {reason}"),
                    });
                }
                GuardrailDecision::Mutate { .. } => {
                    tracing::debug!(
                        "guardrail: Input stage Mutate decision; proceeding with original request"
                    );
                }
                GuardrailDecision::Allow => {}
            }

            let response = inner_fut.await?;

            let response_json = match &response {
                LlmResponse::Chat(r) => match serde_json::to_value(r) {
                    Ok(v) => v,
                    Err(_) => return Ok(response),
                },
                LlmResponse::Embed(r) => match serde_json::to_value(r) {
                    Ok(v) => v,
                    Err(_) => return Ok(response),
                },
                LlmResponse::ListModels(r) => match serde_json::to_value(r) {
                    Ok(v) => v,
                    Err(_) => return Ok(response),
                },
                _ => return Ok(response),
            };

            let output_ctx = GuardrailContext {
                request: &request_json,
                response: Some(&response_json),
                chunk: None,
                metadata: &metadata,
            };

            let output_decision = registry
                .run_stage(GuardrailStage::Output, &output_ctx)
                .await;
            match output_decision {
                GuardrailDecision::Block { reason, code } => Err(HiLlmError::HookRejected {
                    message: format!("guardrail blocked output [code={code}]: {reason}"),
                }),
                GuardrailDecision::Mutate { .. } => {
                    tracing::debug!(
                        "guardrail: Output stage Mutate decision; returning original response"
                    );
                    Ok(response)
                }
                GuardrailDecision::Allow => Ok(response),
            }
        })
    }
}
