use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use cel_interpreter::objects::{Key, Map, Value};
use cel_interpreter::{Context, ParseErrors, Program};

use super::{Guardrail, GuardrailContext, GuardrailDecision, GuardrailStage};

const CEL_EVAL_ERROR_CODE: u32 = 4001;

#[derive(Debug, Clone)]
pub enum CelAction {
    Block { code: u32, reason: String },
    Mutate { new_payload: serde_json::Value },
}

pub struct CelGuardrail {
    guardrail_name: &'static str,
    program: Program,
    on_true: CelAction,
    stages: &'static [GuardrailStage],
    fail_open: bool,
}

impl CelGuardrail {
    pub fn new(
        name: &'static str,
        expression: &str,
        on_true: CelAction,
        stages: &'static [GuardrailStage],
    ) -> Result<Self, ParseErrors> {
        let program = Program::compile(expression)?;
        Ok(Self {
            guardrail_name: name,
            program,
            on_true,
            stages,
            fail_open: false,
        })
    }

    #[must_use]
    pub fn with_fail_open(mut self, fail_open: bool) -> Self {
        self.fail_open = fail_open;
        self
    }
}

impl Guardrail for CelGuardrail {
    fn name(&self) -> &'static str {
        self.guardrail_name
    }

    fn supported_stages(&self) -> &'static [GuardrailStage] {
        self.stages
    }

    fn check<'a>(
        &'a self,
        stage: GuardrailStage,
        ctx: &'a GuardrailContext<'a>,
    ) -> Pin<Box<dyn Future<Output = GuardrailDecision> + Send + 'a>> {
        Box::pin(async move {
            let mut cel_ctx = Context::default();
            cel_ctx.add_variable_from_value("request", json_value_to_cel(ctx.request));

            let response_val = ctx.response.map(json_value_to_cel).unwrap_or_else(|| {
                Value::Map(Map {
                    map: Arc::new(HashMap::new()),
                })
            });
            cel_ctx.add_variable_from_value("response", response_val);

            let chunk_str = ctx.chunk.unwrap_or("").to_string();
            cel_ctx.add_variable_from_value("chunk", Value::String(Arc::new(chunk_str)));

            cel_ctx.add_variable_from_value("metadata", metadata_to_cel(ctx.metadata));

            match self.program.execute(&cel_ctx) {
                Ok(Value::Bool(true)) => match &self.on_true {
                    CelAction::Block { code, reason } => GuardrailDecision::Block {
                        reason: reason.clone(),
                        code: *code,
                    },
                    CelAction::Mutate { new_payload } => GuardrailDecision::Mutate {
                        new_payload: new_payload.clone(),
                    },
                },
                Ok(Value::Bool(false)) => GuardrailDecision::Allow,

                Ok(non_bool) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!(
                        guardrail = self.guardrail_name,
                        stage = ?stage,
                        result = ?non_bool,
                        "CEL expression returned non-bool value; \
                         defaulting to fail-closed (Block/4001) — \
                         set fail_open=true to suppress"
                    );
                    #[cfg(not(feature = "tracing"))]
                    {
                        let _ = stage;
                        let _ = non_bool;
                    }

                    if self.fail_open {
                        GuardrailDecision::Allow
                    } else {
                        GuardrailDecision::Block {
                            reason: "policy evaluation error".to_owned(),
                            code: CEL_EVAL_ERROR_CODE,
                        }
                    }
                }

                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!(
                        guardrail = self.guardrail_name,
                        stage = ?stage,
                        error = %e,
                        "CEL expression evaluation error; \
                         defaulting to fail-closed (Block/4001) — \
                         set fail_open=true to suppress"
                    );
                    #[cfg(not(feature = "tracing"))]
                    {
                        let _ = stage;
                        let _ = e;
                    }

                    if self.fail_open {
                        GuardrailDecision::Allow
                    } else {
                        GuardrailDecision::Block {
                            reason: "policy evaluation error".to_owned(),
                            code: CEL_EVAL_ERROR_CODE,
                        }
                    }
                }
            }
        })
    }
}

fn json_value_to_cel(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(Arc::new(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_value_to_cel).collect();
            Value::List(Arc::new(items))
        }
        serde_json::Value::Object(obj) => {
            let mut map: HashMap<Key, Value> = HashMap::new();
            for (key, val) in obj {
                map.insert(Key::String(Arc::new(key.clone())), json_value_to_cel(val));
            }
            Value::Map(Map { map: Arc::new(map) })
        }
    }
}

fn metadata_to_cel(metadata: &HashMap<String, String>) -> Value {
    let mut map: HashMap<Key, Value> = HashMap::new();
    for (key, val) in metadata {
        map.insert(
            Key::String(Arc::new(key.clone())),
            Value::String(Arc::new(val.clone())),
        );
    }
    Value::Map(Map { map: Arc::new(map) })
}
