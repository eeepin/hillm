use std::task::{Context, Poll};

use tower::Layer;
use tower::Service;
use tracing::Instrument as _;

use super::types::{LLMRequest, LLMResponse};
use crate::client::BoxFuture;
use crate::error::{HiLLMError, HiLLMResult};
use crate::types::FinishReason;

pub struct TracingLayer;

impl<S> Layer<S> for TracingLayer {
    type Service = TracingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TracingService { inner }
    }
}

pub struct TracingService<S> {
    inner: S,
}

impl<S> Clone for TracingService<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<S> Service<LLMRequest> for TracingService<S>
where
    S: Service<LLMRequest, Response = LLMResponse, Error = HiLLMError> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = LLMResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LLMResponse>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: LLMRequest) -> Self::Future {
        let operation_name = req.operation_name();
        let model_str = req.model().unwrap_or("");
        let system = model_str.split_once('/').map_or("", |(prefix, _)| prefix);
        let model = model_str.to_owned();

        let span = tracing::info_span!(
            "gen_ai",
            gen_ai.operation.name = operation_name,
            gen_ai.request.model = %model,
            gen_ai.system = system,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.usage.cost = tracing::field::Empty,
            gen_ai.response.finish_reasons = tracing::field::Empty,
            error.type = tracing::field::Empty,
        );

        let fut = self.inner.call(req);
        Box::pin(
            async move {
                match fut.await {
                    Ok(resp) => {
                        record_response(&tracing::Span::current(), &resp);
                        Ok(resp)
                    }
                    Err(e) => {
                        tracing::Span::current().record("error.type", e.error_type());
                        Err(e)
                    }
                }
            }
            .instrument(span),
        )
    }
}

#[cfg(feature = "otel")]
pub use tracing_opentelemetry;

#[cfg(feature = "otel")]
pub use opentelemetry;

fn record_response(span: &tracing::Span, resp: &LLMResponse) {
    match resp {
        LLMResponse::Chat(r) => {
            span.record("gen_ai.response.id", r.id.as_str());
            span.record("gen_ai.response.model", r.model.as_str());

            let finish_reasons =
                finish_reasons_str(r.choices.iter().map(|c| c.finish_reason.as_ref()));
            if !finish_reasons.is_empty() {
                span.record("gen_ai.response.finish_reasons", finish_reasons.as_str());
            }
        }
        LLMResponse::Embed(r) => {
            span.record("gen_ai.response.model", r.model.as_str());
        }
        LLMResponse::ChatStream(_)
        | LLMResponse::ListModels(_)
        | LLMResponse::ImageGenerate(_)
        | LLMResponse::Speech(_)
        | LLMResponse::Transcribe(_)
        | LLMResponse::Moderate(_)
        | LLMResponse::Rerank(_)
        | LLMResponse::Search(_)
        | LLMResponse::Ocr(_) => {}
    }

    if let Some(usage) = resp.usage() {
        span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
        span.record("gen_ai.usage.output_tokens", usage.completion_tokens);
    }
}

fn finish_reasons_str<'a>(reasons: impl Iterator<Item = Option<&'a FinishReason>>) -> String {
    let first = reasons.filter_map(|r| r.map(finish_reason_name));
    let mut iter = first.peekable();
    let Some(first_name) = iter.next() else {
        return String::new();
    };
    if iter.peek().is_none() {
        return first_name.to_owned();
    }
    iter.fold(first_name.to_owned(), |mut acc, name| {
        acc.push(' ');
        acc.push_str(name);
        acc
    })
}

const fn finish_reason_name(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::FunctionCall => "function_call",
        FinishReason::Other => "other",
    }
}
