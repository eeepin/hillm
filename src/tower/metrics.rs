#[cfg(feature = "otel")]
mod inner {
    use std::sync::{Arc, OnceLock};
    use std::task::{Context, Poll};
    use std::time::Instant;

    use dashmap::DashMap;
    use opentelemetry::KeyValue;
    use opentelemetry::metrics::{Counter, Histogram, Meter};
    use tower::{Layer, Service};

    use super::super::types::{LlmRequest, LlmResponse};
    use crate::client::BoxFuture;
    use crate::error::{HiLlmError, HiLlmResult};

    static METER: OnceLock<Meter> = OnceLock::new();

    pub fn init_meter(meter: Meter) {
        let _ = METER.set(meter);
    }

    pub fn global_meter() -> Option<&'static Meter> {
        METER.get()
    }

    struct Instruments {
        op_duration: Histogram<f64>,
        token_usage: Histogram<u64>,
        #[allow(dead_code)]
        cost_usd: Histogram<f64>,
        cache_hit: Counter<u64>,
        cache_miss: Counter<u64>,
        cache_stale: Counter<u64>,
        circuit_trip: Counter<u64>,
        retry_attempt: Counter<u64>,
        budget_spend: Histogram<f64>,
        budget_rejection: Counter<u64>,
        realtime_session_duration: Histogram<f64>,
        realtime_event_count: Counter<u64>,
        realtime_bytes: Counter<u64>,
    }

    impl Instruments {
        fn new(meter: &Meter) -> Self {
            Self {
                op_duration: meter
                    .f64_histogram("gen_ai.client.operation.duration")
                    .with_description("GenAI client request latency in seconds")
                    .with_unit("s")
                    .build(),
                token_usage: meter
                    .u64_histogram("gen_ai.client.token.usage")
                    .with_description("Token counts for GenAI operations")
                    .with_unit("{token}")
                    .build(),
                cost_usd: meter
                    .f64_histogram("gen_ai.client.cost.usd")
                    .with_description("Estimated cost of GenAI operations in USD")
                    .with_unit("USD")
                    .build(),
                cache_hit: meter
                    .u64_counter("gen_ai.cache.hit")
                    .with_description("Number of GenAI response cache hits")
                    .build(),
                cache_miss: meter
                    .u64_counter("gen_ai.cache.miss")
                    .with_description("Number of GenAI response cache misses")
                    .build(),
                cache_stale: meter
                    .u64_counter("gen_ai.cache.stale")
                    .with_description("Number of stale GenAI cache responses served")
                    .build(),
                circuit_trip: meter
                    .u64_counter("gen_ai.circuit.trip")
                    .with_description("Number of circuit breaker trips")
                    .build(),
                retry_attempt: meter
                    .u64_counter("gen_ai.retry.attempt")
                    .with_description("Number of retry attempts (excluding first try)")
                    .build(),
                budget_spend: meter
                    .f64_histogram("gen_ai.budget.spend_usd")
                    .with_description("Cumulative spend in USD per budget dimension")
                    .with_unit("USD")
                    .build(),
                budget_rejection: meter
                    .u64_counter("gen_ai.budget.rejection")
                    .with_description("Number of requests rejected due to budget limits")
                    .build(),
                realtime_session_duration: meter
                    .f64_histogram("gen_ai.realtime.session.duration")
                    .with_description("Realtime WebSocket session lifetime in seconds")
                    .with_unit("s")
                    .build(),
                realtime_event_count: meter
                    .u64_counter("gen_ai.realtime.event.count")
                    .with_description("Number of Realtime events forwarded, by direction and type")
                    .build(),
                realtime_bytes: meter
                    .u64_counter("gen_ai.realtime.bytes")
                    .with_description("Audio bytes forwarded over Realtime WebSocket sessions")
                    .with_unit("By")
                    .build(),
            }
        }
    }

    type BaseAttrsKey = (Arc<str>, Arc<str>);
    static BASE_ATTRS_CACHE: OnceLock<DashMap<BaseAttrsKey, Arc<[KeyValue]>>> = OnceLock::new();

    struct CachedTokenAttrs {
        input: Arc<[KeyValue]>,
        output: Arc<[KeyValue]>,
    }

    static TOKEN_ATTRS_CACHE: OnceLock<DashMap<BaseAttrsKey, CachedTokenAttrs>> = OnceLock::new();

    fn base_attrs_cache() -> &'static DashMap<BaseAttrsKey, Arc<[KeyValue]>> {
        BASE_ATTRS_CACHE.get_or_init(DashMap::new)
    }

    fn token_attrs_cache() -> &'static DashMap<BaseAttrsKey, CachedTokenAttrs> {
        TOKEN_ATTRS_CACHE.get_or_init(DashMap::new)
    }

    fn get_or_build_base_attrs(
        system: &str,
        model: &str,
        response_model: &str,
        operation: &str,
    ) -> Arc<[KeyValue]> {
        let system_arc = Arc::<str>::from(system);
        let model_arc = Arc::<str>::from(model);
        let key = (Arc::clone(&system_arc), Arc::clone(&model_arc));

        let cache = base_attrs_cache();

        if let Some(entry) = cache.get(&key) {
            return Arc::clone(&entry);
        }

        let attrs = Arc::from(
            vec![
                KeyValue::new("gen_ai.system", system_arc.to_string()),
                KeyValue::new("gen_ai.request.model", model_arc.to_string()),
                KeyValue::new("gen_ai.response.model", response_model.to_owned()),
                KeyValue::new("gen_ai.operation.name", operation.to_owned()),
            ]
            .into_boxed_slice(),
        );

        cache.entry(key).or_insert_with(|| Arc::clone(&attrs));

        attrs
    }

    fn get_or_build_token_attrs(
        system: &str,
        model: &str,
        response_model: &str,
        operation: &str,
    ) -> CachedTokenAttrs {
        let system_arc = Arc::<str>::from(system);
        let model_arc = Arc::<str>::from(model);
        let key = (Arc::clone(&system_arc), Arc::clone(&model_arc));

        let cache = token_attrs_cache();

        if let Some(entry) = cache.get(&key) {
            return CachedTokenAttrs {
                input: Arc::clone(&entry.input),
                output: Arc::clone(&entry.output),
            };
        }

        let base = get_or_build_base_attrs(&system_arc, &model_arc, response_model, operation);

        let mut input_attrs = base.to_vec();
        input_attrs.push(KeyValue::new("gen_ai.token.type", "input"));
        let input_arc = Arc::from(input_attrs.into_boxed_slice());

        let mut output_attrs = base.to_vec();
        output_attrs.push(KeyValue::new("gen_ai.token.type", "output"));
        let output_arc = Arc::from(output_attrs.into_boxed_slice());

        let cached = CachedTokenAttrs {
            input: Arc::clone(&input_arc),
            output: Arc::clone(&output_arc),
        };
        cache.entry(key).or_insert_with(|| CachedTokenAttrs {
            input: Arc::clone(&input_arc),
            output: Arc::clone(&output_arc),
        });

        cached
    }

    static INSTRUMENTS: OnceLock<Arc<Instruments>> = OnceLock::new();

    fn instruments() -> Option<Arc<Instruments>> {
        if let Some(cached) = INSTRUMENTS.get() {
            return Some(Arc::clone(cached));
        }

        if let Some(meter) = global_meter() {
            let new_instruments = Arc::new(Instruments::new(meter));
            let result = INSTRUMENTS
                .set(Arc::clone(&new_instruments))
                .ok()
                .map(|_| Arc::clone(&new_instruments));
            return result.or_else(|| INSTRUMENTS.get().map(Arc::clone));
        }

        None
    }

    #[derive(Clone)]
    pub struct MetricsLayer;

    impl<S> Layer<S> for MetricsLayer {
        type Service = MetricsService<S>;

        fn layer(&self, inner: S) -> Self::Service {
            MetricsService { inner }
        }
    }

    pub struct MetricsService<S> {
        inner: S,
    }

    impl<S: Clone> Clone for MetricsService<S> {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
            }
        }
    }

    impl<S> Service<LlmRequest> for MetricsService<S>
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
            let start = Instant::now();

            let operation = req.operation_name();
            let model_str = req.model().unwrap_or("").to_owned();
            let system = model_str
                .split_once('/')
                .map(|(prefix, _)| prefix.to_owned())
                .unwrap_or_default();

            let fut = self.inner.call(req);

            Box::pin(async move {
                let result = fut.await;
                let elapsed = start.elapsed().as_secs_f64();

                if let Some(instr) = instruments() {
                    let response_model = match &result {
                        Ok(resp) => match resp {
                            LlmResponse::Chat(r) => r.model.clone(),
                            LlmResponse::Embed(r) => r.model.clone(),
                            _ => model_str.clone(),
                        },
                        Err(_) => model_str.clone(),
                    };

                    let base_attrs =
                        get_or_build_base_attrs(&system, &model_str, &response_model, operation);

                    instr.op_duration.record(elapsed, base_attrs.as_ref());

                    if let Ok(resp) = &result
                        && let Some(usage) = resp.usage()
                    {
                        let token_attrs = get_or_build_token_attrs(
                            &system,
                            &model_str,
                            &response_model,
                            operation,
                        );

                        instr
                            .token_usage
                            .record(usage.prompt_tokens, token_attrs.input.as_ref());

                        instr
                            .token_usage
                            .record(usage.completion_tokens, token_attrs.output.as_ref());
                    }
                }

                result
            })
        }
    }

    pub fn record_cache_hit(system: &str, model: &str, operation: &str) {
        if let Some(instr) = instruments() {
            instr.cache_hit.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.operation.name", operation.to_owned()),
                ],
            );
        }
    }

    pub fn record_cache_miss(system: &str, model: &str, operation: &str) {
        if let Some(instr) = instruments() {
            instr.cache_miss.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.operation.name", operation.to_owned()),
                ],
            );
        }
    }

    pub fn record_cache_stale(system: &str, model: &str, operation: &str) {
        if let Some(instr) = instruments() {
            instr.cache_stale.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.operation.name", operation.to_owned()),
                ],
            );
        }
    }

    pub fn record_circuit_trip(system: &str, model: &str) {
        if let Some(instr) = instruments() {
            instr.circuit_trip.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                ],
            );
        }
    }

    pub fn record_retry_attempt(system: &str, model: &str, operation: &str) {
        if let Some(instr) = instruments() {
            instr.retry_attempt.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.operation.name", operation.to_owned()),
                ],
            );
        }
    }

    pub fn record_cache_tier_hit(system: &str, model: &str, tier: &str) {
        if let Some(instr) = instruments() {
            instr.cache_hit.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.cache.tier", tier.to_owned()),
                ],
            );
        }
    }

    pub fn record_cache_tier_miss(system: &str, model: &str, tier: &str) {
        if let Some(instr) = instruments() {
            instr.cache_miss.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", system.to_owned()),
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.cache.tier", tier.to_owned()),
                ],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_budget_spend(
        model: &str,
        provider: &str,
        tenant_id: Option<&str>,
        user_id: Option<&str>,
        api_key_id: Option<&str>,
        cost_usd: f64,
    ) {
        if let Some(instr) = instruments() {
            let mut attrs = vec![
                KeyValue::new("gen_ai.request.model", model.to_owned()),
                KeyValue::new("gen_ai.system", provider.to_owned()),
            ];
            if let Some(tenant) = tenant_id {
                attrs.push(KeyValue::new("gen_ai.budget.tenant_id", tenant.to_owned()));
            }
            if let Some(user) = user_id {
                attrs.push(KeyValue::new("gen_ai.budget.user_id", user.to_owned()));
            }
            if let Some(key) = api_key_id {
                attrs.push(KeyValue::new("gen_ai.budget.api_key_id", key.to_owned()));
            }
            instr.budget_spend.record(cost_usd, &attrs);
        }
    }

    pub fn record_budget_rejection(model: &str, provider: &str, dimension: &str) {
        if let Some(instr) = instruments() {
            instr.budget_rejection.add(
                1,
                &[
                    KeyValue::new("gen_ai.request.model", model.to_owned()),
                    KeyValue::new("gen_ai.system", provider.to_owned()),
                    KeyValue::new("gen_ai.budget.dimension", dimension.to_owned()),
                ],
            );
        }
    }

    pub fn record_realtime_session_duration(provider: &str, duration_secs: f64) {
        if let Some(instr) = instruments() {
            instr.realtime_session_duration.record(
                duration_secs,
                &[KeyValue::new("gen_ai.system", provider.to_owned())],
            );
        }
    }

    pub fn record_realtime_event(provider: &str, direction: &str, event_type: &str) {
        if let Some(instr) = instruments() {
            instr.realtime_event_count.add(
                1,
                &[
                    KeyValue::new("gen_ai.system", provider.to_owned()),
                    KeyValue::new("gen_ai.realtime.direction", direction.to_owned()),
                    KeyValue::new("gen_ai.realtime.event_type", event_type.to_owned()),
                ],
            );
        }
    }

    pub fn record_realtime_bytes(provider: &str, direction: &str, byte_count: u64) {
        if let Some(instr) = instruments() {
            instr.realtime_bytes.add(
                byte_count,
                &[
                    KeyValue::new("gen_ai.system", provider.to_owned()),
                    KeyValue::new("gen_ai.realtime.direction", direction.to_owned()),
                ],
            );
        }
    }
}

#[cfg(not(feature = "otel"))]
mod inner {
    use std::task::{Context, Poll};

    use tower::{Layer, Service};

    use super::super::types::{LlmRequest, LlmResponse};
    use crate::client::BoxFuture;
    use crate::error::{HiLlmError, HiLlmResult};

    #[derive(Clone)]
    pub struct MetricsLayer;

    impl<S> Layer<S> for MetricsLayer {
        type Service = MetricsService<S>;

        fn layer(&self, inner: S) -> Self::Service {
            MetricsService { inner }
        }
    }

    pub struct MetricsService<S> {
        inner: S,
    }

    impl<S: Clone> Clone for MetricsService<S> {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
            }
        }
    }

    impl<S> Service<LlmRequest> for MetricsService<S>
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
            Box::pin(self.inner.call(req))
        }
    }

    #[inline]
    pub fn record_cache_hit(_system: &str, _model: &str, _operation: &str) {}

    #[inline]
    pub fn record_cache_miss(_system: &str, _model: &str, _operation: &str) {}

    #[inline]
    pub fn record_cache_stale(_system: &str, _model: &str, _operation: &str) {}

    #[inline]
    pub fn record_circuit_trip(_system: &str, _model: &str) {}

    #[inline]
    pub fn record_retry_attempt(_system: &str, _model: &str, _operation: &str) {}

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn record_budget_spend(
        _model: &str,
        _provider: &str,
        _tenant_id: Option<&str>,
        _user_id: Option<&str>,
        _api_key_id: Option<&str>,
        _cost_usd: f64,
    ) {
    }

    #[inline]
    pub fn record_budget_rejection(_model: &str, _provider: &str, _dimension: &str) {}

    #[inline]
    pub fn record_cache_tier_hit(_system: &str, _model: &str, _tier: &str) {}

    #[inline]
    pub fn record_cache_tier_miss(_system: &str, _model: &str, _tier: &str) {}

    #[inline]
    pub fn record_realtime_session_duration(_provider: &str, _duration_secs: f64) {}

    #[inline]
    pub fn record_realtime_event(_provider: &str, _direction: &str, _event_type: &str) {}

    #[inline]
    pub fn record_realtime_bytes(_provider: &str, _direction: &str, _byte_count: u64) {}
}

// Re-export the active implementation.
pub use inner::*;
