use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime};

use dashmap::DashMap;
use tower::{Layer, Service};

use super::types::{LlmRequest, LlmResponse};
use crate::client::BoxFuture;
use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::cost;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetDimension {
    Global,
    Model(String),
    Tenant(String),
    User(String),
    ApiKey(String),
}

#[derive(Debug, Clone)]
pub enum BudgetVerdict {
    Allow,
    Reject {
        reason: String,
        dimension: BudgetDimension,
    },
}

pub struct CostRecordContext<'a> {
    pub model: &'a str,
    pub provider: &'a str,
    pub tenant_id: Option<&'a str>,
    pub user_id: Option<&'a str>,
    pub api_key_id: Option<&'a str>,
    pub cost_usd: f64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub timestamp: SystemTime,
}

pub struct CostCheckContext<'a> {
    pub model: &'a str,
    pub provider: &'a str,
    pub tenant_id: Option<&'a str>,
    pub user_id: Option<&'a str>,
    pub api_key_id: Option<&'a str>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Default)]
pub struct BudgetSnapshot {
    pub global_spend_usd: f64,
    pub per_model: HashMap<String, f64>,
    pub per_tenant: HashMap<String, f64>,
    pub per_user: HashMap<String, f64>,
    pub per_api_key: HashMap<String, f64>,
    pub limit_global: Option<f64>,
    pub limits_per_user: HashMap<String, f64>,
    pub limits_per_api_key: HashMap<String, f64>,
    pub limits_per_tenant: HashMap<String, f64>,
}

pub trait BudgetLedger: Send + Sync + 'static {
    fn record<'a>(
        &'a self,
        ctx: &'a CostRecordContext<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
    fn check<'a>(
        &'a self,
        ctx: &'a CostCheckContext<'a>,
    ) -> Pin<Box<dyn Future<Output = BudgetVerdict> + Send + 'a>>;
    fn snapshot(&self) -> BudgetSnapshot;
}

#[derive(Debug)]
struct WindowEntry {
    spend_mc: AtomicU64,
    window_start_secs: AtomicU64,
    window_secs: u64,
}

impl WindowEntry {
    fn new(window: Duration) -> Self {
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            spend_mc: AtomicU64::new(0),
            window_start_secs: AtomicU64::new(now_secs),
            window_secs: window.as_secs(),
        }
    }

    fn spend_usd(&self, now: SystemTime) -> f64 {
        let now_secs = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let start = self.window_start_secs.load(Ordering::Acquire);
        if now_secs.saturating_sub(start) >= self.window_secs {
            let old_mc = self.spend_mc.load(Ordering::Acquire);
            if self
                .window_start_secs
                .compare_exchange(start, now_secs, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.spend_mc.fetch_sub(old_mc, Ordering::AcqRel);
            }
        }
        per_token_to_per_millon(self.spend_mc.load(Ordering::Acquire))
    }

    fn add(&self, usd: f64, now: SystemTime) {
        let _ = self.spend_usd(now);
        self.spend_mc
            .fetch_add(per_millon_to_per_token(usd), Ordering::AcqRel);
    }
}

#[derive(Debug, Clone, Default)]
pub struct DimensionLimits {
    pub global: Option<f64>,
    pub per_model: HashMap<String, f64>,
    pub per_tenant: HashMap<String, f64>,
    pub per_user: HashMap<String, f64>,
    pub per_api_key: HashMap<String, f64>,
}

#[derive(Debug)]
pub struct InMemoryBudgetLedger {
    limits: DimensionLimits,
    window: Duration,
    global: Arc<WindowEntry>,
    per_model: Arc<DashMap<String, WindowEntry>>,
    per_tenant: Arc<DashMap<String, WindowEntry>>,
    per_user: Arc<DashMap<String, WindowEntry>>,
    per_api_key: Arc<DashMap<String, WindowEntry>>,
}

impl InMemoryBudgetLedger {
    #[must_use]
    pub fn new(limits: DimensionLimits, window: Duration) -> Self {
        Self {
            global: Arc::new(WindowEntry::new(window)),
            per_model: Arc::new(DashMap::new()),
            per_tenant: Arc::new(DashMap::new()),
            per_user: Arc::new(DashMap::new()),
            per_api_key: Arc::new(DashMap::new()),
            limits,
            window,
        }
    }

    #[must_use]
    pub fn from_config(config: &BudgetConfig) -> Self {
        let limits = DimensionLimits {
            global: config.global_limit,
            per_model: config.model_limits.clone(),
            ..Default::default()
        };
        // Default window: 30 days — resets monthly.
        Self::new(limits, Duration::from_secs(30 * 24 * 3600))
    }

    pub fn export_csv(&self, mut writer: impl io::Write) -> io::Result<()> {
        let snap = self.snapshot();
        writeln!(writer, "dimension,spend_usd")?;
        writeln!(writer, "global,{}", snap.global_spend_usd)?;
        for (model, spend) in &snap.per_model {
            writeln!(writer, "model:{model},{spend}")?;
        }
        for (tenant, spend) in &snap.per_tenant {
            writeln!(writer, "tenant:{tenant},{spend}")?;
        }
        for (user, spend) in &snap.per_user {
            writeln!(writer, "user:{user},{spend}")?;
        }
        for (key, spend) in &snap.per_api_key {
            writeln!(writer, "api_key:{key},{spend}")?;
        }
        Ok(())
    }

    pub fn reset(&self) {
        let now = SystemTime::now();
        // Force window expiry on the global entry by back-dating start.
        let zero_secs = SystemTime::UNIX_EPOCH
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.global.spend_mc.store(0, Ordering::Relaxed);
        self.global
            .window_start_secs
            .store(zero_secs, Ordering::Relaxed);
        let _ = self.global.spend_usd(now); // re-arm window

        self.per_model.clear();
        self.per_tenant.clear();
        self.per_user.clear();
        self.per_api_key.clear();
    }

    fn entry_spend(map: &DashMap<String, WindowEntry>, key: &str, now: SystemTime) -> f64 {
        map.get(key).map(|e| e.spend_usd(now)).unwrap_or(0.0)
    }

    fn entry_add(
        map: &DashMap<String, WindowEntry>,
        key: &str,
        usd: f64,
        window: Duration,
        now: SystemTime,
    ) {
        map.entry(key.to_owned())
            .or_insert_with(|| WindowEntry::new(window))
            .add(usd, now);
    }

    fn check_limit(
        spend: f64,
        limit: f64,
        dimension: BudgetDimension,
        key: &str,
    ) -> Option<BudgetVerdict> {
        if spend >= limit {
            Some(BudgetVerdict::Reject {
                reason: format!("{key} budget exceeded: spent ${spend:.6}, limit ${limit:.6}"),
                dimension,
            })
        } else {
            None
        }
    }
}

impl BudgetLedger for InMemoryBudgetLedger {
    fn record<'a>(
        &'a self,
        ctx: &'a CostRecordContext<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let now = ctx.timestamp;
            self.global.add(ctx.cost_usd, now);
            Self::entry_add(&self.per_model, ctx.model, ctx.cost_usd, self.window, now);
            if let Some(tenant) = ctx.tenant_id {
                Self::entry_add(&self.per_tenant, tenant, ctx.cost_usd, self.window, now);
            }
            if let Some(user) = ctx.user_id {
                Self::entry_add(&self.per_user, user, ctx.cost_usd, self.window, now);
            }
            if let Some(key) = ctx.api_key_id {
                Self::entry_add(&self.per_api_key, key, ctx.cost_usd, self.window, now);
            }
        })
    }

    fn check<'a>(
        &'a self,
        ctx: &'a CostCheckContext<'a>,
    ) -> Pin<Box<dyn Future<Output = BudgetVerdict> + Send + 'a>> {
        Box::pin(async move {
            let now = ctx.timestamp;

            // Global
            if let Some(limit) = self.limits.global {
                let spend = self.global.spend_usd(now);
                if let Some(v) = Self::check_limit(spend, limit, BudgetDimension::Global, "global")
                {
                    return v;
                }
            }

            // Per-model
            if let Some(&limit) = self.limits.per_model.get(ctx.model) {
                let spend = Self::entry_spend(&self.per_model, ctx.model, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::Model(ctx.model.to_owned()),
                    &format!("model:{}", ctx.model),
                ) {
                    return v;
                }
            }

            // Per-tenant
            if let Some(tenant) = ctx.tenant_id
                && let Some(&limit) = self.limits.per_tenant.get(tenant)
            {
                let spend = Self::entry_spend(&self.per_tenant, tenant, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::Tenant(tenant.to_owned()),
                    &format!("tenant:{tenant}"),
                ) {
                    return v;
                }
            }

            // Per-user
            if let Some(user) = ctx.user_id
                && let Some(&limit) = self.limits.per_user.get(user)
            {
                let spend = Self::entry_spend(&self.per_user, user, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::User(user.to_owned()),
                    &format!("user:{user}"),
                ) {
                    return v;
                }
            }

            // Per-API-key
            if let Some(key) = ctx.api_key_id
                && let Some(&limit) = self.limits.per_api_key.get(key)
            {
                let spend = Self::entry_spend(&self.per_api_key, key, now);
                if let Some(v) = Self::check_limit(
                    spend,
                    limit,
                    BudgetDimension::ApiKey(key.to_owned()),
                    &format!("api_key:{key}"),
                ) {
                    return v;
                }
            }

            BudgetVerdict::Allow
        })
    }

    fn snapshot(&self) -> BudgetSnapshot {
        let now = SystemTime::now();

        let global_spend_usd = self.global.spend_usd(now);

        let per_model = self
            .per_model
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        let per_tenant = self
            .per_tenant
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        let per_user = self
            .per_user
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        let per_api_key = self
            .per_api_key
            .iter()
            .map(|e| (e.key().clone(), e.value().spend_usd(now)))
            .collect();

        BudgetSnapshot {
            global_spend_usd,
            per_model,
            per_tenant,
            per_user,
            per_api_key,
            limit_global: self.limits.global,
            limits_per_user: self.limits.per_user.clone(),
            limits_per_api_key: self.limits.per_api_key.clone(),
            limits_per_tenant: self.limits.per_tenant.clone(),
        }
    }
}

#[must_use]
pub fn should_hedge<L: BudgetLedger>(
    ledger: &L,
    ctx: &CostCheckContext<'_>,
    estimated_cost_usd: f64,
    safety_margin_pct: f64,
) -> bool {
    let snap = ledger.snapshot();
    // A hedge issues two copies of the request.
    let hedge_cost = 2.0 * estimated_cost_usd;
    let margin = safety_margin_pct.clamp(0.0, 0.999);

    // Returns `true` when `spend + hedge_cost` fits within the effective limit.
    let has_headroom = |spend: f64, limit: f64| -> bool {
        let effective_limit = limit * (1.0 - margin);
        spend + hedge_cost < effective_limit
    };

    // Global dimension.
    if let Some(global_limit) = snap.limit_global
        && !has_headroom(snap.global_spend_usd, global_limit)
    {
        return false;
    }

    // Per-user dimension.
    if let Some(user) = ctx.user_id
        && let Some(&user_limit) = snap.limits_per_user.get(user)
    {
        let user_spend = snap.per_user.get(user).copied().unwrap_or(0.0);
        if !has_headroom(user_spend, user_limit) {
            return false;
        }
    }

    // Per-API-key dimension.
    if let Some(key) = ctx.api_key_id
        && let Some(&key_limit) = snap.limits_per_api_key.get(key)
    {
        let key_spend = snap.per_api_key.get(key).copied().unwrap_or(0.0);
        if !has_headroom(key_spend, key_limit) {
            return false;
        }
    }

    // Per-tenant dimension.
    if let Some(tenant) = ctx.tenant_id
        && let Some(&tenant_limit) = snap.limits_per_tenant.get(tenant)
    {
        let tenant_spend = snap.per_tenant.get(tenant).copied().unwrap_or(0.0);
        if !has_headroom(tenant_spend, tenant_limit) {
            return false;
        }
    }

    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Enforcement {
    Hard,
    Soft,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BudgetConfig {
    pub global_limit: Option<f64>,
    pub model_limits: HashMap<String, f64>,
    pub enforcement: Enforcement,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            global_limit: None,
            model_limits: HashMap::new(),
            enforcement: Enforcement::Hard,
        }
    }
}

#[derive(Debug)]
pub struct BudgetState {
    global_spend: AtomicU64,
    model_spend: DashMap<String, AtomicU64>,
}

impl BudgetState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            global_spend: AtomicU64::new(0),
            model_spend: DashMap::new(),
        }
    }

    #[must_use]
    pub fn global_spend(&self) -> f64 {
        per_token_to_per_millon(self.global_spend.load(Ordering::Relaxed))
    }

    #[must_use]
    pub fn model_spend(&self, model: &str) -> f64 {
        self.model_spend
            .get(model)
            .map(|v| per_token_to_per_millon(v.load(Ordering::Relaxed)))
            .unwrap_or(0.0)
    }

    pub fn reset(&self) {
        self.global_spend.store(0, Ordering::Relaxed);
        self.model_spend.clear();
    }

    fn record(&self, model: &str, usd: f64) {
        let mc = per_millon_to_per_token(usd);
        self.global_spend.fetch_add(mc, Ordering::Relaxed);
        self.model_spend
            .entry(model.to_owned())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(mc, Ordering::Relaxed);
    }
}

impl Default for BudgetState {
    fn default() -> Self {
        Self::new()
    }
}

fn per_millon_to_per_token(usd: f64) -> u64 {
    if usd <= 0.0 {
        return 0;
    }
    (usd * 1_000_000.0).round() as u64
}

fn per_token_to_per_millon(mc: u64) -> f64 {
    mc as f64 / 1_000_000.0
}

pub struct BudgetLayer {
    config: BudgetConfig,
    state: Arc<BudgetState>,
    provider: String,
}

impl BudgetLayer {
    #[must_use]
    pub fn new(config: BudgetConfig, state: Arc<BudgetState>, provider: impl Into<String>) -> Self {
        Self {
            config,
            state,
            provider: provider.into(),
        }
    }
}

impl<S> Layer<S> for BudgetLayer {
    type Service = BudgetService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        BudgetService {
            inner,
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            provider: self.provider.clone(),
        }
    }
}

pub struct BudgetService<S> {
    inner: S,
    config: BudgetConfig,
    state: Arc<BudgetState>,
    provider: String,
}

impl<S: Clone> Clone for BudgetService<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            provider: self.provider.clone(),
        }
    }
}

impl<S> Service<LlmRequest> for BudgetService<S>
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
        let model = req.model().unwrap_or("unknown").to_owned();
        let config = self.config.clone();
        let state = Arc::clone(&self.state);
        let provider = self.provider.clone();

        if config.enforcement == Enforcement::Hard
            && let Some(err) = check_budget(&config, &state, &model)
        {
            return Box::pin(async move { Err(err) });
        }

        let fut = self.inner.call(req);

        Box::pin(async move {
            let resp = fut.await?;

            if let Some(usage) = resp.usage()
                && let Ok(Some(cost)) = cost::completion_cost(
                    &provider,
                    &model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )
                .await
            {
                state.record(&model, cost);
                if config.enforcement == Enforcement::Soft {
                    emit_soft_warnings(&config, &state, &model);
                }
            }

            Ok(resp)
        })
    }
}

fn check_budget(config: &BudgetConfig, state: &BudgetState, model: &str) -> Option<HiLlmError> {
    if let Some(limit) = config.global_limit
        && state.global_spend() >= limit
    {
        return Some(HiLlmError::BudgetExceeded {
            message: format!(
                "global budget exceeded: spent ${:.6}, limit ${:.6}",
                state.global_spend(),
                limit,
            ),
            model: None,
        });
    }

    if let Some(&limit) = config.model_limits.get(model)
        && state.model_spend(model) >= limit
    {
        return Some(HiLlmError::BudgetExceeded {
            message: format!(
                "model {model} budget exceeded: spent ${:.6}, limit ${:.6}",
                state.model_spend(model),
                limit,
            ),
            model: Some(model.to_owned()),
        });
    }

    None
}

fn emit_soft_warnings(config: &BudgetConfig, state: &BudgetState, model: &str) {
    if let Some(limit) = config.global_limit
        && state.global_spend() >= limit
    {
        tracing::warn!(
            spend = state.global_spend(),
            limit,
            "global budget exceeded (soft enforcement)"
        );
    }

    if let Some(&limit) = config.model_limits.get(model)
        && state.model_spend(model) >= limit
    {
        tracing::warn!(
            model,
            spend = state.model_spend(model),
            limit,
            "model budget exceeded (soft enforcement)"
        );
    }
}
