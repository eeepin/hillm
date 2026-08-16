# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **P1 architecture** — Per-instance `OutboundPolicyValidator`, `CustomProviderRegistry`, and versioned `ProviderRegistrySnapshot` replace the process-global statics, enabling multi-tenant isolation. `ClientConfig` gains an `outbound_policy` field; `ClientBuilder` gains a `.outbound_policy()` method.
- `OutboundPolicy::server_default()` returns `DenyPrivate`, the recommended default for server-side deployments.
- `HILLM_OUTBOUND_POLICY` environment variable selects the default policy at startup.
- `refresh_registry()`, `registry_snapshot()`, `registry_fetched_at()`, `registry_source()` functions for explicit provider registry refresh.
- Comprehensive test coverage for all previously zero-test modules: `realtime` (42 tests), `guardrail/builtin` (28 tests), `tower/fallback`, `fallback_chain`, `hedge`, `cooldown`, `cache_negative`, `cache_policy`, plus expanded `router` and `tenant/in_memory` tests. Total: 462 tests with `tower` feature.
- `README.md`, `LICENSE-MIT`, `Cargo.toml` package metadata (description, license, keywords, categories, rust-version).
- `GuardrailDecision` now derives `Clone`.

### Changed

- `client/mod.rs` split into `client/impls/{chat,raw,file,batch,response,anthropic}.rs` (2272 → 635 lines).
- `provider/anthropic/compat.rs` split into `compat.rs` (structs + impls) and `compat_convert.rs` (conversion functions).
- Root-level `pub use types::*` wildcard replaced with explicit, curated re-exports.
- `ProviderRegistry` evolved from one-shot `OnceCell`/`OnceLock` to `RwLock<Option<ProviderRegistrySnapshot>>` with version metadata.

### Fixed

- Pre-existing clippy warnings in `idempotency.rs`, `sse.rs`, `cache.rs`, `guardrail/registry.rs`.

## [0.1.0] — 2025-XX-XX

Initial release.

- Unified `LLMClient`, `ResponseClient`, `FileClient`, `BatchClient`, `AnthropicMessagesClient` traits.
- Three explicit API routes: `OpenAIChatCompletions`, `OpenAIResponses`, `AnthropicMessages`.
- Tower middleware: cache, singleflight, circuit, fallback, hedge, budget, rate limit, router, health, idempotency, guardrails, hooks.
- Multi-tenant credential resolution (in-memory, etcd).
- SSE decoder with byte-boundary-independent parsing.
- WASM support via `wasm-http` feature.
