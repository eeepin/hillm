use std::hash::{BuildHasher, Hash, Hasher};

use ahash::RandomState;

const HASH_KEY_SEED_0: u64 = 0x6865_6c6c_6f6c_6c6d; // hellollm
const HASH_KEY_SEED_1: u64 = 0x6861_7368_5f6b_6579; // hash_key
const HASH_KEY_SEED_2: u64 = 0x7676_7669_705f_7631; // vvvip_v1
const HASH_KEY_SEED_3: u64 = 0x5f72_616e_646f_6d5f; // _random_

fn hash_random_state() -> &'static RandomState {
    use std::sync::OnceLock;
    static STATE: OnceLock<RandomState> = OnceLock::new();
    STATE.get_or_init(|| {
        RandomState::generate_with(
            HASH_KEY_SEED_0,
            HASH_KEY_SEED_1,
            HASH_KEY_SEED_2,
            HASH_KEY_SEED_3,
        )
    })
}

#[inline]
fn seeded_hasher() -> impl Hasher {
    hash_random_state().build_hasher()
}

pub struct HashKeyInput<'a> {
    pub model: &'a str,
    pub messages_json: &'a str,
    pub params_json: &'a str,
    pub tenant_id: Option<&'a str>,
    pub system_prompt: Option<&'a str>,
}

pub trait HashKeyStrategy: Send + Sync + 'static {
    fn key_for(&self, input: &HashKeyInput<'_>) -> (u64, String);
}

#[derive(Debug, Clone, Default)]
pub struct ExactHashStrategy;

impl HashKeyStrategy for ExactHashStrategy {
    fn key_for(&self, input: &HashKeyInput<'_>) -> (u64, String) {
        let body = format!(
            "{}|{}|{}|{}|{}",
            input.model,
            input.messages_json,
            input.params_json,
            input.tenant_id.unwrap_or(""),
            input.system_prompt.unwrap_or(""),
        );
        let mut hasher = seeded_hasher();
        body.hash(&mut hasher);
        (hasher.finish(), body)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SystemPromptAwareStrategy;

impl HashKeyStrategy for SystemPromptAwareStrategy {
    fn key_for(&self, input: &HashKeyInput<'_>) -> (u64, String) {
        let body = format!(
            "{}|{}|{}|{}",
            input.model,
            input.messages_json,
            input.params_json,
            input.system_prompt.unwrap_or(""),
        );
        let mut hasher = seeded_hasher();
        body.hash(&mut hasher);
        (hasher.finish(), body)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TenantScopedStrategy;

impl HashKeyStrategy for TenantScopedStrategy {
    fn key_for(&self, input: &HashKeyInput<'_>) -> (u64, String) {
        let body = format!(
            "tenant:{}|{}|{}|{}|{}",
            input.tenant_id.unwrap_or("__global__"),
            input.model,
            input.messages_json,
            input.params_json,
            input.system_prompt.unwrap_or(""),
        );
        let mut hasher = seeded_hasher();
        body.hash(&mut hasher);
        (hasher.finish(), body)
    }
}
