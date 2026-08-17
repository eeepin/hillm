use crate::error::HiLLMError;
#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::net::IpAddr;
#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};
use url::Url;

#[derive(Debug, Clone, Default)]
pub enum OutboundPolicy {
    #[default]
    Off,
    DenyPrivate,
    Allowlist(Vec<Url>),
}

impl OutboundPolicy {
    /// Returns `DenyPrivate`, the recommended default for server-side
    /// deployments that process untrusted input (e.g. user-supplied URLs or
    /// model-controlled endpoints). This prevents SSRF against private
    /// address ranges (loopback, link-local, metadata services, etc.).
    #[must_use]
    pub fn server_default() -> Self {
        OutboundPolicy::DenyPrivate
    }

    /// Returns `true` if this policy is `Off` (no DNS or address checks).
    #[must_use]
    pub fn is_off(&self) -> bool {
        matches!(self, OutboundPolicy::Off)
    }
}

/// Resolve the default outbound policy from the `HILLM_OUTBOUND_POLICY`
/// environment variable. Recognized values:
///
/// - `"off"` (default when unset or empty) — no DNS or address checks
/// - `"deny_private"` — reject private, loopback, link-local, multicast
///
/// Unknown values emit a warning on stderr and fall back to `Off`.
fn default_policy_from_env() -> OutboundPolicy {
    match std::env::var("HILLM_OUTBOUND_POLICY")
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "deny_private" => OutboundPolicy::DenyPrivate,
        "off" | "" => OutboundPolicy::Off,
        other => {
            eprintln!("hillm: warning: unknown HILLM_OUTBOUND_POLICY value '{other}', using Off");
            OutboundPolicy::Off
        }
    }
}

// ---------------------------------------------------------------------------
// Instance-based policy validator
// ---------------------------------------------------------------------------

/// Per-instance outbound policy validator.
///
/// Wraps an [`OutboundPolicy`] behind a `RwLock` so it can be mutated at
/// runtime while being shared across threads. Unlike the process-global
/// [`set_outbound_policy`] / [`current_policy`] functions, a validator is
/// scoped to a single client or component, preventing different tenants from
/// interfering with each other.
///
/// # Examples
///
/// ```
/// use hillm::provider::OutboundPolicy;
/// use hillm::provider::outbound_policy::OutboundPolicyValidator;
///
/// let validator = OutboundPolicyValidator::new(OutboundPolicy::DenyPrivate);
/// // validator.validate_url_sync("http://127.0.0.1/") would fail (loopback)
/// ```
#[derive(Debug)]
pub struct OutboundPolicyValidator {
    policy: RwLock<OutboundPolicy>,
}

impl OutboundPolicyValidator {
    /// Create a new validator with the given initial policy.
    pub fn new(policy: OutboundPolicy) -> Self {
        Self {
            policy: RwLock::new(policy),
        }
    }

    /// Replace the current policy.
    pub fn set_policy(&self, policy: OutboundPolicy) {
        *self.policy.write().expect("outbound policy lock poisoned") = policy;
    }

    /// Read the current policy.
    #[must_use]
    pub fn current_policy(&self) -> OutboundPolicy {
        self.policy
            .read()
            .expect("outbound policy lock poisoned")
            .clone()
    }

    /// Validate a URL against this validator's policy (sync).
    ///
    /// Always parses the URL and rejects non-http/https schemes, even when
    /// the policy is `Off`. When the policy is not `Off`, performs additional
    /// address-range checks on the host.
    pub fn validate_url_sync(&self, raw_url: &str) -> Result<(), HiLLMError> {
        let policy = self.current_policy();
        validate_url_with_policy(&policy, raw_url)
    }

    /// Validate a URL against this validator's policy (async).
    ///
    /// In addition to the sync checks, `DenyPrivate` performs DNS resolution
    /// and rejects URLs whose host resolves to a forbidden address.
    #[cfg(all(
        any(feature = "default-http", feature = "wasm-http"),
        not(target_arch = "wasm32")
    ))]
    pub async fn validate_url(&self, raw_url: &str) -> Result<(), HiLLMError> {
        let policy = self.current_policy();
        validate_url_with_policy_async(&policy, raw_url).await
    }
}

impl Default for OutboundPolicyValidator {
    fn default() -> Self {
        Self::new(default_policy_from_env())
    }
}

impl Clone for OutboundPolicyValidator {
    fn clone(&self) -> Self {
        Self {
            policy: RwLock::new(self.current_policy()),
        }
    }
}

// ---------------------------------------------------------------------------
// Process-global convenience API (delegates to a global validator)
// ---------------------------------------------------------------------------

static GLOBAL_VALIDATOR: OnceLock<OutboundPolicyValidator> = OnceLock::new();

fn global_validator() -> &'static OutboundPolicyValidator {
    GLOBAL_VALIDATOR.get_or_init(OutboundPolicyValidator::default)
}

pub fn set_outbound_policy(policy: OutboundPolicy) {
    global_validator().set_policy(policy);
}

pub fn current_policy() -> OutboundPolicy {
    global_validator().current_policy()
}

#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
pub async fn validate_outbound_url(raw_url: &str) -> Result<(), HiLLMError> {
    global_validator().validate_url(raw_url).await
}

pub fn validate_outbound_url_sync(raw_url: &str) -> Result<(), HiLLMError> {
    global_validator().validate_url_sync(raw_url)
}

// ---------------------------------------------------------------------------
// Core validation logic (extracted for reuse by both global and instance APIs)
// ---------------------------------------------------------------------------

#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
async fn validate_url_with_policy_async(
    policy: &OutboundPolicy,
    raw_url: &str,
) -> Result<(), HiLLMError> {
    // Always parse URL and validate scheme, even when policy is Off
    let url = Url::parse(raw_url).map_err(|e| HiLLMError::OutboundForbidden {
        url: raw_url.to_string(),
        reason: format!("invalid URL: {e}"),
    })?;

    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(HiLLMError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("scheme '{other}' is not allowed; only http/https"),
            });
        }
    }

    // When policy is Off, skip DNS and address range checks
    match policy {
        OutboundPolicy::Off => Ok(()),
        OutboundPolicy::DenyPrivate => check_deny_private(&url, raw_url).await,
        OutboundPolicy::Allowlist(allowed) => check_allowlist(&url, raw_url, allowed),
    }
}

fn validate_url_with_policy(policy: &OutboundPolicy, raw_url: &str) -> Result<(), HiLLMError> {
    // Always parse URL and validate scheme, even when policy is Off
    let url = Url::parse(raw_url).map_err(|e| HiLLMError::OutboundForbidden {
        url: raw_url.to_string(),
        reason: format!("invalid URL: {e}"),
    })?;

    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(HiLLMError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("scheme '{other}' is not allowed; only http/https"),
            });
        }
    }

    // When policy is Off, skip address range checks
    if matches!(policy, OutboundPolicy::Off) {
        return Ok(());
    }

    match url.host() {
        Some(url::Host::Ipv4(v4)) if is_forbidden(IpAddr::V4(v4)) => {
            return Err(HiLLMError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("host is a forbidden address {v4}"),
            });
        }
        Some(url::Host::Ipv6(v6)) if is_forbidden(IpAddr::V6(v6)) => {
            return Err(HiLLMError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("host is a forbidden address {v6}"),
            });
        }
        _ => {}
    }

    if let OutboundPolicy::Allowlist(allowed) = policy {
        return check_allowlist(&url, raw_url, allowed);
    }

    Ok(())
}

#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
async fn check_deny_private(url: &Url, raw: &str) -> Result<(), HiLLMError> {
    let host = url
        .host_str()
        .ok_or_else(|| HiLLMError::OutboundForbidden {
            url: raw.to_string(),
            reason: "URL has no host".into(),
        })?;

    let port = url.port_or_known_default().unwrap_or(0);

    let addrs = tokio::net::lookup_host(format!("{host}:{port}"))
        .await
        .map_err(|e| HiLLMError::OutboundForbidden {
            url: raw.to_string(),
            reason: format!("DNS resolution failed: {e}"),
        })?;

    for sa in addrs {
        if is_forbidden(sa.ip()) {
            return Err(HiLLMError::OutboundForbidden {
                url: raw.to_string(),
                reason: format!("host resolves to forbidden address {}", sa.ip()),
            });
        }
    }
    Ok(())
}

fn check_allowlist(url: &Url, raw: &str, allowed: &[Url]) -> Result<(), HiLLMError> {
    let origin_match = allowed.iter().any(|a| {
        a.scheme() == url.scheme()
            && a.host_str() == url.host_str()
            && a.port_or_known_default() == url.port_or_known_default()
    });
    if origin_match {
        Ok(())
    } else {
        Err(HiLLMError::OutboundForbidden {
            url: raw.to_string(),
            reason: "URL not in outbound allowlist".into(),
        })
    }
}

pub fn is_forbidden(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_unspecified()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_multicast()
                || v4.is_broadcast()
                || is_cgnat(v4)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || is_unique_local_v6(v6)
                || is_link_local_v6(v6)
                || v6
                    .to_ipv4_mapped()
                    .map(|m| is_forbidden(IpAddr::V4(m)))
                    .unwrap_or(false)
        }
    }
}

fn is_cgnat(ip: std::net::Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();
    a == 100 && (64..=127).contains(&b)
}

fn is_unique_local_v6(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

fn is_link_local_v6(ip: std::net::Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

// GuardedResolver

/// DNS resolver that enforces an [`OutboundPolicyValidator`] at the reqwest
/// DNS layer. Every DNS lookup is checked against the validator's current
/// policy; addresses in forbidden ranges are rejected before they reach the
/// TCP connect.
///
/// When constructed without an explicit validator (via [`guarded_resolver`]),
/// the process-global policy is used. When constructed with
/// [`GuardedResolver::new`], the given validator is used — enabling
/// per-client isolation.
#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
pub struct GuardedResolver {
    validator: Option<Arc<OutboundPolicyValidator>>,
}

#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
impl GuardedResolver {
    /// Create a resolver bound to a specific validator instance.
    ///
    /// The validator is shared via `Arc` so that policy changes made through
    /// the original validator are visible to this resolver immediately.
    #[must_use]
    pub fn new(validator: Arc<OutboundPolicyValidator>) -> Self {
        Self {
            validator: Some(validator),
        }
    }

    /// Create a resolver that reads from the process-global policy.
    ///
    /// Equivalent to [`guarded_resolver()`].
    #[must_use]
    pub fn from_global() -> Self {
        Self { validator: None }
    }
}

#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
impl Resolve for GuardedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        // Snapshot the validator reference so the future is 'static.
        let validator = self.validator.clone();
        Box::pin(async move {
            let policy = match &validator {
                Some(v) => v.current_policy(),
                None => current_policy(),
            };
            let host = name.as_str().to_string();

            let addrs: Vec<_> = tokio::net::lookup_host(format!("{host}:0"))
                .await
                .map_err(|e| {
                    let err: Box<dyn std::error::Error + Send + Sync> = Box::new(e);
                    err
                })?
                .collect();

            if !matches!(policy, OutboundPolicy::Off) {
                for sa in &addrs {
                    if is_forbidden(sa.ip()) {
                        let err: Box<dyn std::error::Error + Send + Sync> = format!(
                            "outbound DNS resolution for '{host}' produced \
                                forbidden address {}",
                            sa.ip()
                        )
                        .into();
                        return Err(err);
                    }
                }
            }

            let iter: Addrs = Box::new(addrs.into_iter());
            Ok(iter)
        })
    }
}

#[cfg(all(
    any(feature = "default-http", feature = "wasm-http"),
    not(target_arch = "wasm32")
))]
pub fn guarded_resolver() -> Arc<GuardedResolver> {
    Arc::new(GuardedResolver::from_global())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn with_policy<F: FnOnce()>(policy: OutboundPolicy, f: F) {
        set_outbound_policy(policy);
        f();
        set_outbound_policy(OutboundPolicy::Off);
    }

    #[test]
    fn is_forbidden_recognizes_private_ranges() {
        let cases: &[(&str, bool)] = &[
            ("10.0.0.1", true),
            ("172.16.0.1", true),
            ("192.168.1.1", true),
            ("127.0.0.1", true),
            ("169.254.0.1", true),
            ("100.100.0.1", true),     // CGNAT
            ("0.0.0.0", true),         // unspecified
            ("255.255.255.255", true), // broadcast
            ("224.0.0.1", true),       // multicast
            ("8.8.8.8", false),        // public DNS — allowed
            ("1.1.1.1", false),        // Cloudflare — allowed
        ];
        for (addr, expected) in cases {
            let ip: IpAddr = addr.parse().expect("valid IP");
            assert_eq!(
                is_forbidden(ip),
                *expected,
                "is_forbidden({addr}) should be {expected}"
            );
        }
    }

    #[test]
    fn is_forbidden_ipv6_loopback() {
        let ip: IpAddr = "::1".parse().expect("::1 is a valid IPv6 address");
        assert!(is_forbidden(ip));
    }

    #[test]
    fn is_forbidden_ipv6_ula() {
        let ip: IpAddr = "fc00::1".parse().expect("fc00::1 is a valid IPv6 address");
        assert!(is_forbidden(ip));
    }

    #[test]
    fn is_forbidden_ipv6_link_local() {
        let ip: IpAddr = "fe80::1".parse().expect("fe80::1 is a valid IPv6 address");
        assert!(is_forbidden(ip));
    }

    #[test]
    fn is_forbidden_ipv6_public() {
        let ip: IpAddr = "2001:4860:4860::8888"
            .parse()
            .expect("Google DNS is a valid IPv6 address"); // Google DNS
        assert!(!is_forbidden(ip));
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_off_rejects_non_http_scheme() {
        with_policy(OutboundPolicy::Off, || {
            // Even with Off policy, non-http/https schemes should be rejected
            let result = validate_outbound_url_sync("ftp://example.com/");
            assert!(
                result.is_err(),
                "ftp:// scheme should be rejected even with Off policy"
            );
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("scheme"),
                "error should mention 'scheme': {err}"
            );

            let result = validate_outbound_url_sync("file:///etc/passwd");
            assert!(
                result.is_err(),
                "file:// scheme should be rejected even with Off policy"
            );
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_deny_private_rejects_loopback() {
        with_policy(OutboundPolicy::DenyPrivate, || {
            let result = validate_outbound_url_sync("http://127.0.0.1/");
            assert!(result.is_err(), "loopback should be rejected");
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("forbidden"),
                "error message should mention 'forbidden': {err}"
            );
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_deny_private_rejects_metadata_ip() {
        with_policy(OutboundPolicy::DenyPrivate, || {
            let result = validate_outbound_url_sync("http://169.254.169.254/");
            assert!(result.is_err(), "metadata IP should be rejected");
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_deny_private_rejects_ula() {
        with_policy(OutboundPolicy::DenyPrivate, || {
            let result = validate_outbound_url_sync("http://[fc00::1]/");
            assert!(result.is_err(), "ULA address should be rejected");
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_deny_private_rejects_link_local_v6() {
        with_policy(OutboundPolicy::DenyPrivate, || {
            let result = validate_outbound_url_sync("http://[fe80::1]/");
            assert!(result.is_err(), "IPv6 link-local should be rejected");
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_deny_private_rejects_unknown_scheme() {
        with_policy(OutboundPolicy::DenyPrivate, || {
            let result = validate_outbound_url_sync("ftp://example.com/");
            assert!(result.is_err(), "ftp:// scheme should be rejected");
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("scheme"),
                "error should mention 'scheme': {err}"
            );
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_allowlist_accepts_exact_origin() {
        let allowed =
            vec![Url::parse("https://api.openai.com").expect("openai URL should be valid")];
        with_policy(OutboundPolicy::Allowlist(allowed), || {
            let result = validate_outbound_url_sync("https://api.openai.com/v1/chat/completions");
            assert!(
                result.is_ok(),
                "same-origin with different path should pass"
            );
        });
    }

    #[test]
    #[serial(outbound_policy)]
    fn validate_sync_allowlist_rejects_other_host() {
        let allowed =
            vec![Url::parse("https://api.openai.com").expect("openai URL should be valid")];
        with_policy(OutboundPolicy::Allowlist(allowed), || {
            let result = validate_outbound_url_sync("https://api.anthropic.com/");
            assert!(result.is_err(), "different host should be rejected");
        });
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_off_rejects_non_http_scheme() {
        set_outbound_policy(OutboundPolicy::Off);
        // Even with Off policy, non-http/https schemes should be rejected
        let result = validate_outbound_url("ftp://example.com/").await;
        assert!(
            result.is_err(),
            "ftp:// scheme should be rejected even with Off policy"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("scheme"),
            "error should mention 'scheme': {err}"
        );

        let result = validate_outbound_url("file:///etc/passwd").await;
        assert!(
            result.is_err(),
            "file:// scheme should be rejected even with Off policy"
        );
        set_outbound_policy(OutboundPolicy::Off);
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_loopback() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://127.0.0.1/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(
            result.is_err(),
            "loopback should be rejected by DenyPrivate"
        );
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_metadata_ip() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://169.254.169.254/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "AWS metadata IP should be rejected");
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_ula() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://[fc00::1]/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "ULA address should be rejected");
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_link_local_v6() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://[fe80::1]/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "IPv6 link-local should be rejected");
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_unknown_scheme() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("ftp://example.com/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "ftp:// scheme should be rejected");
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_allowlist_accepts_exact_origin() {
        let allowed =
            vec![Url::parse("https://api.openai.com").expect("openai URL should be valid")];
        set_outbound_policy(OutboundPolicy::Allowlist(allowed));
        let result = validate_outbound_url("https://api.openai.com/v1/chat/completions").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(
            result.is_ok(),
            "same-origin with different path should pass"
        );
    }

    #[cfg(any(feature = "default-http", feature = "wasm-http"))]
    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_allowlist_rejects_other_host() {
        let allowed =
            vec![Url::parse("https://api.openai.com").expect("openai URL should be valid")];
        set_outbound_policy(OutboundPolicy::Allowlist(allowed));
        let result = validate_outbound_url("https://api.anthropic.com/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "different host should be rejected");
    }

    // -----------------------------------------------------------------------
    // OutboundPolicyValidator (instance-based API) tests
    // -----------------------------------------------------------------------

    #[test]
    fn validator_is_independent_of_global_policy() {
        // Set global to Off; validator with DenyPrivate must still reject loopback.
        set_outbound_policy(OutboundPolicy::Off);

        let validator = OutboundPolicyValidator::new(OutboundPolicy::DenyPrivate);
        let result = validator.validate_url_sync("http://127.0.0.1/");
        assert!(
            result.is_err(),
            "instance validator must reject loopback regardless of global policy"
        );
    }

    #[test]
    fn validator_mutations_do_not_affect_global() {
        set_outbound_policy(OutboundPolicy::Off);

        let validator = OutboundPolicyValidator::new(OutboundPolicy::Off);
        validator.set_policy(OutboundPolicy::DenyPrivate);
        assert!(matches!(
            validator.current_policy(),
            OutboundPolicy::DenyPrivate
        ));
        // Global must remain Off.
        assert!(matches!(current_policy(), OutboundPolicy::Off));
    }

    #[test]
    fn two_validators_are_independent() {
        let v1 = OutboundPolicyValidator::new(OutboundPolicy::DenyPrivate);
        let v2 = OutboundPolicyValidator::new(OutboundPolicy::Off);

        // v1 rejects loopback, v2 allows it.
        assert!(v1.validate_url_sync("http://127.0.0.1/").is_err());
        assert!(v2.validate_url_sync("http://127.0.0.1/").is_ok());
    }

    #[test]
    fn validator_clone_is_independent() {
        let v1 = OutboundPolicyValidator::new(OutboundPolicy::Off);
        let v2 = v1.clone();

        v1.set_policy(OutboundPolicy::DenyPrivate);
        // Clone should not see the mutation (it has its own RwLock).
        assert!(matches!(v2.current_policy(), OutboundPolicy::Off));
    }

    #[test]
    fn server_default_returns_deny_private() {
        assert!(matches!(
            OutboundPolicy::server_default(),
            OutboundPolicy::DenyPrivate
        ));
    }

    #[test]
    fn is_off_returns_true_only_for_off() {
        assert!(OutboundPolicy::Off.is_off());
        assert!(!OutboundPolicy::DenyPrivate.is_off());
        assert!(!OutboundPolicy::Allowlist(vec![]).is_off());
    }

    #[test]
    fn validator_allowlist_enforces_per_instance() {
        let allowed = vec![Url::parse("https://api.openai.com").unwrap()];
        let validator = OutboundPolicyValidator::new(OutboundPolicy::Allowlist(allowed));

        assert!(
            validator
                .validate_url_sync("https://api.openai.com/v1/chat/completions")
                .is_ok()
        );
        assert!(
            validator
                .validate_url_sync("https://api.anthropic.com/")
                .is_err()
        );
    }

    #[test]
    fn validator_rejects_non_http_scheme_even_when_off() {
        let validator = OutboundPolicyValidator::new(OutboundPolicy::Off);
        assert!(validator.validate_url_sync("ftp://example.com/").is_err());
        assert!(validator.validate_url_sync("file:///etc/passwd").is_err());
    }

    #[test]
    fn validator_runtime_policy_change_takes_effect() {
        let validator = OutboundPolicyValidator::new(OutboundPolicy::Off);
        assert!(validator.validate_url_sync("http://127.0.0.1/").is_ok());

        validator.set_policy(OutboundPolicy::DenyPrivate);
        assert!(validator.validate_url_sync("http://127.0.0.1/").is_err());

        validator.set_policy(OutboundPolicy::Off);
        assert!(validator.validate_url_sync("http://127.0.0.1/").is_ok());
    }
}
