use crate::error::HiLlmError;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::net::IpAddr;
use std::sync::{Arc, OnceLock, RwLock};
use url::Url;

#[derive(Debug, Default, Clone)]
pub enum OutboundPolicy {
    #[default]
    Off,
    DenyPrivate,
    Allowlist(Vec<Url>),
}

static GLOBAL_POLICY: OnceLock<RwLock<OutboundPolicy>> = OnceLock::new();

fn policy_lock() -> &'static RwLock<OutboundPolicy> {
    GLOBAL_POLICY.get_or_init(|| RwLock::new(OutboundPolicy::default()))
}

pub fn set_outbound_policy(policy: OutboundPolicy) {
    *policy_lock()
        .write()
        .expect("outbound policy lock poisoned") = policy;
}

pub fn current_policy() -> OutboundPolicy {
    policy_lock()
        .read()
        .expect("outbound policy lock poisoned")
        .clone()
}

pub async fn validate_outbound_url(raw_url: &str) -> Result<(), HiLlmError> {
    let policy = current_policy();
    if matches!(policy, OutboundPolicy::Off) {
        return Ok(());
    }

    let url = Url::parse(raw_url).map_err(|e| HiLlmError::OutboundForbidden {
        url: raw_url.to_string(),
        reason: format!("invalid URL: {e}"),
    })?;

    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(HiLlmError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("scheme '{other}' is not allowed; only http/https"),
            });
        }
    }

    match policy {
        OutboundPolicy::Off => Ok(()),
        OutboundPolicy::DenyPrivate => check_deny_private(&url, raw_url).await,
        OutboundPolicy::Allowlist(allowed) => check_allowlist(&url, raw_url, &allowed),
    }
}

pub fn validate_outbound_url_sync(raw_url: &str) -> Result<(), HiLlmError> {
    let policy = current_policy();
    if matches!(policy, OutboundPolicy::Off) {
        return Ok(());
    }

    let url = Url::parse(raw_url).map_err(|e| HiLlmError::OutboundForbidden {
        url: raw_url.to_string(),
        reason: format!("invalid URL: {e}"),
    })?;

    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(HiLlmError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("scheme '{other}' is not allowed; only http/https"),
            });
        }
    }

    match url.host() {
        Some(url::Host::Ipv4(v4)) if is_forbidden(IpAddr::V4(v4)) => {
            return Err(HiLlmError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("host is a forbidden address {v4}"),
            });
        }
        Some(url::Host::Ipv6(v6)) if is_forbidden(IpAddr::V6(v6)) => {
            return Err(HiLlmError::OutboundForbidden {
                url: raw_url.to_string(),
                reason: format!("host is a forbidden address {v6}"),
            });
        }
        _ => {}
    }

    if let OutboundPolicy::Allowlist(allowed) = policy {
        return check_allowlist(&url, raw_url, &allowed);
    }

    Ok(())
}

async fn check_deny_private(url: &Url, raw: &str) -> Result<(), HiLlmError> {
    let host = url
        .host_str()
        .ok_or_else(|| HiLlmError::OutboundForbidden {
            url: raw.to_string(),
            reason: "URL has no host".into(),
        })?;

    let port = url.port_or_known_default().unwrap_or(0);

    let addrs = tokio::net::lookup_host(format!("{host}:{port}"))
        .await
        .map_err(|e| HiLlmError::OutboundForbidden {
            url: raw.to_string(),
            reason: format!("DNS resolution failed: {e}"),
        })?;

    for sa in addrs {
        if is_forbidden(sa.ip()) {
            return Err(HiLlmError::OutboundForbidden {
                url: raw.to_string(),
                reason: format!("host resolves to forbidden address {}", sa.ip()),
            });
        }
    }
    Ok(())
}

fn check_allowlist(url: &Url, raw: &str, allowed: &[Url]) -> Result<(), HiLlmError> {
    let origin_match = allowed.iter().any(|a| {
        a.scheme() == url.scheme()
            && a.host_str() == url.host_str()
            && a.port_or_known_default() == url.port_or_known_default()
    });
    if origin_match {
        Ok(())
    } else {
        Err(HiLlmError::OutboundForbidden {
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

pub struct GuardedResolver;

impl Resolve for GuardedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let policy = current_policy();
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

pub fn guarded_resolver() -> Arc<GuardedResolver> {
    Arc::new(GuardedResolver)
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
    fn validate_sync_off_passes_everything() {
        with_policy(OutboundPolicy::Off, || {
            assert!(validate_outbound_url_sync("http://127.0.0.1/").is_ok());
            assert!(validate_outbound_url_sync("http://169.254.169.254/").is_ok());
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

    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_off_passes_everything() {
        set_outbound_policy(OutboundPolicy::Off);
        assert!(validate_outbound_url("http://127.0.0.1/").await.is_ok());
        assert!(
            validate_outbound_url("http://169.254.169.254/")
                .await
                .is_ok()
        );
    }

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

    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_metadata_ip() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://169.254.169.254/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "AWS metadata IP should be rejected");
    }

    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_ula() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://[fc00::1]/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "ULA address should be rejected");
    }

    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_link_local_v6() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("http://[fe80::1]/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "IPv6 link-local should be rejected");
    }

    #[tokio::test]
    #[serial(outbound_policy)]
    async fn validate_async_deny_private_rejects_unknown_scheme() {
        set_outbound_policy(OutboundPolicy::DenyPrivate);
        let result = validate_outbound_url("ftp://example.com/").await;
        set_outbound_policy(OutboundPolicy::Off);
        assert!(result.is_err(), "ftp:// scheme should be rejected");
    }

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
}
