use std::time::Duration;

#[derive(Clone, Debug)]
pub struct TransportConfig {
    pub pool_max_idle_per_host: usize,
    pub pool_idle_timeout: Option<Duration>,
    pub tcp_keepalive: Option<Duration>,
    pub dns_cache_ttl: Option<Duration>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            pool_max_idle_per_host: 32,
            pool_idle_timeout: Some(Duration::from_secs(90)),
            tcp_keepalive: Some(Duration::from_secs(60)),
            dns_cache_ttl: Some(Duration::from_secs(30)),
        }
    }
}

impl TransportConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pool_max_idle_per_host(mut self, count: usize) -> Self {
        self.pool_max_idle_per_host = count;
        self
    }

    pub fn with_pool_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.pool_idle_timeout = timeout;
        self
    }

    pub fn with_tcp_keepalive(mut self, interval: Option<Duration>) -> Self {
        self.tcp_keepalive = interval;
        self
    }

    pub fn with_dns_cache_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.dns_cache_ttl = ttl;
        self
    }

    pub fn apply_to_builder(&self, builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        let builder = builder
            .pool_max_idle_per_host(self.pool_max_idle_per_host)
            .pool_idle_timeout(self.pool_idle_timeout)
            .tcp_keepalive(self.tcp_keepalive);

        // dns_cache_ttl: deferred — reqwest 0.13 has no DNS-cache TTL setter on
        // ClientBuilder.  The field is stored for future use.
        let _ = self.dns_cache_ttl;

        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = TransportConfig::default();
        assert_eq!(cfg.pool_max_idle_per_host, 32);
        assert_eq!(cfg.pool_idle_timeout, Some(Duration::from_secs(90)));
        assert_eq!(cfg.tcp_keepalive, Some(Duration::from_secs(60)));
        assert_eq!(cfg.dns_cache_ttl, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_builder_chain() {
        let cfg = TransportConfig::new()
            .with_pool_max_idle_per_host(16)
            .with_pool_idle_timeout(Some(Duration::from_secs(45)))
            .with_tcp_keepalive(Some(Duration::from_secs(120)))
            .with_dns_cache_ttl(Some(Duration::from_secs(60)));

        assert_eq!(cfg.pool_max_idle_per_host, 16);
        assert_eq!(cfg.pool_idle_timeout, Some(Duration::from_secs(45)));
        assert_eq!(cfg.tcp_keepalive, Some(Duration::from_secs(120)));
        assert_eq!(cfg.dns_cache_ttl, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_disable_pooling() {
        let cfg = TransportConfig::new().with_pool_max_idle_per_host(0);
        assert_eq!(cfg.pool_max_idle_per_host, 0);
    }

    #[test]
    fn test_disable_keepalive() {
        let cfg = TransportConfig::new().with_tcp_keepalive(None);
        assert_eq!(cfg.tcp_keepalive, None);
    }

    #[test]
    fn test_disable_dns_cache() {
        let cfg = TransportConfig::new().with_dns_cache_ttl(None);
        assert_eq!(cfg.dns_cache_ttl, None);
    }

    #[test]
    fn test_apply_to_builder_builds_client_with_non_default_config() {
        let cfg = TransportConfig::new()
            .with_pool_max_idle_per_host(128)
            .with_pool_idle_timeout(Some(Duration::from_secs(45)))
            .with_tcp_keepalive(Some(Duration::from_secs(30)))
            .with_dns_cache_ttl(Some(Duration::from_secs(60)));

        let builder = reqwest::Client::builder();
        let builder = cfg.apply_to_builder(builder);
        let client = builder.build();
        assert!(
            client.is_ok(),
            "reqwest::Client::build() failed with non-default TransportConfig"
        );
    }

    #[test]
    fn test_apply_to_builder_with_pooling_disabled() {
        let cfg = TransportConfig::new()
            .with_pool_max_idle_per_host(0)
            .with_pool_idle_timeout(None)
            .with_tcp_keepalive(None)
            .with_dns_cache_ttl(None);

        let client = cfg.apply_to_builder(reqwest::Client::builder()).build();
        assert!(
            client.is_ok(),
            "reqwest::Client::build() failed with pooling disabled"
        );
    }
}
