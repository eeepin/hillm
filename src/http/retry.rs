use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn should_retry(
    status: u16,
    attempt: u32,
    max_retries: u32,
    retry_after: Option<Duration>,
) -> Option<Duration> {
    if attempt >= max_retries {
        return None;
    }

    if !matches!(status, 429 | 500 | 502 | 503 | 504) {
        return None;
    }

    if status == 429
        && let Some(server_delay) = retry_after
    {
        return Some(server_delay.min(Duration::from_secs(60)));
    }

    let base_delay = Duration::from_secs(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
    let capped = base_delay.min(Duration::from_secs(30));

    Some(jittered(capped))
}

fn jittered(delay: Duration) -> Duration {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let jitter_factor = 0.5 + (f64::from(nanos % 1000) / 2000.0);
    delay.mul_f64(jitter_factor)
}

pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();

    if let Ok(secs) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }

    None
}
