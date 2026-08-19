use crate::error::{HiLlmError, HiLlmResult};

// 1 MiB
pub const SSE_BUFFER_MAX_BYTES: usize = 1024 * 1024;

// 16 MiB
pub const EVENT_STREAM_BUFFER_MAX_BYTES: usize = 16 * 1024 * 1024;

// 32 MiB
pub const RESPONSE_BODY_MAX_BYTES: usize = 32 * 1024 * 1024;

pub const CHUNK_ACCUMULATION_MAX_BYTES: usize = RESPONSE_BODY_MAX_BYTES;

/// Bound check helper. Assert that `current_len + incoming` does not exceed `limit`.
pub fn check_bound(
    context: &str,
    current_len: usize,
    incoming: usize,
    limit: usize,
) -> HiLlmResult<()> {
    if current_len.saturating_add(incoming) > limit {
        #[cfg(feature = "tracing")]
        tracing::warn!(
            context,
            current_len,
            incoming,
            limit,
            "buffer limit exceeded; aborting stream"
        );
        return Err(HiLlmError::Streaming {
            message: format!("{context} buffer exceeded {limit} bytes; aborting"),
        });
    }
    Ok(())
}
