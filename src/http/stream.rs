use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use memchr::memchr;
use pin_project_lite::pin_project;
#[cfg(feature = "default-http")]
pub use tokio_util::sync::CancellationToken;

use crate::error::{HiLlmError, HiLlmResult};
use crate::types::ChatCompletionChunk;

use super::request::with_retry;

const MAX_BUFFER_BYTES: usize = 1024 * 1024; // 1 MiB

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        skip_all,
        fields(
            http.method = "POST",
            http.url = %url,
            http.status_code = tracing::field::Empty,
            http.retry_count = tracing::field::Empty,
        )
    )
)]
pub async fn post_stream<P>(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
    parse_event: P,
) -> HiLlmResult<crate::client::BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>
where
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>> + Send + 'static,
{
    let mut retry_count = 0u32;

    let resp = with_retry(max_retries, || {
        let mut builder = client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.clone());
        if let Some((name, value)) = auth_header {
            builder = builder.header(name, value);
        }
        for (name, value) in extra_headers {
            builder = builder.header(*name, *value);
        }
        retry_count += 1;
        builder.send()
    })
    .await?;

    #[cfg(feature = "tracing")]
    {
        let span = tracing::Span::current();
        span.record("http.status_code", resp.status().as_u16());
        span.record("http.retry_count", retry_count.saturating_sub(1));
    }

    let byte_stream = resp.bytes_stream();
    let stream = SseParser::new(byte_stream, parse_event, None);
    Ok(Box::pin(stream))
}

#[cfg(feature = "default-http")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)] // The cancel token is the necessary 8th arg.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        skip_all,
        fields(
            http.method = "POST",
            http.url = %url,
            http.status_code = tracing::field::Empty,
            http.retry_count = tracing::field::Empty,
        )
    )
)]
pub async fn post_stream_with_cancel<P>(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
    parse_event: P,
    cancel: CancellationToken,
) -> HiLlmResult<crate::client::BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>
where
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>> + Send + 'static,
{
    let mut retry_count = 0u32;

    let resp = with_retry(max_retries, || {
        let mut builder = client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.clone());
        if let Some((name, value)) = auth_header {
            builder = builder.header(name, value);
        }
        for (name, value) in extra_headers {
            builder = builder.header(*name, *value);
        }
        retry_count += 1;
        builder.send()
    })
    .await?;

    #[cfg(feature = "tracing")]
    {
        let span = tracing::Span::current();
        span.record("http.status_code", resp.status().as_u16());
        span.record("http.retry_count", retry_count.saturating_sub(1));
    }

    let byte_stream = resp.bytes_stream();
    let stream = SseParser::new(byte_stream, parse_event, Some(cancel));
    Ok(Box::pin(stream))
}

#[cfg(feature = "default-http")]
type CancelField = Option<CancellationToken>;

#[cfg(not(feature = "default-http"))]
type CancelField = Option<std::convert::Infallible>;

pin_project! {
    struct SseParser<S, P> {
        #[pin]
        inner: S,
        buffer: String,
        cursor: usize,
        done: bool,
        parse_event: P,
        cancel: CancelField,
    }

    impl<S, P> PinnedDrop for SseParser<S, P> {
        fn drop(this: Pin<&mut Self>) {
            let _ = this;
        }
    }
}

impl<S, P> SseParser<S, P>
where
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    fn new(inner: S, parse_event: P, cancel: CancelField) -> Self {
        Self {
            inner,
            buffer: String::with_capacity(4096),
            cursor: 0,
            done: false,
            parse_event,
            cancel,
        }
    }
}

impl<S, P> Stream for SseParser<S, P>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>>,
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        #[cfg(feature = "default-http")]
        if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            #[cfg(feature = "tracing")]
            tracing::debug!("SSE stream cancelled by downstream disconnect");
            *this.done = true;
            return Poll::Ready(None);
        }

        loop {
            if let Some(offset) = memchr(b'\n', &this.buffer.as_bytes()[*this.cursor..]) {
                let newline_pos = *this.cursor + offset;
                let line = this.buffer[*this.cursor..newline_pos]
                    .trim_end_matches('\r')
                    .trim();

                if line.is_empty() || line.starts_with(':') {
                    *this.cursor = newline_pos + 1;
                    compact_if_needed(this.buffer, this.cursor);
                    continue;
                }

                if let Some(raw) = line.strip_prefix("data:") {
                    let data = raw.strip_prefix(' ').unwrap_or(raw).trim();
                    if data == "[DONE]" {
                        *this.cursor = newline_pos + 1;
                        compact_if_needed(this.buffer, this.cursor);
                        return Poll::Ready(None);
                    }

                    let result = (this.parse_event)(data);
                    *this.cursor = newline_pos + 1;
                    compact_if_needed(this.buffer, this.cursor);
                    match result {
                        Ok(None) => continue,
                        Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }

                *this.cursor = newline_pos + 1;
                compact_if_needed(this.buffer, this.cursor);
                continue;
            }

            if *this.done {
                let remaining = this.buffer.len() - *this.cursor;
                if remaining > 0 {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        leftover_bytes = remaining,
                        preview = &this.buffer[*this.cursor..(*this.cursor + remaining.min(64))],
                        "SSE stream ended with unterminated data in buffer; dropping partial line"
                    );
                    this.buffer.clear();
                    *this.cursor = 0;
                }
                return Poll::Ready(None);
            }

            #[cfg(feature = "default-http")]
            if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
                #[cfg(feature = "tracing")]
                tracing::debug!("SSE stream cancelled while waiting for next chunk");
                *this.done = true;
                return Poll::Ready(None);
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if this.buffer.len() + bytes.len() > MAX_BUFFER_BYTES {
                        *this.done = true;
                        return Poll::Ready(Some(Err(HiLlmError::Streaming {
                            message: format!(
                                "SSE buffer exceeded {MAX_BUFFER_BYTES} bytes; stream aborted"
                            ),
                        })));
                    }
                    match std::str::from_utf8(&bytes) {
                        Ok(s) => this.buffer.push_str(s),
                        Err(e) => {
                            *this.done = true;
                            return Poll::Ready(Some(Err(HiLlmError::Streaming {
                                message: format!("invalid UTF-8 in SSE stream: {e}"),
                            })));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(HiLlmError::from(e))));
                }
                Poll::Ready(None) => {
                    *this.done = true;
                    continue;
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

fn compact_if_needed(buffer: &mut String, cursor: &mut usize) {
    if *cursor > buffer.len() / 2 {
        buffer.drain(..*cursor);
        *cursor = 0;
    }
}
