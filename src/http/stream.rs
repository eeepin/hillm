use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use pin_project_lite::pin_project;
#[cfg(feature = "default-http")]
pub use tokio_util::sync::CancellationToken;

use crate::error::HiLlmResult;
use crate::sse::SSEStream;
use crate::types::ChatCompletionChunk;

use super::request::with_retry;

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
    /// Thin wrapper around SSEStream that adds cancel-token support and
    /// maps SSEEvent to ChatCompletionChunk via the parse_event closure.
    struct SseParser<S, P> {
        #[pin]
        stream: SSEStream<S>,
        parse_event: P,
        cancel: CancelField,
    }
}

impl<S, P> SseParser<S, P>
where
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    fn new(inner: S, parse_event: P, cancel: CancelField) -> Self {
        Self {
            stream: SSEStream::new(inner),
            parse_event,
            cancel,
        }
    }
}

impl<S, P> Stream for SseParser<S, P>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Cancel check
        #[cfg(feature = "default-http")]
        if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            #[cfg(feature = "tracing")]
            tracing::debug!("SSE stream cancelled by downstream disconnect");
            return Poll::Ready(None);
        }

        loop {
            match this.stream.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Some(Ok(event))) => {
                    // [DONE] handling — OpenAI-specific sentinel
                    if event.data == "[DONE]" {
                        return Poll::Ready(None);
                    }
                    match (this.parse_event)(&event.data) {
                        Ok(None) => continue,
                        Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
            }
        }
    }
}
