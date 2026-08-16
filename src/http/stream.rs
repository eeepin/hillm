use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use pin_project_lite::pin_project;

use crate::error::HiLLMResult;
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
) -> HiLLMResult<crate::client::BoxStream<'static, HiLLMResult<ChatCompletionChunk>>>
where
    P: Fn(&str) -> HiLLMResult<Option<ChatCompletionChunk>> + Send + 'static,
{
    let resp =
        send_stream_request(client, url, auth_header, extra_headers, body, max_retries).await?;
    let byte_stream = resp.bytes_stream();
    let stream = SSEParser::new(byte_stream, parse_event);
    Ok(Box::pin(stream))
}

/// POST a request expecting an SSE stream whose events are decoded into a
/// protocol-native event type by `parse_event`.
///
/// This is the transport-level helper for API routes whose stream events are
/// not Chat Completions chunks (e.g. the OpenAI Responses API). The parser
/// decides when the stream is finished; the transport layer stays
/// protocol-agnostic.
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
pub async fn post_typed_stream<T, P>(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
    parse_event: P,
) -> HiLLMResult<crate::client::BoxStream<'static, HiLLMResult<T>>>
where
    T: Send + 'static,
    P: Fn(&str) -> HiLLMResult<Option<T>> + Send + 'static,
{
    let resp =
        send_stream_request(client, url, auth_header, extra_headers, body, max_retries).await?;
    let byte_stream = resp.bytes_stream();
    let stream = TypedSSEParser::new(byte_stream, parse_event);
    Ok(Box::pin(stream))
}

async fn send_stream_request(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
) -> HiLLMResult<reqwest::Response> {
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

    Ok(resp)
}

pin_project! {
    /// Thin wrapper around SSEStream that maps SSEEvent to ChatCompletionChunk
    /// via the parse_event closure.
    struct SSEParser<S, P> {
        #[pin]
        stream: SSEStream<S>,
        parse_event: P,
    }
}

impl<S, P> SSEParser<S, P>
where
    P: Fn(&str) -> HiLLMResult<Option<ChatCompletionChunk>>,
{
    fn new(inner: S, parse_event: P) -> Self {
        Self {
            stream: SSEStream::new(inner),
            parse_event,
        }
    }
}

impl<S, P> Stream for SSEParser<S, P>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
    P: Fn(&str) -> HiLLMResult<Option<ChatCompletionChunk>>,
{
    type Item = HiLLMResult<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.stream.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Some(Ok(event))) => match (this.parse_event)(&event.data) {
                    Ok(None) => continue,
                    Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                    Err(e) => return Poll::Ready(Some(Err(e))),
                },
            }
        }
    }
}

pin_project! {
    /// SSE stream whose events are decoded into a caller-chosen type `T`
    /// via the parse_event closure.
    struct TypedSSEParser<S, P, T> {
        #[pin]
        stream: SSEStream<S>,
        parse_event: P,
        _marker: std::marker::PhantomData<T>,
    }
}

impl<S, P, T> TypedSSEParser<S, P, T>
where
    P: Fn(&str) -> HiLLMResult<Option<T>>,
{
    fn new(inner: S, parse_event: P) -> Self {
        Self {
            stream: SSEStream::new(inner),
            parse_event,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<S, P, T> Stream for TypedSSEParser<S, P, T>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
    P: Fn(&str) -> HiLLMResult<Option<T>>,
{
    type Item = HiLLMResult<T>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            match this.stream.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Some(Ok(event))) => match (this.parse_event)(&event.data) {
                    Ok(None) => continue,
                    Ok(Some(item)) => return Poll::Ready(Some(Ok(item))),
                    Err(e) => return Poll::Ready(Some(Err(e))),
                },
            }
        }
    }
}
