use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use pin_project_lite::pin_project;

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
    let stream = SSEParser::new(byte_stream, parse_event);
    Ok(Box::pin(stream))
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
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
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
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<ChatCompletionChunk>;

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
