use std::cell::RefCell;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures_core::Stream;
use pin_project_lite::pin_project;

use crate::error::{HiLlmError, HiLlmResult};
use crate::provider::StreamFormat;
use crate::sse::SSEStream;
use crate::types::ChatCompletionChunk;

#[cfg(feature = "default-http")]
pub use tokio_util::sync::CancellationToken;

#[cfg(feature = "default-http")]
type CancelField = Option<CancellationToken>;

const MAX_POOL_BUFFER_CAPACITY: usize = 64 * 1024;

thread_local! {
    static EGRESS_BYTES_POOL: RefCell<Option<BytesMut>> = const { RefCell::new(None) };
}

pub(crate) fn pool_acquire() -> BytesMut {
    EGRESS_BYTES_POOL.with(|cell| {
        cell.borrow_mut()
            .take()
            .map(|mut buf| {
                buf.clear();
                buf
            })
            .unwrap_or_else(|| BytesMut::with_capacity(4096))
    })
}

pub(crate) fn pool_release(buf: BytesMut) {
    if buf.capacity() <= MAX_POOL_BUFFER_CAPACITY {
        EGRESS_BYTES_POOL.with(|cell| {
            *cell.borrow_mut() = Some(buf);
        });
    }
}

pub trait ChunkMiddleware: Send + Sync {
    fn process(&self, chunk: ChatCompletionChunk) -> HiLlmResult<Option<ChatCompletionChunk>>;
}

impl<M: ChunkMiddleware + ?Sized> ChunkMiddleware for Arc<M> {
    fn process(&self, chunk: ChatCompletionChunk) -> HiLlmResult<Option<ChatCompletionChunk>> {
        (**self).process(chunk)
    }
}

pin_project! {
    pub struct IngressStream<S, P> {
        #[pin]
        stream: SSEStream<S>,
        parse_event: P,
        cancel: CancelField,
    }
}

impl<S, P> IngressStream<S, P>
where
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    pub fn new_sse(inner: S, parse_event: P, cancel: CancelField) -> Self {
        Self {
            stream: SSEStream::new(inner),
            parse_event,
            cancel,
        }
    }
}

impl<S, P, E> Stream for IngressStream<S, P>
where
    S: Stream<Item = Result<Bytes, E>>,
    E: Into<HiLlmError>,
    P: Fn(&str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        #[cfg(feature = "default-http")]
        if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
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

pin_project! {
    pub struct StreamPipeline<S> {
        #[pin]
        inner: S,
        middleware: Vec<Box<dyn ChunkMiddleware>>,
        cancel: CancelField,
        done: bool,
    }
}

impl<S> StreamPipeline<S> {
    pub fn new(inner: S, middleware: Vec<Box<dyn ChunkMiddleware>>, cancel: CancelField) -> Self {
        Self {
            inner,
            middleware,
            cancel,
            done: false,
        }
    }
}

impl<S> Stream for StreamPipeline<S>
where
    S: Stream<Item = HiLlmResult<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.done {
            return Poll::Ready(None);
        }

        #[cfg(feature = "default-http")]
        if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            *this.done = true;
            return Poll::Ready(None);
        }

        loop {
            #[cfg(feature = "default-http")]
            if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
                *this.done = true;
                return Poll::Ready(None);
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    *this.done = true;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(Some(Ok(chunk))) => {
                    let mut accumulator: Option<ChatCompletionChunk> = Some(chunk);
                    let mut error: Option<HiLlmError> = None;

                    for mw in this.middleware.iter() {
                        match accumulator.take() {
                            None => break,
                            Some(c) => match mw.process(c) {
                                Ok(Some(next)) => accumulator = Some(next),
                                Ok(None) => {
                                    // Middleware dropped the chunk.
                                    accumulator = None;
                                    break;
                                }
                                Err(e) => {
                                    error = Some(e);
                                    break;
                                }
                            },
                        }
                    }

                    if let Some(e) = error {
                        return Poll::Ready(Some(Err(e)));
                    }

                    match accumulator {
                        None => {
                            continue;
                        }
                        Some(final_chunk) => return Poll::Ready(Some(Ok(final_chunk))),
                    }
                }
            }
        }
    }
}

enum EgressMode {
    Passthrough,
    ParseAndEncode(EgressEncoding),
}

enum EgressEncoding {
    OpenAiSSE,
}

pin_project! {
    pub struct EgressStream<S> {
        #[pin]
        inner: S,
        mode: EgressMode,
        cancel: CancelField,
        done: bool,
    }
}

impl<S> EgressStream<S> {
    pub fn new(
        inner: S,
        ingress_format: StreamFormat,
        egress_format: StreamFormat,
        middleware_count: usize,
        cancel: CancelField,
    ) -> Self {
        let mode = if ingress_format == egress_format && middleware_count == 0 {
            EgressMode::Passthrough
        } else {
            let encoding = match egress_format {
                StreamFormat::SSE | StreamFormat::AwsEventStream => EgressEncoding::OpenAiSSE,
            };
            EgressMode::ParseAndEncode(encoding)
        };

        Self {
            inner,
            mode,
            cancel,
            done: false,
        }
    }
}

impl<S> Stream for EgressStream<S>
where
    S: Stream<Item = HiLlmResult<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.done {
            return Poll::Ready(None);
        }

        #[cfg(feature = "default-http")]
        if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            *this.done = true;
            return Poll::Ready(None);
        }

        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                *this.done = true;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(Some(Ok(chunk))) => {
                #[cfg(feature = "default-http")]
                if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
                    *this.done = true;
                    return Poll::Ready(None);
                }

                match this.mode {
                    EgressMode::Passthrough => Poll::Ready(Some(encode_sse_chunk(&chunk))),
                    EgressMode::ParseAndEncode(EgressEncoding::OpenAiSSE) => {
                        Poll::Ready(Some(encode_sse_chunk(&chunk)))
                    }
                }
            }
        }
    }
}

fn encode_sse_chunk(chunk: &ChatCompletionChunk) -> HiLlmResult<Bytes> {
    let json = serde_json::to_string(chunk).map_err(|e| HiLlmError::Streaming {
        message: format!("failed to serialise chunk: {e}"),
    })?;

    let mut buf = pool_acquire();
    buf.extend_from_slice(b"data: ");
    buf.extend_from_slice(json.as_bytes());
    buf.extend_from_slice(b"\n\n");

    let frozen = Bytes::copy_from_slice(&buf);
    buf.clear();
    pool_release(buf);
    Ok(frozen)
}
