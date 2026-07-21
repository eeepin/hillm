use std::cell::RefCell;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures_core::Stream;
use pin_project_lite::pin_project;

use crate::error::{LiterLlmError, Result};
use crate::provider::StreamFormat;
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
    fn process(&self, chunk: ChatCompletionChunk) -> Result<Option<ChatCompletionChunk>>;
}

impl<M: ChunkMiddleware + ?Sized> ChunkMiddleware for Arc<M> {
    fn process(&self, chunk: ChatCompletionChunk) -> Result<Option<ChatCompletionChunk>> {
        (**self).process(chunk)
    }
}

pin_project! {
    pub struct IngressStream<S, P> {
        #[pin]
        inner: S,
        buffer: String,
        cursor: usize,
        done: bool,
        parse_event: P,
        cancel: CancelField,
    }
}

impl<S, P> IngressStream<S, P>
where
    P: Fn(&str) -> Result<Option<ChatCompletionChunk>>,
{
    pub fn new_sse(inner: S, parse_event: P, cancel: CancelField) -> Self {
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

impl<S, P, E> Stream for IngressStream<S, P>
where
    S: Stream<Item = std::result::Result<Bytes, E>>,
    E: Into<LiterLlmError>,
    P: Fn(&str) -> Result<Option<ChatCompletionChunk>>,
{
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        #[cfg(feature = "default-http")]
        if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
            *this.done = true;
            return Poll::Ready(None);
        }

        loop {
            if let Some(offset) = memchr_newline(&this.buffer.as_bytes()[*this.cursor..]) {
                let newline_pos = *this.cursor + offset;
                let line = this.buffer[*this.cursor..newline_pos]
                    .trim_end_matches('\r')
                    .trim();

                if line.is_empty() || line.starts_with(':') {
                    *this.cursor = newline_pos + 1;
                    compact_buffer(this.buffer, this.cursor);
                    continue;
                }

                if let Some(raw) = line.strip_prefix("data:") {
                    let data = raw.strip_prefix(' ').unwrap_or(raw).trim();
                    if data == "[DONE]" {
                        *this.cursor = newline_pos + 1;
                        compact_buffer(this.buffer, this.cursor);
                        return Poll::Ready(None);
                    }
                    let result = (this.parse_event)(data);
                    *this.cursor = newline_pos + 1;
                    compact_buffer(this.buffer, this.cursor);
                    match result {
                        Ok(None) => continue,
                        Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }

                *this.cursor = newline_pos + 1;
                compact_buffer(this.buffer, this.cursor);
                continue;
            }

            if *this.done {
                let remaining = this.buffer.len() - *this.cursor;
                if remaining > 0 {
                    this.buffer.clear();
                    *this.cursor = 0;
                }
                return Poll::Ready(None);
            }

            #[cfg(feature = "default-http")]
            if this.cancel.as_ref().is_some_and(|t| t.is_cancelled()) {
                *this.done = true;
                return Poll::Ready(None);
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    const MAX_BUFFER_BYTES: usize = 1024 * 1024; // 1 MiB
                    if this.buffer.len() + bytes.len() > MAX_BUFFER_BYTES {
                        *this.done = true;
                        return Poll::Ready(Some(Err(LiterLlmError::Streaming {
                            message: format!(
                                "SSE buffer exceeded {MAX_BUFFER_BYTES} bytes; stream aborted"
                            ),
                        })));
                    }
                    match std::str::from_utf8(&bytes) {
                        Ok(s) => this.buffer.push_str(s),
                        Err(e) => {
                            *this.done = true;
                            return Poll::Ready(Some(Err(LiterLlmError::Streaming {
                                message: format!("invalid UTF-8 in SSE stream: {e}"),
                            })));
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e.into())));
                }
                Poll::Ready(None) => {
                    *this.done = true;
                    continue;
                }
                Poll::Pending => return Poll::Pending,
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
    S: Stream<Item = Result<ChatCompletionChunk>>,
{
    type Item = Result<ChatCompletionChunk>;

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
                    let mut error: Option<LiterLlmError> = None;

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
