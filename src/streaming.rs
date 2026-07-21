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
