use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;
use tower::Service;

use super::types::{LlmRequest, LlmRequestKind, LlmResponse};
use crate::client::{BoxFuture, LlmClient};
use crate::error::{HiLlmError, HiLlmResult};
use crate::types::ChatCompletionChunk;

pub struct LlmService<C> {
    inner: Arc<C>,
}

impl<C> LlmService<C> {
    #[must_use]
    pub fn new(client: C) -> Self {
        Self {
            inner: Arc::new(client),
        }
    }

    #[must_use]
    pub fn new_from_arc(client: Arc<C>) -> Self {
        Self { inner: client }
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }
}

impl<C> Clone for LlmService<C> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C> Service<LlmRequest> for LlmService<C>
where
    C: LlmClient + Send + Sync + 'static,
{
    type Response = LlmResponse;
    type Error = HiLlmError;
    type Future = BoxFuture<'static, HiLlmResult<LlmResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLlmResult<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: LlmRequest) -> Self::Future {
        let client = Arc::clone(&self.inner);
        Box::pin(async move {
            match req.kind {
                LlmRequestKind::Chat(r) => {
                    let resp = client.chat(r).await?;
                    Ok(LlmResponse::Chat(resp))
                }
                LlmRequestKind::ChatStream(r) => {
                    let stream = client.chat_stream(r).await?;
                    let chunks = collect_stream(stream).await?;
                    let static_stream: crate::client::BoxStream<
                        'static,
                        HiLlmResult<ChatCompletionChunk>,
                    > = Box::pin(OwnedChunksStream { chunks });
                    Ok(LlmResponse::ChatStream(static_stream))
                }
                LlmRequestKind::Embed(r) => {
                    let resp = client.embed(r).await?;
                    Ok(LlmResponse::Embed(resp))
                }
                LlmRequestKind::ListModels => {
                    let resp = client.list_models().await?;
                    Ok(LlmResponse::ListModels(resp))
                }
                LlmRequestKind::ImageGenerate(r) => {
                    let resp = client.image_generate(r).await?;
                    Ok(LlmResponse::ImageGenerate(resp))
                }
                LlmRequestKind::Speech(r) => {
                    let resp = client.speech(r).await?;
                    Ok(LlmResponse::Speech(resp))
                }
                LlmRequestKind::Transcribe(r) => {
                    let resp = client.transcribe(r).await?;
                    Ok(LlmResponse::Transcribe(resp))
                }
                LlmRequestKind::Moderate(r) => {
                    let resp = client.moderate(r).await?;
                    Ok(LlmResponse::Moderate(resp))
                }
                LlmRequestKind::Rerank(r) => {
                    let resp = client.rerank(r).await?;
                    Ok(LlmResponse::Rerank(resp))
                }
                LlmRequestKind::Search(r) => {
                    let resp = client.search(r).await?;
                    Ok(LlmResponse::Search(resp))
                }
                LlmRequestKind::Ocr(r) => {
                    let resp = client.ocr(r).await?;
                    Ok(LlmResponse::Ocr(resp))
                }
            }
        })
    }
}

async fn collect_stream<'a>(
    mut stream: crate::client::BoxStream<'a, HiLlmResult<ChatCompletionChunk>>,
) -> HiLlmResult<VecDeque<ChatCompletionChunk>> {
    let mut chunks = VecDeque::new();
    loop {
        let item = std::future::poll_fn(|cx| Pin::as_mut(&mut stream).poll_next(cx)).await;
        match item {
            Some(Ok(chunk)) => chunks.push_back(chunk),
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    Ok(chunks)
}

struct OwnedChunksStream {
    chunks: VecDeque<ChatCompletionChunk>,
}

impl Stream for OwnedChunksStream {
    type Item = HiLlmResult<ChatCompletionChunk>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.chunks.pop_front().map(Ok))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunks.len(), Some(self.chunks.len()))
    }
}
