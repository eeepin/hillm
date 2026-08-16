use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;
use tower::Service;

use super::types::{LLMRequest, LLMRequestKind, LLMResponse};
use crate::client::{
    AudioClient, BoxFuture, ChatCompletionClient, EmbeddingClient, ImageClient, ModelClient,
    ModerationClient, OcrClient, RerankClient, SearchClient,
};
use crate::error::{HiLLMError, HiLLMResult};
use crate::types::ChatCompletionChunk;

pub struct LLMService<C> {
    inner: Arc<C>,
}

impl<C> LLMService<C> {
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

impl<C> Clone for LLMService<C> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C> Service<LLMRequest> for LLMService<C>
where
    C: ChatCompletionClient
        + EmbeddingClient
        + ImageClient
        + AudioClient
        + ModerationClient
        + RerankClient
        + SearchClient
        + OcrClient
        + ModelClient
        + Send
        + Sync
        + 'static,
{
    type Response = LLMResponse;
    type Error = HiLLMError;
    type Future = BoxFuture<'static, HiLLMResult<LLMResponse>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<HiLLMResult<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: LLMRequest) -> Self::Future {
        let client = Arc::clone(&self.inner);
        Box::pin(async move {
            match req.kind {
                LLMRequestKind::Chat(r) => {
                    let resp = client.chat(r).await?;
                    Ok(LLMResponse::Chat(resp))
                }
                LLMRequestKind::ChatStream(r) => {
                    let stream = client.chat_stream(r).await?;
                    let chunks = collect_stream(stream).await?;
                    let static_stream: crate::client::BoxStream<
                        'static,
                        HiLLMResult<ChatCompletionChunk>,
                    > = Box::pin(OwnedChunksStream { chunks });
                    Ok(LLMResponse::ChatStream(static_stream))
                }
                LLMRequestKind::Embed(r) => {
                    let resp = client.embed(r).await?;
                    Ok(LLMResponse::Embed(resp))
                }
                LLMRequestKind::ListModels => {
                    let resp = client.list_models().await?;
                    Ok(LLMResponse::ListModels(resp))
                }
                LLMRequestKind::ImageGenerate(r) => {
                    let resp = client.image_generate(r).await?;
                    Ok(LLMResponse::ImageGenerate(resp))
                }
                LLMRequestKind::Speech(r) => {
                    let resp = client.speech(r).await?;
                    Ok(LLMResponse::Speech(resp))
                }
                LLMRequestKind::Transcribe(r) => {
                    let resp = client.transcribe(r).await?;
                    Ok(LLMResponse::Transcribe(resp))
                }
                LLMRequestKind::Moderate(r) => {
                    let resp = client.moderate(r).await?;
                    Ok(LLMResponse::Moderate(resp))
                }
                LLMRequestKind::Rerank(r) => {
                    let resp = client.rerank(r).await?;
                    Ok(LLMResponse::Rerank(resp))
                }
                LLMRequestKind::Search(r) => {
                    let resp = client.search(r).await?;
                    Ok(LLMResponse::Search(resp))
                }
                LLMRequestKind::Ocr(r) => {
                    let resp = client.ocr(r).await?;
                    Ok(LLMResponse::Ocr(resp))
                }
            }
        })
    }
}

async fn collect_stream<'a>(
    mut stream: crate::client::BoxStream<'a, HiLLMResult<ChatCompletionChunk>>,
) -> HiLLMResult<VecDeque<ChatCompletionChunk>> {
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
    type Item = HiLLMResult<ChatCompletionChunk>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.chunks.pop_front().map(Ok))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.chunks.len(), Some(self.chunks.len()))
    }
}
