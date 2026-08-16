//! SSE (Server-Sent Events) decoder and stream wrapper.
//!
//! This module provides a transport-agnostic SSE decoder that processes raw bytes
//! and produces complete SSE events. The decoder is protocol-agnostic and knows
//! nothing about business-level sentinels like `[DONE]`.

use bytes::{Buf, Bytes, BytesMut};
use futures_core::Stream;
use pin_project_lite::pin_project;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::error::{HiLLMError, HiLLMResult};
use crate::util::bound::SSE_BUFFER_MAX_BYTES;

/// A complete SSE event, assembled per the SSE specification.
///
/// All fields are validated UTF-8. Multiple `data:` lines are joined with `\n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SSEEvent {
    /// Concatenation of all `data:` lines, joined with `\n`.
    /// Empty string if no `data:` lines were present.
    pub data: String,
    /// The `event:` field, if present.
    pub event: Option<String>,
    /// The `id:` field, if present.
    pub id: Option<String>,
    /// The `retry:` field, parsed as milliseconds, if present and valid.
    pub retry: Option<u64>,
}

/// Internal builder for assembling SSE event fields.
struct EventBuilder {
    data: Vec<u8>,
    event: Vec<u8>,
    id: Vec<u8>,
    retry: Option<u64>,
    data_line_count: u32,
}

impl EventBuilder {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            event: Vec::new(),
            id: Vec::new(),
            retry: None,
            data_line_count: 0,
        }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.event.clear();
        self.id.clear();
        self.retry = None;
        self.data_line_count = 0;
    }

    fn build(&self) -> HiLLMResult<SSEEvent> {
        Ok(SSEEvent {
            data: String::from_utf8(self.data.clone()).map_err(|e| HiLLMError::Streaming {
                message: format!("invalid UTF-8 in data field: {e}"),
            })?,
            event: if self.event.is_empty() {
                None
            } else {
                Some(
                    String::from_utf8(self.event.clone()).map_err(|e| HiLLMError::Streaming {
                        message: format!("invalid UTF-8 in event field: {e}"),
                    })?,
                )
            },
            id: if self.id.is_empty() {
                None
            } else {
                Some(
                    String::from_utf8(self.id.clone()).map_err(|e| HiLLMError::Streaming {
                        message: format!("invalid UTF-8 in id field: {e}"),
                    })?,
                )
            },
            retry: self.retry,
        })
    }
}

/// Incremental SSE decoder. Input: arbitrary `Bytes` chunks.
/// Output: zero or more complete `SSEEvent`s per `decode()` call.
///
/// This type is NOT a Stream. It is a pure state machine that the caller
/// drives from whatever polling context is appropriate (a Stream impl,
/// a sync loop, a wasm callback, etc.).
pub struct SSEDecoder {
    buf: BytesMut,
    current: EventBuilder,
    has_data: bool,
}

impl SSEDecoder {
    /// Create a new SSE decoder.
    pub fn new() -> Self {
        Self {
            buf: BytesMut::with_capacity(4096),
            current: EventBuilder::new(),
            has_data: false,
        }
    }

    /// Feed a chunk of bytes. Returns all complete events decoded so far.
    /// May return an empty Vec if no event boundary has been reached yet.
    ///
    /// # Errors
    /// Returns `HiLLMError::Streaming` if:
    /// - Internal buffer exceeds `SSE_BUFFER_MAX_BYTES`.
    /// - A complete field contains invalid UTF-8.
    pub fn decode(&mut self, chunk: Bytes) -> HiLLMResult<Vec<SSEEvent>> {
        // Check buffer overflow
        if self.buf.len() + chunk.len() > SSE_BUFFER_MAX_BYTES {
            return Err(HiLLMError::Streaming {
                message: format!(
                    "SSE buffer exceeded {SSE_BUFFER_MAX_BYTES} bytes; stream aborted"
                ),
            });
        }

        self.buf.extend_from_slice(&chunk);
        let mut events = Vec::new();

        while let Some(newline_pos) = memchr::memchr(b'\n', &self.buf) {
            // Extract line (excluding \n)
            let line_bytes = &self.buf[..newline_pos];
            let line_len = newline_pos + 1;

            // Strip trailing \r if present
            let line_bytes = if line_bytes.last() == Some(&b'\r') {
                &line_bytes[..line_bytes.len() - 1]
            } else {
                line_bytes
            };

            // Process the line
            let line =
                String::from_utf8(line_bytes.to_vec()).map_err(|e| HiLLMError::Streaming {
                    message: format!("invalid UTF-8 in SSE line: {e}"),
                })?;

            // Advance buffer
            self.buf.advance(line_len);

            // Parse the line
            let line = line.trim();

            // Empty line = event boundary
            if line.is_empty() {
                if self.has_data {
                    // Dispatch event
                    events.push(self.current.build()?);
                    self.current.reset();
                    self.has_data = false;
                }
                // If no data, just continue (blank lines between events are harmless)
                continue;
            }

            // Comment line (starts with :)
            if line.starts_with(':') {
                continue;
            }

            // Field line: split on first ':'
            let (field_name, value) = if let Some(colon_pos) = line.find(':') {
                let name = &line[..colon_pos];
                let mut val = &line[colon_pos + 1..];
                // Strip single leading space per spec
                if val.starts_with(' ') {
                    val = &val[1..];
                }
                (name, val)
            } else {
                // No colon: treat as field name with empty value
                (line, "")
            };

            // Dispatch by field name
            match field_name {
                "data" => {
                    if self.current.data_line_count > 0 {
                        self.current.data.push(b'\n');
                    }
                    self.current.data.extend_from_slice(value.as_bytes());
                    self.current.data_line_count += 1;
                    self.has_data = true;
                }
                "event" => {
                    self.current.event.clear();
                    self.current.event.extend_from_slice(value.as_bytes());
                }
                "id" => {
                    self.current.id.clear();
                    self.current.id.extend_from_slice(value.as_bytes());
                }
                "retry" => {
                    // Parse as u64, ignore if invalid
                    if let Ok(ms) = value.parse::<u64>() {
                        self.current.retry = Some(ms);
                    }
                }
                _ => {
                    // Unknown field: ignore per spec
                }
            }
        }

        Ok(events)
    }

    /// Signal EOF. Returns any trailing event that had data but no final
    /// blank line (compatibility policy).
    ///
    /// # Errors
    /// Returns `HiLLMError::Streaming` if the buffer ends mid-UTF-8-sequence
    /// or contains an incomplete field (data without a terminating newline).
    pub fn finish(&mut self) -> HiLLMResult<Option<SSEEvent>> {
        // If there are remaining bytes without a terminating newline,
        // that is an incomplete field — return a truncation error.
        if !self.buf.is_empty() {
            let residue = &self.buf[..];

            // Strip trailing \r if present
            let residue = if residue.last() == Some(&b'\r') {
                &residue[..residue.len() - 1]
            } else {
                residue
            };

            if !residue.is_empty() {
                // Validate UTF-8 first for a clear error message
                let text =
                    String::from_utf8(residue.to_vec()).map_err(|_| HiLLMError::Streaming {
                        message: "SSE stream ended with incomplete UTF-8 sequence".to_string(),
                    })?;

                return Err(HiLLMError::Streaming {
                    message: format!(
                        "SSE stream ended with incomplete field (no terminating newline): {:?}",
                        text
                    ),
                });
            }

            self.buf.clear();
        }

        // If we have data but no final blank line, dispatch the event
        if self.has_data {
            let event = self.current.build()?;
            self.current.reset();
            self.has_data = false;
            return Ok(Some(event));
        }

        Ok(None)
    }
}

impl Default for SSEDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pin_project! {
    /// Wraps a byte-stream and an `SSEDecoder`, producing `SSEEvent`s.
    /// This is transport-agnostic: the inner stream can be reqwest's
    /// `BytesStream`, a wasm fetch stream, or a test mock.
    pub struct SSEStream<S> {
        #[pin]
        inner: S,
        decoder: SSEDecoder,
        done: bool,
        pending: VecDeque<SSEEvent>,
    }
}

impl<S> SSEStream<S> {
    /// Create a new SSE stream wrapper.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            decoder: SSEDecoder::new(),
            done: false,
            pending: VecDeque::new(),
        }
    }
}

impl<S, E> Stream for SSEStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
    E: Into<HiLLMError>,
{
    type Item = HiLLMResult<SSEEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Drain pending events first
        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }

        // If already done, return None
        if *this.done {
            return Poll::Ready(None);
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Inner stream ended
                    *this.done = true;
                    match this.decoder.finish() {
                        Ok(Some(event)) => return Poll::Ready(Some(Ok(event))),
                        Ok(None) => return Poll::Ready(None),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    *this.done = true;
                    return Poll::Ready(Some(Err(e.into())));
                }
                Poll::Ready(Some(Ok(chunk))) => {
                    match this.decoder.decode(chunk) {
                        Ok(events) => {
                            if events.is_empty() {
                                // No complete events yet, poll again
                                continue;
                            }
                            // Enqueue all but the first, return the first
                            let mut events = events.into_iter();
                            let first = events.next().unwrap();
                            this.pending.extend(events);
                            return Poll::Ready(Some(Ok(first)));
                        }
                        Err(e) => {
                            *this.done = true;
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Basic Event Parsing ==========

    #[test]
    fn single_data_line() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[0].event, None);
        assert_eq!(events[0].id, None);
        assert_eq!(events[0].retry, None);
    }

    #[test]
    fn multiple_data_lines_joined() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(
                b"data: line1\ndata: line2\ndata: line3\n\n",
            ))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn event_field() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"event: message\ndata: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
        assert_eq!(events[0].event, Some("message".to_string()));
    }

    #[test]
    fn id_field() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"id: 123\ndata: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, Some("123".to_string()));
    }

    #[test]
    fn retry_field() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"retry: 5000\ndata: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry, Some(5000));
    }

    #[test]
    fn retry_field_invalid_ignored() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"retry: not-a-number\ndata: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].retry, None);
    }

    #[test]
    fn comment_lines_ignored() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b": this is a comment\ndata: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn heartbeat_comment_only() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b": heartbeat\n\n"))
            .unwrap();
        assert_eq!(events.len(), 0);
    }

    // ========== Chunk Boundary Tests ==========

    #[test]
    fn json_split_across_chunks() {
        let payload = b"data: {\"key\":\"value\"}\n\n";

        // Split at every position
        for split_at in 0..payload.len() {
            let mut decoder = SSEDecoder::new();
            let chunk1 = Bytes::copy_from_slice(&payload[..split_at]);
            let chunk2 = Bytes::copy_from_slice(&payload[split_at..]);

            let events1 = decoder.decode(chunk1).unwrap();
            let events2 = decoder.decode(chunk2).unwrap();

            let total_events = events1.len() + events2.len();
            assert_eq!(
                total_events,
                1,
                "Failed at split position {}/{}",
                split_at,
                payload.len()
            );

            let all_events: Vec<_> = events1.into_iter().chain(events2).collect();
            let event = all_events.iter().find(|e| !e.data.is_empty()).unwrap();
            assert_eq!(event.data, "{\"key\":\"value\"}");
        }
    }

    #[test]
    fn utf8_chinese_split_across_chunks() {
        let payload = "data: 你好世界\n\n".as_bytes();

        // Split at every byte position (Chinese chars are 3 bytes each)
        for split_at in 0..payload.len() {
            let mut decoder = SSEDecoder::new();
            let chunk1 = Bytes::copy_from_slice(&payload[..split_at]);
            let chunk2 = Bytes::copy_from_slice(&payload[split_at..]);

            let events1 = decoder.decode(chunk1).unwrap();
            let events2 = decoder.decode(chunk2).unwrap();

            let total_events = events1.len() + events2.len();
            assert_eq!(
                total_events,
                1,
                "Failed at split position {}/{}",
                split_at,
                payload.len()
            );
        }
    }

    #[test]
    fn utf8_emoji_split_across_chunks() {
        let payload = "data: 🎉🎊\n\n".as_bytes();

        // Split at every byte position (emoji are 4 bytes each)
        for split_at in 0..payload.len() {
            let mut decoder = SSEDecoder::new();
            let chunk1 = Bytes::copy_from_slice(&payload[..split_at]);
            let chunk2 = Bytes::copy_from_slice(&payload[split_at..]);

            let events1 = decoder.decode(chunk1).unwrap();
            let events2 = decoder.decode(chunk2).unwrap();

            let total_events = events1.len() + events2.len();
            assert_eq!(
                total_events,
                1,
                "Failed at split position {}/{}",
                split_at,
                payload.len()
            );
        }
    }

    #[test]
    fn field_name_split_across_chunks() {
        let mut decoder = SSEDecoder::new();

        // Split "data:" across chunks
        let events1 = decoder.decode(Bytes::from_static(b"dat")).unwrap();
        assert_eq!(events1.len(), 0);

        let events2 = decoder.decode(Bytes::from_static(b"a: hello\n\n")).unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].data, "hello");
    }

    #[test]
    fn crlf_split_across_chunks() {
        let mut decoder = SSEDecoder::new();

        // Split \r\n across chunks
        let events1 = decoder
            .decode(Bytes::from_static(b"data: hello\r"))
            .unwrap();
        assert_eq!(events1.len(), 0);

        let events2 = decoder.decode(Bytes::from_static(b"\n\n")).unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].data, "hello");
    }

    #[test]
    fn blank_line_split_across_chunks() {
        let mut decoder = SSEDecoder::new();

        // Split the terminating blank line
        let events1 = decoder
            .decode(Bytes::from_static(b"data: hello\n"))
            .unwrap();
        assert_eq!(events1.len(), 0);

        let events2 = decoder.decode(Bytes::from_static(b"\n")).unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].data, "hello");
    }

    #[test]
    fn multiple_events_in_single_chunk() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(
                b"data: event1\n\ndata: event2\n\ndata: event3\n\n",
            ))
            .unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "event1");
        assert_eq!(events[1].data, "event2");
        assert_eq!(events[2].data, "event3");
    }

    #[test]
    fn complete_event_plus_partial_next() {
        let mut decoder = SSEDecoder::new();

        let events1 = decoder
            .decode(Bytes::from_static(b"data: event1\n\ndata: eve"))
            .unwrap();
        assert_eq!(events1.len(), 1);
        assert_eq!(events1[0].data, "event1");

        let events2 = decoder.decode(Bytes::from_static(b"nt2\n\n")).unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].data, "event2");
    }

    // ========== CRLF Handling ==========

    #[test]
    fn lf_lf_terminates_event() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn crlf_crlf_terminates_event() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: hello\r\n\r\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn crlf_lf_terminates_event() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: hello\r\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn lf_crlf_terminates_event() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: hello\n\r\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
    }

    // ========== Edge Cases ==========

    #[test]
    fn empty_data_line() {
        let mut decoder = SSEDecoder::new();
        let events = decoder.decode(Bytes::from_static(b"data:\n\n")).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn data_with_leading_space() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn data_without_space_after_colon() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data:hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn field_without_colon() {
        let mut decoder = SSEDecoder::new();
        let events = decoder.decode(Bytes::from_static(b"data\n\n")).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "");
    }

    #[test]
    fn unknown_fields_ignored() {
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"unknown: value\ndata: hello\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn buffer_overflow_error() {
        let mut decoder = SSEDecoder::new();
        let large_chunk = vec![b'a'; SSE_BUFFER_MAX_BYTES + 1];
        let result = decoder.decode(Bytes::from(large_chunk));
        assert!(result.is_err());
        if let Err(HiLLMError::Streaming { message }) = result {
            assert!(message.contains("exceeded"));
        }
    }

    #[test]
    fn invalid_utf8_error() {
        let mut decoder = SSEDecoder::new();
        // Invalid UTF-8 sequence
        let result = decoder.decode(Bytes::from_static(b"data: \xff\xfe\n\n"));
        assert!(result.is_err());
        if let Err(HiLLMError::Streaming { message }) = result {
            assert!(message.contains("UTF-8"));
        }
    }

    // ========== EOF Tests ==========

    #[test]
    fn finish_with_pending_event() {
        let mut decoder = SSEDecoder::new();
        // Field line is complete (has newline), but event has no terminating blank line.
        // Compatibility policy: dispatch the event anyway.
        decoder
            .decode(Bytes::from_static(b"data: hello\n"))
            .unwrap();
        let event = decoder.finish().unwrap();
        assert!(event.is_some());
        assert_eq!(event.unwrap().data, "hello");
    }

    #[test]
    fn finish_with_incomplete_field() {
        let mut decoder = SSEDecoder::new();
        // No terminating newline — incomplete field, should error.
        decoder.decode(Bytes::from_static(b"data: hel")).unwrap();
        let result = decoder.finish();
        assert!(result.is_err());
        if let Err(HiLLMError::Streaming { message }) = result {
            assert!(message.contains("incomplete field"));
        }
    }

    #[test]
    fn finish_with_incomplete_utf8() {
        let mut decoder = SSEDecoder::new();
        // Incomplete UTF-8 sequence (first byte of 3-byte char)
        decoder.decode(Bytes::from_static(b"data: \xe4")).unwrap();
        let result = decoder.finish();
        assert!(result.is_err());
        if let Err(HiLLMError::Streaming { message }) = result {
            assert!(message.contains("incomplete UTF-8"));
        }
    }

    #[test]
    fn finish_empty_buffer() {
        let mut decoder = SSEDecoder::new();
        let event = decoder.finish().unwrap();
        assert!(event.is_none());
    }

    #[test]
    fn finish_only_whitespace() {
        let mut decoder = SSEDecoder::new();
        decoder.decode(Bytes::from_static(b"   \n\n")).unwrap();
        let event = decoder.finish().unwrap();
        assert!(event.is_none());
    }

    // ========== Stream Wrapper Tests ==========

    #[cfg(feature = "tower")]
    #[tokio::test]
    async fn stream_wrapper_basic() {
        let chunks = vec![
            Ok::<Bytes, HiLLMError>(Bytes::from_static(b"data: event1\n\n")),
            Ok(Bytes::from_static(b"data: event2\n\n")),
        ];
        let stream = futures_util::stream::iter(chunks);
        let mut sse_stream = SSEStream::new(stream);

        use futures_util::StreamExt;
        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        let event3 = sse_stream.next().await;
        assert!(event3.is_none());
    }

    #[cfg(feature = "tower")]
    #[tokio::test]
    async fn stream_wrapper_error_propagation() {
        let chunks = vec![
            Ok::<Bytes, HiLLMError>(Bytes::from_static(b"data: hello\n\n")),
            Err(HiLLMError::Streaming {
                message: "test error".to_string(),
            }),
        ];
        let stream = futures_util::stream::iter(chunks);
        let mut sse_stream = SSEStream::new(stream);

        use futures_util::StreamExt;
        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "hello");

        let error = sse_stream.next().await.unwrap().unwrap_err();
        if let HiLLMError::Streaming { message } = error {
            assert_eq!(message, "test error");
        }
    }

    #[cfg(feature = "tower")]
    #[tokio::test]
    async fn stream_wrapper_multiple_events_from_single_chunk() {
        let chunks = vec![Ok::<Bytes, HiLLMError>(Bytes::from_static(
            b"data: event1\n\ndata: event2\n\ndata: event3\n\n",
        ))];
        let stream = futures_util::stream::iter(chunks);
        let mut sse_stream = SSEStream::new(stream);

        use futures_util::StreamExt;
        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        let event3 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event3.data, "event3");

        let event4 = sse_stream.next().await;
        assert!(event4.is_none());
    }

    // ========== Integration-Style Tests ==========

    #[test]
    fn openai_style_stream() {
        let mut decoder = SSEDecoder::new();
        let payload = b"data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\ndata: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n";

        let events = decoder.decode(Bytes::copy_from_slice(payload)).unwrap();
        assert_eq!(events.len(), 3);
        assert!(events[0].data.contains("Hello"));
        assert!(events[1].data.contains("world"));
        assert_eq!(events[2].data, "[DONE]");
    }

    #[test]
    fn anthropic_style_stream_with_event_field() {
        let mut decoder = SSEDecoder::new();
        let payload = b"event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

        let events = decoder.decode(Bytes::copy_from_slice(payload)).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event, Some("message_start".to_string()));
        assert_eq!(events[1].event, Some("content_block_delta".to_string()));
        assert_eq!(events[2].event, Some("message_stop".to_string()));
    }

    // ========== Cancellation Tests ==========

    #[cfg(feature = "tower")]
    #[tokio::test]
    async fn stream_drop_stops_polling() {
        use futures_util::StreamExt;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let poll_count = Arc::new(AtomicUsize::new(0));
        let poll_count_clone = poll_count.clone();

        // Create a stream that tracks how many times it's polled
        let stream = futures_util::stream::unfold((), move |()| {
            let count = poll_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Some((
                    Ok::<Bytes, HiLLMError>(Bytes::from_static(b"data: test\n\n")),
                    (),
                ))
            }
        });

        let sse_stream = SSEStream::new(stream);
        tokio::pin!(sse_stream);

        // Poll once to get an event
        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "test");

        // Drop the stream
        let _ = sse_stream;

        // Record the poll count after drop
        let final_count = poll_count.load(Ordering::SeqCst);

        // Wait a bit to ensure no more polls happen
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Poll count should not have increased after drop
        assert_eq!(
            poll_count.load(Ordering::SeqCst),
            final_count,
            "Stream was polled after being dropped"
        );
    }

    // ========== [DONE] Isolation Tests ==========

    #[test]
    fn done_sentinel_is_transparent_to_decoder() {
        // The SSE decoder should treat [DONE] as regular data.
        // Protocol-specific handling (e.g., OpenAI stopping on [DONE])
        // belongs in the provider codec, not the transport decoder.
        let mut decoder = SSEDecoder::new();
        let events = decoder
            .decode(Bytes::from_static(b"data: [DONE]\n\n"))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "[DONE]");
    }
}
