use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures_core::Stream;
use pin_project_lite::pin_project;

use crate::client::BoxStream;
use crate::error::{HiLlmError, HiLlmResult};
use crate::types::ChatCompletionChunk;

use super::request::with_retry;

const MIN_FRAME_SIZE: usize = 16;

const MAX_FRAME_SIZE: usize = 16 * 1024 * 1024;

const HEADER_TYPE_STRING: u8 = 7;

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
pub async fn post_eventstream<P>(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
    parse_event: P,
) -> HiLlmResult<BoxStream<'static, HiLlmResult<ChatCompletionChunk>>>
where
    P: Fn(&str, &str) -> HiLlmResult<Option<ChatCompletionChunk>> + Send + 'static,
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
    let stream = EventStreamParser::new(byte_stream, parse_event);
    Ok(Box::pin(stream))
}

struct EventHeader {
    name: String,
    value: String,
}

fn parse_headers(mut data: &[u8]) -> HiLlmResult<Vec<EventHeader>> {
    let mut headers = Vec::new();
    while !data.is_empty() {
        let name_len = data[0] as usize;
        data = &data[1..];
        if data.len() < name_len {
            return Err(HiLlmError::Streaming {
                message: "EventStream header name truncated".into(),
            });
        }
        let name = std::str::from_utf8(&data[..name_len])
            .map_err(|_| HiLlmError::Streaming {
                message: "EventStream header name is not UTF-8".into(),
            })?
            .to_owned();
        data = &data[name_len..];

        if data.is_empty() {
            return Err(HiLlmError::Streaming {
                message: "EventStream header type byte missing".into(),
            });
        }
        let value_type = data[0];
        data = &data[1..];

        if value_type == HEADER_TYPE_STRING {
            // String: 2-byte big-endian length + UTF-8 value.
            if data.len() < 2 {
                return Err(HiLlmError::Streaming {
                    message: "EventStream string header length truncated".into(),
                });
            }
            let value_len = u16::from_be_bytes([data[0], data[1]]) as usize;
            data = &data[2..];
            if data.len() < value_len {
                return Err(HiLlmError::Streaming {
                    message: "EventStream string header value truncated".into(),
                });
            }
            let value = std::str::from_utf8(&data[..value_len])
                .map_err(|_| HiLlmError::Streaming {
                    message: "EventStream header value is not UTF-8".into(),
                })?
                .to_owned();
            data = &data[value_len..];
            headers.push(EventHeader { name, value });
        } else {
            // Skip non-string header types based on their wire sizes.
            // AWS EventStream spec: bool types have no value bytes (the type
            // byte itself encodes true/false).
            let skip = match value_type {
                0 => 0, // bool_true  — no value bytes
                1 => 0, // bool_false — no value bytes
                2 => 1, // byte
                3 => 2, // short
                4 => 4, // int
                5 => 8, // long
                6 => {
                    // bytes: 2-byte length prefix
                    if data.len() < 2 {
                        return Err(HiLlmError::Streaming {
                            message: "EventStream bytes header length truncated".into(),
                        });
                    }
                    let len = u16::from_be_bytes([data[0], data[1]]) as usize;
                    2 + len
                }
                8 => 8,  // timestamp
                9 => 16, // uuid
                _ => {
                    return Err(HiLlmError::Streaming {
                        message: format!("unknown EventStream header type: {value_type}"),
                    });
                }
            };
            if data.len() < skip {
                return Err(HiLlmError::Streaming {
                    message: "EventStream header value data truncated".into(),
                });
            }
            data = &data[skip..];
        }
    }
    Ok(headers)
}

fn crc32(data: &[u8]) -> u32 {
    static TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

pin_project! {
    struct EventStreamParser<S, P> {
        #[pin]
        inner: S,
        buffer: BytesMut,
        done: bool,
        parse_event: P,
    }
}

impl<S, P> EventStreamParser<S, P> {
    fn new(inner: S, parse_event: P) -> Self {
        Self {
            inner,
            buffer: BytesMut::new(),
            done: false,
            parse_event,
        }
    }
}

impl<S, P> Stream for EventStreamParser<S, P>
where
    S: Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>,
    P: Fn(&str, &str) -> HiLlmResult<Option<ChatCompletionChunk>>,
{
    type Item = HiLlmResult<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            if this.buffer.len() >= MIN_FRAME_SIZE {
                let total_length = u32::from_be_bytes([
                    this.buffer[0],
                    this.buffer[1],
                    this.buffer[2],
                    this.buffer[3],
                ]) as usize;

                if !(MIN_FRAME_SIZE..=MAX_FRAME_SIZE).contains(&total_length) {
                    return Poll::Ready(Some(Err(HiLlmError::Streaming {
                        message: format!(
                            "EventStream frame size {total_length} is out of range [{MIN_FRAME_SIZE}, {MAX_FRAME_SIZE}]"
                        ),
                    })));
                }

                if this.buffer.len() < total_length {
                } else {
                    let frame = this.buffer.split_to(total_length);

                    let headers_length =
                        u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;

                    let prelude_crc_expected =
                        u32::from_be_bytes([frame[8], frame[9], frame[10], frame[11]]);
                    let prelude_crc_actual = crc32(&frame[..8]);
                    if prelude_crc_expected != prelude_crc_actual {
                        return Poll::Ready(Some(Err(HiLlmError::Streaming {
                            message: format!(
                                "EventStream prelude CRC mismatch: expected {prelude_crc_expected:#010X}, got {prelude_crc_actual:#010X}"
                            ),
                        })));
                    }

                    let message_crc_expected = u32::from_be_bytes([
                        frame[total_length - 4],
                        frame[total_length - 3],
                        frame[total_length - 2],
                        frame[total_length - 1],
                    ]);
                    let message_crc_actual = crc32(&frame[..total_length - 4]);
                    if message_crc_expected != message_crc_actual {
                        return Poll::Ready(Some(Err(HiLlmError::Streaming {
                            message: format!(
                                "EventStream message CRC mismatch: expected {message_crc_expected:#010X}, got {message_crc_actual:#010X}"
                            ),
                        })));
                    }

                    let headers_start = 12;
                    let headers_end = headers_start + headers_length;
                    if headers_end > total_length - 4 {
                        return Poll::Ready(Some(Err(HiLlmError::Streaming {
                            message: "EventStream headers extend past frame boundary".into(),
                        })));
                    }

                    let headers = match parse_headers(&frame[headers_start..headers_end]) {
                        Ok(h) => h,
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    };

                    let mut event_type = "";
                    let mut message_type = "";
                    for h in &headers {
                        match h.name.as_str() {
                            ":event-type" => event_type = &h.value,
                            ":message-type" => message_type = &h.value,
                            _ => {}
                        }
                    }

                    if message_type == "exception" {
                        let payload = &frame[headers_end..total_length - 4];
                        let payload_str = std::str::from_utf8(payload).unwrap_or("<binary>");
                        return Poll::Ready(Some(Err(HiLlmError::Streaming {
                            message: format!(
                                "Bedrock EventStream exception ({event_type}): {payload_str}"
                            ),
                        })));
                    }

                    if message_type != "event" {
                        continue;
                    }

                    let payload = &frame[headers_end..total_length - 4];
                    let payload_str = match std::str::from_utf8(payload) {
                        Ok(s) => s,
                        Err(e) => {
                            return Poll::Ready(Some(Err(HiLlmError::Streaming {
                                message: format!("EventStream payload is not UTF-8: {e}"),
                            })));
                        }
                    };

                    match (this.parse_event)(event_type, payload_str) {
                        Ok(None) => {
                            continue;
                        }
                        Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
            }

            if *this.done {
                if !this.buffer.is_empty() {
                    let leftover = this.buffer.len();
                    this.buffer.clear();
                    return Poll::Ready(Some(Err(HiLlmError::Streaming {
                        message: format!(
                            "EventStream ended with {leftover} bytes of incomplete frame data"
                        ),
                    })));
                }
                return Poll::Ready(None);
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if this.buffer.len() + bytes.len() > MAX_FRAME_SIZE {
                        *this.done = true;
                        return Poll::Ready(Some(Err(HiLlmError::Streaming {
                            message: format!("EventStream buffer exceeded {MAX_FRAME_SIZE} bytes"),
                        })));
                    }
                    this.buffer.extend_from_slice(&bytes);
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(HiLlmError::from(e))));
                }
                Poll::Ready(None) => {
                    *this.done = true;
                    if this.buffer.is_empty() {
                        return Poll::Ready(None);
                    }
                    continue;
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}
