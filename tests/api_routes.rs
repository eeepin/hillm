//! Route integration tests (TODO.md P0 Step 4).
//!
//! These tests run the three API routes against a local mock HTTP server and
//! assert URLs, headers, JSON bodies, native responses, native streaming
//! (across arbitrary byte splits) and the provider/api-type selection rules —
//! without touching the public network.

#![cfg(all(feature = "default-http", not(target_arch = "wasm32")))]

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use hillm::client::{
    AnthropicMessagesClient, ClientConfigBuilder, DefaultClient, LlmClient, ResponseClient,
};
use hillm::provider::APIType;
use hillm::types::ChatCompletionRequest;
use hillm::types::anthropic::{
    AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest, AnthropicRole,
    AnthropicStopReason, AnthropicStreamEvent,
};
use hillm::types::response::{CreateResponseRequest, ResponsesStreamEvent};
use hillm::{HiLlmError, Message};
use serial_test::serial;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

// ============================ Mock HTTP server ============================

#[derive(Debug, Clone)]
struct RecordedRequest {
    method: String,
    path: String,
    /// Lower-cased header names.
    headers: Vec<(String, String)>,
    body: serde_json::Value,
}

impl RecordedRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

#[derive(Clone)]
struct MockServer {
    addr: SocketAddr,
    records: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl MockServer {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let addr = listener.local_addr().expect("mock server addr");
        let records: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));

        let server_records = Arc::clone(&records);
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    break;
                };
                let records = Arc::clone(&server_records);
                tokio::spawn(async move {
                    let _ = handle_connection(socket, &records).await;
                });
            }
        });

        Self { addr, records }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn recorded(&self) -> Vec<RecordedRequest> {
        self.records.lock().expect("records lock").clone()
    }

    fn request_count(&self) -> usize {
        self.records.lock().expect("records lock").len()
    }
}

/// Serves one keep-alive connection until it closes or errors.
async fn handle_connection(
    mut socket: tokio::net::TcpStream,
    records: &Arc<Mutex<Vec<RecordedRequest>>>,
) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        // Read until the end of the header block.
        let header_end = loop {
            if let Some(pos) = find_header_end(&buf) {
                break pos;
            }
            let mut chunk = [0u8; 4096];
            let n = socket.read(&mut chunk).await?;
            if n == 0 {
                return Ok(());
            }
            buf.extend_from_slice(&chunk[..n]);
        };

        let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let mut lines = header_text.lines();
        let request_line = lines.next().unwrap_or_default();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();

        let mut headers: Vec<(String, String)> = Vec::new();
        let mut content_length: usize = 0;
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                let name = name.trim().to_ascii_lowercase();
                let value = value.trim().to_string();
                if name == "content-length" {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((name, value));
            }
        }

        // Read the body, if any.
        let body_start = header_end + 4;
        let mut body = buf[body_start..].to_vec();
        while body.len() < content_length {
            let mut chunk = [0u8; 4096];
            let n = socket.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
        let remaining = if body.len() > content_length {
            Some(body[content_length..].to_vec())
        } else {
            None
        };
        body.truncate(content_length);

        let body_json: serde_json::Value =
            serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
        records.lock().expect("records lock").push(RecordedRequest {
            method: method.clone(),
            path: path.clone(),
            headers,
            body: body_json,
        });

        // Reset the read buffer for the next request on this connection.
        buf = remaining.unwrap_or_default();

        let close_after = serve_response(&mut socket, &method, &path, &body).await?;
        if close_after {
            return Ok(());
        }
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Writes the response for a route. Returns true when the connection must be
/// closed afterwards (streaming responses without content-length).
async fn serve_response(
    socket: &mut tokio::net::TcpStream,
    method: &str,
    path: &str,
    body: &[u8],
) -> std::io::Result<bool> {
    match (method, path) {
        ("POST", "/chat/completions") => {
            if wants_stream(body) {
                write_sse(socket, &chat_stream_sse(), 3).await?;
                return Ok(true);
            }
            write_json(socket, &chat_response_json()).await?;
        }
        ("POST", "/responses") => {
            if wants_stream(body) {
                write_sse(socket, &responses_stream_sse(), 5).await?;
                return Ok(true);
            }
            write_json(socket, &responses_object_json()).await?;
        }
        ("POST", "/messages") => {
            if wants_stream(body) {
                write_sse(socket, &anthropic_stream_sse(), 7).await?;
                return Ok(true);
            }
            write_json(socket, &anthropic_response_json()).await?;
        }
        _ => {
            let body = serde_json::json!({"error": {"message": "not found"}}).to_string();
            let resp = format!(
                "HTTP/1.1 404 Not Found\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await?;
        }
    }
    Ok(false)
}

fn wants_stream(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false)
}

async fn write_json(
    socket: &mut tokio::net::TcpStream,
    value: &serde_json::Value,
) -> std::io::Result<()> {
    let body = value.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    socket.write_all(resp.as_bytes()).await?;
    socket.flush().await
}

/// Writes an SSE payload split into fragments of `fragment_size` bytes with a
/// small pause between fragments, so the client must reassemble events across
/// arbitrary byte boundaries.
async fn write_sse(
    socket: &mut tokio::net::TcpStream,
    sse: &str,
    fragment_size: usize,
) -> std::io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
    socket.write_all(head.as_bytes()).await?;
    socket.flush().await?;

    let bytes = sse.as_bytes();
    for chunk in bytes.chunks(fragment_size.max(1)) {
        socket.write_all(chunk).await?;
        socket.flush().await?;
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    socket.shutdown().await
}

// ============================ Fixtures ============================

fn chat_response_json() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-mock-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from chat"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
}

fn chat_stream_sse() -> String {
    let chunk = |delta: &str| {
        serde_json::json!({
            "id": "chatcmpl-mock-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "delta": {"content": delta},
                "finish_reason": null
            }]
        })
    };
    format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chunk("Hello "),
        chunk("world")
    )
}

fn responses_object_json() -> serde_json::Value {
    serde_json::json!({
        "id": "resp-mock-1",
        "object": "response",
        "created_at": 1700000000,
        "model": "mock-model",
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello from responses"}]
        }],
        "usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}
    })
}

fn responses_stream_sse() -> String {
    let created = serde_json::json!({
        "type": "response.created",
        "response": {"id": "resp-mock-1", "status": "in_progress"}
    });
    let delta = serde_json::json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello"
    });
    let completed = serde_json::json!({
        "type": "response.completed",
        "response": {"id": "resp-mock-1", "status": "completed"}
    });
    format!(
        "event: response.created\ndata: {}\n\n\
         event: response.output_text.delta\ndata: {}\n\n\
         event: response.completed\ndata: {}\n\n",
        created, delta, completed
    )
}

fn anthropic_response_json() -> serde_json::Value {
    serde_json::json!({
        "id": "msg-mock-1",
        "type": "message",
        "role": "assistant",
        "model": "mock-model",
        "content": [{"type": "text", "text": "Hello from messages"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 9,
            "output_tokens": 2,
            "cache_read_input_tokens": 4
        }
    })
}

fn anthropic_stream_sse() -> String {
    let start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": "msg-mock-1",
            "model": "mock-model",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": null,
            "usage": {"input_tokens": 9, "output_tokens": 0}
        }
    });
    let delta = serde_json::json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"}
    });
    let stop = serde_json::json!({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": null},
        "usage": {"output_tokens": 2}
    });
    let message_stop = serde_json::json!({"type": "message_stop"});
    format!(
        "event: message_start\ndata: {}\n\n\
         event: content_block_delta\ndata: {}\n\n\
         event: message_delta\ndata: {}\n\n\
         event: message_stop\ndata: {}\n\n",
        start, delta, stop, message_stop
    )
}

fn user_message(text: &str) -> Message {
    serde_json::from_value(serde_json::json!({"role": "user", "content": text}))
        .expect("valid user message")
}

fn simple_chat_request(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![user_message("Hello")],
        ..Default::default()
    }
}

fn simple_anthropic_request(model: &str) -> AnthropicMessagesRequest {
    AnthropicMessagesRequest {
        model: model.to_string(),
        messages: vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: vec![AnthropicContentBlock::Text {
                text: "Hello".to_string(),
                cache_control: None,
            }],
        }],
        max_tokens: 64,
        ..Default::default()
    }
}

// ==================== Route 1: OpenAI Chat Completions ====================

#[tokio::test]
async fn openai_chat_route_url_headers_and_body() {
    let server = MockServer::start().await;
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::OpenAIChatCompletions)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let resp = client
        .chat(simple_chat_request("mock-model"))
        .await
        .unwrap();

    // Native Chat Completions fields survive the round trip.
    assert_eq!(resp.id, "chatcmpl-mock-1");
    assert_eq!(resp.object, "chat.completion");
    assert_eq!(
        resp.choices[0].message.text().as_deref(),
        Some("Hello from chat")
    );
    let usage = resp.usage.as_ref().expect("usage present");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 3);

    // Wire assertions.
    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    let req = &recorded[0];
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/chat/completions");
    assert_eq!(req.header("authorization"), Some("Bearer sk-test-key"));
    assert_eq!(req.body["model"], "mock-model");
    assert_eq!(req.body["stream"], false);
    assert_eq!(req.body["messages"][0]["role"], "user");
}

#[tokio::test]
async fn openai_chat_stream_survives_arbitrary_chunk_split() {
    let server = MockServer::start().await;
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::OpenAIChatCompletions)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let stream = client
        .chat_stream(simple_chat_request("mock-model"))
        .await
        .expect("stream opens");
    let chunks: Vec<hillm::HiLlmResult<hillm::types::ChatCompletionChunk>> = stream.collect().await;
    let chunks: Vec<hillm::types::ChatCompletionChunk> = chunks
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("all chunks parse");

    // The mock writes SSE three bytes at a time; the event sequence must
    // still be exactly the two content deltas (the [DONE] sentinel ends the
    // Chat decoder without emitting an event).
    assert_eq!(chunks.len(), 2);
    let text: String = chunks
        .iter()
        .flat_map(|c| c.choices.iter())
        .filter_map(|ch| ch.delta.content.clone())
        .collect();
    assert_eq!(text, "Hello world");

    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].path, "/chat/completions");
    assert_eq!(recorded[0].body["stream"], true);
}

// ==================== Route 2: OpenAI Responses ====================

#[tokio::test]
async fn openai_responses_route_url_headers_and_native_body() {
    let server = MockServer::start().await;
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::OpenAIResponses)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let req = CreateResponseRequest {
        model: "mock-model".to_string(),
        input: serde_json::json!("Hello"),
        ..Default::default()
    };
    let resp = client.create_response(req).await.unwrap();

    // Native Responses fields survive.
    assert_eq!(resp.id, "resp-mock-1");
    assert_eq!(resp.object, "response");
    assert_eq!(resp.status, "completed");
    assert_eq!(resp.output.len(), 1);
    assert_eq!(resp.output[0].item_type, "message");
    let usage = resp.usage.as_ref().expect("usage present");
    assert_eq!(usage.input_tokens, 7);
    assert_eq!(usage.output_tokens, 4);

    // Wire assertions: /responses, not /chat/completions.
    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    let req = &recorded[0];
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/responses");
    assert_eq!(req.header("authorization"), Some("Bearer sk-test-key"));
    assert_eq!(req.body["model"], "mock-model");
    assert_eq!(req.body["input"], "Hello");
    assert_eq!(req.body["stream"], false);
}

#[tokio::test]
async fn openai_responses_stream_emits_native_events() {
    let server = MockServer::start().await;
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::OpenAIResponses)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let req = CreateResponseRequest {
        model: "mock-model".to_string(),
        input: serde_json::json!("Hello"),
        ..Default::default()
    };
    let stream = client
        .create_response_stream(req)
        .await
        .expect("stream opens");
    let events: Vec<hillm::HiLlmResult<ResponsesStreamEvent>> = stream.collect().await;
    let events: Vec<ResponsesStreamEvent> = events
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("all events parse");

    // Native event sequence — no conversion to Chat Completions chunks.
    assert_eq!(events.len(), 3);
    assert!(matches!(
        events[0],
        ResponsesStreamEvent::ResponseCreated { .. }
    ));
    match &events[1] {
        ResponsesStreamEvent::OutputTextDelta { delta, .. } => assert_eq!(delta, "Hello"),
        other => panic!("expected text delta, got {other:?}"),
    }
    assert!(matches!(
        events[2],
        ResponsesStreamEvent::ResponseCompleted { .. }
    ));

    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].path, "/responses");
    assert_eq!(recorded[0].body["stream"], true);
}

// ==================== Route 3: Anthropic Messages ====================

#[tokio::test]
async fn anthropic_messages_route_url_headers_and_native_body() {
    let server = MockServer::start().await;
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::AnthropicMessages)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let resp = client
        .create_message(simple_anthropic_request("mock-model"))
        .await
        .unwrap();

    // Native Anthropic fields survive: content blocks, stop reason and
    // cache usage are not flattened into Chat shapes.
    assert_eq!(resp.id, "msg-mock-1");
    assert_eq!(resp.stop_reason, AnthropicStopReason::EndTurn);
    assert_eq!(resp.usage.input_tokens, 9);
    assert_eq!(resp.usage.output_tokens, 2);
    assert_eq!(resp.usage.cache_read_input_tokens, Some(4));
    match &resp.content[0] {
        hillm::types::anthropic::AnthropicResponseContentBlock::Text { text } => {
            assert_eq!(text, "Hello from messages")
        }
        other => panic!("expected text block, got {other:?}"),
    }

    // Wire assertions: /messages and a native Anthropic body.
    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    let req = &recorded[0];
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/messages");
    assert_eq!(req.body["model"], "mock-model");
    assert_eq!(req.body["max_tokens"], 64);
    assert_eq!(req.body["stream"], false);
    assert_eq!(req.body["messages"][0]["role"], "user");
    assert_eq!(req.body["messages"][0]["content"][0]["type"], "text");
}

#[tokio::test]
async fn anthropic_messages_stream_emits_native_events() {
    let server = MockServer::start().await;
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::AnthropicMessages)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let stream = client
        .create_message_stream(simple_anthropic_request("mock-model"))
        .await
        .expect("stream opens");
    let events: Vec<hillm::HiLlmResult<AnthropicStreamEvent>> = stream.collect().await;
    let events: Vec<AnthropicStreamEvent> = events
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("all events parse");

    // Native event sequence: message_start → content_block_delta →
    // message_delta → message_stop.
    assert_eq!(events.len(), 4);
    match &events[0] {
        AnthropicStreamEvent::MessageStart { message } => {
            assert_eq!(message.id, "msg-mock-1");
            assert_eq!(message.usage.input_tokens, 9);
        }
        other => panic!("expected message_start, got {other:?}"),
    }
    match &events[1] {
        AnthropicStreamEvent::ContentBlockDelta {
            delta: hillm::types::anthropic::AnthropicDelta::TextDelta { text },
            ..
        } => assert_eq!(text, "Hello"),
        other => panic!("expected text delta, got {other:?}"),
    }
    match &events[2] {
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            assert_eq!(delta.stop_reason, Some(AnthropicStopReason::EndTurn));
            assert_eq!(usage.output_tokens, 2);
        }
        other => panic!("expected message_delta, got {other:?}"),
    }
    assert!(matches!(events[3], AnthropicStreamEvent::MessageStop));

    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].path, "/messages");
    assert_eq!(recorded[0].body["stream"], true);
}

// ============ Selection rules and pre-send failures ============

#[tokio::test]
async fn unsupported_api_type_fails_before_sending() {
    // Named provider + unsupported API type: must fail at build time with a
    // structured error — no request may be sent.
    let config = ClientConfigBuilder::new("sk-test-key")
        .api_type(APIType::AnthropicMessages)
        .build();
    let err = match DefaultClient::new(config, Some("openai".to_string())) {
        Err(err) => err,
        Ok(_) => panic!("expected APITypeUnsupported"),
    };
    assert!(
        matches!(err, HiLlmError::APITypeUnsupported { .. }),
        "expected APITypeUnsupported, got: {err}"
    );
}

#[tokio::test]
async fn streaming_other_routes_fails_before_sending_on_chat_instance() {
    let server = MockServer::start().await;
    // An instance bound to OpenAI Chat Completions.
    let config = ClientConfigBuilder::new("sk-test-key")
        .base_url(server.url())
        .api_type(APIType::OpenAIChatCompletions)
        .build();
    let client = DefaultClient::new(config, None).expect("client builds");

    let req = CreateResponseRequest {
        model: "mock-model".to_string(),
        input: serde_json::json!("Hello"),
        ..Default::default()
    };
    let err = match client.create_response_stream(req).await {
        Err(err) => err,
        Ok(_) => panic!("Responses streaming must be rejected"),
    };
    assert!(
        matches!(err, HiLlmError::EndpointNotSupported { .. }),
        "expected EndpointNotSupported, got: {err}"
    );

    let err = match client
        .create_message_stream(simple_anthropic_request("mock-model"))
        .await
    {
        Err(err) => err,
        Ok(_) => panic!("Anthropic streaming must be rejected"),
    };
    assert!(
        matches!(err, HiLlmError::EndpointNotSupported { .. }),
        "expected EndpointNotSupported, got: {err}"
    );

    assert_eq!(
        server.request_count(),
        0,
        "no request may be sent for an unsupported api type"
    );
}

/// The explicit compatibility adapter: `chat()` against the `anthropic`
/// provider selected with the Chat Completions API type. The request is
/// converted Chat → Anthropic and sent to `/messages` with Anthropic auth
/// headers; the response is converted Anthropic → Chat.
#[tokio::test]
#[serial]
async fn chat_on_anthropic_uses_explicit_compat_adapter() {
    let server = MockServer::start().await;

    // Point the built-in anthropic provider at the mock server.
    let previous = std::env::var("ANTHROPIC_BASE_URL").ok();
    unsafe {
        std::env::set_var("ANTHROPIC_BASE_URL", server.url());
    }

    let config = ClientConfigBuilder::new("sk-ant-test")
        .api_type(APIType::OpenAIChatCompletions)
        .build();
    let client = DefaultClient::new(config, Some("anthropic".to_string()))
        .expect("compat adapter instance builds");

    let result = client.chat(simple_chat_request("mock-model")).await;

    // Restore the environment regardless of the outcome.
    match previous {
        Some(v) => unsafe { std::env::set_var("ANTHROPIC_BASE_URL", v) },
        None => unsafe { std::env::remove_var("ANTHROPIC_BASE_URL") },
    }

    let resp = result.expect("compat chat call succeeds");

    // The caller sees Chat Completions shapes…
    assert_eq!(resp.object, "chat.completion");
    assert_eq!(
        resp.choices[0].message.text().as_deref(),
        Some("Hello from messages")
    );

    // …but the wire used the Anthropic protocol: /messages endpoint,
    // x-api-key auth and an Anthropic-shaped body (converted by the adapter).
    let recorded = server.recorded();
    assert_eq!(recorded.len(), 1);
    let req = &recorded[0];
    assert_eq!(req.path, "/messages");
    assert_eq!(req.header("x-api-key"), Some("sk-ant-test"));
    assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(req.body["messages"][0]["role"], "user");
    assert_eq!(req.body["messages"][0]["content"][0]["type"], "text");
    // The adapter applies the Anthropic default max_tokens.
    assert_eq!(req.body["max_tokens"], 4096);
    // If the Chat `stream` flag leaks through, it must at least stay false.
    match req.body.get("stream").and_then(|v| v.as_bool()) {
        Some(false) | None => {}
        Some(true) => panic!("the adapter must not turn the call into a stream"),
    }
}
