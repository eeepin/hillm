use std::future::Future;

use bytes::Bytes;

use crate::error::{HiLlmError, HiLlmResult};
use crate::http::retry;

pub(crate) fn retry_after_from_response(resp: &reqwest::Response) -> Option<std::time::Duration> {
    let value = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    retry::parse_retry_after(value)
}

pub(crate) async fn with_retry<F, Fut>(
    max_retries: u32,
    mut send: F,
) -> HiLlmResult<reqwest::Response>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = std::result::Result<reqwest::Response, reqwest::Error>>,
{
    let mut attempt = 0u32;

    loop {
        let resp = send().await?;
        let status = resp.status().as_u16();

        if resp.status().is_success() {
            return Ok(resp);
        }

        let server_retry_after = retry_after_from_response(&resp);

        if let Some(delay) = retry::should_retry(status, attempt, max_retries, server_retry_after) {
            attempt += 1;
            tokio::time::sleep(delay).await;
            continue;
        }

        let text = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(failed to read body: {e})"));
        return Err(HiLlmError::from_status(status, &text, server_retry_after));
    }
}

pub async fn post_json_raw(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
) -> HiLlmResult<serde_json::Value> {
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
    resp.json::<serde_json::Value>()
        .await
        .map_err(HiLlmError::from)
}

pub async fn post_binary(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    body: Bytes,
    max_retries: u32,
) -> HiLlmResult<Bytes> {
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

    resp.bytes().await.map_err(HiLlmError::from)
}

pub async fn post_multipart(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    form: reqwest::multipart::Form,
) -> HiLlmResult<serde_json::Value> {
    let mut builder = client.post(url).multipart(form);
    if let Some((name, value)) = auth_header {
        builder = builder.header(name, value);
    }
    for (name, value) in extra_headers {
        builder = builder.header(*name, *value);
    }

    let resp = builder.send().await?;

    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let server_retry_after = retry_after_from_response(&resp);
        let text = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(failed to read body: {e})"));
        return Err(HiLlmError::from_status(status, &text, server_retry_after));
    }

    resp.json::<serde_json::Value>()
        .await
        .map_err(HiLlmError::from)
}

pub async fn get_json_raw(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    max_retries: u32,
) -> HiLlmResult<serde_json::Value> {
    let mut retry_count = 0u32;

    let resp = with_retry(max_retries, || {
        let mut builder = client.get(url);
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

    resp.json::<serde_json::Value>()
        .await
        .map_err(HiLlmError::from)
}

pub async fn delete_json(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    max_retries: u32,
) -> HiLlmResult<serde_json::Value> {
    let mut retry_count = 0u32;

    let resp = with_retry(max_retries, || {
        let mut builder = client.delete(url);
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

    resp.json::<serde_json::Value>()
        .await
        .map_err(HiLlmError::from)
}

pub async fn get_binary(
    client: &reqwest::Client,
    url: &str,
    auth_header: Option<(&str, &str)>,
    extra_headers: &[(&str, &str)],
    max_retries: u32,
) -> HiLlmResult<Bytes> {
    let mut retry_count = 0u32;

    let resp = with_retry(max_retries, || {
        let mut builder = client.get(url);
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

    resp.bytes().await.map_err(HiLlmError::from)
}
