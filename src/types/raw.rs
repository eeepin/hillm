#[derive(Debug, Clone)]
pub struct RawExchange<T> {
    pub data: T,
    pub raw_request: serde_json::Value,
    pub raw_response: Option<serde_json::Value>,
}

#[derive(Debug)]
pub struct RawStreamExchange<S> {
    pub stream: S,
    pub raw_request: serde_json::Value,
}
