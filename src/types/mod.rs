pub mod audio;
pub mod batch;
pub mod chat;
pub mod embedding;
pub mod file;
pub mod image;
pub mod model;
pub mod moderation;
pub mod ocr;
pub mod raw;
pub mod rerank;
pub mod response;
pub mod search;

pub use audio::*;
pub use batch::*;
pub use chat::*;
pub use embedding::*;
pub use file::*;
pub use image::*;
pub use model::*;
pub use moderation::*;
pub use ocr::*;
pub use raw::*;
pub use rerank::*;
pub use response::*;
pub use search::*;

use serde::{Deserialize, Serialize};

// Messages

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
    Developer(DeveloperMessage),
}

impl Default for Message {
    fn default() -> Self {
        Self::Assistant(AssistantMessage::default())
    }
}

impl Message {
    pub fn user_with_parts(parts: Vec<ContentPart>) -> Self {
        Self::User(UserMessage {
            content: MessageContent::Parts(parts),
            name: None,
        })
    }
    pub fn system_with_parts(parts: Vec<ContentPart>) -> Self {
        Self::System(SystemMessage {
            content: MessageContent::Parts(parts),
            name: None,
        })
    }
    pub fn assistant_with_parts(parts: Vec<ContentPart>) -> Self {
        Self::Assistant(AssistantMessage {
            content: Some(MessageContent::Parts(parts)),
            name: None,
            tool_calls: None,
            refusal: None,
        })
    }
}

/// System message
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// User message
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Assistant message
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl AssistantMessage {
    pub fn text(&self) -> Option<String> {
        self.content.as_ref()?.as_text()
    }

    pub fn refusal_text(&self) -> Option<&str> {
        if let Some(r) = self.refusal.as_deref() {
            return Some(r);
        }
        if let Some(MessageContent::Parts(parts)) = self.content.as_ref() {
            for part in parts {
                if let ContentPart::Refusal { refusal } = part {
                    return Some(refusal.as_str());
                }
            }
        }
        None
    }

    pub fn output_images(&self) -> Vec<ImageUrl> {
        let Some(MessageContent::Parts(parts)) = self.content.as_ref() else {
            return vec![];
        };
        parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::OutputImage { image_url } => Some(image_url.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn output_audio(&self) -> Vec<AudioContent> {
        let Some(MessageContent::Parts(parts)) = self.content.as_ref() else {
            return vec![];
        };
        parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::OutputAudio { audio } => Some(audio.clone()),
                _ => None,
            })
            .collect()
    }
}

/// Tool message
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolMessage {
    pub content: MessageContent,
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Developer message
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeveloperMessage {
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content: String or Array of content parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl MessageContent {
    pub fn as_text(&self) -> Option<String> {
        match self {
            MessageContent::Text(s) => Some(s.clone()),
            MessageContent::Parts(parts) => {
                let texts: Vec<&str> = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join(""))
                }
            }
        }
    }
}

impl std::fmt::Display for MessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_text().as_deref().unwrap_or(""))
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

/// Content part in message, can be text, image, document, audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Refusal { refusal: String },
    ImageUrl { image_url: ImageUrl },
    Document { document: DocumentContent },
    InputAudio { input_audio: AudioContent },
    OutputImage { image_url: ImageUrl },
    OutputAudio { audio: AudioContent },
}

impl Default for ContentPart {
    fn default() -> Self {
        Self::Text {
            text: String::new(),
        }
    }
}

impl ContentPart {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    pub fn image_data_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    pub fn image_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail),
            },
        }
    }

    pub fn image_png(bytes: &[u8]) -> Self {
        Self::image_data_url(crate::image::encode_data_url(
            bytes,
            Some(crate::image::IMAGE_PNG),
        ))
    }

    pub fn image_jpeg(bytes: &[u8]) -> Self {
        Self::image_data_url(crate::image::encode_data_url(
            bytes,
            Some(crate::image::IMAGE_JPEG),
        ))
    }

    pub fn image_webp(bytes: &[u8]) -> Self {
        Self::image_data_url(crate::image::encode_data_url(
            bytes,
            Some(crate::image::IMAGE_WEBP),
        ))
    }

    pub fn image_tiff(bytes: &[u8]) -> Self {
        Self::image_data_url(crate::image::encode_data_url(
            bytes,
            Some(crate::image::IMAGE_TIFF),
        ))
    }
}

/// An image URL with detailed level.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Image detail level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

/// Document content for vision models.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentContent {
    pub data: String,
    pub media_type: String,
}

/// Audio content for speech models.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioContent {
    pub data: String,
    pub format: String,
}

// Tool

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolType {
    #[default]
    #[serde(rename = "function")]
    Function,
}

/// tool that model can invoke.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: ToolType,
    pub function: FunctionCall,
}

/// Function call, the detailed tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool choice, define how to choose tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Specific(SpecificToolChoice),
}

/// Tool choice mode.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    #[default]
    Auto,
    Required,
    None,
}

/// Directively call a specific tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecificToolChoice {
    #[serde(rename = "type")]
    pub choice_type: ToolType,
    pub function: SpecificFunction,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecificFunction {
    pub name: String,
}

// Response

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: JsonSchemaFormat,
    },
}

impl ResponseFormat {
    pub fn json_schema(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self::JsonSchema {
            json_schema: JsonSchemaFormat::new(name, schema),
        }
    }

    pub fn json_object() -> Self {
        Self::JsonObject
    }

    pub fn text() -> Self {
        Self::Text
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl JsonSchemaFormat {
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            schema,
            strict: Some(true),
        }
    }

    #[must_use]
    pub fn strict(mut self, on: bool) -> Self {
        self.strict = Some(on);
        self
    }

    #[must_use]
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }
}

// Usage

/// Token usage
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_details: Option<CostDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_byok: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub audio_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CostDetails {
    pub upstream_inference_completions_cost: f64,
    pub upstream_inference_prompt_cost: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_inference_cost: Option<f64>,
}

// Stop Sequence

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl Default for StopSequence {
    fn default() -> Self {
        Self::Single(String::new())
    }
}

//  Modality

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Audio,
    Image,
    Video,
    Pdf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_part_text_constructor() {
        let part = ContentPart::text("hi");
        let json = serde_json::to_string(&part).expect("serialization should not fail");
        assert_eq!(json, r#"{"type":"text","text":"hi"}"#);
    }

    #[test]
    fn content_part_image_data_url_constructor() {
        let part = ContentPart::image_data_url("data:image/png;base64,aGk=");
        let json = serde_json::to_string(&part).expect("serialization should not fail");
        assert_eq!(
            json,
            r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}}"#
        );
    }

    #[test]
    fn content_part_image_with_detail() {
        let part = ContentPart::image_with_detail("https://example.com/img.png", ImageDetail::High);
        let json = serde_json::to_string(&part).expect("serialization should not fail");
        assert_eq!(
            json,
            r#"{"type":"image_url","image_url":{"url":"https://example.com/img.png","detail":"high"}}"#
        );
    }

    #[test]
    fn content_part_image_png_round_trip() {
        let part = ContentPart::image_png(b"hi");
        match &part {
            ContentPart::ImageUrl { image_url } => {
                assert!(
                    image_url.url.starts_with("data:image/png;base64,"),
                    "expected png data URL, got: {}",
                    image_url.url
                );
            }
            other => panic!("expected ImageUrl variant, got: {other:?}"),
        }
    }

    #[test]
    fn message_user_with_parts() {
        let msg = Message::user_with_parts(vec![
            ContentPart::text("hello"),
            ContentPart::image_data_url("data:image/png;base64,aGk="),
        ]);
        let json = serde_json::to_string(&msg).expect("serialization should not fail");
        assert_eq!(
            json,
            r#"{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}}]}"#
        );
    }

    // ── ResponseFormat / JsonSchemaFormat constructors ──────────────────────

    #[test]
    fn json_schema_new_defaults_strict_true() {
        let fmt = JsonSchemaFormat::new("S", serde_json::json!({}));
        assert_eq!(fmt.strict, Some(true));
        assert_eq!(fmt.description, None);
        assert_eq!(fmt.name, "S");
    }

    #[test]
    fn json_schema_strict_toggle() {
        let fmt = JsonSchemaFormat::new("S", serde_json::json!({})).strict(false);
        assert_eq!(fmt.strict, Some(false));
    }

    #[test]
    fn json_schema_description_attaches() {
        let fmt = JsonSchemaFormat::new("S", serde_json::json!({})).description("d");
        assert_eq!(fmt.description.as_deref(), Some("d"));
    }

    #[test]
    fn response_format_json_schema_serializes() {
        let fmt = ResponseFormat::json_schema("S", serde_json::json!({"type": "object"}));
        let value = serde_json::to_value(&fmt).expect("serialization must succeed");
        assert_eq!(
            value,
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "S",
                    "schema": {"type": "object"},
                    "strict": true
                }
            })
        );
        // description must be absent (skip_serializing_if = "Option::is_none")
        assert!(value["json_schema"].get("description").is_none());
    }

    #[test]
    fn response_format_json_object_serializes() {
        let value = serde_json::to_value(ResponseFormat::json_object())
            .expect("serialization must succeed");
        assert_eq!(value, serde_json::json!({"type": "json_object"}));
    }

    #[test]
    fn response_format_text_serializes() {
        let value =
            serde_json::to_value(ResponseFormat::text()).expect("serialization must succeed");
        assert_eq!(value, serde_json::json!({"type": "text"}));
    }

    #[test]
    fn chat_request_serializes_response_format() {
        use crate::types::chat::ChatCompletionRequest;
        let request = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            response_format: Some(ResponseFormat::json_schema(
                "PersonSchema",
                serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}}),
            )),
            ..Default::default()
        };
        let value = serde_json::to_value(&request).expect("serialization must succeed");
        let rf = &value["response_format"];
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "PersonSchema");
        assert_eq!(rf["json_schema"]["strict"], true);
    }

    // AssistantMessage tests for llm response.

    #[test]
    fn assistant_content_text_deserializes_from_scalar_string() {
        let json = r#"{"role":"assistant","content":"hi"}"#;
        let msg: Message = serde_json::from_str(json).expect("must deserialise");
        let Message::Assistant(a) = msg else {
            panic!("expected assistant")
        };
        assert_eq!(a.content, Some(MessageContent::Text("hi".into())));
    }

    #[test]
    fn assistant_content_parts_deserializes_from_array() {
        let json = r#"{"role":"assistant","content":[{"type":"text","text":"hi"}]}"#;
        let msg: Message = serde_json::from_str(json).expect("must deserialise");
        let Message::Assistant(a) = msg else {
            panic!("expected assistant")
        };
        assert_eq!(
            a.content,
            Some(MessageContent::Parts(vec![ContentPart::Text {
                text: "hi".into()
            }]))
        );
    }

    #[test]
    fn assistant_content_display_renders_text() {
        assert_eq!(
            MessageContent::Text("hi there".into()).to_string(),
            "hi there"
        );
        let parts = MessageContent::Parts(vec![
            ContentPart::Text { text: "a".into() },
            ContentPart::OutputImage {
                image_url: ImageUrl::default(),
            },
            ContentPart::Text { text: "b".into() },
        ]);
        assert_eq!(parts.to_string(), "ab");
        // Image-only / non-text content renders as an empty string.
        let image_only = MessageContent::Parts(vec![ContentPart::OutputImage {
            image_url: ImageUrl::default(),
        }]);
        assert_eq!(image_only.to_string(), "");
    }

    #[test]
    fn assistant_message_text_helper_with_text() {
        let a = AssistantMessage {
            content: Some(MessageContent::Text("hello".into())),
            ..Default::default()
        };
        assert_eq!(a.text(), Some("hello".into()));
    }

    #[test]
    fn assistant_message_text_helper_with_parts() {
        let a = AssistantMessage {
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "foo".into() },
                ContentPart::Text { text: "bar".into() },
            ])),
            ..Default::default()
        };
        assert_eq!(a.text(), Some("foobar".into()));
    }

    #[test]
    fn assistant_message_text_helper_with_refusal_only_is_none() {
        let a = AssistantMessage {
            content: Some(MessageContent::Parts(vec![ContentPart::Refusal {
                refusal: "I cannot do that.".into(),
            }])),
            ..Default::default()
        };
        assert_eq!(a.text(), None);
    }

    #[test]
    fn assistant_part_output_image_serializes() {
        let part = ContentPart::OutputImage {
            image_url: ImageUrl {
                url: "data:image/png;base64,aGk=".into(),
                detail: None,
            },
        };
        let json = serde_json::to_string(&part).expect("must serialise");
        assert_eq!(
            json,
            r#"{"type":"output_image","image_url":{"url":"data:image/png;base64,aGk="}}"#
        );
    }

    #[test]
    fn assistant_part_output_audio_serializes() {
        let part = ContentPart::OutputAudio {
            audio: AudioContent {
                data: "aGk=".into(),
                format: "wav".into(),
            },
        };
        let json = serde_json::to_string(&part).expect("must serialise");
        assert_eq!(
            json,
            r#"{"type":"output_audio","audio":{"data":"aGk=","format":"wav"}}"#
        );
    }

    #[test]
    fn message_system_with_parts_serializes() {
        let msg = Message::system_with_parts(vec![ContentPart::text("You are helpful.")]);
        let json = serde_json::to_string(&msg).expect("must serialise");
        assert_eq!(
            json,
            r#"{"role":"system","content":[{"type":"text","text":"You are helpful."}]}"#
        );
    }

    #[test]
    fn message_assistant_with_parts_round_trips() {
        let msg = Message::assistant_with_parts(vec![ContentPart::Text { text: "ok".into() }]);
        let json = serde_json::to_string(&msg).expect("must serialise");
        assert_eq!(
            json,
            r#"{"role":"assistant","content":[{"type":"text","text":"ok"}]}"#
        );
    }

    #[test]
    fn assistant_output_images_and_audio_helpers() {
        let url = ImageUrl {
            url: "data:image/png;base64,aGk=".into(),
            detail: None,
        };
        let audio = AudioContent {
            data: "aGk=".into(),
            format: "wav".into(),
        };
        let a = AssistantMessage {
            content: Some(MessageContent::Parts(vec![
                ContentPart::OutputImage {
                    image_url: url.clone(),
                },
                ContentPart::OutputAudio {
                    audio: audio.clone(),
                },
            ])),
            ..Default::default()
        };
        assert_eq!(a.output_images(), vec![url.clone()]);
        assert_eq!(a.output_audio(), vec![audio.clone()]);
    }

    #[test]
    fn system_message_content_from_string_back_compat() {
        let json = r#"{"role":"system","content":"You are a helpful assistant."}"#;
        let msg: Message = serde_json::from_str(json).expect("must deserialise");
        let Message::System(s) = msg else {
            panic!("expected system")
        };
        assert_eq!(
            s.content,
            MessageContent::Text("You are a helpful assistant.".into())
        );
    }
}
