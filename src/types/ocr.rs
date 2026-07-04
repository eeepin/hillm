use serde::{Deserialize, Serialize};

use super::Usage;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrRequest {
    pub model: String,
    pub document: OcrDocument,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_image_base64: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    Url { url: String },
    #[serde(rename = "base64")]
    Base64 { data: String, media_type: String },
}

impl Default for OcrDocument {
    fn default() -> Self {
        Self::Url { url: String::new() }
    }
}

/// An OCR response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResponse {
    pub pages: Vec<OcrPage>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrPage {
    pub index: u32,
    pub markdown: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<OcrImage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<PageDimensions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrImage {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDimensions {
    pub width: u32,
    pub height: u32,
}
