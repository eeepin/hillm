use base64::Engine as _;
use serde::{Deserialize, Serialize};

pub const IMAGE_PNG: &str = "image/png";
pub const IMAGE_JPEG: &str = "image/jpeg";
pub const IMAGE_WEBP: &str = "image/webp";
pub const IMAGE_TIFF: &str = "image/tiff";

pub fn encode_data_url(bytes: &[u8], mime: Option<&str>) -> String {
    let mime = mime.unwrap_or(IMAGE_PNG);
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{b64}")
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DecodedDataUrl {
    pub mime: String,
    pub data: Vec<u8>,
}

pub fn decode_data_url(url: &str) -> Option<DecodedDataUrl> {
    let rest = url.strip_prefix("data:")?;
    let marker = ";base64,";
    let marker_pos = rest.find(marker)?;
    let mime = rest[..marker_pos].to_owned();
    let b64 = &rest[marker_pos + marker.len()..];
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    Some(DecodedDataUrl { mime, data: bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_default_mime_is_png() {
        let url = encode_data_url(b"hi", None);
        assert!(
            url.starts_with("data:image/png;base64,"),
            "expected png prefix, got: {url}"
        );
    }

    #[test]
    fn encode_explicit_mime() {
        let url = encode_data_url(b"hi", Some(IMAGE_JPEG));
        assert!(
            url.starts_with("data:image/jpeg;base64,"),
            "expected jpeg prefix, got: {url}"
        );
    }

    #[test]
    fn decode_round_trip() {
        let payload = b"round-trip bytes \x00\x01\x02";
        for mime in [IMAGE_PNG, IMAGE_JPEG, IMAGE_WEBP, IMAGE_TIFF] {
            let url = encode_data_url(payload, Some(mime));
            let decoded = decode_data_url(&url)
                .unwrap_or_else(|| panic!("round-trip failed for mime={mime}"));
            assert_eq!(decoded.mime, mime, "mime mismatch for {mime}");
            assert_eq!(decoded.data, payload, "bytes mismatch for {mime}");
        }
    }

    #[test]
    fn decode_rejects_non_data_url() {
        assert!(decode_data_url("https://example.com/img.png").is_none());
    }

    #[test]
    fn decode_rejects_malformed_base64() {
        assert!(decode_data_url("data:image/png;base64,!@#$").is_none());
    }

    #[test]
    fn decode_rejects_missing_base64_marker() {
        assert!(decode_data_url("data:image/png,plaintext").is_none());
    }

    #[test]
    fn byte_patterns_round_trip() {
        let test_cases: &[&[u8]] = &[
            b"",
            b"\x00",
            b"\xff\xfe\xfd",
            b"hello world",
            &[0u8; 256],
            &(0u8..=255u8).collect::<Vec<_>>(),
        ];
        for &bytes in test_cases {
            let url = encode_data_url(bytes, Some(IMAGE_PNG));
            let decoded = decode_data_url(&url)
                .unwrap_or_else(|| panic!("round-trip failed for input len={}", bytes.len()));
            assert_eq!(decoded.mime, IMAGE_PNG);
            assert_eq!(decoded.data, bytes);
        }
    }
}
