mod batch;
mod live;

use crate::providers::Provider;

use super::{LanguageQuality, LanguageSupport};

pub(crate) fn recognize_http_path(parsed: &url::Url) -> String {
    let base_path = parsed.path().trim_end_matches('/');
    let suffix = Provider::Watsonx.ws_path();
    if base_path.is_empty() || base_path == "/" {
        suffix.to_string()
    } else if base_path.ends_with(suffix) {
        base_path.to_string()
    } else {
        format!("{base_path}{suffix}")
    }
}

#[derive(Clone, Default)]
pub struct WatsonxAdapter;

impl WatsonxAdapter {
    pub fn language_support_live(_languages: &[hypr_language::Language]) -> LanguageSupport {
        LanguageSupport::Supported {
            quality: LanguageQuality::NoData,
        }
    }

    pub fn language_support_batch(_languages: &[hypr_language::Language]) -> LanguageSupport {
        LanguageSupport::Supported {
            quality: LanguageQuality::NoData,
        }
    }

    pub fn is_supported_languages_live(languages: &[hypr_language::Language]) -> bool {
        Self::language_support_live(languages).is_supported()
    }

    pub fn is_supported_languages_batch(languages: &[hypr_language::Language]) -> bool {
        Self::language_support_batch(languages).is_supported()
    }

    pub(crate) fn build_ws_url_from_base(api_base: &str) -> (url::Url, Vec<(String, String)>) {
        super::build_ws_url_from_base_with(Provider::Watsonx, api_base, |parsed| {
            let host = parsed
                .host_str()
                .unwrap_or_else(|| Provider::Watsonx.default_ws_host());
            let path = recognize_http_path(parsed);
            let mut url: url::Url = format!("wss://{}{}", host, path)
                .parse()
                .expect("invalid_ws_url");
            super::set_scheme_from_host(&mut url);
            url
        })
    }
}

#[cfg(test)]
mod tests {
    use hypr_ws_client::client::Message;
    use owhisper_interface::ListenParams;
    use owhisper_interface::stream::StreamResponse;

    use super::*;
    use crate::adapter::RealtimeSttAdapter;

    #[test]
    fn initial_message_multimedia_sets_inactivity_and_low_latency() {
        let adapter = WatsonxAdapter::default();
        let params = ListenParams {
            model: Some("en-US_Multimedia".to_string()),
            sample_rate: 16000,
            ..Default::default()
        };
        let Some(Message::Text(t)) = adapter.initial_message(None, &params, 1) else {
            panic!("expected text start message");
        };
        let v: serde_json::Value = serde_json::from_str(t.as_str()).expect("json");
        assert_eq!(v["inactivity_timeout"], 3600);
        assert_eq!(v["low_latency"], true);
    }

    #[test]
    fn initial_message_stereo_sets_two_channels_in_content_type() {
        let adapter = WatsonxAdapter::default();
        let params = ListenParams {
            model: Some("en-US_Multimedia".to_string()),
            sample_rate: 16000,
            ..Default::default()
        };
        let Some(Message::Text(t)) = adapter.initial_message(None, &params, 2) else {
            panic!("expected text start message");
        };
        assert!(
            t.contains("channels=2"),
            "expected stereo content-type, got {t}"
        );
    }

    #[test]
    fn parse_result_respects_channel_index_when_present() {
        let adapter = WatsonxAdapter::default();
        let raw = r#"{"result_index":0,"results":[{"final":false,"channel_index":[1],"alternatives":[{"transcript":"hi","confidence":0.9,"timestamps":[]}]}]}"#;
        let out = adapter.parse_response(raw);
        assert_eq!(out.len(), 1);
        let StreamResponse::TranscriptResponse { channel_index, .. } = &out[0] else {
            panic!("expected transcript");
        };
        assert_eq!(channel_index.as_slice(), [1_i32]);
    }

    #[test]
    fn initial_message_broadband_omits_low_latency() {
        let adapter = WatsonxAdapter::default();
        let params = ListenParams {
            model: Some("en-US_BroadbandModel".to_string()),
            sample_rate: 16000,
            ..Default::default()
        };
        let Some(Message::Text(t)) = adapter.initial_message(None, &params, 1) else {
            panic!("expected text start message");
        };
        let v: serde_json::Value = serde_json::from_str(t.as_str()).expect("json");
        assert_eq!(v["inactivity_timeout"], 3600);
        assert!(v.get("low_latency").is_none());
    }

    #[test]
    fn parse_top_level_string_error_emits_error_response() {
        let adapter = WatsonxAdapter::default();
        let raw = r#"{"error":"Unsupported model for this session."}"#;
        let out = adapter.parse_response(raw);
        assert_eq!(out.len(), 1);
        let StreamResponse::ErrorResponse {
            error_message,
            provider,
            ..
        } = &out[0]
        else {
            panic!("expected error response");
        };
        assert_eq!(provider, "watsonx");
        assert_eq!(error_message, "Unsupported model for this session.");
    }

    #[test]
    fn parse_interim_without_timestamps_uses_nonzero_word_span() {
        let adapter = WatsonxAdapter::default();
        let raw = r#"{"result_index":1,"results":[{"final":false,"alternatives":[{"transcript":"hello there","confidence":0.9,"timestamps":[]}]}]}"#;
        let out = adapter.parse_response(raw);
        assert_eq!(out.len(), 1);
        let StreamResponse::TranscriptResponse {
            channel,
            channel_index,
            ..
        } = &out[0]
        else {
            panic!("expected transcript");
        };
        assert_eq!(channel_index.as_slice(), [0_i32]);
        let w = &channel.alternatives[0].words[0];
        assert!(w.start > 0.0 && w.end > w.start);
    }

    #[test]
    fn test_build_ws_url_from_base_regional() {
        let (url, params) = WatsonxAdapter::build_ws_url_from_base(
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com",
        );
        assert_eq!(
            url.as_str(),
            "wss://api.us-south.speech-to-text.watson.cloud.ibm.com/v1/recognize"
        );
        assert!(params.is_empty());
    }

    #[test]
    fn test_build_ws_url_from_base_instance_url() {
        let (url, params) = WatsonxAdapter::build_ws_url_from_base(
            "https://api.us-east.speech-to-text.watson.cloud.ibm.com/instances/90bb9d85-165b-4755-bdd7-1b50ff1ea112",
        );
        assert_eq!(
            url.as_str(),
            "wss://api.us-east.speech-to-text.watson.cloud.ibm.com/instances/90bb9d85-165b-4755-bdd7-1b50ff1ea112/v1/recognize"
        );
        assert!(params.is_empty());
    }

    #[test]
    fn test_build_ws_url_from_base_proxy() {
        let (url, params) =
            WatsonxAdapter::build_ws_url_from_base("https://api.hyprnote.com?provider=watsonx");
        assert_eq!(url.as_str(), "wss://api.hyprnote.com/listen");
        assert_eq!(
            params,
            vec![("provider".to_string(), "watsonx".to_string())]
        );
    }

    #[test]
    fn test_is_watsonx_host() {
        assert!(Provider::Watsonx.is_host("api.us-south.speech-to-text.watson.cloud.ibm.com"));
        assert!(!Provider::Watsonx.is_host("api.deepgram.com"));
    }
}
