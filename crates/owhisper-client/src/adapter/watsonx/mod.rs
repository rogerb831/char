mod batch;
mod live;

use serde::{Deserialize, Deserializer};

use owhisper_interface::ListenParams;

use crate::providers::{Provider, is_meta_model};

use super::{LanguageQuality, LanguageSupport};

pub(super) fn resolved_watsonx_model<'a>(
    params: &'a ListenParams,
    default: &'static str,
) -> &'a str {
    match params.model.as_deref().map(str::trim) {
        None | Some("") => default,
        Some(m) if is_meta_model(m) => default,
        Some(m) => m,
    }
}

pub(super) fn deserialize_vec_skip_null<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let opt = Option::<Vec<T>>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct WatsonSpeakerLabel {
    from: f64,
    to: f64,
    speaker: i32,
}

pub(super) fn speaker_id_for_word_start(
    word_start: f64,
    labels: &[WatsonSpeakerLabel],
) -> Option<i32> {
    const EPS: f64 = 1e-4;
    labels.iter().find_map(|lb| {
        if word_start + EPS >= lb.from && word_start < lb.to + EPS {
            Some(lb.speaker)
        } else {
            None
        }
    })
}

pub(super) fn assign_speakers_by_label_index(
    words: &mut [owhisper_interface::stream::Word],
    labels: &[WatsonSpeakerLabel],
) {
    if labels.is_empty() {
        return;
    }
    let last = labels.last().map(|lb| lb.speaker);
    for (i, w) in words.iter_mut().enumerate() {
        w.speaker = Some(
            labels
                .get(i)
                .map(|lb| lb.speaker)
                .unwrap_or_else(|| last.unwrap_or(0)),
        );
    }
}

pub(super) fn watsonx_next_gen_model(model: Option<&str>) -> bool {
    match model {
        None => true,
        Some(m) => m.contains("Multimedia") || m.contains("Telephony"),
    }
}

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
    fn initial_message_multimedia_sets_low_latency_and_speaker_labels() {
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
        assert_eq!(v["speaker_labels"], true);
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
        assert_eq!(channel_index.as_slice(), [1_i32, 2_i32]);
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
        assert!(v.get("speaker_labels").is_none());
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
        assert_eq!(channel_index.as_slice(), [2_i32, 2_i32]);
        let w = &channel.alternatives[0].words[0];
        assert!(w.start > 0.0 && w.end > w.start);
    }

    #[test]
    fn parse_speaker_labels_populate_word_speaker_and_mixed_channel() {
        let adapter = WatsonxAdapter::default();
        let raw = r#"{"result_index":0,"speaker_labels":[{"from":0.0,"to":30.0,"speaker":0}],"results":[{"final":false,"alternatives":[{"transcript":"hello","confidence":0.9,"timestamps":["hello",1.0,2.0]}]}]}"#;
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
        assert_eq!(channel_index.as_slice(), [2_i32, 2_i32]);
        assert_eq!(channel.alternatives[0].words[0].speaker, Some(0));
    }

    #[test]
    fn parse_accepts_camel_case_speaker_labels_key() {
        let adapter = WatsonxAdapter::default();
        let raw = r#"{"result_index":0,"speakerLabels":[{"from":0.0,"to":30.0,"speaker":1}],"results":[{"final":true,"alternatives":[{"transcript":"hello","confidence":0.9,"timestamps":["hello",1.0,2.0]}]}]}"#;
        let out = adapter.parse_response(raw);
        let StreamResponse::TranscriptResponse { channel, .. } = &out[0] else {
            panic!("expected transcript");
        };
        assert_eq!(channel.alternatives[0].words[0].speaker, Some(1));
    }

    #[test]
    fn parse_timestamps_nested_array_carries_optional_speaker_index() {
        let adapter = WatsonxAdapter::default();
        let raw = r#"{"result_index":0,"results":[{"final":true,"alternatives":[{"transcript":"a b","confidence":0.9,"timestamps":[["a",1.0,2.0,0],["b",3.0,4.0,1]]}]}]}"#;
        let out = adapter.parse_response(raw);
        let StreamResponse::TranscriptResponse { channel, .. } = &out[0] else {
            panic!("expected transcript");
        };
        let words = &channel.alternatives[0].words;
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].speaker, Some(0));
        assert_eq!(words[1].speaker, Some(1));
    }

    #[test]
    fn build_ws_url_next_gen_adds_speaker_labels_query_param() {
        let adapter = WatsonxAdapter::default();
        let params = ListenParams {
            model: Some("en-US_Multimedia".to_string()),
            sample_rate: 16000,
            ..Default::default()
        };
        let url = adapter.build_ws_url(
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com",
            &params,
            1,
        );
        let q: std::collections::HashMap<String, String> = url.query_pairs().into_owned().collect();
        assert_eq!(q.get("speaker_labels").map(String::as_str), Some("true"));
        assert!(q.contains_key("model"));
    }

    #[test]
    fn build_ws_url_broadband_omits_speaker_labels_query_param() {
        let adapter = WatsonxAdapter::default();
        let params = ListenParams {
            model: Some("en-US_BroadbandModel".to_string()),
            sample_rate: 16000,
            ..Default::default()
        };
        let url = adapter.build_ws_url(
            "https://api.us-south.speech-to-text.watson.cloud.ibm.com",
            &params,
            1,
        );
        let q: std::collections::HashMap<String, String> = url.query_pairs().into_owned().collect();
        assert!(!q.contains_key("speaker_labels"));
    }

    #[test]
    fn resolved_watsonx_model_empty_or_whitespace_uses_default() {
        let def = "en-US_Multimedia";
        let mut p = ListenParams::default();
        assert_eq!(resolved_watsonx_model(&p, def), def);
        p.model = Some(String::new());
        assert_eq!(resolved_watsonx_model(&p, def), def);
        p.model = Some("   ".to_string());
        assert_eq!(resolved_watsonx_model(&p, def), def);
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
