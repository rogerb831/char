use hypr_ws_client::client::Message;
use owhisper_interface::ListenParams;
use owhisper_interface::stream::{Alternatives, Channel, Metadata, StreamResponse, Word};
use serde::Deserialize;
use serde_json::Value;

use super::WatsonxAdapter;
use crate::adapter::RealtimeSttAdapter;
use crate::adapter::parsing::{WordBuilder, calculate_time_span};
use crate::providers::{Provider, is_meta_model};

impl RealtimeSttAdapter for WatsonxAdapter {
    fn provider_name(&self) -> &'static str {
        "watsonx"
    }

    fn is_supported_languages(
        &self,
        languages: &[hypr_language::Language],
        _model: Option<&str>,
    ) -> bool {
        WatsonxAdapter::is_supported_languages_live(languages)
    }

    fn supports_native_multichannel(&self) -> bool {
        true
    }

    fn build_ws_url(&self, api_base: &str, params: &ListenParams, _channels: u8) -> url::Url {
        let (mut url, existing_params) = WatsonxAdapter::build_ws_url_from_base(api_base);

        let default = Provider::Watsonx.default_live_model();
        let model = match params.model.as_deref() {
            Some(m) if is_meta_model(m) => default,
            Some(m) => m,
            None => default,
        };

        {
            let mut query_pairs = url.query_pairs_mut();
            query_pairs.append_pair("model", model);
            for (key, value) in &existing_params {
                query_pairs.append_pair(key, value);
            }
        }

        url
    }

    fn build_auth_header(&self, api_key: Option<&str>) -> Option<(&'static str, String)> {
        api_key.and_then(|k| Provider::Watsonx.build_auth_header(k))
    }

    fn keep_alive_message(&self) -> Option<Message> {
        None
    }

    fn finalize_message(&self) -> Message {
        Message::Text(r#"{"action":"stop"}"#.into())
    }

    fn initial_message(
        &self,
        _api_key: Option<&str>,
        params: &ListenParams,
        channels: u8,
    ) -> Option<Message> {
        let sample_rate = if params.sample_rate == 0 {
            16000
        } else {
            params.sample_rate
        };
        let content_type = format!("audio/l16;rate={sample_rate};channels={channels}");
        let start = StartMessage {
            action: "start",
            content_type: content_type.as_str(),
            interim_results: true,
            word_confidence: true,
            timestamps: true,
            inactivity_timeout: IBM_WS_INACTIVITY_TIMEOUT_SECS,
            low_latency: watsonx_low_latency_for_model(params.model.as_deref()),
        };
        let json = serde_json::to_string(&start).ok()?;
        Some(Message::Text(json.into()))
    }

    fn parse_response(&self, raw: &str) -> Vec<StreamResponse> {
        if let Some(pe) = Provider::Watsonx.detect_error(raw.as_bytes()) {
            return vec![StreamResponse::ErrorResponse {
                error_code: Some(pe.http_code as i32),
                error_message: pe.message,
                provider: "watsonx".to_string(),
            }];
        }

        if let Ok(state) = serde_json::from_str::<StateMessage>(raw) {
            if state.state.is_some() {
                tracing::debug!(hyprnote.payload = %raw, "watsonx_state_message");
                return vec![];
            }
        }

        if let Ok(err) = serde_json::from_str::<ErrorEnvelope>(raw) {
            if err.error.is_some() {
                let msg = err
                    .error
                    .as_ref()
                    .and_then(|e| e.message.clone())
                    .or_else(|| err.error.as_ref().and_then(|e| e.error.clone()))
                    .unwrap_or_else(|| "watsonx_error".to_string());
                return vec![StreamResponse::ErrorResponse {
                    error_code: err.error.as_ref().and_then(|e| e.code),
                    error_message: msg,
                    provider: "watsonx".to_string(),
                }];
            }
        }

        let payload: ResultsPayload = match serde_json::from_str(raw) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    error = ?e,
                    hyprnote.payload.size_bytes = raw.len() as u64,
                    "watsonx_json_parse_failed"
                );
                return vec![];
            }
        };

        if payload.results.is_empty() {
            return vec![];
        }
        let result_index = payload.result_index.unwrap_or(0);
        let results = payload.results;

        let mut out = Vec::new();
        for block in results {
            let Some(alt) = block.alternatives.first() else {
                continue;
            };
            let transcript = alt.transcript.trim();
            if transcript.is_empty() && alt.timestamps.is_empty() {
                continue;
            }

            let conf = alt.confidence.unwrap_or(0.0);
            let mut words = words_from_timestamps(&alt.timestamps, transcript, conf);
            if should_use_synthetic_watsonx_timing(&words) && !transcript.is_empty() {
                let (start_sec, end_sec) = synthetic_span_for_watsonx_result(result_index);
                words = vec![
                    WordBuilder::new(transcript)
                        .start(start_sec)
                        .end(end_sec)
                        .confidence(conf)
                        .build(),
                ];
            }
            let (start, duration) = calculate_time_span(&words);
            let is_final = block.final_;
            let channel = Channel {
                alternatives: vec![Alternatives {
                    transcript: transcript.to_string(),
                    words,
                    confidence: conf,
                    languages: vec![],
                }],
            };

            let channel_index = block
                .channel_index
                .as_ref()
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0);

            out.push(StreamResponse::TranscriptResponse {
                is_final,
                speech_final: is_final,
                from_finalize: false,
                start,
                duration,
                channel,
                metadata: Metadata::default(),
                channel_index: vec![channel_index],
            });
        }

        out
    }
}

/// IBM interim hypotheses often omit timestamps; we previously emitted words at (0,0). That breaks
/// `TranscriptProcessor::apply_partial` (0 ms range interacts badly with `before`/`after` filters and
/// duplicates text). Bucketing by `result_index` matches IBM semantics: same index replaces prior text.
fn synthetic_span_for_watsonx_result(result_index: u32) -> (f64, f64) {
    const BUCKET_SEC: f64 = 600.0;
    let start = f64::from(result_index) * BUCKET_SEC + 0.001;
    let end = start + BUCKET_SEC - 0.002;
    (start, end)
}

fn should_use_synthetic_watsonx_timing(words: &[Word]) -> bool {
    words.is_empty() || words.iter().all(|w| w.start == 0.0 && w.end == 0.0)
}

/// IBM closes streaming sessions when only silence is detected for the inactivity window (default
/// ~30s). `-1` is documented as unlimited but is not reliably honored for next-gen models; dual WS
/// (mic + system audio) often leaves the speaker leg silent and hits that default.
const IBM_WS_INACTIVITY_TIMEOUT_SECS: i32 = 3600;

fn watsonx_low_latency_for_model(model: Option<&str>) -> Option<bool> {
    let use_ll = match model {
        None => true,
        Some(m) => m.contains("Multimedia") || m.contains("Telephony"),
    };
    use_ll.then_some(true)
}

#[derive(serde::Serialize)]
struct StartMessage<'a> {
    action: &'a str,
    #[serde(rename = "content-type")]
    content_type: &'a str,
    interim_results: bool,
    word_confidence: bool,
    timestamps: bool,
    inactivity_timeout: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    low_latency: Option<bool>,
}

#[derive(Deserialize)]
struct StateMessage {
    state: Option<String>,
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct ErrorBody {
    code: Option<i32>,
    message: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct ResultsPayload {
    #[serde(default)]
    result_index: Option<u32>,
    #[serde(default)]
    results: Vec<ResultBlock>,
}

#[derive(Deserialize)]
struct ResultBlock {
    #[serde(default, rename = "final")]
    final_: bool,
    #[serde(default)]
    channel_index: Option<Vec<i32>>,
    #[serde(default)]
    alternatives: Vec<WatsonAlternative>,
}

#[derive(Deserialize)]
struct WatsonAlternative {
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    timestamps: Vec<Value>,
}

pub(super) fn words_from_timestamps(
    timestamps: &[Value],
    transcript: &str,
    confidence: f64,
) -> Vec<owhisper_interface::stream::Word> {
    if timestamps.is_empty() {
        if transcript.is_empty() {
            return vec![];
        }
        return vec![WordBuilder::new(transcript).confidence(confidence).build()];
    }

    let mut words = Vec::new();
    let mut i = 0;
    while i + 2 < timestamps.len() {
        let w = timestamps[i].as_str().map(str::to_string);
        let start = timestamps[i + 1]
            .as_f64()
            .or_else(|| timestamps[i + 1].as_u64().map(|u| u as f64));
        let end = timestamps[i + 2]
            .as_f64()
            .or_else(|| timestamps[i + 2].as_u64().map(|u| u as f64));
        if let (Some(word), Some(s), Some(e)) = (w, start, end) {
            words.push(
                WordBuilder::new(word)
                    .start(s)
                    .end(e)
                    .confidence(confidence)
                    .build(),
            );
            i += 3;
        } else {
            i += 1;
        }
    }

    if words.is_empty() && !transcript.is_empty() {
        words.push(WordBuilder::new(transcript).confidence(confidence).build());
    }

    words
}
