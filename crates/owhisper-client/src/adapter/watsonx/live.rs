use std::sync::{Mutex, OnceLock};

use hypr_ws_client::client::Message;
use owhisper_interface::ListenParams;
use owhisper_interface::stream::{Alternatives, Channel, Metadata, StreamResponse, Word};
use serde::Deserialize;
use serde_json::Value;

use super::{
    WatsonSpeakerLabel, WatsonxAdapter, assign_speakers_by_label_index, deserialize_vec_skip_null,
    resolved_watsonx_model, speaker_id_for_word_start, watsonx_next_gen_model,
};
use crate::adapter::RealtimeSttAdapter;
use crate::adapter::parsing::{WordBuilder, calculate_time_span};
use crate::providers::Provider;

fn watsonx_pending_speaker_labels() -> &'static Mutex<Vec<WatsonSpeakerLabel>> {
    static CELL: OnceLock<Mutex<Vec<WatsonSpeakerLabel>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(Vec::new()))
}

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
        let model = resolved_watsonx_model(params, default);

        {
            let mut query_pairs = url.query_pairs_mut();
            query_pairs.append_pair("model", model);
            if watsonx_next_gen_model(params.model.as_deref()) {
                query_pairs.append_pair("speaker_labels", "true");
            }
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
            speaker_labels: watsonx_next_gen_model(params.model.as_deref()).then_some(true),
            inactivity_timeout: IBM_WS_INACTIVITY_TIMEOUT_SECS,
            low_latency: watsonx_next_gen_model(params.model.as_deref()).then_some(true),
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
            if !payload.speaker_labels.is_empty() {
                if let Ok(mut g) = watsonx_pending_speaker_labels().lock() {
                    *g = payload.speaker_labels.clone();
                }
            }
            return vec![];
        }

        let mut merged_labels = payload.speaker_labels.clone();
        if merged_labels.is_empty() {
            if let Ok(mut pl) = watsonx_pending_speaker_labels().lock() {
                if !pl.is_empty() {
                    merged_labels = std::mem::take(&mut *pl);
                }
            }
        } else if let Ok(mut pl) = watsonx_pending_speaker_labels().lock() {
            pl.clear();
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

            let labels_for_block: &[WatsonSpeakerLabel] = if block.speaker_labels.is_empty() {
                merged_labels.as_slice()
            } else {
                block.speaker_labels.as_slice()
            };

            let conf = alt.confidence.unwrap_or(0.0);
            let mut words = words_from_timestamps(&alt.timestamps, transcript, conf);
            if should_use_synthetic_watsonx_timing(&words) && !transcript.is_empty() {
                words = synthetic_token_words_for_watsonx(
                    result_index,
                    transcript,
                    conf,
                    labels_for_block,
                );
            } else {
                assign_speakers_to_words(&mut words, labels_for_block);
                if !labels_for_block.is_empty() && words.iter().all(|w| w.speaker.is_none()) {
                    assign_speakers_by_label_index(&mut words, labels_for_block);
                }
            }
            let channel_index_vec =
                watsonx_output_channel_index(block.channel_index.as_ref(), &words);
            // #region agent log
            {
                let sp_vals: Vec<i32> = words.iter().filter_map(|w| w.speaker).collect();
                let mut uniq = sp_vals.clone();
                uniq.sort_unstable();
                uniq.dedup();
                let with_sp = words.iter().filter(|w| w.speaker.is_some()).count();
                owhisper_interface::agent_debug::append_ndjson_line(&serde_json::json!({
                    "hypothesisId": "H1",
                    "location": "watsonx/live.rs:parse_response",
                    "message": "watsonx_emit_block",
                    "data": {
                        "n_words": words.len(),
                        "words_with_speaker": with_sp,
                        "distinct_word_speakers": uniq,
                        "channel_index": channel_index_vec,
                        "n_labels": labels_for_block.len(),
                        "label_speakers": labels_for_block.iter().map(|l| l.speaker).collect::<Vec<i32>>(),
                        "is_final": block.final_,
                        "raw_has_speaker_labels_key": raw.contains("\"speaker_labels\"")
                            || raw.contains("\"speakerLabels\""),
                    },
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0),
                }));
            }
            // #endregion
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

            out.push(StreamResponse::TranscriptResponse {
                is_final,
                speech_final: is_final,
                from_finalize: false,
                start,
                duration,
                channel,
                metadata: Metadata::default(),
                channel_index: channel_index_vec,
            });
        }

        out
    }
}

fn assign_speakers_to_words(words: &mut [Word], labels: &[WatsonSpeakerLabel]) {
    if labels.is_empty() {
        return;
    }
    for w in words.iter_mut() {
        if let Some(s) = speaker_id_for_word_start(w.start, labels) {
            w.speaker = Some(s);
        }
    }
}

fn synthetic_token_words_for_watsonx(
    result_index: u32,
    transcript: &str,
    conf: f64,
    labels: &[WatsonSpeakerLabel],
) -> Vec<Word> {
    let tokens: Vec<&str> = transcript.split_whitespace().collect();
    let (bucket_start, bucket_end) = synthetic_span_for_watsonx_result(result_index);
    if tokens.is_empty() {
        let mut w = vec![
            WordBuilder::new(transcript)
                .start(bucket_start)
                .end(bucket_end)
                .confidence(conf)
                .build(),
        ];
        assign_speakers_to_words(&mut w, labels);
        if !labels.is_empty() && w.iter().all(|word| word.speaker.is_none()) {
            assign_speakers_by_label_index(&mut w, labels);
        }
        return w;
    }
    let n = tokens.len();
    let span = bucket_end - bucket_start;
    let step = span / n as f64;
    let mut words: Vec<Word> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let s = bucket_start + step * i as f64;
            let e = bucket_start + step * (i + 1) as f64;
            WordBuilder::new(*t)
                .start(s)
                .end(e)
                .confidence(conf)
                .build()
        })
        .collect();
    assign_speakers_by_label_index(&mut words, labels);
    words
}

/// `MixedCapture` (2) when words carry IBM `speaker` so the UI can show "Speaker N" instead of
/// treating all `DirectMic` text as "You". Otherwise map IBM `channel_index` 0/1 to mic vs remote.
fn watsonx_output_channel_index(
    block_channel_index: Option<&Vec<i32>>,
    words: &[Word],
) -> Vec<i32> {
    if words.iter().any(|w| w.speaker.is_some()) {
        return vec![2, 2];
    }
    if let Some(ci) = block_channel_index.and_then(|v| v.first().copied()) {
        if ci == 0 || ci == 1 {
            return vec![ci, 2];
        }
    }
    vec![2, 2]
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

#[derive(serde::Serialize)]
struct StartMessage<'a> {
    action: &'a str,
    #[serde(rename = "content-type")]
    content_type: &'a str,
    interim_results: bool,
    word_confidence: bool,
    timestamps: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_labels: Option<bool>,
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
    #[serde(default, deserialize_with = "deserialize_vec_skip_null")]
    results: Vec<ResultBlock>,
    #[serde(
        default,
        deserialize_with = "deserialize_vec_skip_null",
        alias = "speakerLabels"
    )]
    speaker_labels: Vec<WatsonSpeakerLabel>,
}

#[derive(Deserialize)]
struct ResultBlock {
    #[serde(default, rename = "final")]
    final_: bool,
    #[serde(default)]
    channel_index: Option<Vec<i32>>,
    #[serde(
        default,
        deserialize_with = "deserialize_vec_skip_null",
        alias = "speakerLabels"
    )]
    speaker_labels: Vec<WatsonSpeakerLabel>,
    #[serde(default, deserialize_with = "deserialize_vec_skip_null")]
    alternatives: Vec<WatsonAlternative>,
}

#[derive(Deserialize)]
struct WatsonAlternative {
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_vec_skip_null")]
    timestamps: Vec<Value>,
}

fn json_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_u64().map(|u| u as f64))
        .or_else(|| v.as_i64().map(|i| i as f64))
}

fn json_i32(v: &Value) -> Option<i32> {
    v.as_i64()
        .and_then(|i| i32::try_from(i).ok())
        .or_else(|| v.as_u64().and_then(|u| i32::try_from(u).ok()))
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
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

    if timestamps
        .first()
        .is_some_and(|v| v.is_array() || v.is_object())
    {
        let mut words = Vec::new();
        for entry in timestamps {
            if let Some(obj) = entry.as_object() {
                let w = obj
                    .get("word")
                    .or_else(|| obj.get("token"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let start = obj
                    .get("start")
                    .or_else(|| obj.get("from"))
                    .and_then(json_f64);
                let end = obj.get("end").or_else(|| obj.get("to")).and_then(json_f64);
                let sp = obj
                    .get("speaker")
                    .or_else(|| obj.get("speaker_id"))
                    .and_then(json_i32);
                if let (Some(word), Some(s), Some(e)) = (w, start, end) {
                    words.push(
                        WordBuilder::new(word)
                            .start(s)
                            .end(e)
                            .confidence(confidence)
                            .speaker(sp)
                            .build(),
                    );
                }
                continue;
            }
            let Some(arr) = entry.as_array() else {
                continue;
            };
            if arr.len() < 3 {
                continue;
            }
            let w = arr[0].as_str().map(str::to_string);
            let start = arr[1]
                .as_f64()
                .or_else(|| arr[1].as_u64().map(|u| u as f64));
            let end = arr[2]
                .as_f64()
                .or_else(|| arr[2].as_u64().map(|u| u as f64));
            let sp = arr.get(3).and_then(json_i32);
            if let (Some(word), Some(s), Some(e)) = (w, start, end) {
                words.push(
                    WordBuilder::new(word)
                        .start(s)
                        .end(e)
                        .confidence(confidence)
                        .speaker(sp)
                        .build(),
                );
            }
        }
        if !words.is_empty() {
            return words;
        }
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
            let sp = (i + 3 < timestamps.len())
                .then(|| json_i32(&timestamps[i + 3]))
                .flatten();
            let step = if sp.is_some() { 4 } else { 3 };
            words.push(
                WordBuilder::new(word)
                    .start(s)
                    .end(e)
                    .confidence(confidence)
                    .speaker(sp)
                    .build(),
            );
            i += step;
        } else {
            i += 1;
        }
    }

    if words.is_empty() && !transcript.is_empty() {
        words.push(WordBuilder::new(transcript).confidence(confidence).build());
    }

    words
}
