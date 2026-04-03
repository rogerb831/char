use std::path::{Path, PathBuf};

use owhisper_interface::ListenParams;
use owhisper_interface::batch::{Alternatives, Channel, Response as BatchResponse, Results, Word};
use serde::Deserialize;
use serde_json::Value;

use super::{
    WatsonxAdapter, deserialize_vec_skip_null, recognize_http_path, resolved_watsonx_model,
    speaker_id_for_word_start, watsonx_next_gen_model,
};
use crate::adapter::{BatchFuture, BatchSttAdapter, ClientWithMiddleware};
use crate::error::Error;
use crate::providers::Provider;

/// `ChannelProfile::MixedCapture` — same as live Watsonx when words carry `speaker`.
const WATSONX_BATCH_DIARIZATION_CHANNEL: i32 = 2;

impl BatchSttAdapter for WatsonxAdapter {
    fn provider_name(&self) -> &'static str {
        "watsonx"
    }

    fn is_supported_languages(
        &self,
        languages: &[hypr_language::Language],
        _model: Option<&str>,
    ) -> bool {
        WatsonxAdapter::is_supported_languages_batch(languages)
    }

    fn transcribe_file<'a, P: AsRef<Path> + Send + 'a>(
        &'a self,
        client: &'a ClientWithMiddleware,
        api_base: &'a str,
        api_key: &'a str,
        params: &'a ListenParams,
        file_path: P,
    ) -> BatchFuture<'a> {
        let path = file_path.as_ref().to_path_buf();
        Box::pin(do_transcribe_file(client, api_base, api_key, params, path))
    }
}

#[derive(Deserialize)]
struct BatchRecognizeResponse {
    #[serde(default, deserialize_with = "deserialize_vec_skip_null")]
    results: Vec<BatchResultBlock>,
    #[serde(
        default,
        deserialize_with = "deserialize_vec_skip_null",
        alias = "speakerLabels"
    )]
    speaker_labels: Vec<super::WatsonSpeakerLabel>,
}

#[derive(Deserialize)]
struct BatchResultBlock {
    #[serde(default, deserialize_with = "deserialize_vec_skip_null")]
    alternatives: Vec<BatchAlternative>,
    #[serde(
        default,
        deserialize_with = "deserialize_vec_skip_null",
        alias = "speakerLabels"
    )]
    speaker_labels: Vec<super::WatsonSpeakerLabel>,
}

#[derive(Deserialize)]
struct BatchAlternative {
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_vec_skip_null")]
    timestamps: Vec<Value>,
}

async fn do_transcribe_file(
    client: &ClientWithMiddleware,
    api_base: &str,
    api_key: &str,
    params: &ListenParams,
    file_path: PathBuf,
) -> Result<BatchResponse, Error> {
    let file_bytes = tokio::fs::read(&file_path)
        .await
        .map_err(|e| Error::AudioProcessing(e.to_string()))?;

    let default = Provider::Watsonx.default_batch_model();
    let model = resolved_watsonx_model(params, default);

    let mut url = url::Url::parse(api_base.trim_end_matches('/'))
        .map_err(|e| Error::AudioProcessing(format!("invalid api_base: {e}")))?;
    url.set_path(&recognize_http_path(&url));
    {
        let mut pairs = url.query_pairs_mut();
        pairs.clear();
        pairs.append_pair("model", model);
        if watsonx_next_gen_model(params.model.as_deref()) {
            pairs.append_pair("speaker_labels", "true");
            pairs.append_pair("timestamps", "true");
        }
    }

    let auth = Provider::Watsonx
        .build_auth_header(api_key)
        .ok_or_else(|| Error::AudioProcessing("IBM watsonx: missing API key".to_string()))?;

    let mime_type = mime_type_from_extension(&file_path);

    let response = client
        .post(url.as_str())
        .header(auth.0, auth.1)
        .header("Content-Type", mime_type)
        .body(file_bytes)
        .send()
        .await
        .map_err(Error::HttpMiddleware)?;

    let status = response.status();
    let body = response.bytes().await.map_err(Error::Http)?;
    if !status.is_success() {
        return Err(Error::UnexpectedStatus {
            status,
            body: String::from_utf8_lossy(&body).to_string(),
        });
    }

    let parsed: BatchRecognizeResponse = serde_json::from_slice(&body).map_err(|e| {
        Error::AudioProcessing(format!(
            "watsonx batch json: {e}: {}",
            String::from_utf8_lossy(&body)
        ))
    })?;

    Ok(build_batch_response_from_ibm(parsed))
}

fn build_batch_response_from_ibm(parsed: BatchRecognizeResponse) -> BatchResponse {
    let mut segments: Vec<String> = Vec::new();
    let mut all_words: Vec<Word> = Vec::new();
    let mut conf_acc = 0.0f64;
    let mut conf_n = 0u32;
    let merged_labels = parsed.speaker_labels.clone();

    for block in parsed.results {
        let Some(alt) = block.alternatives.first() else {
            continue;
        };
        let t = alt.transcript.trim();
        if !t.is_empty() {
            segments.push(t.to_string());
        }
        let conf = alt.confidence.unwrap_or(0.0);
        if conf > 0.0 {
            conf_acc += conf;
            conf_n += 1;
        }
        let labels: &[super::WatsonSpeakerLabel] = if block.speaker_labels.is_empty() {
            merged_labels.as_slice()
        } else {
            block.speaker_labels.as_slice()
        };
        let mut stream_words = super::live::words_from_timestamps(&alt.timestamps, t, conf);
        for w in &mut stream_words {
            if let Some(s) = speaker_id_for_word_start(w.start, labels) {
                w.speaker = Some(s);
            }
        }
        all_words.extend(stream_words.into_iter().map(Into::into));
    }

    if all_words.iter().any(|w| w.speaker.is_some()) {
        for w in &mut all_words {
            w.channel = WATSONX_BATCH_DIARIZATION_CHANNEL;
        }
    }

    let transcript = segments.join(" ");
    let confidence = if conf_n > 0 {
        conf_acc / f64::from(conf_n)
    } else {
        0.0
    };

    BatchResponse {
        metadata: serde_json::json!({}),
        results: Results {
            channels: vec![Channel {
                alternatives: vec![Alternatives {
                    transcript,
                    words: all_words,
                    confidence,
                }],
            }],
        },
    }
}

fn mime_type_from_extension(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("wav") => "audio/wav",
        Some("flac") => "audio/flac",
        Some("ogg" | "opus") => "audio/ogg",
        Some("mp3") => "audio/mp3",
        Some("mpeg" | "mpga") => "audio/mpeg",
        Some("webm") => "audio/webm",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibm_multiblock_top_level_labels_set_speaker_and_mixed_channel() {
        let json = r#"{
            "result_index": 0,
            "results": [
                {
                    "final": true,
                    "alternatives": [{
                        "transcript": "hi",
                        "confidence": 0.9,
                        "timestamps": [["hi", 1.0, 1.2]]
                    }]
                },
                {
                    "final": true,
                    "alternatives": [{
                        "transcript": "there",
                        "confidence": 0.9,
                        "timestamps": [["there", 2.0, 2.3]]
                    }]
                }
            ],
            "speaker_labels": [
                {"from": 1.0, "to": 1.2, "speaker": 0},
                {"from": 2.0, "to": 2.3, "speaker": 1}
            ]
        }"#;
        let parsed: BatchRecognizeResponse = serde_json::from_str(json).unwrap();
        let out = build_batch_response_from_ibm(parsed);
        let words = &out.results.channels[0].alternatives[0].words;
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].speaker, Some(0));
        assert_eq!(words[1].speaker, Some(1));
        assert!(words.iter().all(|w| w.channel == 2));
    }

    #[test]
    fn without_speaker_labels_words_stay_channel_zero() {
        let json = r#"{
            "results": [{
                "final": true,
                "alternatives": [{
                    "transcript": "hi",
                    "confidence": 0.9,
                    "timestamps": [["hi", 1.0, 1.2]]
                }]
            }]
        }"#;
        let parsed: BatchRecognizeResponse = serde_json::from_str(json).unwrap();
        let out = build_batch_response_from_ibm(parsed);
        let w = &out.results.channels[0].alternatives[0].words[0];
        assert_eq!(w.speaker, None);
        assert_eq!(w.channel, 0);
    }
}
