use std::path::{Path, PathBuf};

use owhisper_interface::ListenParams;
use owhisper_interface::batch::{Alternatives, Channel, Response as BatchResponse, Results, Word};
use serde::Deserialize;
use serde_json::Value;

use super::{WatsonxAdapter, recognize_http_path};
use crate::adapter::{BatchFuture, BatchSttAdapter, ClientWithMiddleware};
use crate::error::Error;
use crate::providers::{Provider, is_meta_model};

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
    #[serde(default)]
    results: Vec<BatchResultBlock>,
}

#[derive(Deserialize)]
struct BatchResultBlock {
    #[serde(default)]
    alternatives: Vec<BatchAlternative>,
}

#[derive(Deserialize)]
struct BatchAlternative {
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
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
    let model = match params.model.as_deref() {
        Some(m) if is_meta_model(m) => default,
        Some(m) => m,
        None => default,
    };

    let mut url = url::Url::parse(api_base.trim_end_matches('/'))
        .map_err(|e| Error::AudioProcessing(format!("invalid api_base: {e}")))?;
    url.set_path(&recognize_http_path(&url));
    {
        let mut pairs = url.query_pairs_mut();
        pairs.clear();
        pairs.append_pair("model", model);
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

    let mut segments: Vec<String> = Vec::new();
    let mut all_words: Vec<Word> = Vec::new();
    let mut conf_acc = 0.0f64;
    let mut conf_n = 0u32;

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
        all_words.extend(
            super::live::words_from_timestamps(&alt.timestamps, t, conf)
                .into_iter()
                .map(Into::into),
        );
    }

    let transcript = segments.join(" ");
    let confidence = if conf_n > 0 {
        conf_acc / f64::from(conf_n)
    } else {
        0.0
    };

    Ok(BatchResponse {
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
    })
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
