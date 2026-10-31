mod atlas_cloud;
mod caiyun;
mod claude;
mod cursor;
mod deepl;
mod deepseek;
mod gemini;
mod google_cloud;
mod grok;
mod lm_studio;
mod minimax;
mod openai;
mod openai_compatible;
mod openrouter;

use anyhow::Context;
use futures::{FutureExt, future::BoxFuture, future::join_all};
use koharu_usage::{RequestSettings, UsageOutcome, wire};
use reqwest::{Client, RequestBuilder, StatusCode, header::HeaderMap};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::time::Duration;

pub use atlas_cloud::AtlasCloudConfig;
pub use caiyun::CaiyunConfig;
pub use claude::ClaudeConfig;
pub use cursor::CursorConfig;
pub use deepl::DeepLConfig;
pub use deepseek::DeepSeekConfig;
pub use gemini::GeminiConfig;
pub use google_cloud::GoogleCloudConfig;
pub use grok::GrokConfig;
pub use lm_studio::LmStudioConfig;
pub use minimax::MiniMaxConfig;
pub use openai::OpenAiConfig;
pub use openai_compatible::OpenAiCompatibleConfig;
pub use openrouter::OpenRouterConfig;

use crate::{
    Error, GenerationConfig, Model, ModelSelection, Provider, ProvidersConfig, Result,
    TranslationRequest,
    usage::{InFlight, UsageTrace},
};

pub(crate) async fn translate(
    client: &Client,
    providers: &ProvidersConfig,
    selection: &ModelSelection,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let model = || {
        selection
            .model
            .as_deref()
            .with_context(|| format!("{} requires a selected model", selection.provider))
    };
    let result = match selection.provider {
        Provider::AtlasCloud => {
            atlas_cloud::translate(
                client,
                &providers.atlas_cloud,
                model()?,
                generation,
                request,
            )
            .await
        }
        Provider::OpenAi => {
            openai::translate(client, &providers.openai, model()?, generation, request).await
        }
        Provider::Gemini => {
            gemini::translate(client, &providers.gemini, model()?, generation, request).await
        }
        Provider::Claude => {
            claude::translate(client, &providers.claude, model()?, generation, request).await
        }
        Provider::Grok => {
            grok::translate(client, &providers.grok, model()?, generation, request).await
        }
        Provider::Cursor => {
            cursor::translate(client, &providers.cursor, model()?, generation, request).await
        }
        Provider::MiniMax => {
            minimax::translate(client, &providers.minimax, model()?, generation, request).await
        }
        Provider::DeepSeek => {
            deepseek::translate(client, &providers.deepseek, model()?, generation, request).await
        }
        Provider::OpenAiCompatible => {
            openai_compatible::compatible(
                client,
                &providers.openai_compatible,
                model()?,
                generation,
                request,
            )
            .await
        }
        Provider::OpenRouter => {
            openrouter::translate(client, &providers.openrouter, model()?, generation, request)
                .await
        }
        Provider::LmStudio => {
            lm_studio::translate(client, &providers.lm_studio, model()?, generation, request).await
        }
        Provider::DeepL => deepl::translate(client, &providers.deepl, request).await,
        Provider::GoogleCloudTranslation => {
            google_cloud::translate(client, &providers.google_cloud_translation, request).await
        }
        Provider::Caiyun => caiyun::translate(client, &providers.caiyun, request).await,
        Provider::Local => unreachable!("local translation has its own backend"),
    };
    if result.is_err()
        && let Some(usage) = request.usage.as_ref()
    {
        // The provider answered, but nothing translatable came out of it.
        usage.reject(UsageOutcome::InvalidOutput);
    }
    result
}

pub(crate) async fn models(client: &Client, providers: &ProvidersConfig) -> Vec<Model> {
    let mut models = Vec::new();
    let pending: Vec<BoxFuture<'_, Result<Vec<Model>>>> = vec![
        atlas_cloud::models(client).boxed(),
        openai::models(client).boxed(),
        gemini::models(client).boxed(),
        claude::models(client).boxed(),
        grok::models(client).boxed(),
        cursor::models(client).boxed(),
        minimax::models(client).boxed(),
        deepseek::models(client).boxed(),
        openai_compatible::models(client, &providers.openai_compatible).boxed(),
        openrouter::models(client).boxed(),
        lm_studio::models(client, &providers.lm_studio).boxed(),
        deepl::models().boxed(),
        google_cloud::models().boxed(),
        caiyun::models().boxed(),
    ];
    for result in join_all(pending).await {
        append_models(&mut models, result);
    }
    models.sort_by(|left, right| {
        left.provider
            .to_string()
            .cmp(&right.provider.to_string())
            .then_with(|| left.model.cmp(&right.model))
    });
    models.dedup_by(|left, right| left.provider == right.provider && left.model == right.model);
    models
}

fn append_models(models: &mut Vec<Model>, result: Result<Vec<Model>>) {
    match result {
        Ok(mut provider_models) => models.append(&mut provider_models),
        Err(error) => tracing::warn!(%error, "failed to list translation models"),
    }
}

/// How an inference request reaches the usage journal: the trace of the
/// translation attempt it belongs to, the settings that went on the wire, and
/// how to read the provider's usage out of the response body.
///
/// A request without a trace (built outside `Translator::translate`) is sent
/// like any other and is not journaled.
pub(super) struct Inference<'a> {
    trace: Option<&'a UsageTrace>,
    settings: RequestSettings,
    report: fn(&Value) -> wire::Reported,
}

impl<'a> Inference<'a> {
    pub(super) fn new(
        request: &'a TranslationRequest,
        settings: RequestSettings,
        report: fn(&Value) -> wire::Reported,
    ) -> Self {
        Self {
            trace: request.usage.as_ref(),
            settings,
            report,
        }
    }

    /// For translation services that take no generation settings and report
    /// characters rather than tokens: the request is journaled without usage.
    pub(super) fn service(request: &'a TranslationRequest) -> Self {
        Self::new(request, RequestSettings::default(), |_| {
            wire::Reported::default()
        })
    }
}

/// Sends a request that is not an inference, such as model discovery. These
/// are never journaled.
pub(super) async fn send_json<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
) -> Result<T> {
    let response = request
        .send()
        .await
        .with_context(|| format!("{provider} request failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} response"))?;

    if !status.is_success() {
        return Err(status_error(provider, status, &text));
    }
    decode(provider, &text)
}

fn status_error(provider: &'static str, status: StatusCode, text: &str) -> Error {
    let lower = text.to_ascii_lowercase();
    if status == StatusCode::TOO_MANY_REQUESTS
        || lower.contains("insufficient_quota")
        || lower.contains("resource_exhausted")
        || lower.contains("rate limit exceeded")
        || lower.contains("credit balance is too low")
    {
        return Error::QuotaExceeded { provider };
    }
    if lower.contains("invalid_prompt")
        || lower.contains("usage policy")
        || lower.contains("safety policy")
    {
        return Error::PolicyRejected { provider };
    }
    Error::Api {
        provider,
        status: status.as_u16(),
        message: text.chars().take(4096).collect(),
    }
}

fn decode<T: DeserializeOwned>(provider: &'static str, text: &str) -> Result<T> {
    serde_json::from_str(text)
        .with_context(|| format!("failed to decode {provider} response"))
        .map_err(Into::into)
}

/// Sends one inference request and journals it.
pub(super) async fn send_inference<T: DeserializeOwned>(
    provider: &'static str,
    inference: &Inference<'_>,
    request: RequestBuilder,
) -> Result<T> {
    send_attempt(provider, inference, request, 0).await
}

/// Like [`send_inference`], resending after transient failures. Every resend is
/// a separate request with a record of its own.
pub(super) async fn send_inference_with_transient_retry<T: DeserializeOwned>(
    provider: &'static str,
    inference: &Inference<'_>,
    request: RequestBuilder,
) -> Result<T> {
    const ATTEMPTS: usize = 3;
    let template = request.try_clone();
    let mut first = Some(request);
    for attempt in 0..ATTEMPTS {
        let current = if let Some(request) = first.take() {
            request
        } else if let Some(request) = template.as_ref().and_then(RequestBuilder::try_clone) {
            request
        } else {
            break;
        };
        match send_attempt(provider, inference, current, attempt as u32).await {
            Ok(response) => return Ok(response),
            Err(error)
                if attempt + 1 < ATTEMPTS
                    && template.is_some()
                    && is_transient_provider_error(&error) =>
            {
                let delay = Duration::from_millis(250 * (attempt as u64 + 1));
                tracing::warn!(
                    provider,
                    attempt = attempt + 1,
                    ?delay,
                    %error,
                    "retrying transient translation provider failure"
                );
                tokio::time::sleep(delay).await;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("the first translation provider request is always available")
}

async fn send_attempt<T: DeserializeOwned>(
    provider: &'static str,
    inference: &Inference<'_>,
    request: RequestBuilder,
    transport_retry: u32,
) -> Result<T> {
    let Some(trace) = inference.trace else {
        return send_json(provider, request).await;
    };
    // Dropping the future before the request resolves journals it as cancelled.
    let mut flight = InFlight::start(trace, inference.settings.clone(), transport_retry);
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            flight.finish(if error.is_timeout() {
                UsageOutcome::Timeout
            } else {
                UsageOutcome::NetworkError
            });
            return Err(Error::Other(
                anyhow::Error::new(error).context(format!("{provider} request failed")),
            ));
        }
    };
    let status = response.status();
    flight.record().http_status = Some(status.as_u16());
    flight.record().provider_request_id = request_id(response.headers());
    let text = match response.text().await {
        Ok(text) => text,
        Err(error) => {
            flight.finish(if error.is_timeout() {
                UsageOutcome::Timeout
            } else {
                UsageOutcome::Incomplete
            });
            return Err(Error::Other(
                anyhow::Error::new(error).context(format!("failed to read {provider} response")),
            ));
        }
    };

    // Usage is read from the generic body so that a response that is paid for
    // but does not match the expected shape still counts.
    let body = serde_json::from_str::<Value>(&text).ok();
    if !status.is_success() {
        let error = status_error(provider, status, &text);
        flight.record().error_code = body.as_ref().and_then(wire::error_code);
        flight.finish(failure_outcome(&error));
        return Err(error);
    }
    if let Some(body) = &body {
        flight.record().report((inference.report)(body));
    }
    match decode(provider, &text) {
        Ok(response) => {
            flight.finish(UsageOutcome::Succeeded);
            Ok(response)
        }
        Err(error) => {
            flight.finish(UsageOutcome::DecodeError);
            Err(error)
        }
    }
}

/// The provider's own identifier for the request, from the response headers
/// OpenAI-style (`x-request-id`) and Anthropic (`request-id`) APIs use.
fn request_id(headers: &HeaderMap) -> Option<String> {
    ["x-request-id", "request-id"]
        .into_iter()
        .find_map(|name| headers.get(name)?.to_str().ok().map(str::to_owned))
}

/// How a request that failed with `error` ended. Transports that cannot
/// observe the exchange themselves classify the error they were handed.
pub(super) fn failure_outcome(error: &Error) -> UsageOutcome {
    match error {
        Error::QuotaExceeded { .. } => UsageOutcome::QuotaExceeded,
        Error::PolicyRejected { .. } => UsageOutcome::PolicyRejected,
        Error::Api { .. } => UsageOutcome::HttpError,
        Error::Other(error) => {
            if let Some(transport) = error
                .chain()
                .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
            {
                if transport.is_timeout() {
                    UsageOutcome::Timeout
                } else {
                    UsageOutcome::NetworkError
                }
            } else if error
                .chain()
                .any(|cause| cause.downcast_ref::<serde_json::Error>().is_some())
            {
                UsageOutcome::DecodeError
            } else {
                UsageOutcome::ProviderError
            }
        }
        _ => UsageOutcome::ProviderError,
    }
}

fn is_transient_provider_error(error: &Error) -> bool {
    match error {
        Error::Api { status, .. } => matches!(status, 408 | 425 | 502 | 503 | 504),
        Error::Other(error) => error.chain().any(|cause| {
            cause.downcast_ref::<reqwest::Error>().is_some_and(|error| {
                error.is_timeout() || error.is_connect() || error.is_body() || error.is_decode()
            })
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::OnceLock,
        thread,
    };

    use koharu_usage::{UsageJournal, UsageQuery, UsageRecord};
    use serde_json::Value;

    use super::*;
    use crate::Language;

    #[tokio::test]
    async fn retries_incomplete_provider_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 4096];
                let _ = stream.read(&mut request).unwrap();
                if attempt == 0 {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64\r\nConnection: close\r\n\r\n{\"value\":",
                        )
                        .unwrap();
                } else {
                    let body = br#"{"value":"complete"}"#;
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .unwrap();
                    stream.write_all(body).unwrap();
                }
            }
        });

        let inference = Inference {
            trace: None,
            settings: RequestSettings::default(),
            report: wire::openai_chat,
        };
        let response: Value = send_inference_with_transient_retry(
            "openai-compatible",
            &inference,
            Client::new().get(format!("http://{address}/chat/completions")),
        )
        .await
        .unwrap();

        assert_eq!(response["value"], "complete");
        server.join().unwrap();
    }

    #[tokio::test]
    async fn maps_policy_rejection_without_echoing_provider_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            let body =
                br#"{"error":{"code":"invalid_prompt","message":"sensitive provider details"}}"#;
            write!(
                stream,
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });

        let error = send_json::<Value>(
            "openai-compatible",
            Client::new().get(format!("http://{address}/chat/completions")),
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            Error::PolicyRejected {
                provider: "openai-compatible"
            }
        ));
        assert!(!error.to_string().contains("sensitive provider details"));
        server.join().unwrap();
    }

    /// One journal for the whole test binary, away from the user's real one.
    fn journal() -> &'static UsageJournal {
        static DIRECTORY: OnceLock<tempfile::TempDir> = OnceLock::new();
        let directory = DIRECTORY.get_or_init(|| tempfile::tempdir().unwrap());
        UsageJournal::configure(directory.path().join("usage.sqlite3")).unwrap();
        UsageJournal::shared().unwrap()
    }

    fn trace(model: &str) -> UsageTrace {
        journal();
        let selection = ModelSelection {
            provider: Provider::OpenAiCompatible,
            model: Some(model.to_owned()),
            ..ModelSelection::default()
        };
        let request = TranslationRequest::new(["a", "b"], Language::Japanese, Language::English);
        UsageTrace::new(&selection, &request, 1)
    }

    /// Records written for `model`, oldest first.
    fn records(model: &str) -> Vec<UsageRecord> {
        let mut entries = journal()
            .query(&UsageQuery {
                model: Some(model.to_owned()),
                ..UsageQuery::default()
            })
            .unwrap()
            .entries;
        entries.reverse();
        entries
    }

    fn http(status: &str, headers: &str, body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    /// Answers one connection per canned response, in order.
    fn serve(responses: Vec<Vec<u8>>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/chat/completions", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 4096];
                let _ = stream.read(&mut request).unwrap();
                stream.write_all(&response).unwrap();
            }
        });
        (url, server)
    }

    fn chat_inference(trace: &UsageTrace) -> Inference<'_> {
        Inference {
            trace: Some(trace),
            settings: RequestSettings {
                reasoning: Some(true),
                reasoning_effort: Some("high".to_owned()),
                temperature: Some(0.5),
                max_tokens: Some(2048),
                ..RequestSettings::default()
            },
            report: wire::openai_chat,
        }
    }

    #[tokio::test]
    async fn every_http_attempt_is_journaled_including_transient_retries() {
        let (url, server) = serve(vec![
            http(
                "503 Service Unavailable",
                "x-request-id: req-503\r\n",
                r#"{"error":{"type":"overloaded_error","message":"do not store me"}}"#,
            ),
            http(
                "200 OK",
                "x-request-id: req-ok\r\n",
                r#"{"id":"chatcmpl-1","model":"served-model","usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":100},"completion_tokens_details":{"reasoning_tokens":10}}}"#,
            ),
        ]);
        let trace = trace("journal-retries");

        let _: Value = send_inference_with_transient_retry(
            "openai-compatible",
            &chat_inference(&trace),
            Client::new().get(url),
        )
        .await
        .unwrap();
        server.join().unwrap();

        let records = records("journal-retries");
        assert_eq!(records.len(), 2);
        let (failed, succeeded) = (&records[0], &records[1]);

        assert_eq!(failed.outcome, UsageOutcome::HttpError);
        assert_eq!(failed.http_status, Some(503));
        assert_eq!(failed.transport_retry, 0);
        assert_eq!(failed.error_code.as_deref(), Some("overloaded_error"));
        assert_eq!(failed.provider_request_id.as_deref(), Some("req-503"));
        assert!(failed.tokens.is_unreported());
        assert_eq!(failed.cost, None);

        assert_eq!(succeeded.outcome, UsageOutcome::Succeeded);
        assert_eq!(succeeded.http_status, Some(200));
        assert_eq!(succeeded.transport_retry, 1);
        assert_eq!(succeeded.provider_request_id.as_deref(), Some("req-ok"));
        assert_eq!(
            succeeded.provider_response_id.as_deref(),
            Some("chatcmpl-1")
        );
        assert_eq!(succeeded.resolved_model.as_deref(), Some("served-model"));
        assert_eq!(
            succeeded.requested_model.as_deref(),
            Some("journal-retries")
        );
        assert_eq!(
            (
                succeeded.tokens.input,
                succeeded.tokens.output,
                succeeded.tokens.total,
                succeeded.tokens.cache_read,
                succeeded.tokens.reasoning,
            ),
            (Some(120), Some(30), Some(150), Some(100), Some(10))
        );

        for record in &records {
            assert_eq!(record.provider, "openai-compatible");
            assert_eq!(record.operation, koharu_usage::UsageOperation::Translation);
            assert_eq!(record.attempt, 1);
            assert!(record.latency_ms.is_some());
            assert_eq!(record.settings.reasoning_effort.as_deref(), Some("high"));
            assert_eq!(record.settings.max_tokens, Some(2048));
            assert_eq!(
                record.context.source_language.as_deref(),
                Some(Language::Japanese.tag())
            );
            assert_eq!(
                record.context.target_language.as_deref(),
                Some(Language::English.tag())
            );
            assert_eq!(record.context.segments, Some(2));
            assert_eq!(record.context.images, Some(0));
        }
        assert!(
            !serde_json::to_string(&records)
                .unwrap()
                .contains("do not store me")
        );
    }

    #[tokio::test]
    async fn paid_responses_that_do_not_decode_still_count() {
        #[derive(serde::Deserialize)]
        struct NeedsChoices {
            #[allow(dead_code)]
            choices: Vec<Value>,
        }
        let (url, server) = serve(vec![http(
            "200 OK",
            "",
            r#"{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        )]);
        let trace = trace("journal-undecodable");

        let result = send_inference::<NeedsChoices>(
            "openai-compatible",
            &chat_inference(&trace),
            Client::new().get(url),
        )
        .await;
        server.join().unwrap();

        assert!(result.is_err());
        let records = records("journal-undecodable");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, UsageOutcome::DecodeError);
        assert_eq!(records[0].http_status, Some(200));
        assert_eq!(records[0].tokens.total, Some(15));
    }

    #[tokio::test]
    async fn failures_are_classified_and_keep_only_short_codes() {
        let (url, server) = serve(vec![
            http(
                "429 Too Many Requests",
                "",
                r#"{"error":{"code":"insufficient_quota","message":"secret"}}"#,
            ),
            http(
                "400 Bad Request",
                "",
                r#"{"error":{"code":"invalid_prompt","message":"secret"}}"#,
            ),
            http(
                "500 Internal Server Error",
                "",
                r#"{"error":{"message":"secret only"}}"#,
            ),
        ]);
        let trace = trace("journal-failures");

        for _ in 0..3 {
            let _ = send_inference::<Value>(
                "openai-compatible",
                &chat_inference(&trace),
                Client::new().get(&url),
            )
            .await
            .unwrap_err();
        }
        server.join().unwrap();

        let records = records("journal-failures");
        let summary: Vec<_> = records
            .iter()
            .map(|record| {
                (
                    record.outcome,
                    record.http_status,
                    record.error_code.as_deref(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (
                    UsageOutcome::QuotaExceeded,
                    Some(429),
                    Some("insufficient_quota")
                ),
                (
                    UsageOutcome::PolicyRejected,
                    Some(400),
                    Some("invalid_prompt")
                ),
                (UsageOutcome::HttpError, Some(500), None),
            ]
        );
        assert!(!serde_json::to_string(&records).unwrap().contains("secret"));
    }

    #[tokio::test]
    async fn unreachable_hosts_are_network_errors_without_usage() {
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/chat/completions", closed.local_addr().unwrap());
        drop(closed);
        let trace = trace("journal-unreachable");

        let _ = send_inference::<Value>(
            "openai-compatible",
            &chat_inference(&trace),
            Client::new().get(url),
        )
        .await
        .unwrap_err();

        let records = records("journal-unreachable");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, UsageOutcome::NetworkError);
        assert_eq!(records[0].http_status, None);
        assert!(records[0].tokens.is_unreported());
    }

    #[tokio::test]
    async fn abandoned_requests_are_journaled_as_cancelled() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/chat/completions", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            // Never answer: the job is stopped while the request is in flight.
            thread::sleep(Duration::from_millis(1500));
        });
        let trace = trace("journal-cancelled");

        let outcome = tokio::time::timeout(
            Duration::from_millis(200),
            send_inference::<Value>(
                "openai-compatible",
                &chat_inference(&trace),
                Client::new().get(url),
            ),
        )
        .await;
        assert!(outcome.is_err(), "the request must still be in flight");
        server.join().unwrap();

        let records = records("journal-cancelled");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, UsageOutcome::Cancelled);
        assert_eq!(records[0].http_status, None);
    }

    #[tokio::test]
    async fn rejecting_output_downgrades_only_a_succeeded_attempt() {
        let (url, server) = serve(vec![http("200 OK", "", r#"{"usage":{"total_tokens":3}}"#)]);
        let trace = trace("journal-rejected");
        let _: Value = send_inference(
            "openai-compatible",
            &chat_inference(&trace),
            Client::new().get(url),
        )
        .await
        .unwrap();
        server.join().unwrap();
        assert_eq!(
            records("journal-rejected")[0].outcome,
            UsageOutcome::Succeeded
        );

        trace.reject(UsageOutcome::ValidationRejected);
        trace.reject(UsageOutcome::InvalidOutput);

        let records = records("journal-rejected");
        assert_eq!(records[0].outcome, UsageOutcome::ValidationRejected);
        assert_eq!(records[0].tokens.total, Some(3));
    }
}
