use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use specta::Type;
use uuid::Uuid;

/// Declares a snake_case wire enum that is also stored as that text.
macro_rules! text_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$variant_meta:meta])* $variant:ident => $text:literal,)+ }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
        pub enum $name {
            $($(#[$variant_meta])* #[serde(rename = $text)] $variant,)+
        }

        impl $name {
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)+
                }
            }

            pub(crate) fn parse(text: &str) -> Option<Self> {
                match text {
                    $($text => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

text_enum! {
    /// What kind of work an inference request served.
    UsageOperation {
        Translation => "translation",
        Agent => "agent",
        Ocr => "ocr",
    }
}

text_enum! {
    /// How a request ended. Everything except `Succeeded` is a failure, but a
    /// failed request may still carry usage the provider billed.
    UsageOutcome {
        Succeeded => "succeeded",
        /// The request never produced a response (connection, TLS, DNS).
        NetworkError => "network_error",
        Timeout => "timeout",
        /// Non-success HTTP status other than quota and policy rejections.
        HttpError => "http_error",
        QuotaExceeded => "quota_exceeded",
        PolicyRejected => "policy_rejected",
        /// HTTP success whose body reported a provider-level error.
        ProviderError => "provider_error",
        /// HTTP success whose body could not be decoded.
        DecodeError => "decode_error",
        /// A decoded response that did not yield usable translated output.
        InvalidOutput => "invalid_output",
        /// Usable output that the configured translation validator rejected.
        ValidationRejected => "validation_rejected",
        /// The response ended before it was complete.
        Incomplete => "incomplete",
        Cancelled => "cancelled",
    }
}

impl UsageOutcome {
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Succeeded)
    }
}

/// Token counters exactly as the provider reported them, normalized so that
/// subsets are never double counted.
///
/// `None` means the provider did not report the counter; it is never zero.
///
/// * `input` includes `cache_read`, `cache_write`, `input_audio` and
///   `input_image`.
/// * `output` includes `reasoning` and `output_audio`.
/// * `total` is the provider's own total, or `input + output` when the
///   provider omits it and both are known.
///
/// A sum over several requests must add `input`, `output` and `total` only;
/// the subset counters are reported next to them, never on top of them.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
pub struct TokenUsage {
    pub input: Option<u32>,
    pub output: Option<u32>,
    pub total: Option<u32>,
    pub reasoning: Option<u32>,
    pub cache_read: Option<u32>,
    pub cache_write: Option<u32>,
    pub input_audio: Option<u32>,
    pub input_image: Option<u32>,
    pub output_audio: Option<u32>,
}

impl TokenUsage {
    /// Whether the provider reported none of the headline counters.
    #[must_use]
    pub fn is_unreported(&self) -> bool {
        self.input.is_none() && self.output.is_none() && self.total.is_none()
    }
}

/// A cost the provider itself put in its response. Koharu never derives one
/// from a price list, so a request without this has an unknown cost.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ProviderCost {
    pub amount: f64,
    /// ISO 4217 code, or the provider's own unit (for example `credits`).
    pub currency: String,
    /// The response field the amount was read from, such as `usage.cost`.
    pub source: String,
}

/// Generation settings as they were sent on the wire. A provider that ignores
/// a setting reports it as `None`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
pub struct RequestSettings {
    pub reasoning: Option<bool>,
    /// The reasoning level label the provider received, in its own vocabulary.
    /// `None` when the provider's default applied.
    pub reasoning_effort: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

/// What the request was about. Identifiers are opaque; text is never recorded.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
pub struct UsageContext {
    pub project_id: Option<String>,
    pub page_id: Option<String>,
    pub job_id: Option<String>,
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub segments: Option<u32>,
    pub images: Option<u32>,
    pub context_entries: Option<u32>,
}

/// One inference request attempt.
///
/// Prompts, model output, images, credentials, reasoning text and provider
/// error messages are never part of a record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct UsageRecord {
    /// Opaque, time-ordered identifier.
    pub id: String,
    /// When the request was sent, in milliseconds since the Unix epoch.
    #[specta(type = f64)]
    pub started_at: i64,
    pub operation: UsageOperation,
    /// Stable provider identifier, such as `openai`, `local` or `codex`.
    pub provider: String,
    pub requested_model: Option<String>,
    /// The model the provider says served the request.
    pub resolved_model: Option<String>,
    pub provider_request_id: Option<String>,
    pub provider_response_id: Option<String>,
    pub outcome: UsageOutcome,
    /// A short machine-readable provider error code, never a message.
    pub error_code: Option<String>,
    pub http_status: Option<u16>,
    pub latency_ms: Option<u32>,
    /// Zero-based validation attempt of the logical request.
    pub attempt: u32,
    /// Zero-based resend of the same HTTP request after a transient failure.
    pub transport_retry: u32,
    pub settings: RequestSettings,
    pub context: UsageContext,
    pub tokens: TokenUsage,
    pub cost: Option<ProviderCost>,
}

impl UsageRecord {
    /// Starts a record for a request sent now. The outcome defaults to
    /// `Incomplete` until the caller learns how the request ended.
    #[must_use]
    pub fn new(operation: UsageOperation, provider: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            started_at: unix_millis(),
            operation,
            provider: provider.into(),
            requested_model: None,
            resolved_model: None,
            provider_request_id: None,
            provider_response_id: None,
            outcome: UsageOutcome::Incomplete,
            error_code: None,
            http_status: None,
            latency_ms: None,
            attempt: 0,
            transport_retry: 0,
            settings: RequestSettings::default(),
            context: UsageContext::default(),
            tokens: TokenUsage::default(),
            cost: None,
        }
    }

    /// A copy whose free-form fields are bounded and free of control
    /// characters, and whose numbers can be stored and serialized. Invalid
    /// values become unknown rather than being repaired.
    pub(crate) fn sanitized(&self) -> Self {
        let mut record = self.clone();
        record.id = tidy(&record.id, 64).unwrap_or_else(|| Uuid::now_v7().to_string());
        record.provider = tidy(&record.provider, 64).unwrap_or_else(|| "unknown".to_owned());
        record.requested_model = tidy_option(&record.requested_model, 256);
        record.resolved_model = tidy_option(&record.resolved_model, 256);
        record.provider_request_id = tidy_option(&record.provider_request_id, 160);
        record.provider_response_id = tidy_option(&record.provider_response_id, 160);
        record.error_code = tidy_option(&record.error_code, 64);
        record.http_status = record
            .http_status
            .filter(|status| (100..=599).contains(status));
        record.settings.reasoning_effort = tidy_option(&record.settings.reasoning_effort, 32);
        record.settings.temperature = record
            .settings
            .temperature
            .filter(|value| value.is_finite());
        record.settings.top_p = record.settings.top_p.filter(|value| value.is_finite());
        record.settings.frequency_penalty = record
            .settings
            .frequency_penalty
            .filter(|value| value.is_finite());
        record.settings.presence_penalty = record
            .settings
            .presence_penalty
            .filter(|value| value.is_finite());
        record.context.project_id = tidy_option(&record.context.project_id, 128);
        record.context.page_id = tidy_option(&record.context.page_id, 128);
        record.context.job_id = tidy_option(&record.context.job_id, 128);
        record.context.source_language = tidy_option(&record.context.source_language, 32);
        record.context.target_language = tidy_option(&record.context.target_language, 32);
        record.cost = record.cost.take().and_then(|cost| {
            let currency = tidy(&cost.currency, 16)?;
            let source = tidy(&cost.source, 64)?;
            (cost.amount.is_finite() && cost.amount >= 0.0).then_some(ProviderCost {
                amount: cost.amount,
                currency,
                source,
            })
        });
        record
    }
}

fn tidy(value: &str, limit: usize) -> Option<String> {
    let value: String = value
        .chars()
        .filter(|character| !character.is_control())
        .collect();
    let value = value.trim();
    (!value.is_empty()).then(|| value.chars().take(limit).collect())
}

fn tidy_option(value: &Option<String>, limit: usize) -> Option<String> {
    value.as_deref().and_then(|value| tidy(value, limit))
}

pub(crate) fn unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_records_are_unique_and_pessimistic() {
        let first = UsageRecord::new(UsageOperation::Translation, "openai");
        let second = UsageRecord::new(UsageOperation::Translation, "openai");
        assert_ne!(first.id, second.id);
        assert_eq!(first.outcome, UsageOutcome::Incomplete);
        assert!(first.tokens.is_unreported());
        assert!(first.cost.is_none());
    }

    #[test]
    fn sanitizing_bounds_text_and_drops_unusable_numbers() {
        let mut record = UsageRecord::new(UsageOperation::Agent, "  codex\n");
        record.requested_model = Some("m".repeat(1000));
        record.error_code = Some("  ".to_owned());
        record.http_status = Some(7);
        record.settings.temperature = Some(f32::NAN);
        record.cost = Some(ProviderCost {
            amount: f64::INFINITY,
            currency: "USD".to_owned(),
            source: "usage.cost".to_owned(),
        });

        let clean = record.sanitized();

        assert_eq!(clean.provider, "codex");
        assert_eq!(clean.requested_model.unwrap().chars().count(), 256);
        assert_eq!(clean.error_code, None);
        assert_eq!(clean.http_status, None);
        assert_eq!(clean.settings.temperature, None);
        assert_eq!(clean.cost, None);
    }

    #[test]
    fn wire_names_round_trip() {
        for outcome in [
            UsageOutcome::Succeeded,
            UsageOutcome::NetworkError,
            UsageOutcome::Timeout,
            UsageOutcome::HttpError,
            UsageOutcome::QuotaExceeded,
            UsageOutcome::PolicyRejected,
            UsageOutcome::ProviderError,
            UsageOutcome::DecodeError,
            UsageOutcome::InvalidOutput,
            UsageOutcome::ValidationRejected,
            UsageOutcome::Incomplete,
            UsageOutcome::Cancelled,
        ] {
            assert_eq!(UsageOutcome::parse(outcome.as_str()), Some(outcome));
            assert_eq!(
                serde_json::to_value(outcome).unwrap(),
                serde_json::Value::String(outcome.as_str().to_owned())
            );
        }
        assert_eq!(UsageOperation::parse("ocr"), Some(UsageOperation::Ocr));
        assert_eq!(UsageOperation::parse("chat"), None);
    }
}
