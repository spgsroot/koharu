//! Normalizes what providers report about a finished request.
//!
//! Each function takes the whole response object of one API family and returns
//! only what the provider stated: absent counters stay `None`, and a cost
//! exists only when the response carries one. Nothing is estimated from text.
//!
//! The functions are shared because the same schemas arrive through several
//! crates: the Responses API is read by the translator (xAI) and the agent
//! (Codex, xAI), and chat completions by every OpenAI-compatible provider.

use serde_json::Value;

use crate::model::{ProviderCost, TokenUsage, UsageRecord};

const TICKS_PER_USD: f64 = 10_000_000_000.0;
const NANOS_PER_USD: f64 = 1_000_000_000.0;

/// The provider's own account of one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reported {
    pub response_id: Option<String>,
    /// The model the provider says served the request.
    pub model: Option<String>,
    pub tokens: TokenUsage,
    pub cost: Option<ProviderCost>,
}

impl UsageRecord {
    /// Copies a provider report onto the record. Values the report lacks keep
    /// what the record already knew, so partial reports can be applied in turn.
    pub fn report(&mut self, reported: Reported) {
        if reported.response_id.is_some() {
            self.provider_response_id = reported.response_id;
        }
        if reported.model.is_some() {
            self.resolved_model = reported.model;
        }
        let known = std::mem::take(&mut self.tokens);
        self.tokens = TokenUsage {
            input: reported.tokens.input.or(known.input),
            output: reported.tokens.output.or(known.output),
            total: reported.tokens.total.or(known.total),
            reasoning: reported.tokens.reasoning.or(known.reasoning),
            cache_read: reported.tokens.cache_read.or(known.cache_read),
            cache_write: reported.tokens.cache_write.or(known.cache_write),
            input_audio: reported.tokens.input_audio.or(known.input_audio),
            input_image: reported.tokens.input_image.or(known.input_image),
            output_audio: reported.tokens.output_audio.or(known.output_audio),
        };
        if reported.cost.is_some() {
            self.cost = reported.cost;
        }
    }
}

/// OpenAI Chat Completions and the providers that copy it: OpenAI, DeepSeek,
/// Atlas Cloud, MiniMax's native route and local OpenAI-compatible servers.
#[must_use]
pub fn openai_chat(body: &Value) -> Reported {
    let usage = body.get("usage");
    let prompt = field(usage, "prompt_tokens_details");
    let completion = field(usage, "completion_tokens_details");
    Reported {
        response_id: text(body.get("id")),
        model: text(body.get("model")),
        tokens: settle(TokenUsage {
            input: count(field(usage, "prompt_tokens")),
            output: count(field(usage, "completion_tokens")),
            total: count(field(usage, "total_tokens")),
            reasoning: count(field(completion, "reasoning_tokens")),
            // DeepSeek's `prompt_cache_hit_tokens` equals `cached_tokens`.
            cache_read: count(field(prompt, "cached_tokens"))
                .or_else(|| count(field(usage, "prompt_cache_hit_tokens"))),
            cache_write: count(field(prompt, "cache_write_tokens")),
            input_audio: count(field(prompt, "audio_tokens")),
            input_image: count(field(prompt, "image_tokens")),
            output_audio: count(field(completion, "audio_tokens")),
        }),
        cost: xai_cost(usage),
    }
}

/// OpenRouter: chat completions plus the credits it charged for the request.
#[must_use]
pub fn openrouter(body: &Value) -> Reported {
    let mut reported = openai_chat(body);
    reported.cost = body
        .get("usage")
        .and_then(|usage| usage.get("cost"))
        .and_then(Value::as_f64)
        .map(|amount| ProviderCost {
            amount,
            currency: "credits".to_owned(),
            source: "usage.cost".to_owned(),
        });
    reported
}

/// The Responses API (OpenAI, xAI, Codex). Pass the response object itself:
/// the body of a non-streaming call, or the `response` field of a streamed
/// `response.completed`, `response.incomplete` or `response.failed` event.
#[must_use]
pub fn openai_responses(response: &Value) -> Reported {
    let usage = response.get("usage");
    let input = field(usage, "input_tokens_details");
    let output = field(usage, "output_tokens_details");
    Reported {
        response_id: text(response.get("id")),
        model: text(response.get("model")),
        tokens: settle(TokenUsage {
            input: count(field(usage, "input_tokens")),
            output: count(field(usage, "output_tokens")),
            total: count(field(usage, "total_tokens")),
            reasoning: count(field(output, "reasoning_tokens")),
            cache_read: count(field(input, "cached_tokens")),
            cache_write: count(field(input, "cache_write_tokens")),
            input_audio: count(field(input, "audio_tokens")),
            input_image: count(field(input, "image_tokens")),
            output_audio: count(field(output, "audio_tokens")),
        }),
        cost: xai_cost(usage),
    }
}

/// Anthropic Messages. Its `input_tokens` counts only the prompt after the
/// last cache breakpoint, so cache reads and writes are added back to keep
/// `input` the whole prompt.
#[must_use]
pub fn anthropic(message: &Value) -> Reported {
    let usage = message.get("usage");
    let read = count(field(usage, "cache_read_input_tokens"));
    let write = count(field(usage, "cache_creation_input_tokens"));
    let input = count(field(usage, "input_tokens")).and_then(|uncached| {
        uncached
            .checked_add(read.unwrap_or(0))?
            .checked_add(write.unwrap_or(0))
    });
    Reported {
        response_id: text(message.get("id")),
        model: text(message.get("model")),
        tokens: settle(TokenUsage {
            input,
            output: count(field(usage, "output_tokens")),
            total: None,
            reasoning: count(field(
                field(usage, "output_tokens_details"),
                "thinking_tokens",
            )),
            cache_read: read,
            cache_write: write,
            ..TokenUsage::default()
        }),
        cost: None,
    }
}

/// Gemini `generateContent`. Its total covers the prompt, tool-use prompt,
/// thoughts and candidates; thoughts are reported apart from candidates, and
/// zero counters are omitted from the JSON.
#[must_use]
pub fn gemini(response: &Value) -> Reported {
    let usage = response.get("usageMetadata");
    let total = count(field(usage, "totalTokenCount"));
    let thoughts = count(field(usage, "thoughtsTokenCount"));
    let input = count(field(usage, "promptTokenCount")).and_then(|prompt| {
        prompt.checked_add(count(field(usage, "toolUsePromptTokenCount")).unwrap_or(0))
    });
    let output = match count(field(usage, "candidatesTokenCount")) {
        Some(candidates) => candidates.checked_add(thoughts.unwrap_or(0)),
        // An omitted zero: what the total holds beyond the prompt was generated.
        None => total
            .zip(input)
            .and_then(|(total, input)| total.checked_sub(input)),
    };
    let modality = |details: &str, name: &str| {
        let entries = field(usage, details)?.as_array()?;
        entries
            .iter()
            .filter(|entry| entry.get("modality").and_then(Value::as_str) == Some(name))
            .try_fold(None::<u32>, |sum, entry| {
                let tokens = count(entry.get("tokenCount"))?;
                Some(Some(sum.unwrap_or(0).checked_add(tokens)?))
            })
            .flatten()
    };
    Reported {
        response_id: text(response.get("responseId")),
        model: text(response.get("modelVersion")),
        tokens: settle(TokenUsage {
            input,
            output,
            total,
            reasoning: thoughts,
            cache_read: count(field(usage, "cachedContentTokenCount")),
            cache_write: None,
            input_audio: modality("promptTokensDetails", "AUDIO"),
            input_image: modality("promptTokensDetails", "IMAGE"),
            output_audio: modality("candidatesTokensDetails", "AUDIO"),
        }),
        cost: None,
    }
}

/// LM Studio's native `/api/v1/chat`, which reports `stats` instead of `usage`.
#[must_use]
pub fn lm_studio(body: &Value) -> Reported {
    let stats = body.get("stats");
    Reported {
        response_id: text(body.get("response_id")),
        model: text(body.get("model_instance_id")),
        tokens: settle(TokenUsage {
            input: count(field(stats, "input_tokens")),
            output: count(field(stats, "total_output_tokens")),
            reasoning: count(field(stats, "reasoning_output_tokens")),
            ..TokenUsage::default()
        }),
        cost: None,
    }
}

/// A short provider error code from an error response, such as
/// `insufficient_quota` or `RESOURCE_EXHAUSTED`. Messages are never read; a
/// code that is not a short identifier is dropped.
#[must_use]
pub fn error_code(body: &Value) -> Option<String> {
    let error = body.get("error");
    let identifier = |code: &str| {
        let valid = !code.is_empty()
            && code.len() <= 64
            && code
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-'));
        // The envelope type of Anthropic errors carries no information.
        (valid && code != "error").then(|| code.to_owned())
    };
    [
        field(error, "status"),
        field(error, "code"),
        field(error, "type"),
        body.get("code"),
        body.get("type"),
    ]
    .into_iter()
    .flatten()
    .find_map(|value| value.as_str().and_then(identifier))
    // Providers such as OpenRouter use numeric codes; keep them as text.
    .or_else(|| {
        [field(error, "code"), body.get("code")]
            .into_iter()
            .flatten()
            .find_map(|value| {
                value
                    .as_u64()
                    .and_then(|code| identifier(&code.to_string()))
            })
    })
}

fn field<'a>(value: Option<&'a Value>, key: &str) -> Option<&'a Value> {
    value?.get(key)
}

fn text(value: Option<&Value>) -> Option<String> {
    value?.as_str().map(str::to_owned)
}

/// A counter that is a JSON integer fitting `u32`; anything else is unknown.
fn count(value: Option<&Value>) -> Option<u32> {
    u32::try_from(value?.as_u64()?).ok()
}

/// xAI's exact charge for the request, in USD ticks or nano-dollars.
fn xai_cost(usage: Option<&Value>) -> Option<ProviderCost> {
    let usd = |field_name: &str, per_usd: f64| {
        let units = field(usage, field_name)?.as_i64()?;
        (units >= 0).then_some(ProviderCost {
            amount: units as f64 / per_usd,
            currency: "USD".to_owned(),
            source: format!("usage.{field_name}"),
        })
    };
    usd("cost_in_usd_ticks", TICKS_PER_USD).or_else(|| usd("cost_in_nano_usd", NANOS_PER_USD))
}

/// Exact arithmetic over the provider's own counters.
fn settle(mut tokens: TokenUsage) -> TokenUsage {
    // xAI reports reasoning beside the output instead of inside it; its total
    // then equals input + output + reasoning. Folding reasoning into `output`
    // keeps the subset invariant without touching OpenAI-style providers,
    // whose total equals input + output.
    if let (Some(input), Some(output), Some(total), Some(reasoning)) =
        (tokens.input, tokens.output, tokens.total, tokens.reasoning)
        && reasoning > 0
        && input.checked_add(output) != Some(total)
        && input
            .checked_add(output)
            .and_then(|sum| sum.checked_add(reasoning))
            == Some(total)
    {
        tokens.output = output.checked_add(reasoning);
    }
    if tokens.total.is_none()
        && let (Some(input), Some(output)) = (tokens.input, tokens.output)
    {
        tokens.total = input.checked_add(output);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn chat_completions_keep_cache_and_reasoning_as_subsets() {
        let reported = openai_chat(&json!({
            "id": "chatcmpl-1",
            "model": "gpt-5.6-2026-01-01",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 300,
                "total_tokens": 1300,
                "prompt_tokens_details": {
                    "cached_tokens": 400, "cache_write_tokens": 50,
                    "audio_tokens": 7, "image_tokens": 256
                },
                "completion_tokens_details": {"reasoning_tokens": 120, "audio_tokens": 3}
            }
        }));

        assert_eq!(reported.response_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(reported.model.as_deref(), Some("gpt-5.6-2026-01-01"));
        assert_eq!(
            reported.tokens,
            TokenUsage {
                input: Some(1000),
                output: Some(300),
                total: Some(1300),
                reasoning: Some(120),
                cache_read: Some(400),
                cache_write: Some(50),
                input_audio: Some(7),
                input_image: Some(256),
                output_audio: Some(3),
            }
        );
        assert_eq!(reported.cost, None);
    }

    #[test]
    fn deepseek_cache_hits_are_read_from_the_top_level_field() {
        let reported = openai_chat(&json!({
            "usage": {
                "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40,
                "prompt_cache_hit_tokens": 20, "prompt_cache_miss_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 4}
            }
        }));
        assert_eq!(reported.tokens.cache_read, Some(20));
        assert_eq!(reported.tokens.reasoning, Some(4));
        assert_eq!(reported.tokens.output, Some(10));
    }

    #[test]
    fn responses_without_usage_stay_unreported() {
        for reported in [
            openai_chat(&json!({"id": "x", "choices": []})),
            openai_responses(&json!({"usage": null})),
            anthropic(&json!({})),
            gemini(&json!({"candidates": []})),
            lm_studio(&json!({"output": []})),
        ] {
            assert!(reported.tokens.is_unreported());
            assert_eq!(reported.tokens, TokenUsage::default());
        }
    }

    #[test]
    fn openai_responses_keep_reasoning_inside_output() {
        let reported = openai_responses(&json!({
            "id": "resp_1",
            "model": "gpt-5.6",
            "usage": {
                "input_tokens": 200, "output_tokens": 80, "total_tokens": 280,
                "input_tokens_details": {"cached_tokens": 128, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 64}
            }
        }));
        assert_eq!(reported.tokens.output, Some(80));
        assert_eq!(reported.tokens.reasoning, Some(64));
        assert_eq!(reported.tokens.total, Some(280));
        assert_eq!(reported.tokens.cache_read, Some(128));
        assert_eq!(reported.tokens.cache_write, Some(0));
    }

    #[test]
    fn xai_reasoning_beside_output_is_folded_into_output() {
        // Mirrors docs.x.ai: input 716 + output 126 + reasoning 167 = 1009.
        let reported = openai_responses(&json!({
            "id": "resp_x",
            "usage": {
                "input_tokens": 716, "output_tokens": 126, "total_tokens": 1009,
                "output_tokens_details": {"reasoning_tokens": 167},
                "cost_in_usd_ticks": 37_756_000_i64
            }
        }));

        assert_eq!(reported.tokens.input, Some(716));
        assert_eq!(reported.tokens.output, Some(293));
        assert_eq!(reported.tokens.reasoning, Some(167));
        assert_eq!(reported.tokens.total, Some(1009));
        let cost = reported.cost.unwrap();
        assert_eq!(cost.currency, "USD");
        assert_eq!(cost.source, "usage.cost_in_usd_ticks");
        assert!((cost.amount - 0.0037756).abs() < 1e-12);
    }

    #[test]
    fn xai_nano_dollar_cost_is_a_fallback_for_ticks() {
        let reported = openai_responses(&json!({
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "cost_in_usd_ticks": null, "cost_in_nano_usd": 2_500_000}
        }));
        let cost = reported.cost.unwrap();
        assert_eq!(cost.source, "usage.cost_in_nano_usd");
        assert!((cost.amount - 0.0025).abs() < 1e-12);
    }

    #[test]
    fn anthropic_input_includes_cache_reads_and_writes() {
        let reported = anthropic(&json!({
            "id": "msg_1",
            "model": "claude-sonnet",
            "usage": {
                "input_tokens": 50,
                "cache_read_input_tokens": 100_000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 400,
                "output_tokens_details": {"thinking_tokens": 150}
            }
        }));
        assert_eq!(reported.tokens.input, Some(100_050));
        assert_eq!(reported.tokens.cache_read, Some(100_000));
        assert_eq!(reported.tokens.cache_write, Some(0));
        assert_eq!(reported.tokens.output, Some(400));
        assert_eq!(reported.tokens.reasoning, Some(150));
        assert_eq!(reported.tokens.total, Some(100_450));
    }

    #[test]
    fn anthropic_without_cache_fields_reports_plain_input() {
        let reported = anthropic(&json!({
            "usage": {"input_tokens": 12, "output_tokens": 3,
                      "cache_read_input_tokens": null, "cache_creation_input_tokens": null}
        }));
        assert_eq!(reported.tokens.input, Some(12));
        assert_eq!(reported.tokens.cache_read, None);
        assert_eq!(reported.tokens.reasoning, None);
        assert_eq!(reported.tokens.total, Some(15));
    }

    #[test]
    fn gemini_counts_thoughts_in_output_and_reads_modalities() {
        let reported = gemini(&json!({
            "responseId": "resp-g",
            "modelVersion": "gemini-3.7-flash",
            "usageMetadata": {
                "promptTokenCount": 600,
                "cachedContentTokenCount": 100,
                "candidatesTokenCount": 40,
                "thoughtsTokenCount": 60,
                "totalTokenCount": 700,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 342},
                    {"modality": "IMAGE", "tokenCount": 258}
                ]
            }
        }));
        assert_eq!(reported.response_id.as_deref(), Some("resp-g"));
        assert_eq!(reported.model.as_deref(), Some("gemini-3.7-flash"));
        assert_eq!(reported.tokens.input, Some(600));
        assert_eq!(reported.tokens.output, Some(100));
        assert_eq!(reported.tokens.reasoning, Some(60));
        assert_eq!(reported.tokens.total, Some(700));
        assert_eq!(reported.tokens.cache_read, Some(100));
        assert_eq!(reported.tokens.input_image, Some(258));
        assert_eq!(reported.tokens.input_audio, None);
    }

    #[test]
    fn gemini_omitted_zero_candidates_are_derived_from_the_total() {
        // A thinking model that spent everything on thoughts omits candidates.
        let reported = gemini(&json!({
            "usageMetadata": {
                "promptTokenCount": 10, "thoughtsTokenCount": 90, "totalTokenCount": 100
            }
        }));
        assert_eq!(reported.tokens.output, Some(90));
        assert_eq!(reported.tokens.reasoning, Some(90));
        assert_eq!(reported.tokens.total, Some(100));
    }

    #[test]
    fn openrouter_reports_credits_charged() {
        let reported = openrouter(&json!({
            "id": "gen-1",
            "usage": {
                "prompt_tokens": 194, "completion_tokens": 2, "total_tokens": 196,
                "cost": 0.95,
                "cost_details": {"upstream_inference_cost": 19},
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 100}
            }
        }));
        let cost = reported.cost.unwrap();
        assert_eq!(cost.amount, 0.95);
        assert_eq!(cost.currency, "credits");
        assert_eq!(cost.source, "usage.cost");
        assert_eq!(reported.tokens.cache_write, Some(100));
        assert_eq!(reported.tokens.cache_read, Some(0));
    }

    #[test]
    fn plain_chat_providers_never_get_a_cost_from_unrelated_fields() {
        assert_eq!(
            openai_chat(&json!({"usage": {"cost": 5, "total_tokens": 1}})).cost,
            None
        );
    }

    #[test]
    fn lm_studio_reads_stats() {
        let reported = lm_studio(&json!({
            "model_instance_id": "ibm/granite",
            "response_id": "resp_lm",
            "stats": {"input_tokens": 646, "total_output_tokens": 586, "reasoning_output_tokens": 0,
                      "tokens_per_second": 29.7}
        }));
        assert_eq!(reported.model.as_deref(), Some("ibm/granite"));
        assert_eq!(reported.tokens.input, Some(646));
        assert_eq!(reported.tokens.output, Some(586));
        assert_eq!(reported.tokens.reasoning, Some(0));
        assert_eq!(reported.tokens.total, Some(1232));
    }

    #[test]
    fn counters_outside_u32_are_unknown_not_saturated() {
        let reported = openai_chat(&json!({
            "usage": {"prompt_tokens": 5_000_000_000_u64, "completion_tokens": -1,
                      "total_tokens": 1.5}
        }));
        assert_eq!(reported.tokens, TokenUsage::default());
    }

    #[test]
    fn error_codes_are_short_identifiers_never_messages() {
        assert_eq!(
            error_code(&json!({"error": {"code": "insufficient_quota", "message": "secret text"}})),
            Some("insufficient_quota".to_owned())
        );
        assert_eq!(
            error_code(&json!({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})),
            Some("RESOURCE_EXHAUSTED".to_owned())
        );
        assert_eq!(
            error_code(&json!({"type": "error", "error": {"type": "overloaded_error"}})),
            Some("overloaded_error".to_owned())
        );
        assert_eq!(
            error_code(&json!({"error": {"code": 402}})),
            Some("402".to_owned())
        );
        assert_eq!(
            error_code(&json!({"error": {"code": "Your prompt said: hello world"}})),
            None
        );
        assert_eq!(error_code(&json!({"message": "plain text"})), None);
    }

    #[test]
    fn reports_merge_without_forgetting_known_values() {
        let mut record = UsageRecord::new(crate::UsageOperation::Agent, "codex");
        record.report(Reported {
            response_id: Some("resp_1".to_owned()),
            tokens: TokenUsage {
                input: Some(10),
                ..TokenUsage::default()
            },
            ..Reported::default()
        });
        record.report(Reported {
            model: Some("gpt".to_owned()),
            tokens: TokenUsage {
                output: Some(4),
                ..TokenUsage::default()
            },
            ..Reported::default()
        });

        assert_eq!(record.provider_response_id.as_deref(), Some("resp_1"));
        assert_eq!(record.resolved_model.as_deref(), Some("gpt"));
        assert_eq!(record.tokens.input, Some(10));
        assert_eq!(record.tokens.output, Some(4));
    }
}
