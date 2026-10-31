// https://developers.openai.com/api/reference/resources/models/methods/list
// Reasoning levels are documented per model ("Reasoning.effort supports ..."):
// https://developers.openai.com/api/docs/guides/reasoning
// https://developers.openai.com/api/docs/models

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::Deserialize;

use super::openai_compatible::{ChatBackend, ResponseMode};
use super::send_json;
use crate::{
    GenerationConfig, Model, Provider, ReasoningEffort, Result, TranslationRequest, display_name,
};

const URL: &str = "https://api.openai.com/v1/chat/completions";
const MODELS_URL: &str = "https://api.openai.com/v1/models";

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct OpenAiConfig {}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("openai")? else {
        return Ok(Vec::new());
    };
    let response: ModelsResponse = send_json(
        "openai",
        client.get(MODELS_URL).bearer_auth(api_key.expose_secret()),
    )
    .await?;
    Ok(response
        .data
        .into_iter()
        .filter(|model| supports_translation(&model.id))
        .map(|model| {
            let listed = documented(&model.id);
            Model {
                provider: Provider::OpenAi,
                name: display_name(&model.id),
                reasoning_efforts: listed.map_or_else(Vec::new, |(_, efforts, _)| efforts.to_vec()),
                reasoning_always_on: listed.is_some_and(|(_, _, can_disable)| !can_disable),
                model: Some(model.id),
                quantizations: Vec::new(),
                vision: true,
                reasoning: true,
            }
        })
        .collect())
}

fn supports_translation(id: &str) -> bool {
    let base = id
        .strip_prefix("ft:")
        .and_then(|id| id.split(':').next())
        .unwrap_or(id)
        .to_ascii_lowercase();
    let supported_family = base.starts_with("gpt-5")
        || base.starts_with("gpt-4.1")
        || base.starts_with("gpt-4o")
        || base == "o3"
        || base.starts_with("o3-");
    let incompatible = [
        "-audio",
        "-codex",
        "-deep-research",
        "-image",
        "-pro",
        "-realtime",
        "-search",
        "-transcribe",
        "-tts",
    ]
    .iter()
    .any(|marker| base.contains(marker));
    supported_family && !incompatible
}

/// What each model page documents about `reasoning_effort`: the values it
/// accepts besides `none`, and whether `none`, the off switch, is among them.
/// The model list reports no capabilities, and a model whose page does not
/// list its values is absent, so no level is offered for it and its off switch
/// keeps working as before.
const DOCUMENTED: &[(&str, &[ReasoningEffort], bool)] = {
    use ReasoningEffort::{High, Low, Max, Medium, Minimal, XHigh};
    &[
        ("gpt-5", &[Minimal, Low, Medium, High], false),
        ("gpt-5.1", &[Low, Medium, High], true),
        ("gpt-5.2", &[Low, Medium, High, XHigh], true),
        ("gpt-5.4", &[Low, Medium, High, XHigh], true),
        ("gpt-5.4-mini", &[Low, Medium, High, XHigh], true),
        ("gpt-5.4-nano", &[Low, Medium, High, XHigh], true),
        ("gpt-5.5", &[Low, Medium, High, XHigh], true),
        ("gpt-5.6-luna", &[Low, Medium, High, XHigh, Max], true),
        ("gpt-5.6-sol", &[Low, Medium, High, XHigh, Max], true),
        ("gpt-5.6-terra", &[Low, Medium, High, XHigh, Max], true),
    ]
};

fn documented(id: &str) -> Option<&'static (&'static str, &'static [ReasoningEffort], bool)> {
    DOCUMENTED
        .iter()
        .find(|(model, _, _)| is_snapshot_of(id, model))
}

/// Whether `id` is the model itself or one of its dated snapshots such as
/// `gpt-5.4-mini-2026-03-17`.
fn is_snapshot_of(id: &str, model: &str) -> bool {
    id.strip_prefix(model).is_some_and(|rest| {
        rest.is_empty()
            || rest.strip_prefix('-').is_some_and(|date| {
                date.len() == 10
                    && date.bytes().enumerate().all(|(index, byte)| match index {
                        4 | 7 => byte == b'-',
                        _ => byte.is_ascii_digit(),
                    })
            })
    })
}

pub(super) async fn translate(
    client: &Client,
    _config: &OpenAiConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("openai")?.context("openai API key is not configured")?;
    let backend = backend(api_key.expose_secret(), model, generation);
    super::openai_compatible::translate(client, backend, request).await
}

fn backend<'a>(api_key: &'a str, model: &'a str, generation: &GenerationConfig) -> ChatBackend<'a> {
    ChatBackend {
        max_tokens: None,
        max_completion_tokens: generation.max_tokens,
        reasoning_effort: reasoning_effort(generation),
        ..ChatBackend::new(
            "openai",
            URL,
            Some(api_key),
            model,
            generation,
            ResponseMode::JsonSchema,
        )
    }
}

/// `none` turns reasoning off. Without a stored level, enabled reasoning keeps
/// the `medium` that OpenAI documents as the default.
fn reasoning_effort(generation: &GenerationConfig) -> Option<&'static str> {
    generation
        .reasoning
        .map(|enabled| match (enabled, generation.effort()) {
            (false, _) => "none",
            (true, Some(effort)) => effort.into(),
            (true, None) => "medium",
        })
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
}

#[cfg(test)]
mod tests {
    use crate::remote::openai_compatible::request_body;

    use super::*;

    #[test]
    fn filters_models_to_the_chat_translation_contract() {
        assert!(supports_translation("gpt-5.6-luna"));
        assert!(supports_translation("gpt-4o-mini"));
        assert!(supports_translation("ft:gpt-4o-mini:org:name:id"));
        assert!(!supports_translation("gpt-5.5-pro"));
        assert!(!supports_translation("gpt-5.3-codex"));
        assert!(!supports_translation("gpt-4o-audio-preview"));
        assert!(!supports_translation("text-embedding-3-large"));
    }

    #[test]
    fn documented_reasoning_support_follows_the_model_page() {
        use ReasoningEffort::{High, Low, Max, Medium, Minimal, XHigh};

        let support =
            |id| documented(id).map(|(_, efforts, can_disable)| (efforts.to_vec(), *can_disable));
        assert_eq!(
            support("gpt-5.6-luna"),
            Some((vec![Low, Medium, High, XHigh, Max], true))
        );
        assert_eq!(
            support("gpt-5.4-mini-2026-03-17"),
            Some((vec![Low, Medium, High, XHigh], true))
        );
        // GPT-5 has no `none`, so its reasoning cannot be turned off.
        assert_eq!(
            support("gpt-5-2025-08-07"),
            Some((vec![Minimal, Low, Medium, High], false))
        );
        // A page that does not list its values, or another family, offers no level.
        assert_eq!(support("gpt-5-mini"), None);
        assert_eq!(support("gpt-4o-mini"), None);
        assert_eq!(support("ft:gpt-5.5:org:name:id"), None);
    }

    #[test]
    fn reasoning_level_reaches_the_chat_completion() {
        let request = TranslationRequest::new(
            ["hello"],
            crate::Language::English,
            crate::Language::Japanese,
        );
        let payload = |reasoning, reasoning_effort| {
            let generation = GenerationConfig {
                reasoning,
                reasoning_effort,
                ..GenerationConfig::default()
            };
            let backend = backend("key", "gpt-5.6-luna", &generation);
            serde_json::to_value(request_body(&backend, &request).unwrap()).unwrap()
        };

        assert_eq!(
            payload(Some(true), Some(ReasoningEffort::High))["reasoning_effort"],
            "high"
        );
        assert_eq!(payload(Some(true), None)["reasoning_effort"], "medium");
        // Turning reasoning off omits the stored level.
        assert_eq!(
            payload(Some(false), Some(ReasoningEffort::High))["reasoning_effort"],
            "none"
        );
        assert!(payload(None, None).get("reasoning_effort").is_none());
    }
}
