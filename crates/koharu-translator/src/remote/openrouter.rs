// https://openrouter.ai/docs/api/reference/overview
// https://openrouter.ai/docs/api/api-reference/models/get-models
// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use koharu_usage::wire;
use reqwest::Client;
use serde::{Deserialize, Deserializer};
use strum::VariantArray;

use super::openai_compatible::{ChatBackend, ReasoningControl, ResponseMode};
use crate::{GenerationConfig, Model, Provider, ReasoningEffort, Result, TranslationRequest};

const CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct OpenRouterConfig {}

pub(super) async fn translate(
    client: &Client,
    _config: &OpenRouterConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key =
        koharu_secrets::get("openrouter")?.context("openrouter API key is not configured")?;
    let backend = backend(api_key.expose_secret(), model, generation);
    super::openai_compatible::translate(client, backend, request).await
}

fn backend<'a>(api_key: &'a str, model: &'a str, generation: &GenerationConfig) -> ChatBackend<'a> {
    ChatBackend {
        reasoning: ReasoningControl::from_generation(generation),
        // OpenRouter states the credits it charged in `usage.cost`.
        report: wire::openrouter,
        ..ChatBackend::new(
            "openrouter",
            CHAT_URL,
            Some(api_key),
            model,
            generation,
            ResponseMode::PromptOnly,
        )
    }
}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("openrouter")? else {
        return Ok(Vec::new());
    };
    let discovered: Result<ModelsResponse> = super::send_json(
        "openrouter",
        client.get(MODELS_URL).bearer_auth(api_key.expose_secret()),
    )
    .await;
    Ok(match discovered {
        Ok(discovered) => discovered
            .data
            .into_iter()
            .map(|model| Model {
                provider: Provider::OpenRouter,
                reasoning_efforts: model.reasoning_efforts(),
                name: model.name,
                model: Some(model.id),
                quantizations: Vec::new(),
                vision: model
                    .architecture
                    .input_modalities
                    .iter()
                    .any(|modality| modality == "image"),
                reasoning: model.reasoning.is_some()
                    || model
                        .supported_parameters
                        .iter()
                        .any(|parameter| parameter == "reasoning"),
                reasoning_always_on: false,
            })
            .collect(),
        Err(error) => {
            tracing::warn!(%error, "failed to list OpenRouter models");
            Vec::new()
        }
    })
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
    name: String,
    architecture: Architecture,
    supported_parameters: Vec<String>,
    reasoning: Option<ReasoningCapabilities>,
}

impl ListedModel {
    /// The levels `reasoning.effort` accepts for this model: its listed
    /// allowlist, every level when the list is `null`, and none when the model
    /// does not expose effort selection.
    fn reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        match self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.supported_efforts.as_ref())
        {
            Some(Some(efforts)) => {
                ReasoningEffort::from_reported(efforts.iter().map(String::as_str))
            }
            Some(None) => ReasoningEffort::VARIANTS.to_vec(),
            None => Vec::new(),
        }
    }
}

#[derive(Deserialize)]
struct ReasoningCapabilities {
    #[serde(default, deserialize_with = "present")]
    supported_efforts: Option<Option<Vec<String>>>,
}

/// Keeps a `null` value apart from a missing key.
fn present<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
struct Architecture {
    input_modalities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use crate::remote::openai_compatible::request_body;

    use super::ReasoningEffort::{High, Low, Max, Medium, Minimal, XHigh};
    use super::*;

    #[test]
    fn reads_reasoning_capability_from_model_list() {
        let response: ModelsResponse = serde_json::from_value(serde_json::json!({
            "data": [
                {
                    "id": "provider/reasoning-model",
                    "name": "Reasoning Model",
                    "architecture": { "input_modalities": ["text"] },
                    "supported_parameters": ["reasoning"],
                    "reasoning": {
                        "supported_efforts": ["high", "medium", "low", "minimal", "none"],
                        "mandatory": false
                    }
                },
                {
                    "id": "provider/gateway-model",
                    "name": "Gateway Model",
                    "architecture": { "input_modalities": ["text"] },
                    "supported_parameters": ["reasoning"],
                    "reasoning": { "supported_efforts": null, "mandatory": true }
                },
                {
                    "id": "provider/switch-only-model",
                    "name": "Switch Only Model",
                    "architecture": { "input_modalities": ["text"] },
                    "supported_parameters": ["reasoning"],
                    "reasoning": { "mandatory": false }
                },
                {
                    "id": "provider/chat-model",
                    "name": "Chat Model",
                    "architecture": { "input_modalities": ["text"] },
                    "supported_parameters": []
                }
            ]
        }))
        .unwrap();
        let [listed, gateway, switch_only, chat] = &response.data[..] else {
            panic!("expected four models");
        };
        assert!(listed.reasoning.is_some());
        assert!(chat.reasoning.is_none());
        // Ascending, without the `none` switch.
        assert_eq!(listed.reasoning_efforts(), [Minimal, Low, Medium, High]);
        // A `null` allowlist accepts every gateway level.
        assert_eq!(
            gateway.reasoning_efforts(),
            [Minimal, Low, Medium, High, XHigh, Max]
        );
        // A model that does not expose effort selection offers no level.
        assert!(switch_only.reasoning_efforts().is_empty());
        assert!(chat.reasoning_efforts().is_empty());
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
            let backend = backend("key", "provider/reasoning-model", &generation);
            serde_json::to_value(request_body(&backend, &request).unwrap()).unwrap()
        };

        assert_eq!(
            payload(Some(true), Some(High))["reasoning"],
            serde_json::json!({ "effort": "high" })
        );
        assert_eq!(
            payload(Some(true), None)["reasoning"],
            serde_json::json!({ "enabled": true })
        );
        // Turning reasoning off omits the stored level.
        assert_eq!(
            payload(Some(false), Some(High))["reasoning"],
            serde_json::json!({ "enabled": false })
        );
        assert!(payload(None, None).get("reasoning").is_none());
    }
}
