// https://api-docs.deepseek.com/api/list-models
// https://api-docs.deepseek.com/guides/thinking_mode

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::Deserialize;

use super::openai_compatible::{ChatBackend, ResponseMode};
use super::send_json;
use crate::{
    GenerationConfig, Model, Provider, ReasoningEffort, Result, TranslationRequest, display_name,
};

const URL: &str = "https://api.deepseek.com/chat/completions";
const MODELS_URL: &str = "https://api.deepseek.com/models";

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct DeepSeekConfig {}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("deepseek")? else {
        return Ok(Vec::new());
    };
    let response: ModelsResponse = send_json(
        "deepseek",
        client.get(MODELS_URL).bearer_auth(api_key.expose_secret()),
    )
    .await?;
    Ok(response
        .data
        .into_iter()
        .map(|model| Model {
            provider: Provider::DeepSeek,
            name: display_name(&model.id),
            reasoning_efforts: model.reasoning_efforts(),
            model: Some(model.id),
            quantizations: Vec::new(),
            vision: false,
            reasoning: true,
            reasoning_always_on: false,
        })
        .collect())
}

pub(super) async fn translate(
    client: &Client,
    _config: &DeepSeekConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("deepseek")?.context("deepseek API key is not configured")?;
    let backend = backend(api_key.expose_secret(), model, generation);
    super::openai_compatible::translate(client, backend, request).await
}

fn backend<'a>(api_key: &'a str, model: &'a str, generation: &GenerationConfig) -> ChatBackend<'a> {
    ChatBackend {
        temperature: generation.temperature.or(Some(1.3)),
        thinking: generation
            .reasoning
            .map(|enabled| if enabled { "enabled" } else { "disabled" }),
        reasoning_effort: generation.effort().map(Into::into),
        ..ChatBackend::new(
            "deepseek",
            URL,
            Some(api_key),
            model,
            generation,
            ResponseMode::JsonObject,
        )
    }
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
    effort: Option<Effort>,
}

impl ListedModel {
    /// The values `reasoning_effort` accepts while thinking mode is on.
    fn reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        let reported = self
            .effort
            .iter()
            .flat_map(|effort| &effort.supported_levels);
        ReasoningEffort::from_reported(reported.map(String::as_str))
    }
}

#[derive(Deserialize)]
struct Effort {
    supported_levels: Vec<String>,
}

#[cfg(test)]
mod tests {
    use crate::remote::openai_compatible::request_body;

    use super::ReasoningEffort::{High, Low, Max};
    use super::*;

    #[test]
    fn reads_reasoning_levels_from_the_model_list() {
        let response: ModelsResponse = serde_json::from_value(serde_json::json!({
            "data": [
                {
                    "id": "deepseek-flash",
                    "effort": { "supported_levels": ["low", "high", "max"], "default_level": "high" }
                },
                { "id": "deepseek-unlisted" }
            ]
        }))
        .unwrap();
        assert_eq!(response.data[0].reasoning_efforts(), [Low, High, Max]);
        assert!(response.data[1].reasoning_efforts().is_empty());
    }

    #[test]
    fn thinking_level_reaches_the_chat_completion() {
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
            let backend = backend("key", "deepseek-flash", &generation);
            serde_json::to_value(request_body(&backend, &request).unwrap()).unwrap()
        };

        let on = payload(Some(true), Some(Max));
        assert_eq!(on["thinking"]["type"], "enabled");
        assert_eq!(on["reasoning_effort"], "max");

        let default = payload(Some(true), None);
        assert_eq!(default["thinking"]["type"], "enabled");
        assert!(default.get("reasoning_effort").is_none());

        // Turning thinking off omits the stored level.
        let off = payload(Some(false), Some(Max));
        assert_eq!(off["thinking"]["type"], "disabled");
        assert!(off.get("reasoning_effort").is_none());
    }
}
