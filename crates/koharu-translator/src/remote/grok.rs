// https://docs.x.ai/developers/model-capabilities/text/generate-text
// https://docs.x.ai/developers/model-capabilities/text/reasoning
// https://docs.x.ai/developers/rest-api-reference/inference/models

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use koharu_usage::{RequestSettings, wire};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{Inference, send_inference, send_json};
use crate::{
    GenerationConfig, Model, Provider, ReasoningEffort, Result, TranslationRequest,
    backend::encode_image, display_name, prompt,
};

const RESPONSES_URL: &str = "https://api.x.ai/v1/responses";
const MODELS_URL: &str = "https://api.x.ai/v1/language-models";

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct GrokConfig {}

pub(super) async fn translate(
    client: &Client,
    _config: &GrokConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = credential()
        .await?
        .context("Grok is not signed in and no API key is configured")?;
    let body = request_body(model, generation, request)?;
    let inference = Inference::new(request, settings(&body), wire::openai_responses);
    let response: Response = send_inference(
        "grok",
        &inference,
        client
            .post(RESPONSES_URL)
            .bearer_auth(api_key.expose_secret())
            .json(&body),
    )
    .await?;
    let text = response
        .output
        .into_iter()
        .flat_map(|item| item.content)
        .find_map(|content| {
            (content.kind == "output_text")
                .then_some(content.text)
                .flatten()
        })
        .context("Grok returned no output text")?;
    Ok(prompt::translations("grok", &text, &request.segments)?)
}

/// The generation settings `body` puts on the wire, for the usage journal.
/// Reasoning cannot be switched off, so a request without a `reasoning` object
/// ran at xAI's default.
fn settings(body: &Request<'_>) -> RequestSettings {
    RequestSettings {
        reasoning: body.reasoning.as_ref().map(|_| true),
        reasoning_effort: body
            .reasoning
            .as_ref()
            .map(|reasoning| reasoning.effort.to_owned()),
        temperature: body.temperature,
        top_p: body.top_p,
        max_tokens: body.max_output_tokens,
        ..RequestSettings::default()
    }
}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = credential().await? else {
        return Ok(Vec::new());
    };
    let response: ModelsResponse = send_json(
        "grok",
        client.get(MODELS_URL).bearer_auth(api_key.expose_secret()),
    )
    .await?;
    Ok(response
        .models
        .into_iter()
        .map(|model| Model {
            provider: Provider::Grok,
            name: display_name(&model.id),
            reasoning_efforts: model.reasoning_efforts(),
            model: Some(model.id),
            quantizations: Vec::new(),
            vision: model
                .input_modalities
                .iter()
                .any(|modality| modality == "image"),
            reasoning: true,
            reasoning_always_on: false,
        })
        .collect())
}

async fn credential() -> Result<Option<koharu_secrets::SecretString>> {
    if let Some(token) = koharu_agent::Xai::access_token().await? {
        return Ok(Some(token));
    }
    Ok(koharu_secrets::get("grok")?)
}

fn request_body<'a>(
    model: &'a str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> anyhow::Result<Request<'a>> {
    let (system, user) = prompt::prompts(request)?;
    let user_content = match request.image.as_deref() {
        Some(image) => InputContent::Parts(vec![
            InputPart::Text { text: user },
            InputPart::Image {
                image_url: encode_image(image)?.data_url(),
            },
        ]),
        None => InputContent::Text(user),
    };
    Ok(Request {
        model,
        input: [
            InputMessage {
                role: "system",
                content: InputContent::Text(system),
            },
            InputMessage {
                role: "user",
                content: user_content,
            },
        ],
        store: false,
        temperature: generation.temperature,
        top_p: generation.top_p,
        max_output_tokens: generation.max_tokens,
        // Grok reasoning models cannot turn reasoning off, so only the enabled
        // switch sends anything, at the documented default of `high` unless a
        // level was chosen.
        reasoning: match generation.reasoning {
            Some(true) => Some(Reasoning {
                effort: generation.effort().map_or("high", Into::into),
            }),
            Some(false) | None => None,
        },
        text: TextConfig {
            format: TextFormat {
                kind: "json_schema",
                name: "manga_translation",
                strict: true,
                schema: prompt::output_schema(request.segments.len()),
            },
        },
    })
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: [InputMessage; 2],
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Reasoning>,
    text: TextConfig,
}

#[derive(Serialize)]
struct InputMessage {
    role: &'static str,
    content: InputContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum InputContent {
    Text(String),
    Parts(Vec<InputPart>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputPart {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "input_image")]
    Image { image_url: String },
}

#[derive(Serialize)]
struct Reasoning {
    effort: &'static str,
}

#[derive(Serialize)]
struct TextConfig {
    format: TextFormat,
}

#[derive(Serialize)]
struct TextFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Deserialize)]
struct Response {
    output: Vec<OutputItem>,
}

#[derive(Deserialize)]
struct OutputItem {
    #[serde(default)]
    content: Vec<OutputContent>,
}

#[derive(Deserialize)]
struct OutputContent {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    models: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
    input_modalities: Vec<String>,
    #[serde(default)]
    capabilities: Capabilities,
}

impl ListedModel {
    /// The values the model accepts for `reasoning.effort`.
    fn reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        ReasoningEffort::from_reported(
            self.capabilities
                .reasoning_effort
                .iter()
                .map(String::as_str),
        )
    }
}

#[derive(Default, Deserialize)]
struct Capabilities {
    #[serde(default)]
    reasoning_effort: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_responses_schema_and_image_input() {
        let request = TranslationRequest::new(
            ["hello"],
            crate::Language::English,
            crate::Language::Japanese,
        )
        .with_image(std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)));
        let body = request_body(
            "grok-4.5",
            &GenerationConfig {
                max_tokens: Some(1024),
                reasoning: Some(true),
                ..GenerationConfig::default()
            },
            &request,
        )
        .unwrap();
        let value = serde_json::to_value(body).unwrap();
        assert_eq!(value["store"], false);
        assert_eq!(value["max_output_tokens"], 1024);
        assert_eq!(value["reasoning"]["effort"], "high");
        assert_eq!(value["text"]["format"]["type"], "json_schema");
        assert_eq!(value["text"]["format"]["strict"], true);
        assert_eq!(value["input"][1]["content"][0]["type"], "input_text");
        assert_eq!(value["input"][1]["content"][1]["type"], "input_image");
        assert!(value["input"][1]["content"][1]["image_url"].is_string());
    }

    #[test]
    fn reasoning_level_reaches_the_responses_request() {
        let request = TranslationRequest::new(
            ["hello"],
            crate::Language::English,
            crate::Language::Japanese,
        );
        let reasoning = |reasoning, reasoning_effort| {
            let generation = GenerationConfig {
                reasoning,
                reasoning_effort,
                ..GenerationConfig::default()
            };
            let body = request_body("grok-4.7", &generation, &request).unwrap();
            serde_json::to_value(body)
                .unwrap()
                .get("reasoning")
                .cloned()
        };

        assert_eq!(
            reasoning(Some(true), Some(ReasoningEffort::XHigh)),
            Some(serde_json::json!({ "effort": "xhigh" }))
        );
        assert_eq!(
            reasoning(Some(true), None),
            Some(serde_json::json!({ "effort": "high" }))
        );
        // Grok cannot turn reasoning off, so a disabled switch sends no level.
        assert_eq!(reasoning(Some(false), Some(ReasoningEffort::XHigh)), None);
    }

    #[test]
    fn reads_accepted_reasoning_levels_from_the_language_model_list() {
        use ReasoningEffort::{High, Low, Medium, XHigh};

        let response: ModelsResponse = serde_json::from_value(serde_json::json!({
            "models": [
                {
                    "id": "grok-4.7",
                    "input_modalities": ["text", "image"],
                    "capabilities": {
                        "reasoning_effort": ["low", "medium", "high", "xhigh"],
                        "default_reasoning_effort": "high"
                    }
                },
                { "id": "grok-4.5", "input_modalities": ["text"], "capabilities": {
                    "reasoning_effort": ["low", "medium", "high"]
                } },
                { "id": "grok-chat", "input_modalities": ["text"] }
            ]
        }))
        .unwrap();
        assert_eq!(
            response.models[0].reasoning_efforts(),
            [Low, Medium, High, XHigh]
        );
        assert_eq!(response.models[1].reasoning_efforts(), [Low, Medium, High]);
        assert!(response.models[2].reasoning_efforts().is_empty());
    }
}
