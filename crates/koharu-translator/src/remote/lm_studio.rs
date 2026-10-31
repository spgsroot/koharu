// https://lmstudio.ai/docs/developer/rest/chat
// https://lmstudio.ai/docs/developer/rest/list

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use koharu_usage::{RequestSettings, wire};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use url::Url;

use super::{Inference, send_inference, send_json};
use crate::{
    GenerationConfig, Model, Provider, ReasoningEffort, Result, TranslationRequest,
    backend::encode_image, display_name, prompt,
};

const DEFAULT_BASE_URL: &str = "http://localhost:1234";

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct LmStudioConfig {
    pub base_url: Option<Url>,
}

impl Default for LmStudioConfig {
    fn default() -> Self {
        Self {
            base_url: Some(Url::parse(DEFAULT_BASE_URL).expect("default LM Studio URL is valid")),
        }
    }
}

pub(super) async fn translate(
    client: &Client,
    config: &LmStudioConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("lm-studio")?;
    let (system, input) = prompt::prompts(request)?;
    let body = request_body(model, generation, &system, &input, request.image.as_deref())?;
    let http = client.post(endpoint(config.base_url.as_ref(), "chat"));
    let http = match api_key {
        Some(api_key) => http.bearer_auth(api_key.expose_secret()),
        None => http,
    };
    let inference = Inference::new(request, settings(&body), wire::lm_studio);
    let response: ChatResponse = send_inference("lm-studio", &inference, http.json(&body)).await?;
    let text = response
        .output
        .into_iter()
        .rev()
        .find_map(|output| {
            (output.kind == "message")
                .then_some(output.content)
                .flatten()
        })
        .context("LM Studio returned no message output")?;
    Ok(prompt::translations("lm-studio", &text, &request.segments)?)
}

/// The generation settings `body` puts on the wire, for the usage journal.
/// `on` and `off` are the switch; any other value is a named level.
fn settings(body: &ChatRequest<'_>) -> RequestSettings {
    RequestSettings {
        reasoning: body.reasoning.map(|setting| setting != "off"),
        reasoning_effort: body
            .reasoning
            .filter(|setting| !matches!(*setting, "on" | "off"))
            .map(str::to_owned),
        temperature: body.temperature,
        max_tokens: body.max_output_tokens,
        ..RequestSettings::default()
    }
}

pub(super) async fn models(client: &Client, config: &LmStudioConfig) -> Result<Vec<Model>> {
    let api_key = koharu_secrets::get("lm-studio")?;
    let request = client.get(endpoint(config.base_url.as_ref(), "models"));
    let request = match api_key {
        Some(api_key) => request.bearer_auth(api_key.expose_secret()),
        None => request,
    };
    let response: ModelsResponse = send_json("lm-studio", request).await?;
    Ok(catalog(response))
}

fn catalog(response: ModelsResponse) -> Vec<Model> {
    response
        .models
        .into_iter()
        .filter(|model| matches!(model.kind.as_str(), "llm" | "vlm"))
        .map(|model| {
            let capabilities = model.capabilities;
            let reasoning = capabilities
                .as_ref()
                .and_then(|capabilities| capabilities.reasoning.as_ref());
            Model {
                provider: Provider::LmStudio,
                name: display_name(&model.key),
                model: Some(model.key),
                quantizations: Vec::new(),
                vision: capabilities
                    .as_ref()
                    .is_some_and(|capabilities| capabilities.vision),
                reasoning: reasoning.is_some(),
                // `allowed_options` mixes the `off`/`on` switch with named levels.
                reasoning_efforts: reasoning.map_or_else(Vec::new, |reasoning| {
                    ReasoningEffort::from_reported(
                        reasoning.allowed_options.iter().map(String::as_str),
                    )
                }),
                reasoning_always_on: false,
            }
        })
        .collect()
}

fn request_body<'a>(
    model: &'a str,
    generation: &GenerationConfig,
    system: &'a str,
    input: &'a str,
    image: Option<&image::DynamicImage>,
) -> anyhow::Result<ChatRequest<'a>> {
    Ok(ChatRequest {
        model,
        input: ChatInput::new(input, image)?,
        system_prompt: system,
        temperature: generation.temperature,
        max_output_tokens: generation.max_tokens,
        reasoning: reasoning_setting(generation),
        store: false,
    })
}

/// `on` and `off` switch reasoning; a chosen level selects one of the named
/// settings the model allows.
fn reasoning_setting(generation: &GenerationConfig) -> Option<&'static str> {
    generation
        .reasoning
        .map(|enabled| match (enabled, generation.effort()) {
            (false, _) => "off",
            (true, Some(effort)) => effort.into(),
            (true, None) => "on",
        })
}

fn endpoint(base_url: Option<&Url>, suffix: &str) -> String {
    let base_url = base_url.map_or(DEFAULT_BASE_URL, Url::as_str);
    format!(
        "{}/api/v1/{}",
        base_url.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    input: ChatInput<'a>,
    system_prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'static str>,
    store: bool,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ChatInput<'a> {
    Text(&'a str),
    Items(Vec<InputItem<'a>>),
}

impl<'a> ChatInput<'a> {
    fn new(text: &'a str, image: Option<&image::DynamicImage>) -> anyhow::Result<Self> {
        let Some(image) = image else {
            return Ok(Self::Text(text));
        };
        Ok(Self::Items(vec![
            InputItem::Text { content: text },
            InputItem::Image {
                data_url: encode_image(image)?.data_url(),
            },
        ]))
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputItem<'a> {
    Text { content: &'a str },
    Image { data_url: String },
}

#[derive(Deserialize)]
struct ChatResponse {
    output: Vec<Output>,
}

#[derive(Deserialize)]
struct Output {
    #[serde(rename = "type")]
    kind: String,
    content: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    models: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    #[serde(rename = "type")]
    kind: String,
    key: String,
    capabilities: Option<ListedModelCapabilities>,
}

#[derive(Deserialize)]
struct ListedModelCapabilities {
    vision: bool,
    reasoning: Option<ListedModelReasoning>,
}

#[derive(Deserialize)]
struct ListedModelReasoning {
    #[serde(default)]
    allowed_options: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_native_v1_endpoints() {
        let base_url = Url::parse("http://localhost:1234/").unwrap();
        assert_eq!(
            endpoint(Some(&base_url), "chat"),
            "http://localhost:1234/api/v1/chat"
        );
        assert_eq!(
            endpoint(Some(&base_url), "models"),
            "http://localhost:1234/api/v1/models"
        );
    }

    #[test]
    fn serializes_native_chat_options() {
        let value = serde_json::to_value(ChatRequest {
            model: "publisher/model",
            input: ChatInput::Text("input"),
            system_prompt: "system",
            temperature: Some(0.2),
            max_output_tokens: Some(1024),
            reasoning: None,
            store: false,
        })
        .unwrap();
        assert_eq!(value["max_output_tokens"], 1024);
        assert!(value.get("reasoning").is_none());
        assert_eq!(value["store"], false);
        assert!(value.get("messages").is_none());
    }

    #[test]
    fn reasoning_level_reaches_the_native_chat_request() {
        let payload = |reasoning, reasoning_effort| {
            let generation = GenerationConfig {
                reasoning,
                reasoning_effort,
                ..GenerationConfig::default()
            };
            let body = request_body("publisher/model", &generation, "system", "input", None);
            serde_json::to_value(body.unwrap()).unwrap()
        };

        assert_eq!(
            payload(Some(true), Some(ReasoningEffort::High))["reasoning"],
            "high"
        );
        assert_eq!(payload(Some(true), None)["reasoning"], "on");
        // Turning reasoning off omits the stored level.
        assert_eq!(
            payload(Some(false), Some(ReasoningEffort::High))["reasoning"],
            "off"
        );
        assert!(payload(None, None).get("reasoning").is_none());
    }

    #[test]
    fn model_discovery_keeps_language_and_vision_models() {
        use ReasoningEffort::{High, Low, Medium};

        let response: ModelsResponse = serde_json::from_value(serde_json::json!({
            "models": [
                {
                    "type": "llm",
                    "key": "publisher/chat-model",
                    "capabilities": { "vision": false }
                },
                {
                    "type": "llm",
                    "key": "publisher/vision-model",
                    "capabilities": {
                        "vision": true,
                        "reasoning": {
                            "allowed_options": ["off", "on"],
                            "default": "on"
                        }
                    }
                },
                {
                    "type": "llm",
                    "key": "publisher/leveled-model",
                    "capabilities": {
                        "vision": false,
                        "reasoning": {
                            "allowed_options": ["low", "medium", "high"],
                            "default": "medium"
                        }
                    }
                },
                {
                    "type": "embedding",
                    "key": "publisher/embed-model"
                }
            ]
        }))
        .unwrap();
        let models = catalog(response)
            .into_iter()
            .map(|model| {
                (
                    model.model.unwrap(),
                    model.vision,
                    model.reasoning,
                    model.reasoning_efforts,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            models,
            [
                ("publisher/chat-model".to_owned(), false, false, vec![]),
                ("publisher/vision-model".to_owned(), true, true, vec![]),
                (
                    "publisher/leveled-model".to_owned(),
                    false,
                    true,
                    vec![Low, Medium, High]
                )
            ]
        );
    }

    #[test]
    fn serializes_native_image_input() {
        let input =
            ChatInput::new("translate", Some(&image::DynamicImage::new_rgb8(1, 1))).unwrap();
        let value = serde_json::to_value(input).unwrap();
        assert_eq!(
            value[0],
            serde_json::json!({ "type": "text", "content": "translate" })
        );
        assert_eq!(value[1]["type"], "image");
        assert!(
            value[1]["data_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
    }
}
