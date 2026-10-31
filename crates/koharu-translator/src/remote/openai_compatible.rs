// Request shape aligned with:
// https://github.com/koharu-rs/koharu/blob/f4ce03999ed1ae2faaec938dd52c2f41a87d03d9/crates/koharu-llm/src/providers/chat_completions.rs

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use koharu_usage::{RequestSettings, wire};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use super::{Inference, send_inference_with_transient_retry, send_json};
use crate::{
    GenerationConfig, Model, Provider, Result, TranslationRequest, backend::encode_image,
    display_name, prompt,
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct OpenAiCompatibleConfig {
    pub base_url: Option<Url>,
}

impl Default for OpenAiCompatibleConfig {
    fn default() -> Self {
        Self {
            base_url: Some(
                Url::parse(DEFAULT_BASE_URL).expect("default OpenAI-compatible URL is valid"),
            ),
        }
    }
}

pub(super) async fn compatible(
    client: &Client,
    config: &OpenAiCompatibleConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("openai-compatible")?;
    let endpoint = endpoint(config.base_url.as_ref(), "chat/completions");
    let backend = ChatBackend {
        // The model list of an arbitrary endpoint reports no capabilities and
        // servers differ in which levels they accept, so only the switch is sent.
        reasoning_effort: generation
            .reasoning
            .map(|enabled| if enabled { "medium" } else { "none" }),
        ..ChatBackend::new(
            "openai-compatible",
            &endpoint,
            api_key.as_ref().map(ExposeSecret::expose_secret),
            model,
            generation,
            ResponseMode::PromptOnly,
        )
    };
    translate(client, backend, request).await
}

pub(super) async fn translate(
    client: &Client,
    backend: ChatBackend<'_>,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let body = request_body(&backend, request)?;
    let http = client.post(backend.endpoint).json(&body);
    let http = match backend.api_key {
        Some(api_key) => http.bearer_auth(api_key),
        None => http,
    };
    let inference = Inference::new(request, settings(&backend), backend.report);
    let response: ChatResponse =
        send_inference_with_transient_retry(backend.provider, &inference, http).await?;
    let text = response
        .choices
        .into_iter()
        .next()
        .context("chat completion returned no choices")?
        .message
        .content;
    Ok(prompt::translations(
        backend.provider,
        &text,
        &request.segments,
    )?)
}

/// The generation settings `backend` puts on the wire, for the usage journal.
fn settings(backend: &ChatBackend<'_>) -> RequestSettings {
    let level = match backend.reasoning {
        Some(ReasoningControl::Level { effort }) => Some(effort),
        _ => None,
    };
    let reasoning = match (
        backend.reasoning,
        backend.thinking,
        backend.reasoning_effort,
    ) {
        (Some(ReasoningControl::Switch { enabled }), _, _) => Some(enabled),
        (Some(ReasoningControl::Level { .. }), _, _) => Some(true),
        (None, Some(kind), _) => Some(kind == "enabled"),
        (None, None, Some(effort)) => Some(effort != "none"),
        (None, None, None) => None,
    };
    RequestSettings {
        reasoning,
        reasoning_effort: backend.reasoning_effort.or(level).map(str::to_owned),
        temperature: backend.temperature,
        top_p: backend.top_p,
        top_k: None,
        max_tokens: backend.max_tokens.or(backend.max_completion_tokens),
        frequency_penalty: backend.frequency_penalty,
        presence_penalty: backend.presence_penalty,
    }
}

/// The chat completion a backend sends for `request`.
pub(super) fn request_body<'a>(
    backend: &ChatBackend<'a>,
    request: &TranslationRequest,
) -> Result<ChatRequest<'a>> {
    let (system, user) = prompt::prompts(request)?;
    let user_content = match request.image.as_deref() {
        Some(image) => MessageContent::Parts(vec![
            ContentPart::Text { text: user },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: encode_image(image)?.data_url(),
                },
            },
        ]),
        None => MessageContent::Text(user),
    };
    Ok(ChatRequest {
        model: backend.model,
        messages: [
            Message {
                role: "system",
                content: MessageContent::Text(system),
            },
            Message {
                role: "user",
                content: user_content,
            },
        ],
        temperature: backend.temperature,
        top_p: backend.top_p,
        max_tokens: backend.max_tokens,
        max_completion_tokens: backend.max_completion_tokens,
        frequency_penalty: backend.frequency_penalty,
        presence_penalty: backend.presence_penalty,
        reasoning_effort: backend.reasoning_effort,
        reasoning: backend.reasoning,
        thinking: backend.thinking.map(|kind| ThinkingConfig { kind }),
        response_format: backend
            .response_mode
            .response_format(request.segments.len()),
    })
}

pub(super) struct ChatBackend<'a> {
    pub(super) provider: &'static str,
    pub(super) endpoint: &'a str,
    pub(super) api_key: Option<&'a str>,
    pub(super) model: &'a str,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) max_tokens: Option<u32>,
    pub(super) max_completion_tokens: Option<u32>,
    pub(super) frequency_penalty: Option<f32>,
    pub(super) presence_penalty: Option<f32>,
    pub(super) reasoning_effort: Option<&'static str>,
    pub(super) reasoning: Option<ReasoningControl>,
    pub(super) thinking: Option<&'static str>,
    pub(super) response_mode: ResponseMode,
    /// Reads the provider's usage out of a response body.
    pub(super) report: fn(&Value) -> wire::Reported,
}

impl<'a> ChatBackend<'a> {
    pub(super) fn new(
        provider: &'static str,
        endpoint: &'a str,
        api_key: Option<&'a str>,
        model: &'a str,
        generation: &GenerationConfig,
        response_mode: ResponseMode,
    ) -> Self {
        Self {
            provider,
            endpoint,
            api_key,
            model,
            temperature: generation.temperature,
            top_p: generation.top_p,
            max_tokens: generation.max_tokens,
            max_completion_tokens: None,
            frequency_penalty: generation.frequency_penalty,
            presence_penalty: generation.presence_penalty,
            reasoning_effort: None,
            reasoning: None,
            thinking: None,
            response_mode,
            report: wire::openai_chat,
        }
    }
}

pub(super) async fn models(client: &Client, config: &OpenAiCompatibleConfig) -> Result<Vec<Model>> {
    let api_key = koharu_secrets::get("openai-compatible")?;
    let request = client.get(endpoint(config.base_url.as_ref(), "models"));
    let request = match api_key {
        Some(api_key) => request.bearer_auth(api_key.expose_secret()),
        None => request,
    };
    Ok(discover_models("openai-compatible", request)
        .await?
        .into_iter()
        .map(|model| Model {
            provider: Provider::OpenAiCompatible,
            name: display_name(&model),
            model: Some(model),
            quantizations: Vec::new(),
            vision: true,
            reasoning: true,
            reasoning_efforts: Vec::new(),
            reasoning_always_on: false,
        })
        .collect())
}

pub(super) async fn discover_models(
    provider: &'static str,
    request: reqwest::RequestBuilder,
) -> Result<Vec<String>> {
    let response: ModelsResponse = send_json(provider, request).await?;
    Ok(response.data.into_iter().map(|model| model.id).collect())
}

fn endpoint(base_url: Option<&Url>, suffix: &str) -> String {
    let base_url = base_url.map_or(DEFAULT_BASE_URL, Url::as_str);
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

#[derive(Serialize)]
pub(super) struct ChatRequest<'a> {
    model: &'a str,
    messages: [Message; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Clone, Copy, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    kind: &'static str,
}

/// The `reasoning` object of OpenRouter-style gateways: a switch, or a level,
/// which implies on.
#[derive(Clone, Copy, Serialize)]
#[serde(untagged)]
pub(super) enum ReasoningControl {
    Switch { enabled: bool },
    Level { effort: &'static str },
}

impl ReasoningControl {
    pub(super) fn from_generation(generation: &GenerationConfig) -> Option<Self> {
        generation
            .reasoning
            .map(|enabled| match generation.effort() {
                Some(effort) => Self::Level {
                    effort: effort.into(),
                },
                None => Self::Switch { enabled },
            })
    }
}

#[derive(Clone, Copy)]
pub(super) enum ResponseMode {
    PromptOnly,
    JsonObject,
    JsonSchema,
}

impl ResponseMode {
    fn response_format(self, expected: usize) -> Option<ResponseFormat> {
        match self {
            Self::PromptOnly => None,
            Self::JsonObject => Some(ResponseFormat {
                kind: "json_object",
                json_schema: None,
            }),
            Self::JsonSchema => Some(ResponseFormat {
                kind: "json_schema",
                json_schema: Some(JsonSchema {
                    name: "manga_translation",
                    strict: true,
                    schema: prompt::output_schema(expected),
                }),
            }),
        }
    }
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<JsonSchema>,
}

#[derive(Serialize)]
struct JsonSchema {
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: MessageContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
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
    use super::*;

    #[test]
    fn endpoint_preserves_base_path() {
        let url = Url::parse("http://localhost:1234/v1").unwrap();
        assert_eq!(
            endpoint(Some(&url), "models"),
            "http://localhost:1234/v1/models"
        );
    }

    #[test]
    fn serializes_provider_specific_response_formats() {
        assert_eq!(
            serde_json::to_value(ResponseMode::JsonObject.response_format(2).unwrap()).unwrap(),
            serde_json::json!({ "type": "json_object" })
        );

        let strict =
            serde_json::to_value(ResponseMode::JsonSchema.response_format(2).unwrap()).unwrap();
        assert_eq!(strict["type"], "json_schema");
        assert_eq!(strict["json_schema"]["name"], "manga_translation");
        assert_eq!(strict["json_schema"]["strict"], true);
        assert_eq!(
            strict["json_schema"]["schema"]["properties"]["translations"]["items"]["properties"]["id"]
                ["maximum"],
            1
        );
        assert!(ResponseMode::PromptOnly.response_format(2).is_none());
    }

    #[test]
    fn serializes_current_completion_fields() {
        let request = ChatRequest {
            model: "gpt-5.6-luna",
            messages: [
                Message {
                    role: "system",
                    content: MessageContent::Text("system".to_owned()),
                },
                Message {
                    role: "user",
                    content: MessageContent::Text("user".to_owned()),
                },
            ],
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: Some(1024),
            frequency_penalty: None,
            presence_penalty: None,
            reasoning_effort: Some("none"),
            reasoning: None,
            thinking: None,
            response_format: None,
        };
        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["max_completion_tokens"], 1024);
        assert_eq!(value["reasoning_effort"], "none");
        assert!(value.get("max_tokens").is_none());
        assert!(value.get("thinking").is_none());
    }

    #[test]
    fn gateway_reasoning_control_is_a_switch_or_a_level() {
        let control = |reasoning, reasoning_effort| {
            let generation = GenerationConfig {
                reasoning,
                reasoning_effort,
                ..GenerationConfig::default()
            };
            serde_json::to_value(ReasoningControl::from_generation(&generation)).unwrap()
        };
        assert_eq!(control(None, None), serde_json::Value::Null);
        assert_eq!(
            control(Some(false), None),
            serde_json::json!({ "enabled": false })
        );
        assert_eq!(
            control(Some(true), None),
            serde_json::json!({ "enabled": true })
        );
        assert_eq!(
            control(Some(true), Some(crate::ReasoningEffort::High)),
            serde_json::json!({ "effort": "high" })
        );
        // A stored level is not sent while reasoning is off.
        assert_eq!(
            control(Some(false), Some(crate::ReasoningEffort::High)),
            serde_json::json!({ "enabled": false })
        );
    }

    #[test]
    fn serializes_text_before_an_attached_image() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "translate".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,image".to_owned(),
                },
            },
        ]);
        let value = serde_json::to_value(content).unwrap();
        assert_eq!(
            value[0],
            serde_json::json!({ "type": "text", "text": "translate" })
        );
        assert_eq!(value[1]["type"], "image_url");
        assert_eq!(value[1]["image_url"]["url"], "data:image/jpeg;base64,image");
    }
}
