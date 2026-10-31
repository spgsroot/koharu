// Ported from:
// https://github.com/koharu-rs/koharu/blob/f4ce03999ed1ae2faaec938dd52c2f41a87d03d9/crates/koharu-llm/src/providers/claude.rs
// Model discovery:
// https://platform.claude.com/docs/en/api/models/list

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use koharu_usage::{RequestSettings, wire};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{Inference, send_inference, send_json};
use crate::{
    GenerationConfig, Model, Provider, ReasoningEffort, Result, TranslationRequest,
    backend::encode_image, prompt,
};

const URL: &str = "https://api.anthropic.com/v1/messages";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models?limit=1000";

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct ClaudeConfig {}

pub(super) async fn models(client: &Client) -> Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("claude")? else {
        return Ok(Vec::new());
    };
    let response: ModelsResponse = send_json(
        "claude",
        client
            .get(MODELS_URL)
            .header("x-api-key", api_key.expose_secret())
            .header("anthropic-version", "2023-06-01"),
    )
    .await?;
    Ok(response
        .data
        .into_iter()
        .map(|model| {
            let reasoning = model.capabilities.thinking.types.adaptive.supported;
            Model {
                provider: Provider::Claude,
                // Effort applies on top of adaptive thinking, the only thinking
                // mode this adapter sends.
                reasoning_efforts: if reasoning {
                    model.capabilities.effort.levels()
                } else {
                    Vec::new()
                },
                model: Some(model.id),
                name: model.display_name,
                quantizations: Vec::new(),
                vision: model.capabilities.image_input.supported,
                reasoning,
                reasoning_always_on: false,
            }
        })
        .collect())
}

pub(super) async fn translate(
    client: &Client,
    _config: &ClaudeConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let api_key = koharu_secrets::get("claude")?.context("claude API key is not configured")?;
    let (system, user) = prompt::prompts(request)?;
    let body = request_body(model, generation, &system, &user, request.image.as_deref())?;
    let inference = Inference::new(request, settings(&body), wire::anthropic);
    let response: Response = send_inference(
        "claude",
        &inference,
        client
            .post(URL)
            .header("x-api-key", api_key.expose_secret())
            .header("anthropic-version", "2023-06-01")
            .json(&body),
    )
    .await?;
    let text = response
        .content
        .into_iter()
        .find_map(|block| (block.kind == "text").then_some(block.text).flatten())
        .context("Claude returned no text content")?;
    Ok(prompt::translations("claude", &text, &request.segments)?)
}

/// The generation settings `body` puts on the wire, for the usage journal.
fn settings(body: &Request<'_>) -> RequestSettings {
    RequestSettings {
        reasoning: body
            .thinking
            .as_ref()
            .map(|thinking| thinking.kind != "disabled"),
        reasoning_effort: body
            .output_config
            .as_ref()
            .map(|output| output.effort.to_owned()),
        temperature: body.temperature,
        max_tokens: Some(body.max_tokens),
        ..RequestSettings::default()
    }
}

fn request_body<'a>(
    model: &'a str,
    generation: &GenerationConfig,
    system: &'a str,
    user: &'a str,
    image: Option<&image::DynamicImage>,
) -> anyhow::Result<Request<'a>> {
    Ok(Request {
        model,
        max_tokens: generation.max_tokens.unwrap_or(8192),
        system,
        messages: [Message::user(user, image)?],
        temperature: generation.temperature,
        thinking: generation.reasoning.map(|enabled| ThinkingConfig {
            kind: if enabled { "adaptive" } else { "disabled" },
        }),
        output_config: generation.effort().map(|effort| OutputConfig {
            effort: effort.into(),
        }),
    })
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: [Message<'a>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct OutputConfig {
    effort: &'static str,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: Vec<ContentBlock<'a>>,
}

impl<'a> Message<'a> {
    fn user(text: &'a str, image: Option<&image::DynamicImage>) -> anyhow::Result<Self> {
        let mut content = vec![ContentBlock::Text { text }];
        if let Some(image) = image {
            content.push(ContentBlock::Image {
                source: ImageSource {
                    kind: "base64",
                    media_type: "image/jpeg",
                    data: encode_image(image)?.data,
                },
            });
        }
        Ok(Self {
            role: "user",
            content,
        })
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock<'a> {
    Text { text: &'a str },
    Image { source: ImageSource },
}

#[derive(Serialize)]
struct ImageSource {
    #[serde(rename = "type")]
    kind: &'static str,
    media_type: &'static str,
    data: String,
}

#[derive(Deserialize)]
struct Response {
    content: Vec<Content>,
}

#[derive(Deserialize)]
struct Content {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
    display_name: String,
    capabilities: ModelCapabilities,
}

#[derive(Deserialize)]
struct ModelCapabilities {
    image_input: CapabilitySupport,
    thinking: ThinkingCapability,
    #[serde(default)]
    effort: EffortCapability,
}

#[derive(Deserialize)]
struct ThinkingCapability {
    types: ThinkingTypes,
}

#[derive(Deserialize)]
struct ThinkingTypes {
    adaptive: CapabilitySupport,
}

#[derive(Default, Deserialize)]
struct CapabilitySupport {
    supported: bool,
}

/// `capabilities.effort` of the model list: which `output_config.effort`
/// values a model accepts. `xhigh` is `null` on models that predate it.
#[derive(Default, Deserialize)]
#[serde(default)]
struct EffortCapability {
    supported: bool,
    low: CapabilitySupport,
    medium: CapabilitySupport,
    high: CapabilitySupport,
    xhigh: Option<CapabilitySupport>,
    max: CapabilitySupport,
}

impl EffortCapability {
    fn levels(&self) -> Vec<ReasoningEffort> {
        let accepts = |support: &CapabilitySupport| self.supported && support.supported;
        [
            (ReasoningEffort::Low, accepts(&self.low)),
            (ReasoningEffort::Medium, accepts(&self.medium)),
            (ReasoningEffort::High, accepts(&self.high)),
            (
                ReasoningEffort::XHigh,
                self.xhigh.as_ref().is_some_and(accepts),
            ),
            (ReasoningEffort::Max, accepts(&self.max)),
        ]
        .into_iter()
        .filter_map(|(level, accepted)| accepted.then_some(level))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::ReasoningEffort::{High, Low, Max, Medium, XHigh};
    use super::*;

    #[test]
    fn serializes_text_and_base64_image_blocks() {
        let message =
            Message::user("translate", Some(&image::DynamicImage::new_rgb8(1, 1))).unwrap();
        let value = serde_json::to_value(message).unwrap();
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][1]["type"], "image");
        assert_eq!(value["content"][1]["source"]["type"], "base64");
        assert_eq!(value["content"][1]["source"]["media_type"], "image/jpeg");
    }

    #[test]
    fn reads_thinking_and_effort_capabilities_from_model_list() {
        let response: ModelsResponse = serde_json::from_value(serde_json::json!({
            "data": [
                {
                    "id": "claude-opus-4-8",
                    "display_name": "Claude Opus 4.8",
                    "capabilities": {
                        "image_input": { "supported": true },
                        "thinking": { "types": { "adaptive": { "supported": true } } },
                        "effort": {
                            "supported": true,
                            "low": { "supported": true },
                            "medium": { "supported": true },
                            "high": { "supported": true },
                            "xhigh": { "supported": true },
                            "max": { "supported": true }
                        }
                    }
                },
                {
                    "id": "claude-sonnet-4-6",
                    "display_name": "Claude Sonnet 4.6",
                    "capabilities": {
                        "image_input": { "supported": true },
                        "thinking": { "types": { "adaptive": { "supported": true } } },
                        "effort": {
                            "supported": true,
                            "low": { "supported": true },
                            "medium": { "supported": true },
                            "high": { "supported": true },
                            "xhigh": null,
                            "max": { "supported": true }
                        }
                    }
                },
                {
                    "id": "claude-haiku-4-5",
                    "display_name": "Claude Haiku 4.5",
                    "capabilities": {
                        "image_input": { "supported": true },
                        "thinking": { "types": { "adaptive": { "supported": false } } },
                        "effort": {
                            "supported": false,
                            "low": { "supported": false },
                            "medium": { "supported": false },
                            "high": { "supported": false },
                            "xhigh": null,
                            "max": { "supported": false }
                        }
                    }
                }
            ]
        }))
        .unwrap();
        let [opus, sonnet, haiku] = &response.data[..] else {
            panic!("expected three models");
        };
        assert!(opus.capabilities.thinking.types.adaptive.supported);
        assert_eq!(
            opus.capabilities.effort.levels(),
            [Low, Medium, High, XHigh, Max]
        );
        assert_eq!(
            sonnet.capabilities.effort.levels(),
            [Low, Medium, High, Max]
        );
        assert!(haiku.capabilities.effort.levels().is_empty());
    }

    #[test]
    fn reasoning_level_reaches_the_messages_request() {
        let payload = |reasoning, reasoning_effort| {
            let generation = GenerationConfig {
                reasoning,
                reasoning_effort,
                ..GenerationConfig::default()
            };
            let body = request_body("claude-opus-4-8", &generation, "system", "user", None);
            serde_json::to_value(body.unwrap()).unwrap()
        };

        let chosen = payload(Some(true), Some(XHigh));
        assert_eq!(chosen["thinking"]["type"], "adaptive");
        assert_eq!(chosen["output_config"]["effort"], "xhigh");

        let default = payload(Some(true), None);
        assert_eq!(default["thinking"]["type"], "adaptive");
        assert!(default.get("output_config").is_none());

        // Turning thinking off omits the stored level.
        let off = payload(Some(false), Some(XHigh));
        assert_eq!(off["thinking"]["type"], "disabled");
        assert!(off.get("output_config").is_none());
    }
}
