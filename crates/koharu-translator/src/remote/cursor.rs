// https://cursor.com/docs/cloud-agent/api/endpoints
// A translation is a no-repository agent: no pull request, no workspace edits.
// Plan mode asks for a text reply. The agent is deleted after the run ends.

use std::time::{Duration, Instant};

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use koharu_usage::{RequestSettings, UsageOutcome};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    GenerationConfig, Model, Provider, TranslationRequest,
    backend::encode_image,
    prompt,
    remote::{failure_outcome, send_json},
    usage::InFlight,
};

const API: &str = "https://api.cursor.com/v1";
const POLL_INTERVAL: Duration = Duration::from_secs(5);
const RUN_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_IMAGE_BYTES: usize = 15 * 1024 * 1024;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct CursorConfig {}

pub(super) async fn translate(
    client: &Client,
    _config: &CursorConfig,
    model: &str,
    _generation: &GenerationConfig,
    request: &TranslationRequest,
) -> crate::Result<Vec<String>> {
    let api_key = koharu_secrets::get("cursor")?.context("cursor API key is not configured")?;
    let key = api_key.expose_secret();
    let body = create_body(model, request)?;
    // One agent run is one request: its polls and the cleanup are not journaled.
    let mut flight = request
        .usage
        .as_ref()
        .map(|trace| InFlight::start(trace, RequestSettings::default(), 0));
    let created: Created = match send_json(
        "cursor",
        client
            .post(format!("{API}/agents"))
            .bearer_auth(key)
            .json(&body),
    )
    .await
    {
        Ok(created) => created,
        Err(error) => {
            if let Some(flight) = flight {
                flight.finish(failure_outcome(&error));
            }
            return Err(error);
        }
    };
    if let Some(flight) = flight.as_mut() {
        let record = flight.record();
        record.provider_request_id = Some(created.agent.id.clone());
        record.provider_response_id = Some(created.run.id.clone());
    }
    let result = wait_for_result(client, key, &created.agent.id, &created.run.id).await;
    if let Some(flight) = flight {
        flight.finish(match &result {
            Ok(_) => UsageOutcome::Succeeded,
            Err(error) => run_outcome(error),
        });
    }
    delete_agent(client, key, &created.agent.id).await;
    let text = result?;
    Ok(prompt::translations("cursor", &text, &request.segments)?)
}

pub(super) async fn models(client: &Client) -> crate::Result<Vec<Model>> {
    let Some(api_key) = koharu_secrets::get("cursor")? else {
        return Ok(Vec::new());
    };
    let response: ModelsResponse = send_json(
        "cursor",
        client
            .get(format!("{API}/models"))
            .bearer_auth(api_key.expose_secret()),
    )
    .await?;
    Ok(response
        .items
        .into_iter()
        .map(|model| {
            // A thinking variant always thinks, and the agent API has no switch.
            let thinking = model.id.to_ascii_lowercase().contains("thinking");
            Model {
                provider: Provider::Cursor,
                name: if model.display_name.is_empty() {
                    model.id.clone()
                } else {
                    model.display_name
                },
                reasoning: thinking,
                reasoning_efforts: Vec::new(),
                reasoning_always_on: thinking,
                model: Some(model.id),
                quantizations: Vec::new(),
                vision: true,
            }
        })
        .collect())
}

fn create_body(model: &str, request: &TranslationRequest) -> anyhow::Result<CreateAgent> {
    let (system, user) = prompt::prompts(request)?;
    let mut images = Vec::new();
    if let Some(image) = request.image.as_deref() {
        let encoded = encode_image(image)?;
        let decoded = encoded.data.len().saturating_mul(3) / 4;
        if decoded > MAX_IMAGE_BYTES {
            anyhow::bail!("cursor accepts images up to 15 MB");
        }
        images.push(PromptImage {
            data: encoded.data,
            mime_type: "image/jpeg",
        });
    }
    Ok(CreateAgent {
        name: "Koharu translation",
        mode: "plan",
        model: SelectedModel {
            id: model.to_owned(),
        },
        prompt: Prompt {
            text: format!(
                "{system}\n\n{user}\n\nReply with only the JSON object. Do not use tools, edit files, or add commentary."
            ),
            images,
        },
    })
}

async fn wait_for_result(
    client: &Client,
    api_key: &str,
    agent_id: &str,
    run_id: &str,
) -> anyhow::Result<String> {
    let url = format!("{API}/agents/{agent_id}/runs/{run_id}");
    let deadline = Instant::now() + RUN_TIMEOUT;
    loop {
        if Instant::now() >= deadline {
            cancel_run(client, api_key, agent_id, run_id).await;
            return Err(RunEnded::TimedOut.into());
        }
        let run: Run = send_json("cursor", client.get(&url).bearer_auth(api_key)).await?;
        match run.status.to_ascii_uppercase().as_str() {
            "CREATING" | "RUNNING" | "WAITING_FOR_BACKGROUND_WORK" => {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            "FINISHED" => {
                return run.result.context("cursor returned no output text");
            }
            "ERROR" | "CANCELLED" | "EXPIRED" => {
                return Err(RunEnded::Status(run.status).into());
            }
            _ => tokio::time::sleep(POLL_INTERVAL).await,
        }
    }
}

/// How a Cursor run ended without producing a reply.
#[derive(Debug, thiserror::Error)]
enum RunEnded {
    #[error("cursor agent run timed out")]
    TimedOut,
    #[error("cursor agent run ended with status {0}")]
    Status(String),
}

fn run_outcome(error: &anyhow::Error) -> UsageOutcome {
    if let Some(ended) = error.downcast_ref::<RunEnded>() {
        return match ended {
            RunEnded::TimedOut => UsageOutcome::Timeout,
            RunEnded::Status(status) if status.eq_ignore_ascii_case("CANCELLED") => {
                UsageOutcome::Cancelled
            }
            RunEnded::Status(status) if status.eq_ignore_ascii_case("EXPIRED") => {
                UsageOutcome::Timeout
            }
            RunEnded::Status(_) => UsageOutcome::ProviderError,
        };
    }
    match error.downcast_ref::<crate::Error>() {
        Some(error) => failure_outcome(error),
        // A finished run without a reply.
        None => UsageOutcome::InvalidOutput,
    }
}

async fn cancel_run(client: &Client, api_key: &str, agent_id: &str, run_id: &str) {
    let result = client
        .post(format!("{API}/agents/{agent_id}/runs/{run_id}/cancel"))
        .bearer_auth(api_key)
        .send()
        .await;
    if let Err(error) = result {
        tracing::warn!(%error, "failed to cancel cursor agent run");
    }
}

async fn delete_agent(client: &Client, api_key: &str, agent_id: &str) {
    match client
        .delete(format!("{API}/agents/{agent_id}"))
        .bearer_auth(api_key)
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {}
        Ok(response) => {
            tracing::warn!(status = %response.status(), "failed to delete cursor agent")
        }
        Err(error) => tracing::warn!(%error, "failed to delete cursor agent"),
    }
}
#[derive(Serialize)]
struct CreateAgent {
    name: &'static str,
    mode: &'static str,
    model: SelectedModel,
    prompt: Prompt,
}

#[derive(Serialize)]
struct SelectedModel {
    id: String,
}

#[derive(Serialize)]
struct Prompt {
    text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<PromptImage>,
}

#[derive(Serialize)]
struct PromptImage {
    data: String,
    #[serde(rename = "mimeType")]
    mime_type: &'static str,
}

#[derive(Deserialize)]
struct Created {
    agent: CreatedAgent,
    run: CreatedRun,
}

#[derive(Deserialize)]
struct CreatedAgent {
    id: String,
}

#[derive(Deserialize)]
struct CreatedRun {
    id: String,
}

#[derive(Deserialize)]
struct Run {
    status: String,
    result: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    items: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
    #[serde(rename = "displayName", default)]
    display_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_no_repo_plan_with_page_image() {
        let request = TranslationRequest::new(
            ["hello"],
            crate::Language::English,
            crate::Language::Japanese,
        )
        .with_image(std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)));
        let body = create_body("composer-2", &request).unwrap();
        let value = serde_json::to_value(body).unwrap();
        assert_eq!(value["mode"], "plan");
        assert_eq!(value["model"]["id"], "composer-2");
        assert!(value.get("repos").is_none());
        assert!(value.get("autoCreatePR").is_none());
        assert_eq!(value["prompt"]["images"][0]["mimeType"], "image/jpeg");
        assert!(
            value["prompt"]["text"]
                .as_str()
                .unwrap()
                .contains("Do not use tools")
        );
    }
}
