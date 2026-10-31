//! Kimi Code subscription backend: RFC 8628 OAuth plus Chat Completions.

mod auth;
mod protocol;
mod stream;

use std::{collections::HashMap, fmt};

use anyhow::{Context as _, Result, bail};
use reqwest::{Client, StatusCode};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::{Control, LoginEvent, Reasoning, codex::Request};
use auth::Auth;

const BASE_URL: &str = "https://api.kimi.com/coding/v1";

pub struct Kimi {
    client: Client,
    auth: Auth,
    models: Mutex<HashMap<String, ModelCapabilities>>,
}

#[derive(Clone, Debug, Default)]
struct ModelCapabilities {
    adjustable_effort: bool,
}

impl Kimi {
    pub fn new() -> Result<Self> {
        let client = koharu_runtime::http_client()?;
        Ok(Self {
            auth: Auth::new(client.clone()),
            client,
            models: Mutex::new(HashMap::new()),
        })
    }

    pub fn account(&self) -> Result<Option<crate::Account>> {
        self.auth.account()
    }

    pub async fn login_device<F>(&self, control: &Control, publish: F) -> Result<crate::Account>
    where
        F: FnMut(LoginEvent),
    {
        self.auth.login_device(control, publish).await
    }

    pub fn logout(&self) -> Result<()> {
        self.auth.logout()
    }

    pub async fn models(&self) -> Result<Vec<crate::CodexModel>> {
        let session = self.auth.session().await?;
        let response = self
            .client
            .get(format!("{BASE_URL}/models"))
            .bearer_auth(&session.access)
            .send()
            .await
            .context("Kimi Code model discovery failed")?;
        let response = if response.status() == StatusCode::UNAUTHORIZED {
            let session = self.auth.force_refresh().await?;
            self.client
                .get(format!("{BASE_URL}/models"))
                .bearer_auth(&session.access)
                .send()
                .await?
        } else {
            response
        };
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!(
                "Kimi Code model discovery returned {status}: {}",
                &body[..body.len().min(4096)]
            );
        }
        let value: Value =
            serde_json::from_str(&body).context("invalid Kimi Code model catalog")?;
        let rows = value
            .get("data")
            .and_then(Value::as_array)
            .context("Kimi Code model catalog must contain a data array")?;
        let mut models = Vec::new();
        let mut capabilities = HashMap::new();
        for row in rows {
            let Some(id) = row
                .get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
            else {
                continue;
            };
            let (reasoning, adjustable) = model_reasoning(row);
            capabilities.insert(
                id.to_owned(),
                ModelCapabilities {
                    adjustable_effort: adjustable,
                },
            );
            models.push(crate::CodexModel {
                name: row
                    .get("display_name")
                    .or_else(|| row.get("displayName"))
                    .and_then(Value::as_str)
                    .unwrap_or(id)
                    .to_owned(),
                id: id.to_owned(),
                reasoning,
            });
        }
        if models.is_empty() {
            bail!("the signed-in Kimi Code account has no available models");
        }
        *self.models.lock().await = capabilities;
        Ok(models)
    }

    pub(crate) async fn respond<F>(
        &self,
        mut request: Request,
        control: &Control,
        publish: F,
    ) -> Result<crate::codex::Turn>
    where
        F: FnMut(crate::codex::Delta),
    {
        control.ensure_running()?;
        if !self
            .models
            .lock()
            .await
            .get(&request.model)
            .is_some_and(|capabilities| capabilities.adjustable_effort)
        {
            request = request.without_reasoning();
        }
        let body = protocol::ChatRequest::new(&request)?;
        let mut session = self.auth.session().await?;
        let mut usage = crate::usage::Inference::new(
            request
                .usage
                .take()
                .context("agent request usage context is missing")?,
        );
        let mut response = usage.sent(self.send(&body, &session, control).await, control)?;
        if response.status() == StatusCode::UNAUTHORIZED {
            usage.entry.outcome = koharu_usage::UsageOutcome::HttpError;
            let retry = usage.retry_entry();
            drop(usage);
            session = self.auth.force_refresh().await?;
            usage = crate::usage::Inference::new(retry);
            response = usage.sent(self.send(&body, &session, control).await, control)?;
        }
        if !response.status().is_success() {
            usage.entry.outcome = if response.status() == StatusCode::TOO_MANY_REQUESTS {
                koharu_usage::UsageOutcome::QuotaExceeded
            } else {
                koharu_usage::UsageOutcome::HttpError
            };
            let status = response.status();
            let mut body = response.text().await.unwrap_or_default();
            if let Ok(value) = serde_json::from_str(&body) {
                usage.entry.report(koharu_usage::wire::openai_chat(&value));
                usage.entry.error_code = koharu_usage::wire::error_code(&value);
            }
            body.truncate(16 * 1024);
            bail!("Kimi Code returned {status}: {body}");
        }
        let result = stream::read(response, control, publish, &mut usage).await;
        if result.is_ok() {
            usage.entry.outcome = koharu_usage::UsageOutcome::Succeeded;
        } else if control.is_cancelled() {
            usage.entry.outcome = koharu_usage::UsageOutcome::Cancelled;
        }
        result
    }

    async fn send(
        &self,
        body: &protocol::ChatRequest,
        session: &auth::Session,
        control: &Control,
    ) -> Result<reqwest::Response> {
        let send = self
            .client
            .post(format!("{BASE_URL}/chat/completions"))
            .bearer_auth(&session.access)
            .header("accept", "text/event-stream")
            .json(body)
            .send();
        tokio::select! {
            response = send => Ok(response?),
            () = control.cancelled() => {
                control.ensure_running()?;
                unreachable!("cancelled control must fail ensure_running")
            }
        }
    }
}

impl fmt::Debug for Kimi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Kimi").finish_non_exhaustive()
    }
}

/// Reads the upstream `/models` reasoning declaration. `adjustable` is false
/// for non-thinking models and for fixed-effort thinking models, so those
/// requests never carry a `reasoning_effort` the server would reject.
fn model_reasoning(row: &Value) -> (Vec<Reasoning>, bool) {
    let reasoning = row.get("reasoning");
    let kind = reasoning
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let legacy_efforts = row
        .get("think_efforts")
        .and_then(|value| {
            if value.get("support").and_then(Value::as_bool) == Some(true) {
                value.get("valid_efforts")
            } else {
                None
            }
        })
        .or_else(|| row.get("support_efforts"))
        .or_else(|| row.get("supportEfforts"));
    let efforts = reasoning
        .and_then(|value| value.get("effort"))
        .and_then(|value| value.get("valid"))
        .or(legacy_efforts)
        .and_then(Value::as_array);
    let supported = match kind {
        "no" | "none" => false,
        "only" | "both" => true,
        _ => {
            row.get("supports_reasoning")
                .or_else(|| row.get("supportsReasoning"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || efforts.is_some()
        }
    };
    if !supported {
        return (Vec::new(), false);
    }
    let levels = efforts
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .filter_map(reasoning_level)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    (levels.clone(), !levels.is_empty())
}

fn reasoning_level(value: &str) -> Option<Reasoning> {
    match value {
        "low" => Some(Reasoning::Low),
        "medium" => Some(Reasoning::Medium),
        "high" => Some(Reasoning::High),
        "xhigh" => Some(Reasoning::Xhigh),
        "max" => Some(Reasoning::Max),
        "ultra" => Some(Reasoning::Ultra),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn model_reasoning_distinguishes_fixed_and_adjustable_thinking() {
        let (levels, adjustable) = model_reasoning(&json!({
            "id": "kimi-for-coding",
            "reasoning": {"type": "both", "effort": {"valid": ["low", "high", "max"], "default": "max"}},
        }));
        assert_eq!(levels, [Reasoning::Low, Reasoning::High, Reasoning::Max]);
        assert!(adjustable);

        let (levels, adjustable) = model_reasoning(&json!({"reasoning": {"type": "only"}}));
        assert!(levels.is_empty());
        assert!(!adjustable);

        let (levels, adjustable) = model_reasoning(&json!({"reasoning": {"type": "no"}}));
        assert!(levels.is_empty());
        assert!(!adjustable);

        let (levels, adjustable) = model_reasoning(&json!({
            "think_efforts": {"support": true, "valid_efforts": ["medium", "high"]},
        }));
        assert_eq!(levels, [Reasoning::Medium, Reasoning::High]);
        assert!(adjustable);
    }
}
