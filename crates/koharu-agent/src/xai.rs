use std::{
    fmt,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use koharu_secrets::{ExposeSecret as _, SecretString};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::{
    Control, LoginEvent, Reasoning,
    codex::{Delta, Request, Turn, read},
};

const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const RESPONSES_URL: &str = "https://api.x.ai/v1/responses";
const MODELS_URL: &str = "https://api.x.ai/v1/language-models";
const SECRET: &str = "agent_xai";
const REFRESH_SKEW_MS: u64 = 5 * 60 * 1000;

#[derive(Clone)]
pub struct Xai {
    client: Client,
    refresh: Arc<Mutex<()>>,
}

impl Xai {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: koharu_runtime::http_client()?,
            refresh: Arc::new(Mutex::new(())),
        })
    }

    pub fn account(&self) -> Result<Option<crate::Account>> {
        Ok(load()?.map(|tokens| account(&tokens)))
    }

    pub async fn login_device<F>(&self, control: &Control, mut publish: F) -> Result<crate::Account>
    where
        F: FnMut(LoginEvent),
    {
        publish(LoginEvent::Progress {
            message: "Requesting a Grok device code".to_owned(),
        });
        let response = self
            .client
            .post(DEVICE_CODE_URL)
            .form(&[("client_id", CLIENT_ID), ("scope", SCOPE)])
            .send()
            .await
            .context("failed to start Grok device authorization")?;
        let value: Value = json_success(response).await?;
        let device_code = string(&value, "device_code")?;
        let user_code = string(&value, "user_code")?;
        let verification_url = https_uri(
            value
                .get("verification_uri_complete")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .or_else(|| value.get("verification_uri").and_then(Value::as_str))
                .context("Grok device authorization did not return a verification URL")?,
        )?;
        let interval = Duration::from_secs(
            value
                .get("interval")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .max(1),
        );
        let expires_in = Duration::from_secs(
            value
                .get("expires_in")
                .and_then(Value::as_u64)
                .unwrap_or(900),
        );
        publish(LoginEvent::DeviceCode {
            verification_url,
            user_code,
        });

        let started = tokio::time::Instant::now();
        let tokens = loop {
            control.ensure_running()?;
            if started.elapsed() >= expires_in {
                bail!("Grok device login expired");
            }
            tokio::select! {
                () = tokio::time::sleep(interval) => {}
                () = control.cancelled() => control.ensure_running()?,
            }
            let response = self
                .client
                .post(TOKEN_URL)
                .form(&[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("client_id", CLIENT_ID),
                    ("device_code", device_code.as_str()),
                ])
                .send()
                .await
                .context("failed to poll Grok device authorization")?;
            let status = response.status();
            let body: Value = response.json().await.unwrap_or(Value::Null);
            if status.is_success() {
                break stored_tokens(&body, None)?;
            }
            match body.get("error").and_then(Value::as_str) {
                Some("authorization_pending") => continue,
                Some("slow_down") => tokio::time::sleep(interval).await,
                Some("access_denied" | "authorization_denied") => {
                    bail!("Grok device authorization was denied");
                }
                Some("expired_token") => bail!("Grok device code expired"),
                Some(error) => {
                    let description = body
                        .get("error_description")
                        .and_then(Value::as_str)
                        .unwrap_or(error);
                    bail!("Grok device authorization failed: {description}");
                }
                None => bail!("Grok device authorization failed ({status})"),
            }
        };
        let signed_in = account(&tokens);
        save(&tokens)?;
        Ok(signed_in)
    }

    pub fn logout(&self) -> Result<()> {
        koharu_secrets::delete(SECRET)?;
        Ok(())
    }

    pub async fn access_token() -> Result<Option<SecretString>> {
        let client = Self::new()?;
        if client.account()?.is_none() {
            return Ok(None);
        }
        Ok(Some(SecretString::from(client.session().await?.access)))
    }

    pub async fn models(&self) -> Result<Vec<crate::CodexModel>> {
        let session = self.session().await?;
        let response = self
            .client
            .get(MODELS_URL)
            .bearer_auth(&session.access)
            .send()
            .await?;
        if response.status() == StatusCode::UNAUTHORIZED {
            let session = self.force_refresh().await?;
            let response = self
                .client
                .get(MODELS_URL)
                .bearer_auth(&session.access)
                .send()
                .await?;
            return parse_models(response).await;
        }
        parse_models(response).await
    }

    pub(crate) async fn respond<F>(
        &self,
        request: Request,
        control: &Control,
        publish: F,
    ) -> Result<Turn>
    where
        F: FnMut(Delta),
    {
        control.ensure_running()?;
        let mut request = request.without_reasoning_summary();
        let mut session = self.session().await?;
        let mut usage = crate::usage::Inference::new(
            request
                .usage
                .take()
                .context("agent request usage context is missing")?,
        );
        let mut response = usage.sent(self.send(&request, &session, control).await, control)?;
        if response.status() == StatusCode::UNAUTHORIZED {
            usage.entry.outcome = koharu_usage::UsageOutcome::HttpError;
            let retry = usage.retry_entry();
            drop(usage);
            session = self.force_refresh().await?;
            usage = crate::usage::Inference::new(retry);
            response = usage.sent(self.send(&request, &session, control).await, control)?;
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
                usage
                    .entry
                    .report(koharu_usage::wire::openai_responses(&value));
                usage.entry.error_code = koharu_usage::wire::error_code(&value);
            }
            body.truncate(16 * 1024);
            bail!("Grok returned {status}: {body}");
        }
        let result = read(response, control, publish, &mut usage).await;
        if result.is_ok() {
            usage.entry.outcome = koharu_usage::UsageOutcome::Succeeded;
        } else if control.is_cancelled() {
            usage.entry.outcome = koharu_usage::UsageOutcome::Cancelled;
        }
        result
    }

    async fn send(
        &self,
        request: &Request,
        session: &Session,
        control: &Control,
    ) -> Result<reqwest::Response> {
        let request_id = request.cache_key().to_owned();
        let send = self
            .client
            .post(RESPONSES_URL)
            .bearer_auth(&session.access)
            .header("accept", "text/event-stream")
            .header("x-grok-conv-id", &request_id)
            .json(request)
            .send();
        tokio::select! {
            response = send => Ok(response?),
            () = control.cancelled() => {
                control.ensure_running()?;
                unreachable!("cancelled control must fail ensure_running")
            }
        }
    }

    async fn session(&self) -> Result<Session> {
        let tokens = load()?.context("Grok is not signed in")?;
        if tokens.expires_at_ms <= now_ms().saturating_add(REFRESH_SKEW_MS) {
            return self.refresh().await;
        }
        Ok(Session {
            access: tokens.access,
        })
    }

    async fn refresh(&self) -> Result<Session> {
        let _refresh = self.refresh.lock().await;
        let current = load()?.context("Grok is not signed in")?;
        if current.expires_at_ms > now_ms().saturating_add(REFRESH_SKEW_MS) {
            return Ok(Session {
                access: current.access,
            });
        }
        self.refresh_tokens(current).await
    }

    async fn force_refresh(&self) -> Result<Session> {
        let _refresh = self.refresh.lock().await;
        let current = load()?.context("Grok is not signed in")?;
        self.refresh_tokens(current).await
    }

    async fn refresh_tokens(&self, current: Stored) -> Result<Session> {
        let response = self
            .client
            .post(TOKEN_URL)
            .form(&[
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", current.refresh.as_str()),
            ])
            .send()
            .await
            .context("failed to refresh Grok OAuth credentials")?;
        if response.status() == StatusCode::UNAUTHORIZED {
            let _ = koharu_secrets::delete(SECRET);
        }
        let body = json_success(response).await?;
        let tokens = stored_tokens(&body, Some(current.refresh))?;
        let access = tokens.access.clone();
        save(&tokens)?;
        Ok(Session { access })
    }
}

impl fmt::Debug for Xai {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Xai").finish_non_exhaustive()
    }
}

struct Session {
    access: String,
}

#[derive(Serialize, Deserialize)]
struct Stored {
    access: String,
    refresh: String,
    expires_at_ms: u64,
}

fn load() -> Result<Option<Stored>> {
    let Some(secret) = koharu_secrets::get(SECRET)? else {
        return Ok(None);
    };
    serde_json::from_str(secret.expose_secret()).context("stored Grok credentials are invalid")
}

fn save(tokens: &Stored) -> Result<()> {
    let encoded = serde_json::to_string(tokens)?;
    koharu_secrets::set(SECRET, &SecretString::from(encoded))
}

fn account(tokens: &Stored) -> crate::Account {
    let claims = jwt(&tokens.access);
    crate::Account {
        id: claims
            .as_ref()
            .and_then(|claims| claims.get("sub"))
            .and_then(Value::as_str)
            .unwrap_or("grok")
            .to_owned(),
        email: claims
            .as_ref()
            .and_then(|claims| claims.get("email"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        plan: Some("SuperGrok".to_owned()),
    }
}

fn stored_tokens(body: &Value, previous_refresh: Option<String>) -> Result<Stored> {
    let access = string(body, "access_token")?;
    let refresh = body
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or(previous_refresh)
        .context("Grok OAuth response did not include a refresh token")?;
    let expires_in = body
        .get("expires_in")
        .and_then(Value::as_u64)
        .unwrap_or(3600);
    Ok(Stored {
        access,
        refresh,
        expires_at_ms: now_ms().saturating_add(expires_in.saturating_mul(1000)),
    })
}

async fn parse_models(response: reqwest::Response) -> Result<Vec<crate::CodexModel>> {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "Grok model discovery returned {status}: {}",
            &body[..body.len().min(4096)]
        );
    }
    let value: Value = serde_json::from_str(&body).context("invalid Grok model catalog")?;
    let rows = value
        .get("models")
        .or_else(|| value.get("data"))
        .and_then(Value::as_array)
        .context("Grok model catalog must contain a models array")?;
    let models = rows
        .iter()
        .filter_map(|row| {
            let id = row.get("id").and_then(Value::as_str)?.trim();
            if id.is_empty() || !is_chat_model(id) {
                return None;
            }
            Some(crate::CodexModel {
                name: row
                    .get("display_name")
                    .or_else(|| row.get("displayName"))
                    .and_then(Value::as_str)
                    .unwrap_or(id)
                    .to_owned(),
                id: id.to_owned(),
                reasoning: vec![Reasoning::Low, Reasoning::Medium, Reasoning::High],
            })
        })
        .collect::<Vec<_>>();
    if models.is_empty() {
        bail!("the signed-in Grok account has no available models");
    }
    Ok(models)
}

fn is_chat_model(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    !id.contains("imagine") && !id.contains("tts") && !id.contains("stt") && !id.contains("voice")
}

async fn json_success(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Grok returned {status}: {}", &body[..body.len().min(4096)]);
    }
    serde_json::from_str(&body).context("Grok returned invalid JSON")
}

fn https_uri(raw: &str) -> Result<String> {
    let url = reqwest::Url::parse(raw).context("Grok returned an invalid verification URL")?;
    if url.scheme() != "https" {
        bail!("Grok returned a non-HTTPS verification URL");
    }
    Ok(url.to_string())
}

fn string(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("Grok response is missing {field}"))
}

fn jwt(value: &str) -> Option<Value> {
    let mut parts = value.split('.');
    let (_header, payload, _signature, None) =
        (parts.next()?, parts.next()?, parts.next()?, parts.next())
    else {
        return None;
    };
    serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).ok()?).ok()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
