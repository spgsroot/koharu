use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use koharu_secrets::{ExposeSecret as _, SecretString};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::{Account, Control, LoginEvent};

const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const DEVICE_CODE_URL: &str = "https://auth.kimi.com/api/oauth/device_authorization";
const TOKEN_URL: &str = "https://auth.kimi.com/api/oauth/token";
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const SECRET: &str = "agent_kimi";
const REFRESH_SKEW_MS: u64 = 5 * 60 * 1000;

pub(super) struct Auth {
    client: Client,
    refresh: Arc<Mutex<()>>,
}

pub(super) struct Session {
    pub(super) access: String,
}

impl Auth {
    pub(super) fn new(client: Client) -> Self {
        Self {
            client,
            refresh: Arc::new(Mutex::new(())),
        }
    }

    pub(super) fn account(&self) -> Result<Option<Account>> {
        Ok(load()?.map(|tokens| account(&tokens)))
    }

    pub(super) async fn login_device<F>(&self, control: &Control, mut publish: F) -> Result<Account>
    where
        F: FnMut(LoginEvent),
    {
        publish(LoginEvent::Progress {
            message: "Requesting a Kimi Code device code".to_owned(),
        });
        let response = send_cancellable(
            self.client
                .post(DEVICE_CODE_URL)
                .form(&[("client_id", CLIENT_ID)]),
            control,
        )
        .await
        .context("failed to start Kimi Code device authorization")?;
        let value = json_success(response, "Kimi Code device authorization").await?;
        let device_code = string(&value, "device_code")?;
        let user_code = string(&value, "user_code")?;
        let verification_url = https_uri(
            value
                .get("verification_uri_complete")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .or_else(|| value.get("verification_uri").and_then(Value::as_str))
                .context("Kimi Code device authorization did not return a verification URL")?,
        )?;
        let mut interval = Duration::from_secs(
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
                bail!("Kimi Code device login expired");
            }
            sleep(interval, control).await?;
            let response = send_cancellable(
                self.client.post(TOKEN_URL).form(&[
                    ("grant_type", DEVICE_GRANT),
                    ("client_id", CLIENT_ID),
                    ("device_code", device_code.as_str()),
                ]),
                control,
            )
            .await
            .context("failed to poll Kimi Code device authorization")?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let value: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            if status.is_success() {
                break stored_tokens(&value, None)?;
            }
            match value.get("error").and_then(Value::as_str) {
                Some("authorization_pending") => continue,
                Some("slow_down") => interval += Duration::from_secs(5),
                Some("access_denied" | "authorization_denied") => {
                    bail!("Kimi Code device authorization was denied");
                }
                Some("expired_token") => bail!("Kimi Code device code expired"),
                Some(error) => {
                    let description = value
                        .get("error_description")
                        .and_then(Value::as_str)
                        .unwrap_or(error);
                    bail!("Kimi Code device authorization failed: {description}");
                }
                None => bail!("Kimi Code device authorization failed ({status})"),
            }
        };
        let signed_in = account(&tokens);
        save(&tokens)?;
        Ok(signed_in)
    }

    pub(super) fn logout(&self) -> Result<()> {
        koharu_secrets::delete(SECRET)?;
        Ok(())
    }

    pub(super) async fn session(&self) -> Result<Session> {
        let tokens = load()?.context("Kimi Code is not signed in")?;
        if tokens.expires_at_ms <= now_ms().saturating_add(REFRESH_SKEW_MS) {
            return self.refresh().await;
        }
        Ok(Session {
            access: tokens.access,
        })
    }

    pub(super) async fn force_refresh(&self) -> Result<Session> {
        let _refresh = self.refresh.lock().await;
        let current = load()?.context("Kimi Code is not signed in")?;
        self.refresh_tokens(current).await
    }

    async fn refresh(&self) -> Result<Session> {
        let _refresh = self.refresh.lock().await;
        let current = load()?.context("Kimi Code is not signed in")?;
        if current.expires_at_ms > now_ms().saturating_add(REFRESH_SKEW_MS) {
            return Ok(Session {
                access: current.access,
            });
        }
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
            .context("failed to refresh Kimi Code OAuth credentials")?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let value: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
            || value.get("error").and_then(Value::as_str) == Some("invalid_grant")
        {
            koharu_secrets::delete(SECRET)?;
        }
        let value = json_value(status, body, "Kimi Code token refresh")?;
        let tokens = stored_tokens(&value, Some(current.refresh))?;
        let access = tokens.access.clone();
        save(&tokens)?;
        Ok(Session { access })
    }
}

async fn send_cancellable(request: RequestBuilder, control: &Control) -> Result<reqwest::Response> {
    control.ensure_running()?;
    tokio::select! {
        response = request.send() => Ok(response?),
        () = control.cancelled() => {
            control.ensure_running()?;
            unreachable!("cancelled control must fail ensure_running")
        }
    }
}

async fn sleep(duration: Duration, control: &Control) -> Result<()> {
    tokio::select! {
        () = tokio::time::sleep(duration) => Ok(()),
        () = control.cancelled() => control.ensure_running(),
    }
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
    serde_json::from_str(secret.expose_secret()).context("stored Kimi Code credentials are invalid")
}

fn save(tokens: &Stored) -> Result<()> {
    let encoded = serde_json::to_string(tokens)?;
    koharu_secrets::set(SECRET, &SecretString::from(encoded))
}

fn account(tokens: &Stored) -> Account {
    let claims = jwt(&tokens.access);
    Account {
        id: claims
            .as_ref()
            .and_then(|claims| claims.get("sub"))
            .and_then(Value::as_str)
            .unwrap_or("kimi")
            .to_owned(),
        email: claims
            .as_ref()
            .and_then(|claims| claims.get("email"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        plan: None,
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
        .context("Kimi Code OAuth response did not include a refresh token")?;
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

async fn json_success(response: reqwest::Response, operation: &str) -> Result<Value> {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    json_value(status, body, operation)
}

fn json_value(status: StatusCode, body: String, operation: &str) -> Result<Value> {
    if !status.is_success() {
        let excerpt = body.chars().take(4096).collect::<String>();
        bail!("{operation} returned {status}: {excerpt}");
    }
    serde_json::from_str(&body).with_context(|| format!("{operation} returned invalid JSON"))
}

fn https_uri(raw: &str) -> Result<String> {
    let url = reqwest::Url::parse(raw).context("Kimi Code returned an invalid verification URL")?;
    if url.scheme() != "https" {
        bail!("Kimi Code returned a non-HTTPS verification URL");
    }
    Ok(url.to_string())
}

fn string(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("Kimi Code response is missing {field}"))
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn token_refresh_preserves_an_unchanged_refresh_token() {
        let before = now_ms();
        let tokens = stored_tokens(
            &json!({"access_token": "access", "expires_in": 60}),
            Some("refresh".to_owned()),
        )
        .unwrap();

        assert_eq!(tokens.access, "access");
        assert_eq!(tokens.refresh, "refresh");
        assert!(tokens.expires_at_ms >= before + 60_000);
    }

    #[test]
    fn token_response_requires_access_and_refresh_tokens() {
        assert!(stored_tokens(&json!({"refresh_token": "refresh"}), None).is_err());
        assert!(stored_tokens(&json!({"access_token": "access"}), None).is_err());
    }

    #[test]
    fn verification_urls_require_https() {
        assert!(https_uri("http://auth.kimi.com/device").is_err());
        assert_eq!(
            https_uri("https://auth.kimi.com/device?user_code=ABCD").unwrap(),
            "https://auth.kimi.com/device?user_code=ABCD"
        );
    }
}
