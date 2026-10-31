mod auth;
mod catalog;
mod protocol;
mod stream;
mod token_store;

use anyhow::{Context as _, Result, bail};
use reqwest::{Client, StatusCode};
use serde::Serialize;
use specta::Type;

use crate::{Control, Reasoning};

pub use auth::Account;
use auth::Auth;
pub(crate) use protocol::{Request, function_output, message, project_context};
pub(crate) use stream::{Delta, Turn, read};

const RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

#[derive(Clone, Debug, Serialize, Type)]
pub struct CodexModel {
    pub id: String,
    pub name: String,
    pub reasoning: Vec<Reasoning>,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LoginEvent {
    Progress {
        message: String,
    },
    DeviceCode {
        verification_url: String,
        user_code: String,
    },
}

#[derive(Clone, Debug)]
pub struct Codex {
    client: Client,
    auth: Auth,
}

impl Codex {
    pub fn new() -> Result<Self> {
        let client = koharu_runtime::http_client()?;
        Ok(Self {
            auth: Auth::new(client.clone()),
            client,
        })
    }

    pub fn account(&self) -> Result<Option<Account>> {
        self.auth.account()
    }

    #[tracing::instrument(skip_all)]
    pub async fn login_device<F>(&self, control: &Control, publish: F) -> Result<Account>
    where
        F: FnMut(LoginEvent),
    {
        self.auth.login_device(control, publish).await
    }

    pub fn logout(&self) -> Result<()> {
        self.auth.logout()
    }

    #[tracing::instrument(skip_all)]
    pub async fn models(&self) -> Result<Vec<CodexModel>> {
        catalog::models(&self.client, &self.auth).await
    }

    pub(crate) async fn respond<F>(
        &self,
        mut request: Request,
        control: &Control,
        publish: F,
    ) -> Result<Turn>
    where
        F: FnMut(Delta),
    {
        control.ensure_running()?;
        let session = self.auth.session().await?;
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
            let session = self.auth.force_refresh().await?;
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
            bail!("Codex returned {status}: {body}");
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
        request: &Request,
        session: &auth::Session,
        control: &Control,
    ) -> Result<reqwest::Response> {
        let request_id = request.prompt_cache_key.clone();
        let send = self
            .client
            .post(RESPONSES_URL)
            .bearer_auth(&session.access)
            .header("chatgpt-account-id", &session.account.id)
            .header("originator", "koharu")
            .header("OpenAI-Beta", "responses=experimental")
            .header("accept", "text/event-stream")
            .header("session_id", &request_id)
            .header("x-client-request-id", request_id)
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
}
