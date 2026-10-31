use std::{fmt, sync::Arc};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use specta::Type;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    Account, AgentProvider, Codex, CodexModel, Config, Control, Host, Kimi, ToolCall, Xai,
    codex::{Delta, Request, function_output, message, project_context},
};

const INSTRUCTIONS: &str = r#"You are Koharu Agent, operating a manga translation project inside Koharu.
The complete current project state is supplied in a koharu_project_context block on every user turn.
Page images are intentionally omitted from that context. Call view_page only when visual inspection is needed, and only for relevant pages.
Use the provided tools whenever the user asks to inspect or change the project. Never claim a change succeeded unless its tool result says it succeeded.
All project changes are revisioned and reversible. Do not ask for permission. Do not produce a plan or expose internal steps; continue using tools until the request is complete or cannot be completed.
Do not invent entity identifiers. Preserve artwork and existing authored content unless the user asks to change them."#;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Started {
        run: RunId,
    },
    TextDelta {
        run: RunId,
        delta: String,
    },
    ReasoningDelta {
        run: RunId,
        delta: String,
    },
    ToolStarted {
        run: RunId,
        call_id: String,
        name: String,
    },
    ToolFinished {
        run: RunId,
        call_id: String,
        name: String,
        changed: bool,
        output: String,
    },
    Completed {
        run: RunId,
        message: String,
    },
    Failed {
        run: RunId,
        message: String,
    },
    Cancelled {
        run: RunId,
    },
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct RunResult {
    pub run: RunId,
    pub message: String,
}

pub struct Agent<H> {
    codex: Codex,
    xai: Xai,
    kimi: Kimi,
    host: Arc<H>,
    config: koharu_config::Config<Config>,
    history: Mutex<Vec<Value>>,
    serial: Mutex<()>,
}

impl<H> Agent<H>
where
    H: Host,
{
    pub fn new(host: H) -> Result<Self> {
        Ok(Self {
            codex: Codex::new()?,
            xai: Xai::new()?,
            kimi: Kimi::new()?,
            host: Arc::new(host),
            config: Config::load()?,
            history: Mutex::new(Vec::new()),
            serial: Mutex::new(()),
        })
    }

    pub fn account(&self) -> Result<Option<Account>> {
        match self.provider()? {
            AgentProvider::Codex => self.codex.account(),
            AgentProvider::Xai => self.xai.account(),
            AgentProvider::Kimi => self.kimi.account(),
        }
    }

    pub async fn login_device<F>(&self, control: &Control, publish: F) -> Result<Account>
    where
        F: FnMut(crate::LoginEvent),
    {
        match self.provider()? {
            AgentProvider::Codex => self.codex.login_device(control, publish).await,
            AgentProvider::Xai => self.xai.login_device(control, publish).await,
            AgentProvider::Kimi => self.kimi.login_device(control, publish).await,
        }
    }

    pub fn logout(&self) -> Result<()> {
        match self.provider()? {
            AgentProvider::Codex => self.codex.logout(),
            AgentProvider::Xai => self.xai.logout(),
            AgentProvider::Kimi => self.kimi.logout(),
        }
    }

    fn provider(&self) -> Result<AgentProvider> {
        Ok(self.config.read()?.provider)
    }

    pub fn config(&self) -> Result<Config> {
        Ok(self.config.read()?.clone())
    }

    pub async fn models(&self) -> Result<Vec<CodexModel>> {
        match self.provider()? {
            AgentProvider::Codex => self.codex.models().await,
            AgentProvider::Xai => self.xai.models().await,
            AgentProvider::Kimi => self.kimi.models().await,
        }
    }

    pub fn save_config(&self, mut config: Config) -> Result<Config> {
        let mut current = self.config.write()?;
        if current.provider != config.provider {
            config.model = None;
        }
        *current = config;
        let saved = current.clone();
        current.save()?;
        Ok(saved)
    }

    pub async fn clear(&self) {
        self.history.lock().await.clear();
    }

    #[tracing::instrument(skip_all)]
    pub async fn run<F>(
        &self,
        run: RunId,
        prompt: String,
        control: Control,
        mut publish: F,
    ) -> Result<RunResult>
    where
        F: FnMut(Event) + Send,
    {
        let _serial = self.serial.lock().await;
        publish(Event::Started { run });
        let result = self.run_inner(run, prompt, &control, &mut publish).await;
        match &result {
            Ok(result) => publish(Event::Completed {
                run,
                message: result.message.clone(),
            }),
            Err(_) if control.is_cancelled() => publish(Event::Cancelled { run }),
            Err(error) => publish(Event::Failed {
                run,
                message: format!("{error:#}"),
            }),
        }
        result
    }

    async fn run_inner<F>(
        &self,
        run: RunId,
        prompt: String,
        control: &Control,
        publish: &mut F,
    ) -> Result<RunResult>
    where
        F: FnMut(Event),
    {
        control.ensure_running()?;
        let mut context = self.host.context().await?;
        // Audit identity is local metadata, not extra input to the model.
        let project_id = context
            .as_object_mut()
            .and_then(|object| object.remove("project_id"))
            .and_then(|value| match value {
                Value::String(id) => Some(id),
                _ => None,
            });
        let config = self.config()?;
        let models = self.models().await?;
        let provider = config.provider.display_name();
        let model = match config.model.as_deref() {
            Some(selected) => models
                .iter()
                .find(|model| model.id == selected)
                .with_context(|| {
                    format!("configured {provider} model {selected} is not available")
                })?,
            None => models
                .first()
                .with_context(|| format!("{provider} returned no available model"))?,
        };
        let reasoning = if model.reasoning.is_empty() || model.reasoning.contains(&config.reasoning)
        {
            config.reasoning
        } else {
            model.reasoning[0]
        };
        let clean_user = message("user", &prompt);
        let context_message = project_context(&context)?;
        let base = self.history.lock().await.clone();
        let mut input = base.clone();
        input.push(context_message);
        input.push(clean_user.clone());
        let mut persisted = base;
        persisted.push(clean_user);
        let tools = self.host.tools();
        let mut attempt = 0_u32;
        let session = run.to_string();

        loop {
            control.ensure_running()?;
            let mut usage = koharu_usage::UsageRecord::new(
                koharu_usage::UsageOperation::Agent,
                config.provider.as_str(),
            );
            usage.attempt = attempt;
            attempt = attempt.saturating_add(1);
            usage.context.job_id = Some(session.clone());
            usage.context.project_id = project_id.clone();
            usage.context.page_id = context
                .pointer("/project/active_page")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let request = Request::new(
                model.id.clone(),
                INSTRUCTIONS.to_owned(),
                input.clone(),
                tools.clone(),
                reasoning,
                session.clone(),
                usage,
            );
            let turn =
                match config.provider {
                    AgentProvider::Codex => {
                        self.codex
                            .respond(request, control, |event| match event {
                                Delta::Text(delta) => publish(Event::TextDelta { run, delta }),
                                Delta::Reasoning(delta) => {
                                    publish(Event::ReasoningDelta { run, delta })
                                }
                            })
                            .await?
                    }
                    AgentProvider::Xai => {
                        self.xai
                            .respond(request.without_reasoning_summary(), control, |event| {
                                match event {
                                    Delta::Text(delta) => publish(Event::TextDelta { run, delta }),
                                    Delta::Reasoning(delta) => {
                                        publish(Event::ReasoningDelta { run, delta })
                                    }
                                }
                            })
                            .await?
                    }
                    AgentProvider::Kimi => {
                        self.kimi
                            .respond(request, control, |event| match event {
                                Delta::Text(delta) => publish(Event::TextDelta { run, delta }),
                                Delta::Reasoning(delta) => {
                                    publish(Event::ReasoningDelta { run, delta })
                                }
                            })
                            .await?
                    }
                };

            input.extend(turn.output.iter().cloned());
            persisted.extend(turn.output.iter().cloned());

            if turn.calls.is_empty() {
                *self.history.lock().await = persisted;
                return Ok(RunResult {
                    run,
                    message: turn.text,
                });
            }

            for call in turn.calls {
                control.ensure_running()?;
                publish(Event::ToolStarted {
                    run,
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                });
                let invocation = self
                    .host
                    .invoke(
                        ToolCall {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments,
                        },
                        control,
                    )
                    .await;
                let (output, changed, images) = match invocation {
                    Ok(invocation) => (
                        json!({ "ok": true, "value": invocation.value }),
                        invocation.changed,
                        invocation.images,
                    ),
                    Err(error) => (
                        json!({ "ok": false, "error": format!("{error:#}") }),
                        false,
                        Vec::new(),
                    ),
                };
                publish(Event::ToolFinished {
                    run,
                    call_id: call.call_id.clone(),
                    name: call.name,
                    changed,
                    output: output.to_string(),
                });
                input.push(function_output(&call.call_id, &output, &images)?);
                persisted.push(function_output(&call.call_id, &output, &[])?);
            }
        }
    }
}
