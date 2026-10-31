//! Streaming Chat Completions → the provider-neutral Responses-style turn.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, anyhow, bail};
use eventsource_stream::Eventsource as _;
use futures::StreamExt as _;
use serde_json::{Value, json};

use crate::{
    Control, ToolCall,
    codex::{Delta, Turn},
};

const MAX_STREAM_BYTES: usize = 16 * 1024 * 1024;

pub(super) async fn read<F>(
    response: reqwest::Response,
    control: &Control,
    mut publish: F,
    usage: &mut crate::usage::Inference,
) -> Result<Turn>
where
    F: FnMut(Delta),
{
    let mut stream = response.bytes_stream().eventsource();
    let mut state = StreamState::default();
    let mut bytes = 0_usize;
    loop {
        let event = tokio::select! {
            event = stream.next() => event,
            () = control.cancelled() => {
                control.ensure_running()?;
                unreachable!("cancelled control must fail ensure_running")
            }
        };
        let Some(event) = event else { break };
        let event = event.map_err(|error| {
            usage.entry.outcome = koharu_usage::UsageOutcome::NetworkError;
            anyhow!("invalid Kimi Code event stream: {error}")
        })?;
        bytes = bytes.saturating_add(event.data.len());
        if bytes > MAX_STREAM_BYTES {
            usage.entry.outcome = koharu_usage::UsageOutcome::DecodeError;
            bail!("Kimi Code response stream exceeded {MAX_STREAM_BYTES} bytes");
        }
        if event.data.trim().is_empty() {
            continue;
        }
        if event.data == "[DONE]" {
            state.completed = true;
            break;
        }
        let value: Value = serde_json::from_str(&event.data)
            .inspect_err(|_| {
                usage.entry.outcome = koharu_usage::UsageOutcome::DecodeError;
            })
            .context("invalid JSON in Kimi Code response stream")?;
        if value.get("error").is_some() {
            usage.entry.outcome = koharu_usage::UsageOutcome::ProviderError;
            usage.entry.error_code = koharu_usage::wire::error_code(&value);
            bail!("Kimi Code returned an error: {}", provider_error(&value));
        }
        usage.entry.report(koharu_usage::wire::openai_chat(&value));
        state.apply(&value, &mut publish)?;
        if state.completed {
            break;
        }
    }
    state.finish()
}

#[derive(Default)]
struct StreamState {
    text: String,
    reasoning: String,
    calls: BTreeMap<u64, PendingCall>,
    completed: bool,
}

#[derive(Default)]
struct PendingCall {
    id: String,
    name: String,
    arguments: String,
}

impl StreamState {
    fn apply<F>(&mut self, value: &Value, publish: &mut F) -> Result<()>
    where
        F: FnMut(Delta),
    {
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            // Usage-only chunks carry no choices.
            return Ok(());
        };
        if let Some(delta) = choice.get("delta") {
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                self.text.push_str(text);
                publish(Delta::Text(text.to_owned()));
            }
            if let Some(reasoning) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
            {
                self.reasoning.push_str(reasoning);
                publish(Delta::Reasoning(reasoning.to_owned()));
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for (position, call) in calls.iter().enumerate() {
                    let index = call
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or(position as u64);
                    let pending = self.calls.entry(index).or_default();
                    if let Some(id) = call.get("id").and_then(Value::as_str) {
                        pending.id.push_str(id);
                    }
                    if let Some(function) = call.get("function") {
                        if let Some(name) = function.get("name").and_then(Value::as_str) {
                            pending.name.push_str(name);
                        }
                        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                            pending.arguments.push_str(arguments);
                        }
                    }
                }
            }
        }
        // A complete non-streamed message is accepted too; some proxies
        // collapse the SSE stream into one terminal chunk.
        if let Some(message) = choice.get("message") {
            if self.text.is_empty()
                && let Some(text) = message.get("content").and_then(Value::as_str)
            {
                self.text.push_str(text);
                publish(Delta::Text(text.to_owned()));
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for (position, call) in calls.iter().enumerate() {
                    let pending = self.calls.entry(position as u64).or_default();
                    pending.id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    pending.name = call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    pending.arguments = call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                }
            }
        }
        if choice
            .get("finish_reason")
            .is_some_and(|reason| !reason.is_null())
        {
            self.completed = true;
        }
        Ok(())
    }

    fn finish(self) -> Result<Turn> {
        if !self.completed {
            bail!("Kimi Code response stream ended before completion");
        }
        let mut output = Vec::new();
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(json!({"type": "output_text", "text": self.text.clone()}));
        }
        let mut message = json!({
            "type": "message",
            "role": "assistant",
            "content": content,
        });
        if !self.reasoning.is_empty() {
            message
                .as_object_mut()
                .expect("message is an object")
                .insert(
                    "reasoning_content".to_owned(),
                    Value::String(self.reasoning.clone()),
                );
        }
        output.push(message);
        let mut calls = Vec::new();
        for (index, call) in self.calls {
            if call.name.is_empty() {
                bail!("Kimi Code returned a tool call without a function name");
            }
            let call_id = if call.id.is_empty() {
                format!("kimi-call-{index}")
            } else {
                call.id
            };
            output.push(json!({
                "type": "function_call",
                "call_id": call_id,
                "name": call.name,
                "arguments": call.arguments,
            }));
            calls.push(ToolCall {
                call_id,
                name: call.name,
                arguments: call.arguments,
            });
        }
        Ok(Turn {
            output,
            calls,
            text: self.text,
        })
    }
}

fn provider_error(value: &Value) -> String {
    value
        .get("message")
        .or_else(|| value.get("error").and_then(|error| error.get("message")))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn streamed_text_reasoning_and_fragmented_tool_calls_form_one_turn() {
        let mut state = StreamState::default();
        let mut deltas = Vec::new();
        let mut publish = |delta: Delta| {
            deltas.push(match delta {
                Delta::Text(text) => format!("text:{text}"),
                Delta::Reasoning(text) => format!("reasoning:{text}"),
            });
        };
        for chunk in [
            json!({"choices": [{"delta": {"reasoning_content": "look", "tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "view_page", "arguments": "{\"page\":"}}]}}]}),
            json!({"choices": [{"delta": {"content": "Checking", "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3}}),
        ] {
            state.apply(&chunk, &mut publish).unwrap();
        }
        let turn = state.finish().unwrap();

        assert_eq!(turn.text, "Checking");
        assert_eq!(turn.calls.len(), 1);
        assert_eq!(turn.calls[0].call_id, "call-1");
        assert_eq!(turn.calls[0].name, "view_page");
        assert_eq!(turn.calls[0].arguments, "{\"page\":1}");
        assert_eq!(turn.output[0]["reasoning_content"], "look");
        assert_eq!(turn.output[1]["type"], "function_call");
        assert_eq!(deltas, ["reasoning:look", "text:Checking"]);
    }

    #[test]
    fn unterminated_streams_are_rejected() {
        assert!(StreamState::default().finish().is_err());
    }
}
