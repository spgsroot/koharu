use anyhow::{Context as _, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{Tool, codex::Request};

#[derive(Debug, Serialize)]
pub(super) struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    stream: bool,
    stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

impl ChatRequest {
    pub(super) fn new(request: &Request) -> Result<Self> {
        let reasoning_effort = match request.reasoning_effort() {
            "" => None,
            effort => Some(effort.to_owned()),
        };
        Ok(Self {
            model: request.model.clone(),
            messages: messages(&request.instructions, &request.input)?,
            tools: request.tools.iter().map(function_tool).collect(),
            tool_choice: "auto",
            parallel_tool_calls: false,
            stream: true,
            stream_options: StreamOptions {
                include_usage: true,
            },
            reasoning_effort,
        })
    }
}

fn function_tool(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
    })
}

#[derive(Default)]
struct AssistantMessage {
    text: String,
    reasoning: String,
    calls: Vec<Value>,
}

impl AssistantMessage {
    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.reasoning.is_empty() && self.calls.is_empty()
    }

    fn into_value(self) -> Value {
        let mut message = json!({
            "role": "assistant",
            "content": if self.text.is_empty() { Value::Null } else { Value::String(self.text) },
        });
        let object = message
            .as_object_mut()
            .expect("assistant message is an object");
        if !self.reasoning.is_empty() {
            object.insert(
                "reasoning_content".to_owned(),
                Value::String(self.reasoning),
            );
        }
        if !self.calls.is_empty() {
            object.insert("tool_calls".to_owned(), Value::Array(self.calls));
        }
        message
    }
}

fn messages(instructions: &str, input: &[Value]) -> Result<Vec<Value>> {
    let mut messages = vec![json!({"role": "system", "content": instructions})];
    let mut assistant = AssistantMessage::default();

    for item in input {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .context("agent message is missing its role")?;
                if role == "assistant" {
                    append_assistant(&mut assistant, item)?;
                    continue;
                }
                flush_assistant(&mut messages, &mut assistant);
                let content = chat_content(item.get("content"))?;
                messages.push(json!({"role": role, "content": content}));
            }
            Some("function_call") => {
                let call_id = required_string(item, "call_id")?;
                let name = required_string(item, "name")?;
                let arguments = required_string(item, "arguments")?;
                assistant.calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }));
            }
            Some("function_call_output") => {
                flush_assistant(&mut messages, &mut assistant);
                append_tool_output(&mut messages, item)?;
            }
            Some(other) => bail!("unsupported agent history item for Kimi: {other}"),
            None => bail!("agent history item is missing its type"),
        }
    }
    flush_assistant(&mut messages, &mut assistant);
    Ok(messages)
}

fn append_assistant(assistant: &mut AssistantMessage, item: &Value) -> Result<()> {
    if let Some(content) = item.get("content") {
        match content {
            Value::String(text) => assistant.text.push_str(text),
            Value::Array(parts) => {
                for part in parts {
                    if let Some("output_text" | "input_text" | "text") =
                        part.get("type").and_then(Value::as_str)
                        && let Some(text) = part.get("text").and_then(Value::as_str)
                    {
                        assistant.text.push_str(text);
                    }
                }
            }
            Value::Null => {}
            _ => bail!("assistant message content must be text or an array"),
        }
    }
    if let Some(reasoning) = item.get("reasoning_content").and_then(Value::as_str) {
        assistant.reasoning.push_str(reasoning);
    }
    Ok(())
}

fn flush_assistant(messages: &mut Vec<Value>, assistant: &mut AssistantMessage) {
    if !assistant.is_empty() {
        messages.push(std::mem::take(assistant).into_value());
    }
}

fn append_tool_output(messages: &mut Vec<Value>, item: &Value) -> Result<()> {
    let call_id = required_string(item, "call_id")?;
    let output = item.get("output").unwrap_or(&Value::Null);
    let mut tool_text = String::new();
    let mut images = Vec::new();
    match output {
        Value::String(text) => tool_text.push_str(text),
        Value::Array(parts) => {
            let mut first_text = true;
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            if !tool_text.is_empty() {
                                tool_text.push('\n');
                            }
                            tool_text.push_str(text);
                            if !first_text {
                                images.push(json!({"type": "text", "text": text}));
                            }
                            first_text = false;
                        }
                    }
                    Some("input_image" | "image_url") => {
                        if let Some(image) = chat_image(part) {
                            images.push(image);
                        }
                    }
                    _ => {}
                }
            }
        }
        value => tool_text.push_str(&serde_json::to_string(value)?),
    }
    messages.push(json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": tool_text,
    }));
    if !images.is_empty() {
        images.insert(
            0,
            json!({"type": "text", "text": format!("Images from tool call {call_id}:")}),
        );
        messages.push(json!({"role": "user", "content": images}));
    }
    Ok(())
}

fn chat_content(content: Option<&Value>) -> Result<Value> {
    match content {
        Some(Value::String(text)) => Ok(Value::String(text.clone())),
        Some(Value::Array(parts)) => {
            let mut output = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            output.push(json!({"type": "text", "text": text}));
                        }
                    }
                    Some("input_image" | "image_url") => {
                        if let Some(image) = chat_image(part) {
                            output.push(image);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Value::Array(output))
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => bail!("agent message content must be text or an array"),
    }
}

fn chat_image(part: &Value) -> Option<Value> {
    let image = part.get("image_url")?;
    match image {
        Value::String(url) => Some(json!({
            "type": "image_url",
            "image_url": {"url": url, "detail": part.get("detail").unwrap_or(&json!("high"))},
        })),
        Value::Object(_) => Some(json!({"type": "image_url", "image_url": image})),
        _ => None,
    }
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .with_context(|| format!("agent history item is missing {field}"))
}

#[cfg(test)]
mod tests {
    use koharu_usage::{UsageOperation, UsageRecord};

    use crate::{
        Reasoning, ToolImage,
        codex::{Request, function_output, message},
    };

    use super::*;

    #[test]
    fn responses_history_becomes_kimi_chat_messages_and_function_tools() {
        let image = ToolImage {
            label: "page 1".to_owned(),
            data_url: "data:image/png;base64,AAAA".to_owned(),
        };
        let input = vec![
            message("user", "translate"),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "checking"}],
                "reasoning_content": "need the page",
            }),
            json!({
                "type": "function_call",
                "call_id": "call-1",
                "name": "view_page",
                "arguments": "{\"page\":1}",
            }),
            function_output("call-1", &json!({"ok": true}), &[image]).unwrap(),
        ];
        let request = Request::new(
            "k3".to_owned(),
            "instructions".to_owned(),
            input,
            vec![Tool::new(
                "view_page",
                "View a page",
                json!({"type": "object"}),
            )],
            Reasoning::Max,
            "session".to_owned(),
            UsageRecord::new(UsageOperation::Agent, "kimi"),
        );
        let body = serde_json::to_value(ChatRequest::new(&request).unwrap()).unwrap();

        assert_eq!(body["model"], "k3");
        assert_eq!(body["reasoning_effort"], "max");
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["tools"][0]["function"]["name"], "view_page");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][2]["content"], "checking");
        assert_eq!(body["messages"][2]["reasoning_content"], "need the page");
        assert_eq!(body["messages"][2]["tool_calls"][0]["id"], "call-1");
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_call_id"], "call-1");
        assert_eq!(body["messages"][4]["role"], "user");
        assert_eq!(
            body["messages"][4]["content"][2]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn reasoning_effort_is_omitted_for_non_adjustable_models() {
        let request = Request::new(
            "kimi-for-coding-highspeed".to_owned(),
            "instructions".to_owned(),
            vec![message("user", "hello")],
            Vec::new(),
            Reasoning::Medium,
            "session".to_owned(),
            UsageRecord::new(UsageOperation::Agent, "kimi"),
        )
        .without_reasoning();
        let body = serde_json::to_value(ChatRequest::new(&request).unwrap()).unwrap();

        assert!(body.get("reasoning_effort").is_none());
    }
}
