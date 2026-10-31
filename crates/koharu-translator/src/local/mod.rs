use std::sync::Arc;

use anyhow::Context;
use koharu_ml::llm::{
    ChatMessage, ChatTemplateOptions, Input, Llm, LoadOptions, MtmdOptions, media_marker,
};
use koharu_usage::{RequestSettings, UsageOutcome};

mod catalog;

pub use catalog::LocalConfig;
use catalog::LocalModelDescriptor;
pub(crate) use catalog::{DEFAULT_MODEL, DEFAULT_QUANTIZATION};

use crate::{
    Device, Error, GenerationConfig, Model, ModelSelection, Provider, Quantization, Result,
    TranslationRequest, prompt, usage::InFlight,
};

#[derive(Debug)]
pub struct LocalTranslator {
    descriptor: LocalModelDescriptor,
    llm: Arc<Llm>,
}

impl LocalTranslator {
    pub async fn load(device: Device, selection: &ModelSelection) -> Result<Self> {
        let model = selection
            .model
            .as_deref()
            .context("local translation requires a selected model")?;
        let descriptor = catalog::MODELS
            .iter()
            .copied()
            .find(|descriptor| descriptor.id == model)
            .with_context(|| format!("unknown local translator '{model}'"))?;
        let resolved = descriptor.resolve(selection).await?;
        let options = LoadOptions {
            mtmd: resolved.projector.map(MtmdOptions::new),
            ..LoadOptions::default()
        };
        let llm = Llm::load_with_options(device, resolved.model, options)
            .await
            .context("failed to load local translation model")?;
        if llm.capabilities().vision != descriptor.projector.is_some() {
            return Err(anyhow::anyhow!(
                "local translator vision capability does not match its catalog"
            )
            .into());
        }
        Ok(Self {
            descriptor,
            llm: Arc::new(llm),
        })
    }

    pub(crate) async fn translate(
        &self,
        request: TranslationRequest,
        generation: GenerationConfig,
    ) -> Result<Vec<String>> {
        let expected = request.segments.len();
        if expected == 0 {
            return Ok(Vec::new());
        }
        if !self
            .descriptor
            .target_languages
            .contains(request.target_language)
        {
            return Err(Error::UnsupportedLanguage {
                provider: "local",
                language: request.target_language,
            });
        }

        let image = request.image.clone();
        let reasoning = generation.reasoning.unwrap_or(false) && self.descriptor.reasoning;
        let prompt = self.render_prompt(&request, reasoning)?;
        let schema = prompt::output_schema(expected);
        let llm = Arc::clone(&self.llm);
        let generation = self.descriptor.generation.options(generation);
        let mut flight = request.usage.as_ref().map(|trace| {
            InFlight::start(
                trace,
                RequestSettings {
                    reasoning: Some(reasoning),
                    reasoning_effort: None,
                    temperature: Some(generation.temperature),
                    top_p: generation.top_p,
                    top_k: generation.top_k.and_then(|top_k| u32::try_from(top_k).ok()),
                    max_tokens: u32::try_from(generation.max_tokens).ok(),
                    frequency_penalty: Some(generation.frequency_penalty),
                    presence_penalty: Some(generation.presence_penalty),
                },
                0,
            )
        });
        let generated = match tokio::task::spawn_blocking(move || {
            let input = image.as_deref().map_or_else(
                || Input::new(&prompt),
                |image| Input::new(&prompt).with_image(image),
            );
            llm.inference_with_json_schema(&input, &generation, &schema)
        })
        .await
        {
            Ok(generated) => generated,
            Err(error) => Err(anyhow::Error::new(error).context("local translation task panicked")),
        };
        // llama.cpp counts the prompt and generated tokens exactly.
        if let Some(mut flight) = flight.take() {
            match &generated {
                Ok(output) => {
                    let tokens = &mut flight.record().tokens;
                    tokens.input = u32::try_from(output.prompt_tokens).ok();
                    tokens.output = u32::try_from(output.generated_tokens).ok();
                    tokens.total = tokens
                        .input
                        .zip(tokens.output)
                        .and_then(|(input, output)| input.checked_add(output));
                    flight.finish(UsageOutcome::Succeeded);
                }
                Err(_) => flight.finish(UsageOutcome::ProviderError),
            }
        }
        let output = generated?;
        let segments = match prompt::translations("local", &output.text, &request.segments) {
            Ok(segments) => segments,
            Err(error) => {
                if let Some(usage) = request.usage.as_ref() {
                    usage.reject(UsageOutcome::InvalidOutput);
                }
                return Err(error.into());
            }
        };
        Ok(segments)
    }

    fn render_prompt(&self, request: &TranslationRequest, reasoning: bool) -> Result<String> {
        let (system, payload) = prompt::prompts(request)?;
        let payload = if request.image.is_some() {
            format!("{}\n{payload}", media_marker())
        } else {
            payload
        };
        Ok(self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(payload)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: reasoning,
                },
            )
            .context("failed to render local translation prompt")?)
    }
}

pub(crate) fn models() -> Vec<Model> {
    catalog::MODELS
        .iter()
        .map(|descriptor| Model {
            provider: Provider::Local,
            model: Some(descriptor.id.to_owned()),
            name: descriptor.name.to_owned(),
            quantizations: descriptor
                .quantizations
                .iter()
                .map(|quantization| Quantization {
                    id: quantization.id.to_owned(),
                    name: quantization.name.to_owned(),
                })
                .collect(),
            vision: descriptor.projector.is_some(),
            reasoning: descriptor.reasoning,
            reasoning_efforts: Vec::new(),
            reasoning_always_on: false,
        })
        .collect()
}

pub(crate) fn supports_vision(selection: &ModelSelection) -> bool {
    selection.model.as_deref().is_some_and(|model| {
        catalog::MODELS
            .iter()
            .find(|descriptor| descriptor.id == model)
            .is_some_and(|descriptor| descriptor.projector.is_some())
    })
}
