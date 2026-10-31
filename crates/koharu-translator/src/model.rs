use koharu_ml::llm::GenerationOptions;
use serde::{Deserialize, Serialize};
use specta::Type;
use strum::{Display, EnumString, IntoStaticStr, VariantArray};

use crate::{Error, Provider, Result};

/// How hard a model reasons while reasoning is on.
///
/// Providers document different subsets of these levels, so a level is only
/// valid for a model that lists it in [`Model::reasoning_efforts`]. Turning
/// reasoning off is [`GenerationConfig::reasoning`]; `none` is never a level.
#[derive(
    Clone,
    Copy,
    Debug,
    Display,
    EnumString,
    Eq,
    Hash,
    IntoStaticStr,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    Deserialize,
    Type,
    VariantArray,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    /// The levels a provider reports as strings, ascending and without
    /// duplicates. Values outside the typed set, such as `none`, `off` and
    /// `on`, are switches rather than levels and are skipped.
    pub(crate) fn from_reported<'a>(reported: impl IntoIterator<Item = &'a str>) -> Vec<Self> {
        let mut levels: Vec<Self> = reported
            .into_iter()
            .filter_map(|value| value.parse().ok())
            .collect();
        levels.sort_unstable();
        levels.dedup();
        levels
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct GenerationConfig {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub reasoning: Option<bool>,
    /// The level to reason at while `reasoning` is on. `None` keeps the
    /// provider's own default depth.
    pub reasoning_effort: Option<ReasoningEffort>,
    pub vision: Option<bool>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            max_tokens: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            reasoning: Some(false),
            reasoning_effort: None,
            vision: Some(true),
        }
    }
}

impl GenerationConfig {
    /// The level to send: a stored level applies only while reasoning is on.
    pub(crate) fn effort(&self) -> Option<ReasoningEffort> {
        if self.reasoning == Some(true) {
            self.reasoning_effort
        } else {
            None
        }
    }

    /// The settings this model can honor.
    ///
    /// Capabilities the model lacks are left out. A model that cannot turn
    /// reasoning off never receives an "off" setting: a chosen level still
    /// applies, and otherwise nothing is sent. A reasoning level the model does
    /// not list is an error: sending it would reach the provider as an
    /// undocumented value, and remapping it would silently change the choice.
    pub(crate) fn for_model(self, model: &ModelSelection) -> Result<Self> {
        let reasoning = match (model.reasoning, model.reasoning_always_on) {
            (false, _) => None,
            (true, true) if self.reasoning != Some(true) => self.reasoning_effort.map(|_| true),
            (true, _) => self.reasoning,
        };
        let applied = Self { reasoning, ..self };
        match applied.effort() {
            Some(effort) if !model.reasoning_efforts.contains(&effort) => {
                Err(Error::UnsupportedReasoningEffort {
                    provider: model.provider.into(),
                    effort,
                })
            }
            effort => Ok(Self {
                reasoning_effort: effort,
                vision: if model.vision { self.vision } else { None },
                ..applied
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub struct ModelSelection {
    pub provider: Provider,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub reasoning_efforts: Vec<ReasoningEffort>,
    /// The provider offers no off setting: the model always reasons.
    #[serde(default)]
    pub reasoning_always_on: bool,
}

impl Default for ModelSelection {
    fn default() -> Self {
        Self {
            provider: Provider::Local,
            model: Some(crate::local::DEFAULT_MODEL.to_owned()),
            quantization: Some(crate::local::DEFAULT_QUANTIZATION.to_owned()),
            vision: true,
            reasoning: true,
            reasoning_efforts: Vec::new(),
            reasoning_always_on: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Type)]
pub struct Quantization {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Type)]
pub struct Model {
    pub provider: Provider,
    pub model: Option<String>,
    pub name: String,
    pub quantizations: Vec<Quantization>,
    pub vision: bool,
    pub reasoning: bool,
    pub reasoning_efforts: Vec<ReasoningEffort>,
    /// The provider offers no off setting: the model always reasons.
    pub reasoning_always_on: bool,
}

impl Model {
    pub(crate) fn service(provider: Provider, name: &str) -> Self {
        Self {
            provider,
            model: None,
            name: name.to_owned(),
            quantizations: Vec::new(),
            vision: false,
            reasoning: false,
            reasoning_efforts: Vec::new(),
            reasoning_always_on: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuantizationDefinition {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
}

impl QuantizationDefinition {
    pub const fn new(id: &'static str, name: &'static str, filename: &'static str) -> Self {
        Self { id, name, filename }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ModelGeneration {
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl ModelGeneration {
    pub(crate) fn options(self, overrides: GenerationConfig) -> GenerationOptions {
        let defaults = GenerationOptions::default();
        GenerationOptions {
            max_tokens: overrides
                .max_tokens
                .or(self.max_tokens)
                .map_or(1000, |value| value as usize),
            temperature: overrides
                .temperature
                .or(self.temperature)
                .unwrap_or(defaults.temperature),
            top_k: overrides.top_k.or(self.top_k).map(|value| value as usize),
            top_p: overrides.top_p.or(self.top_p),
            min_p: overrides.min_p.or(self.min_p),
            repeat_penalty: overrides
                .repeat_penalty
                .or(self.repeat_penalty)
                .unwrap_or(defaults.repeat_penalty),
            frequency_penalty: overrides
                .frequency_penalty
                .or(self.frequency_penalty)
                .unwrap_or(defaults.frequency_penalty),
            presence_penalty: overrides
                .presence_penalty
                .or(self.presence_penalty)
                .unwrap_or(defaults.presence_penalty),
            ..defaults
        }
    }
}

#[must_use]
pub(crate) fn display_name(model: &str) -> String {
    model
        .rsplit_once('/')
        .map_or(model, |(_, model)| model)
        .split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            characters.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(characters).collect()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::ReasoningEffort::{High, Low, Medium, Minimal, XHigh};
    use super::*;

    fn selection(vision: bool, reasoning: bool, efforts: &[ReasoningEffort]) -> ModelSelection {
        ModelSelection {
            provider: Provider::LmStudio,
            model: Some("publisher/model".to_owned()),
            quantization: None,
            vision,
            reasoning,
            reasoning_efforts: efforts.to_vec(),
            reasoning_always_on: false,
        }
    }

    #[test]
    fn generation_omits_unsupported_model_capabilities() {
        let generation = GenerationConfig {
            vision: Some(true),
            reasoning: Some(false),
            ..GenerationConfig::default()
        };
        let supported = generation.for_model(&selection(true, true, &[])).unwrap();
        assert_eq!(supported.vision, Some(true));
        assert_eq!(supported.reasoning, Some(false));

        let unsupported = generation.for_model(&selection(false, false, &[])).unwrap();
        assert_eq!(unsupported.vision, None);
        assert_eq!(unsupported.reasoning, None);
    }

    #[test]
    fn reasoning_level_is_sent_only_while_reasoning_is_on_and_the_model_lists_it() {
        let model = selection(false, true, &[Low, High]);
        let chosen = GenerationConfig {
            reasoning: Some(true),
            reasoning_effort: Some(High),
            ..GenerationConfig::default()
        };
        assert_eq!(
            chosen.for_model(&model).unwrap().reasoning_effort,
            Some(High)
        );

        let off = GenerationConfig {
            reasoning: Some(false),
            ..chosen
        };
        assert_eq!(off.for_model(&model).unwrap().reasoning_effort, None);

        let incapable = selection(false, false, &[]);
        assert_eq!(chosen.for_model(&incapable).unwrap().reasoning_effort, None);

        let error = chosen
            .for_model(&selection(false, true, &[Low, Medium]))
            .unwrap_err();
        assert!(matches!(
            error,
            Error::UnsupportedReasoningEffort { effort: High, .. }
        ));
    }

    #[test]
    fn reasoning_that_cannot_be_turned_off_never_receives_an_off_setting() {
        let model = ModelSelection {
            reasoning_always_on: true,
            ..selection(false, true, &[Low, High])
        };
        let generation = |reasoning, reasoning_effort| GenerationConfig {
            reasoning,
            reasoning_effort,
            ..GenerationConfig::default()
        };
        let applied = |generation: GenerationConfig| {
            let applied = generation.for_model(&model).unwrap();
            (applied.reasoning, applied.reasoning_effort)
        };

        // The default "off" sends nothing, and a chosen level still applies.
        assert_eq!(applied(generation(Some(false), None)), (None, None));
        assert_eq!(
            applied(generation(Some(false), Some(High))),
            (Some(true), Some(High))
        );
        // An explicit "on" keeps working as before.
        assert_eq!(applied(generation(Some(true), None)), (Some(true), None));
        // A level the model does not list is still rejected.
        assert!(
            generation(Some(false), Some(Medium))
                .for_model(&model)
                .is_err()
        );
    }

    #[test]
    fn reasoning_level_survives_save_and_reload_and_older_files_load_without_one() {
        let legacy: GenerationConfig =
            serde_json::from_value(serde_json::json!({ "reasoning": true })).unwrap();
        assert_eq!(legacy.reasoning, Some(true));
        assert_eq!(legacy.reasoning_effort, None);

        let saved = serde_json::to_value(GenerationConfig {
            reasoning_effort: Some(XHigh),
            ..legacy
        })
        .unwrap();
        assert_eq!(saved["reasoning_effort"], "xhigh");
        let reloaded: GenerationConfig = serde_json::from_value(saved).unwrap();
        assert_eq!(reloaded.reasoning_effort, Some(XHigh));
    }

    #[test]
    fn reported_levels_are_ordered_and_leave_switches_out() {
        assert_eq!(
            ReasoningEffort::from_reported(["high", "on", "none", "low", "high", "off", "minimal"]),
            [Minimal, Low, High]
        );
    }

    #[test]
    fn model_selection_defaults_missing_capabilities() {
        let selection: ModelSelection = serde_json::from_value(serde_json::json!({
            "provider": "lm-studio",
            "model": "publisher/model",
            "quantization": null
        }))
        .unwrap();
        assert!(!selection.vision);
        assert!(!selection.reasoning);
        assert!(selection.reasoning_efforts.is_empty());
        assert!(!selection.reasoning_always_on);
    }
}
