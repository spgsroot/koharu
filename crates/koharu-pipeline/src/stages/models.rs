//! Settings for the generative models that the inpainting and decensoring
//! stages can select.
//!
//! Each stage keeps its own copy of every model's settings (see
//! `ArtworkInpaintingConfig`, `InpaintingConfig::manual` and `DecensorConfig`),
//! so a model used by several stages carries independent prompts. The
//! constructors below define what each stage starts from: they differ only where
//! the stage's purpose changes the prompt.

use std::path::PathBuf;

use koharu_ml::rorem_mixed::{DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT};
use serde::{Deserialize, Serialize};
use specta::Type;

/// Instruction for the inpainting stage, which removes lettering.
const TEXT_CLEANUP_PROMPT: &str = "Remove all lettering, Japanese sound effects, onomatopoeia and isolated glyphs, including their outlines and filled interiors. Reconstruct the background and line art hidden by the lettering. Match the surrounding drawing and shading. Preserve speech bubble borders, panel borders and all unrelated artwork. Do not remove censorship or add new details.";

/// Instruction for the manual inpainting brush, which repaints whatever the
/// artist paints over. The artist is expected to replace it per repair.
const MANUAL_INPAINTING_PROMPT: &str = "Remove the unwanted lettering, marks and objects, then reconstruct the background and line art hidden beneath them. Match the surrounding drawing, shading and composition. Preserve speech bubble borders, panel borders and all unrelated artwork. Do not add new details.";

/// Instruction for the decensor stage, which repaints censor bars.
pub(crate) const DECENSOR_PROMPT: &str = "Remove the black censor bars and reconstruct the original artwork. Match surrounding line art, shading, anatomy and composition exactly.";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux2KleinConfig {
    pub prompt: String,
}

impl Flux2KleinConfig {
    pub fn text_cleanup() -> Self {
        Self {
            prompt: TEXT_CLEANUP_PROMPT.to_owned(),
        }
    }

    pub fn decensor() -> Self {
        Self {
            prompt: DECENSOR_PROMPT.to_owned(),
        }
    }
}

/// A table that omits fields falls back to the text-cleanup values; the stage
/// configurations supply their own defaults for entire missing tables.
impl Default for Flux2KleinConfig {
    fn default() -> Self {
        Self::text_cleanup()
    }
}

/// FLUX.2 Klein 9B settings. Every stage that runs the model keeps its own copy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux2Klein9BConfig {
    pub prompt: String,
    pub num_inference_steps: i32,
    pub guidance_scale: f32,
    pub lora_path: Option<PathBuf>,
    pub lora_strength: f32,
    /// Sampler seed. A fixed seed reproduces a repair; a negative one draws a
    /// new random seed for every generated tile.
    #[specta(type = f64)]
    pub seed: i64,
    /// Above zero, at most one. One generates the masked area from the prompt
    /// and the surrounding image; lower values keep more of the source pixels
    /// beneath the mask.
    pub strength: f64,
}

impl Flux2Klein9BConfig {
    pub fn text_cleanup() -> Self {
        Self {
            prompt: TEXT_CLEANUP_PROMPT.to_owned(),
            num_inference_steps: 8,
            guidance_scale: 1.0,
            lora_path: None,
            lora_strength: 1.0,
            seed: -1,
            strength: 1.0,
        }
    }

    pub fn decensor() -> Self {
        Self {
            prompt: DECENSOR_PROMPT.to_owned(),
            ..Self::text_cleanup()
        }
    }

    pub fn manual_inpainting() -> Self {
        Self {
            prompt: MANUAL_INPAINTING_PROMPT.to_owned(),
            ..Self::text_cleanup()
        }
    }
}

/// A table that omits fields falls back to the text-cleanup values; the stage
/// configurations supply their own defaults for entire missing tables.
impl Default for Flux2Klein9BConfig {
    fn default() -> Self {
        Self::text_cleanup()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct DiffusionCreamPyConfig {
    pub endpoint: String,
    pub checkpoint: String,
    pub controlnet: String,
    pub positive_prompt: String,
    pub negative_prompt: String,
    pub seed: u32,
    pub num_inference_steps: u32,
    pub guidance_scale: f32,
    pub denoise: f32,
    pub mask_grow: u32,
    pub mask_blur: u32,
    pub mask_padding: u32,
    pub color_match_strength: f32,
    pub timeout_seconds: u32,
}

impl DiffusionCreamPyConfig {
    pub fn animagine_xl_4() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8188".to_owned(),
            checkpoint: "animagine-xl-4.0-opt.safetensors".to_owned(),
            controlnet: "controlnet-union-sdxl-1.0.safetensors".to_owned(),
            positive_prompt: String::new(),
            negative_prompt:
                "censored, bar censor, text, watermark, signature, lowres, worst quality, low quality"
                    .to_owned(),
            seed: 424_242,
            num_inference_steps: 10,
            guidance_scale: 7.0,
            denoise: 0.5,
            mask_grow: 0,
            mask_blur: 16,
            mask_padding: 128,
            color_match_strength: 1.0,
            timeout_seconds: 900,
        }
    }

    pub fn illustrious_xl_2() -> Self {
        Self {
            checkpoint: "Illustrious-XL-v2.0.safetensors".to_owned(),
            ..Self::animagine_xl_4()
        }
    }
}

impl Default for DiffusionCreamPyConfig {
    fn default() -> Self {
        Self::animagine_xl_4()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct RoremMixedConfig {
    pub prompt: String,
    pub negative_prompt: String,
}

impl RoremMixedConfig {
    pub fn decensor() -> Self {
        Self {
            prompt: DECENSOR_PROMPT.to_owned(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_owned(),
        }
    }
}

impl Default for RoremMixedConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_PROMPT.to_owned(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_owned(),
        }
    }
}
