//! Loaded inpainting backends shared by the inpainting and decensoring stages.
//!
//! Both stages crop, infer, and composite through the same machinery; only the
//! selected model, the mask source, and the commit rules differ. They also share
//! loaded weights: a model selected by both stages is resident once, with each
//! stage keeping its own prompt and sampling settings.

use std::{
    future::Future,
    sync::{Arc, Mutex, Weak},
};

use anyhow::{Context as _, Result, anyhow, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_ml::{
    aot_inpainting::AotInpainting,
    flux2_klein::{Flux2Klein9B, Flux2KleinInpaint, Flux2KleinInpaintOptions},
    lama::{InpaintRequest, LaMa},
    rorem_mixed::{RoremMixed, RoremMixedOptions},
};

use super::comfyui::ComfyUi;
use super::models::{
    DiffusionCreamPyConfig, Flux2Klein9BConfig, Flux2KleinConfig, RoremMixedConfig,
};
use super::tiling::{inpaint_tiled, inpaint_tiled_comfy};
use crate::{DecensorModel, InpaintingModel};

/// A model selected by the inpainting or decensoring stage.
///
/// Direct models only spread surrounding pixels and cannot invent obscured
/// anatomy, so decensoring selects from the generative profiles only.
#[derive(Clone, Copy)]
pub(super) enum Selection<'a> {
    LaMa,
    Aot,
    Flux(&'a Flux2KleinConfig),
    Flux9B(&'a Flux2Klein9BConfig),
    Animagine(&'a DiffusionCreamPyConfig),
    Illustrious(&'a DiffusionCreamPyConfig),
    Rorem(&'a RoremMixedConfig),
}

impl<'a> From<&'a InpaintingModel> for Selection<'a> {
    fn from(model: &'a InpaintingModel) -> Self {
        match model {
            InpaintingModel::LaMa {} => Self::LaMa,
            InpaintingModel::AotInpainting {} => Self::Aot,
            InpaintingModel::Flux2Klein(config) => Self::Flux(config),
            InpaintingModel::Flux2Klein9B(config) => Self::Flux9B(config),
            InpaintingModel::DiffusionCreamPyAnimagineXl4(config) => Self::Animagine(config),
            InpaintingModel::DiffusionCreamPyIllustriousXl2(config) => Self::Illustrious(config),
            InpaintingModel::RoremMixed(config) => Self::Rorem(config),
        }
    }
}

impl<'a> From<&'a DecensorModel> for Selection<'a> {
    fn from(model: &'a DecensorModel) -> Self {
        match model {
            DecensorModel::Flux2Klein9B(config) => Self::Flux9B(config),
            DecensorModel::Flux2Klein(config) => Self::Flux(config),
            DecensorModel::DiffusionCreamPyAnimagineXl4(config) => Self::Animagine(config),
            DecensorModel::DiffusionCreamPyIllustriousXl2(config) => Self::Illustrious(config),
            DecensorModel::RoremMixed(config) => Self::Rorem(config),
        }
    }
}

impl Selection<'_> {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::LaMa => "lama",
            Self::Aot => "aot-inpainting",
            Self::Flux(_) => "flux2-klein",
            Self::Flux9B(_) => "flux2-klein-9b-uncensored",
            Self::Animagine(_) => "diffusioncreampy-animagine-xl-4",
            Self::Illustrious(_) => "diffusioncreampy-illustrious-xl-2",
            Self::Rorem(_) => "rorem-mixed",
        }
    }
}

pub(super) fn validate(selection: Selection<'_>) -> Result<()> {
    match selection {
        Selection::LaMa | Selection::Aot => Ok(()),
        Selection::Flux(config) => {
            ensure!(!config.prompt.contains('\0'), "FLUX.2 prompt contains NUL");
            Ok(())
        }
        Selection::Flux9B(config) => {
            ensure!(!config.prompt.contains('\0'), "FLUX.2 prompt contains NUL");
            ensure!(
                config.num_inference_steps > 0,
                "FLUX.2 inference steps must be greater than zero"
            );
            ensure!(
                config.guidance_scale.is_finite() && config.guidance_scale > 0.0,
                "FLUX.2 guidance scale must be finite and greater than zero"
            );
            ensure!(
                config.strength > 0.0 && config.strength <= 1.0,
                "FLUX.2 strength must be greater than zero and at most one"
            );
            ensure!(
                config.lora_strength.is_finite() && config.lora_strength >= 0.0,
                "FLUX.2 LoRA strength must be finite and non-negative"
            );
            if let Some(path) = config
                .lora_path
                .as_ref()
                .filter(|_| config.lora_strength > 0.0)
            {
                ensure!(
                    path.is_file(),
                    "FLUX.2 LoRA file does not exist: {}",
                    path.display()
                );
            }
            Ok(())
        }
        Selection::Animagine(config) | Selection::Illustrious(config) => {
            ensure!(
                !config.endpoint.contains('\0')
                    && !config.checkpoint.contains('\0')
                    && !config.controlnet.contains('\0')
                    && !config.positive_prompt.contains('\0')
                    && !config.negative_prompt.contains('\0'),
                "DiffusionCreamPy settings contain NUL"
            );
            ensure!(
                !config.checkpoint.trim().is_empty() && !config.controlnet.trim().is_empty(),
                "DiffusionCreamPy model filenames must not be empty"
            );
            ensure!(
                config.num_inference_steps > 0,
                "DiffusionCreamPy inference steps must be greater than zero"
            );
            ensure!(
                config.guidance_scale.is_finite() && config.guidance_scale > 0.0,
                "DiffusionCreamPy guidance scale must be finite and greater than zero"
            );
            ensure!(
                config.denoise.is_finite() && (0.0..=1.0).contains(&config.denoise),
                "DiffusionCreamPy denoise must be between zero and one"
            );
            ensure!(
                config.color_match_strength.is_finite()
                    && (0.0..=1.0).contains(&config.color_match_strength),
                "DiffusionCreamPy color-match strength must be between zero and one"
            );
            ensure!(
                config.timeout_seconds > 0,
                "DiffusionCreamPy timeout must be greater than zero"
            );
            Ok(())
        }
        Selection::Rorem(config) => {
            ensure!(
                !config.prompt.contains('\0') && !config.negative_prompt.contains('\0'),
                "RORem prompt contains NUL"
            );
            Ok(())
        }
    }
}

/// Loaded model weights shared by every stage that selects the same model.
///
/// FLUX.2 Klein 9B alone keeps about 16 GB of parameters in host memory when
/// the accelerator cannot hold them, so a private copy per stage would double
/// that footprint and make both copies compete for the same memory. Slots hold
/// weak references: a model stays loaded while any stage uses it and is freed as
/// soon as every stage has unloaded it.
///
/// The pipeline owns one registry for every configuration generation. A run that
/// keeps an earlier generation while settings change, and the generation that
/// replaces it, therefore share the loaded weights instead of each holding a
/// copy; only their prompts and sampling options differ.
#[derive(Default)]
pub(crate) struct Models {
    lama: Shared<LaMa>,
    aot: Shared<AotInpainting>,
    flux: Shared<Flux2KleinInpaint>,
    flux9b: Shared<Flux2Klein9B>,
    rorem: Shared<RoremMixed>,
}

struct Shared<M>(tokio::sync::Mutex<Weak<Mutex<M>>>);

impl<M> Default for Shared<M> {
    fn default() -> Self {
        Self(tokio::sync::Mutex::new(Weak::new()))
    }
}

impl<M> Shared<M> {
    /// The loaded model, loading it first when no stage currently holds it.
    ///
    /// Concurrent callers wait for the one load instead of starting their own.
    async fn get_or_load(&self, load: impl Future<Output = Result<M>>) -> Result<Arc<Mutex<M>>> {
        let mut slot = self.0.lock().await;
        if let Some(model) = slot.upgrade() {
            return Ok(model);
        }
        let model = Arc::new(Mutex::new(load.await?));
        *slot = Arc::downgrade(&model);
        Ok(model)
    }
}

/// One page of masked inference handed to a backend.
pub(super) struct InpaintTask {
    pub(super) image: Arc<DynamicImage>,
    pub(super) mask: Arc<GrayImage>,
    pub(super) stop: crate::StopToken,
}

pub(super) enum Backend {
    LaMa(Arc<Mutex<LaMa>>),
    Aot(Arc<Mutex<AotInpainting>>),
    Flux {
        model: Arc<Mutex<Flux2KleinInpaint>>,
        config: Flux2KleinConfig,
    },
    Flux9B {
        model: Arc<Mutex<Flux2Klein9B>>,
        config: Flux2Klein9BConfig,
    },
    Comfy {
        client: ComfyUi,
        config: DiffusionCreamPyConfig,
    },
    Rorem {
        model: Arc<Mutex<RoremMixed>>,
        config: RoremMixedConfig,
    },
}

fn flux9b_options(config: &Flux2Klein9BConfig) -> Flux2KleinInpaintOptions {
    Flux2KleinInpaintOptions {
        num_inference_steps: config.num_inference_steps as usize,
        guidance_scale: config.guidance_scale,
        strength: config.strength,
        seed: config.seed,
        lora_path: config.lora_path.clone(),
        lora_strength: config.lora_strength,
        ..Flux2KleinInpaintOptions::default()
    }
}

impl Backend {
    pub(super) async fn load(
        device: koharu_ml::Device,
        selection: Selection<'_>,
        models: &Models,
    ) -> Result<Self> {
        match selection {
            Selection::LaMa => Ok(Self::LaMa(
                models.lama.get_or_load(LaMa::load(device)).await?,
            )),
            Selection::Aot => Ok(Self::Aot(
                models.aot.get_or_load(AotInpainting::load(device)).await?,
            )),
            Selection::Flux(config) => Ok(Self::Flux {
                model: models
                    .flux
                    .get_or_load(Flux2KleinInpaint::load(device))
                    .await?,
                config: config.clone(),
            }),
            Selection::Flux9B(config) => Ok(Self::Flux9B {
                model: models
                    .flux9b
                    .get_or_load(Flux2Klein9B::load(device))
                    .await?,
                config: config.clone(),
            }),
            Selection::Animagine(config) => Ok(Self::Comfy {
                client: ComfyUi::connect(config).await?,
                config: config.clone(),
            }),
            Selection::Illustrious(config) => Ok(Self::Comfy {
                client: ComfyUi::connect(config).await?,
                config: config.clone(),
            }),
            Selection::Rorem(config) => Ok(Self::Rorem {
                model: models.rorem.get_or_load(RoremMixed::load(device)).await?,
                config: config.clone(),
            }),
        }
    }

    pub(super) async fn inpaint(&self, task: InpaintTask) -> Result<RgbImage> {
        match self {
            Self::LaMa(model) => {
                let model = Arc::clone(model);
                tokio::task::spawn_blocking(move || -> Result<RgbImage> {
                    let model = model
                        .lock()
                        .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                    inpaint_tiled(&task.image, &task.mask, &task.stop, |image, mask| {
                        Ok(DynamicImage::ImageRgb8(model.inference(
                            image,
                            mask,
                            &InpaintRequest::default(),
                        )?))
                    })
                })
                .await
                .context("LaMa task panicked")?
            }
            Self::Aot(model) => {
                let model = Arc::clone(model);
                tokio::task::spawn_blocking(move || -> Result<RgbImage> {
                    let model = model
                        .lock()
                        .map_err(|_| anyhow!("AOT model lock is poisoned"))?;
                    inpaint_tiled(&task.image, &task.mask, &task.stop, |image, mask| {
                        Ok(DynamicImage::ImageRgb8(model.inference(image, mask)?))
                    })
                })
                .await
                .context("AOT task panicked")?
            }
            Self::Flux { model, config } => {
                let model = Arc::clone(model);
                let config = config.clone();
                tokio::task::spawn_blocking(move || -> Result<RgbImage> {
                    let model = model
                        .lock()
                        .map_err(|_| anyhow!("FLUX model lock is poisoned"))?;
                    inpaint_tiled(&task.image, &task.mask, &task.stop, |image, mask| {
                        model.inference(
                            &config.prompt,
                            image,
                            None,
                            &DynamicImage::ImageLuma8(mask.clone()),
                            &Flux2KleinInpaintOptions::default(),
                        )
                    })
                })
                .await
                .context("FLUX task panicked")?
            }
            Self::Flux9B { model, config } => {
                let model = Arc::clone(model);
                let config = config.clone();
                tokio::task::spawn_blocking(move || -> Result<RgbImage> {
                    let model = model
                        .lock()
                        .map_err(|_| anyhow!("FLUX.2 Klein 9B model lock is poisoned"))?;
                    let options = flux9b_options(&config);
                    inpaint_tiled(&task.image, &task.mask, &task.stop, |image, mask| {
                        model.inference(
                            &config.prompt,
                            image,
                            &DynamicImage::ImageLuma8(mask.clone()),
                            &options,
                        )
                    })
                })
                .await
                .context("FLUX.2 Klein 9B task panicked")?
            }
            Self::Comfy { client, config } => {
                inpaint_tiled_comfy(task.image, task.mask, &task.stop, client, config).await
            }
            Self::Rorem { model, config } => {
                let model = Arc::clone(model);
                let config = config.clone();
                tokio::task::spawn_blocking(move || -> Result<RgbImage> {
                    let model = model
                        .lock()
                        .map_err(|_| anyhow!("RORem model lock is poisoned"))?;
                    inpaint_tiled(&task.image, &task.mask, &task.stop, |image, mask| {
                        Ok(DynamicImage::ImageRgb8(model.inference(
                            image,
                            mask,
                            &config.prompt,
                            &config.negative_prompt,
                            &RoremMixedOptions::default(),
                        )?))
                    })
                })
                .await
                .context("RORem task panicked")?
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn weights_are_shared_until_every_holder_releases_them() {
        let shared = Shared::<u32>::default();

        let first = shared.get_or_load(async { Ok(1) }).await.unwrap();
        let second = shared
            .get_or_load(async { Err(anyhow!("a loaded model must not load again")) })
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(*second.lock().unwrap(), 1);

        drop(first);
        let still_held = shared.get_or_load(async { Ok(2) }).await.unwrap();
        assert!(Arc::ptr_eq(&still_held, &second));

        drop((second, still_held));
        let reloaded = shared.get_or_load(async { Ok(3) }).await.unwrap();
        assert_eq!(*reloaded.lock().unwrap(), 3);
    }

    #[tokio::test]
    async fn a_failed_load_leaves_the_slot_empty() {
        let shared = Shared::<u32>::default();

        assert!(
            shared
                .get_or_load(async { Err(anyhow!("weights are missing")) })
                .await
                .is_err()
        );

        let model = shared.get_or_load(async { Ok(7) }).await.unwrap();
        assert_eq!(*model.lock().unwrap(), 7);
    }

    #[test]
    fn disabling_lora_does_not_require_the_adapter_file() {
        let mut config = Flux2Klein9BConfig::decensor();
        config.lora_path = Some(
            std::env::temp_dir()
                .join(uuid::Uuid::new_v4().to_string())
                .join("missing.safetensors"),
        );
        assert!(validate(Selection::Flux9B(&config)).is_err());

        config.lora_strength = 0.0;
        assert!(validate(Selection::Flux9B(&config)).is_ok());

        for strength in [-1.0, f32::NAN, f32::INFINITY] {
            config.lora_strength = strength;
            assert!(validate(Selection::Flux9B(&config)).is_err());
        }
    }
}
