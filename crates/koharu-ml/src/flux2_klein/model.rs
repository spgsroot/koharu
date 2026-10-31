//! Native FLUX.2 Klein model assembly.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/cc734292286f85f9c48305d94d7fd22f42838522/docs/flux2.md
//! Qwen3 chat templating and hidden-state layers 9, 18, and 27 are mapped by:
//! https://github.com/leejet/stable-diffusion.cpp/blob/cc734292286f85f9c48305d94d7fd22f42838522/src/conditioning/conditioner.hpp#L2287-L2301
//! The decoder width is inferred from the small-decoder checkpoint by:
//! https://github.com/leejet/stable-diffusion.cpp/blob/cc734292286f85f9c48305d94d7fd22f42838522/src/model/vae/auto_encoder_kl.hpp#L518-L569

use std::{path::PathBuf, sync::Mutex};

use anyhow::{Context as _, Result, anyhow, ensure};
use koharu_diffusion::{Context, ContextParams, ImageGenerationParams, RgbImage, VaeFormat};

use crate::Backend;

#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

#[derive(Debug)]
pub struct Model {
    context: Mutex<Context>,
}

impl Model {
    pub fn new(device: &crate::Device, paths: ModelPaths) -> Result<Self> {
        let context = Context::new(&context_params(device, paths))
            .context("failed to load FLUX.2 Klein components")?;
        ensure!(
            context.supports_image_generation(),
            "the loaded FLUX.2 Klein context does not support image generation"
        );
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub fn forward(&self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| anyhow!("FLUX.2 Klein context lock was poisoned"))?;
        context
            .generate_image(params)
            .context("FLUX.2 Klein inference failed")
    }
}

/// Upper bound for budgeted parameter streaming, in GiB.
///
/// stable-diffusion.cpp clamps the streaming budget at every compute to the
/// VRAM free at that moment minus a 512 MiB margin, so this bound only enables
/// the planner; the live free memory decides residency.
/// https://github.com/leejet/stable-diffusion.cpp/blob/97d2990807fe6d558e395f8764198d7c7e7b411c/src/core/ggml_extend.hpp#L2452-L2487
const STREAMING_BUDGET_GIB: &str = "1024";

fn context_params(device: &crate::Device, paths: ModelPaths) -> ContextParams {
    let use_accelerator = device.backend != Backend::Cpu;
    let use_cuda = device.backend == Backend::Cuda;
    ContextParams {
        diffusion_model_path: Some(paths.transformer),
        llm_path: Some(paths.text_encoder),
        vae_path: Some(paths.vae),
        enable_mmap: true,
        flash_attention: use_cuda,
        diffusion_flash_attention: use_cuda,
        diffusion_conv_direct: use_cuda,
        vae_conv_direct: use_cuda,
        vae_format: VaeFormat::Flux2,
        backend: Some(device.name.clone()),
        // Parameters stay in RAM and stream to the accelerator within the VRAM
        // that is free when each graph runs. Staging a whole denoiser instead
        // oversubscribes the card once other resident models hold VRAM, and the
        // driver then pages allocations through system memory: the same tile
        // ran 90 times slower on a 16 GB card.
        params_backend: use_accelerator.then(|| "*=cpu".to_owned()),
        max_vram: use_accelerator.then(|| STREAMING_BUDGET_GIB.to_owned()),
        stream_layers: use_accelerator,
        ..ContextParams::default()
    }
}
