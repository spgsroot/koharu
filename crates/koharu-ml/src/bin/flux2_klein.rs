use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use image::DynamicImage;
use koharu_ml::flux2_klein::{
    Flux2Klein, Flux2Klein9B, Flux2KleinInpaint, Flux2KleinInpaintOptions, Flux2KleinModelPaths,
    Flux2KleinOptions,
};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE", requires = "input")]
    mask: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(short, long, value_name = "TEXT")]
    prompt: String,

    #[arg(long, value_name = "FILE", conflicts_with = "nine_b")]
    reference: Option<PathBuf>,

    #[arg(long, default_value_t = 4)]
    steps: usize,

    #[arg(long)]
    strength: Option<f64>,

    #[arg(long, default_value_t = -1)]
    seed: i64,

    #[arg(long)]
    padding_mask_crop: Option<u32>,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long, default_value_t = false)]
    nine_b: bool,

    #[arg(long, default_value_t = 1.0)]
    guidance: f32,

    #[arg(long, value_name = "FILE", requires = "nine_b")]
    diffusion_model: Option<PathBuf>,

    #[arg(long, value_name = "FILE", requires = "nine_b")]
    text_encoder: Option<PathBuf>,

    #[arg(long, value_name = "FILE", requires = "nine_b")]
    vae: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        let executable = std::env::current_exe()?;
        if let Some(root) = executable.parent().and_then(|debug| debug.parent()) {
            let release_store = root.join("release").join("store");
            if release_store.is_dir() {
                koharu_runtime::Store::configure(release_store)?;
            }
        }
    }
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let input = cli.input.as_ref().map(image::open).transpose()?;
    let mask = cli.mask.as_ref().map(image::open).transpose()?;
    let reference = cli.reference.as_ref().map(image::open).transpose()?;

    koharu_ml::init().await?;

    let output = if cli.nine_b {
        let input = input.as_ref().context("--nine-b requires --input")?;
        let mask = mask.as_ref().context("--nine-b requires --mask")?;
        let pipeline = match (cli.diffusion_model, cli.text_encoder, cli.vae) {
            (Some(transformer), Some(text_encoder), Some(vae)) => Flux2Klein9B::load_from_paths(
                koharu_ml::device(cli.cpu),
                Flux2KleinModelPaths {
                    transformer,
                    text_encoder,
                    vae,
                },
            )?,
            (None, None, None) => Flux2Klein9B::load(koharu_ml::device(cli.cpu)).await?,
            _ => anyhow::bail!(
                "--diffusion-model, --text-encoder, and --vae must be supplied together"
            ),
        };
        pipeline.inference(
            &cli.prompt,
            input,
            mask,
            &Flux2KleinInpaintOptions {
                num_inference_steps: cli.steps,
                strength: cli.strength.unwrap_or(1.0),
                seed: cli.seed,
                guidance_scale: cli.guidance,
                padding_mask_crop: cli.padding_mask_crop,
                ..Flux2KleinInpaintOptions::default()
            },
        )?
    } else if let Some(mask) = mask.as_ref() {
        let input = input
            .as_ref()
            .context("--input is required when --mask is provided")?;
        let pipeline = Flux2KleinInpaint::load(koharu_ml::device(cli.cpu)).await?;
        let options = Flux2KleinInpaintOptions {
            num_inference_steps: cli.steps,
            strength: cli.strength.unwrap_or(0.8),
            seed: cli.seed,
            padding_mask_crop: cli.padding_mask_crop,
            guidance_scale: cli.guidance,
            ..Flux2KleinInpaintOptions::default()
        };
        pipeline.inference(&cli.prompt, input, reference.as_ref(), mask, &options)?
    } else {
        let pipeline = Flux2Klein::load(koharu_ml::device(cli.cpu)).await?;
        let options = Flux2KleinOptions {
            num_inference_steps: i32::try_from(cli.steps)?,
            seed: cli.seed,
            ..Flux2KleinOptions::default()
        };
        let references = input.into_iter().chain(reference).collect::<Vec<_>>();
        let generated = pipeline
            .inference(&references, &cli.prompt, &options)?
            .into_iter()
            .next()
            .context("FLUX.2 Klein returned no image")?;
        DynamicImage::ImageRgb8(generated)
    };
    output.save(cli.output)?;

    Ok(())
}
