use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    ArtworkInpaintingConfig, BubbleInpaintingKind, CensorDetectionConfig, Committed, Committer,
    DecensorConfig, DecensorKind, DetectionModel, InpaintingConfig, InpaintingKind,
    KoharuLayoutRFDetrSeg2XLConfig, OcrModel, Operation, Pipeline, PipelineConfig, Progress,
    Request, Scope, StageOutput, SystemPrompts, TranslationConfig,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{
    ContentProfile, GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig,
};

#[derive(Debug, Parser)]
#[command(version, about = "Run Koharu's complete in-process pipeline")]
struct Arguments {
    #[arg(short, long, value_name = "INPUT")]
    input: PathBuf,

    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    #[arg(long, value_enum, default_value = "koharu-layout-rfdetr-seg-2xl")]
    detection: DetectionChoice,

    #[arg(long, value_enum, default_value = "paddleocr-vl-1.6")]
    ocr: OcrChoice,

    #[arg(long, value_enum, default_value = "lama")]
    bubble_inpainting: BubbleInpaintingChoice,

    #[arg(long, value_enum, default_value = "flux2-klein-9b-uncensored")]
    artwork_inpainting: ArtworkInpaintingChoice,

    #[arg(long, value_enum, default_value = "flux2-klein-9b-uncensored")]
    decensor: DecensorChoice,

    #[arg(long, default_value = "ja-JP")]
    source_language: Language,

    #[arg(long, default_value = "en-US")]
    target_language: Language,

    #[arg(long)]
    translation_instructions: Option<String>,

    #[arg(long, value_enum, default_value = "manga")]
    content_profile: ContentProfileChoice,

    /// Replaces the selected content profile's built-in prompt.
    #[arg(long)]
    translation_system_prompt: Option<String>,

    #[arg(long, default_value = "gemma4-12b-it")]
    llm: String,

    #[arg(long)]
    cpu: bool,
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<Committed> {
        Ok(Committed::Applied(
            self.0.commit(output.patch).await?.snapshot,
        ))
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DetectionChoice {
    #[value(name = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrChoice {
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[value(name = "manga-ocr")]
    MangaOcr,
    #[value(name = "baberu-ocr")]
    BaberuOcr,
    #[value(name = "hayai-ocr")]
    HayaiOcr,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BubbleInpaintingChoice {
    #[value(name = "lama")]
    LaMa,
    #[value(name = "aot-inpainting")]
    AotInpainting,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ArtworkInpaintingChoice {
    #[value(name = "flux2-klein-9b-uncensored")]
    Flux2Klein9B,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "diffusioncreampy-animagine-xl-4")]
    DiffusionCreamPyAnimagineXl4,
    #[value(name = "diffusioncreampy-illustrious-xl-2")]
    DiffusionCreamPyIllustriousXl2,
    #[value(name = "rorem-mixed")]
    RoremMixed,
    #[value(name = "lama")]
    LaMa,
    #[value(name = "aot-inpainting")]
    AotInpainting,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DecensorChoice {
    #[value(name = "flux2-klein-9b-uncensored")]
    Flux2Klein9B,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "diffusioncreampy-animagine-xl-4")]
    DiffusionCreamPyAnimagineXl4,
    #[value(name = "diffusioncreampy-illustrious-xl-2")]
    DiffusionCreamPyIllustriousXl2,
    #[value(name = "rorem-mixed")]
    RoremMixed,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ContentProfileChoice {
    Hentai,
    Manga,
    Comic,
}

impl Arguments {
    fn pipeline_config(&self) -> PipelineConfig {
        let content_profile = match self.content_profile {
            ContentProfileChoice::Hentai => ContentProfile::Hentai,
            ContentProfileChoice::Manga => ContentProfile::Manga,
            ContentProfileChoice::Comic => ContentProfile::Comic,
        };
        let mut system_prompts = SystemPrompts::default();
        match content_profile {
            ContentProfile::Hentai => {
                system_prompts.hentai = self.translation_system_prompt.clone()
            }
            ContentProfile::Manga => system_prompts.manga = self.translation_system_prompt.clone(),
            ContentProfile::Comic => system_prompts.comic = self.translation_system_prompt.clone(),
        }
        PipelineConfig {
            detection: match self.detection {
                DetectionChoice::KoharuLayoutRFDetrSeg2XL => {
                    DetectionModel::KoharuLayoutRFDetrSeg2XL(
                        KoharuLayoutRFDetrSeg2XLConfig::default(),
                    )
                }
            },
            ocr: match self.ocr {
                OcrChoice::PaddleOcrVl1_6 => OcrModel::PaddleOcrVl1_6,
                OcrChoice::MangaOcr => OcrModel::MangaOcr,
                OcrChoice::BaberuOcr => OcrModel::BaberuOcr,
                OcrChoice::HayaiOcr => OcrModel::HayaiOcr,
            },
            translation: TranslationConfig {
                model: ModelSelection {
                    provider: Provider::Local,
                    model: Some(self.llm.clone()),
                    quantization: None,
                    vision: true,
                    reasoning: true,
                    reasoning_efforts: Vec::new(),
                    reasoning_always_on: false,
                },
                generation: GenerationConfig::default(),
                source_language: self.source_language,
                target_language: self.target_language,
                content_profile,
                system_prompts,
                instructions: self.translation_instructions.clone(),
                validators: Vec::new(),
            },
            inpainting: InpaintingConfig {
                bubbles: match self.bubble_inpainting {
                    BubbleInpaintingChoice::LaMa => BubbleInpaintingKind::LaMa,
                    BubbleInpaintingChoice::AotInpainting => BubbleInpaintingKind::AotInpainting,
                },
                artwork: ArtworkInpaintingConfig {
                    model: match self.artwork_inpainting {
                        ArtworkInpaintingChoice::Flux2Klein9B => InpaintingKind::Flux2Klein9B,
                        ArtworkInpaintingChoice::Flux2Klein => InpaintingKind::Flux2Klein,
                        ArtworkInpaintingChoice::DiffusionCreamPyAnimagineXl4 => {
                            InpaintingKind::DiffusionCreamPyAnimagineXl4
                        }
                        ArtworkInpaintingChoice::DiffusionCreamPyIllustriousXl2 => {
                            InpaintingKind::DiffusionCreamPyIllustriousXl2
                        }
                        ArtworkInpaintingChoice::RoremMixed => InpaintingKind::RoremMixed,
                        ArtworkInpaintingChoice::LaMa => InpaintingKind::LaMa,
                        ArtworkInpaintingChoice::AotInpainting => InpaintingKind::AotInpainting,
                    },
                    ..ArtworkInpaintingConfig::default()
                },
                ..InpaintingConfig::default()
            },
            decensor: DecensorConfig {
                model: match self.decensor {
                    DecensorChoice::Flux2Klein9B => DecensorKind::Flux2Klein9B,
                    DecensorChoice::Flux2Klein => DecensorKind::Flux2Klein,
                    DecensorChoice::DiffusionCreamPyAnimagineXl4 => {
                        DecensorKind::DiffusionCreamPyAnimagineXl4
                    }
                    DecensorChoice::DiffusionCreamPyIllustriousXl2 => {
                        DecensorKind::DiffusionCreamPyIllustriousXl2
                    }
                    DecensorChoice::RoremMixed => DecensorKind::RoremMixed,
                },
                ..DecensorConfig::default()
            },
            censor_detection: CensorDetectionConfig::default(),
            processor: Default::default(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    initialize_with_retry().await;

    let source = fs::read(&arguments.input)
        .with_context(|| format!("failed to read {}", arguments.input.display()))?;
    let decoded = image::load_from_memory(&source).context("failed to decode input image")?;
    let name = arguments
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("input");
    let mut session = Session::memory().await?;
    let mut page = None;
    let patch = session.snapshot().patch(|edit| {
        let id = edit.add_page(
            PageDraft::new(
                name,
                f64::from(decoded.width()),
                f64::from(decoded.height()),
            ),
            At::End,
        )?;
        edit.set_asset(
            id,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(source),
                image_media_type(&arguments.input),
                AssetMetadata {
                    width: Some(decoded.width()),
                    height: Some(decoded.height()),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        page = Some(id);
        Ok(())
    })?;
    session.commit(patch).await?;
    let page = page.expect("page ID is assigned by the edit");

    let device = koharu_ml::device(arguments.cpu);
    let pipeline = Pipeline::from_config(
        Config::memory(arguments.pipeline_config()),
        Config::memory(ProvidersConfig::default()),
        device,
    )?;
    let snapshot = session.snapshot();
    let mut committer = SessionCommitter(&mut session);
    let report = pipeline
        .execute(
            snapshot,
            Request {
                operation: Operation::Full,
                scope: Scope::Pages(vec![page]),
                progress: Some(Arc::new(|event| {
                    if let Progress::Finished { stage, elapsed, .. } = event {
                        eprintln!("{stage} finished in {:.2}s", elapsed.as_secs_f64());
                    }
                })),
                ..Request::default()
            },
            &mut committer,
        )
        .await?;
    eprintln!("pipeline finished in {:.2}s", report.elapsed.as_secs_f64());

    let renderer = Renderer::new()?;
    let render_started = Instant::now();
    let snapshot = session.snapshot();
    let frame = renderer.render(&snapshot, page).await?;
    let rasterizer = Rasterizer::new()?;
    let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::default())?;
    let render_elapsed = render_started.elapsed();
    raster
        .image
        .save(&arguments.output)
        .with_context(|| format!("failed to write {}", arguments.output.display()))?;
    eprintln!(
        "rendered {} in {:.2}s",
        arguments.output.display(),
        render_elapsed.as_secs_f64()
    );
    Ok(())
}

async fn initialize_with_retry() {
    let mut delay = Duration::from_secs(1);
    let mut attempt = 0_u64;
    loop {
        attempt += 1;
        match koharu_ml::init().await {
            Ok(()) => return,
            Err(error) => {
                let jitter = Duration::from_millis((attempt.wrapping_mul(137)) % 251);
                let wait = delay + jitter;
                eprintln!(
                    "runtime initialization attempt {attempt} failed: {error}; retrying in {:.1}s",
                    wait.as_secs_f64()
                );
                tokio::time::sleep(wait).await;
                delay = delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        }
    }
}

fn image_media_type(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}
