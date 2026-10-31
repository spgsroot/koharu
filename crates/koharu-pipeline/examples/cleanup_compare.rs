//! Detection + inpainting over a list of pages, for the measurements in
//! `docs/fork/cleanup-evidence.md`. Writes each page's lettering masks, the
//! page composited with its cleanup layer, and per-stage timings.
//!
//! Usage: cleanup_compare <store> <out_dir> <label> <bubbles>:<artwork> <inputs...>
//! bubbles: lama|aot; artwork: lama|aot|flux2-klein|flux2-klein-9b

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context as _, Result, bail};
use koharu_config::Config;
use koharu_pipeline::{
    ArtworkInpaintingConfig, BubbleInpaintingKind, Committed, Committer, InpaintingConfig,
    InpaintingKind, Operation, Pipeline, PipelineConfig, Progress, Request, Scope, Stage,
    StageOutput,
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, PageDraft, RasterLayer, RasterLayerKind, Session,
};
use koharu_translator::ProvidersConfig;

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<Committed> {
        Ok(Committed::Applied(
            self.0.commit(output.patch).await?.snapshot,
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        bail!("usage: cleanup_compare <store> <out_dir> <label> <bubbles>:<artwork> <inputs...>");
    }
    koharu_runtime::Store::configure(PathBuf::from(&args[1]))?;
    let out = PathBuf::from(&args[2]);
    fs::create_dir_all(&out)?;
    let label = args[3].clone();
    let (bubbles, artwork) = args[4]
        .split_once(':')
        .context("variant is <bubbles>:<artwork>")?;
    let bubbles = match bubbles {
        "lama" => BubbleInpaintingKind::LaMa,
        "aot" => BubbleInpaintingKind::AotInpainting,
        other => bail!("unknown bubble model {other}"),
    };
    let artwork = match artwork {
        "lama" => InpaintingKind::LaMa,
        "aot" => InpaintingKind::AotInpainting,
        "flux2-klein" => InpaintingKind::Flux2Klein,
        "flux2-klein-9b" => InpaintingKind::Flux2Klein9B,
        other => bail!("unknown artwork model {other}"),
    };
    koharu_ml::init().await?;
    let config = PipelineConfig {
        inpainting: InpaintingConfig {
            bubbles,
            artwork: ArtworkInpaintingConfig {
                model: artwork,
                ..ArtworkInpaintingConfig::default()
            },
            ..InpaintingConfig::default()
        },
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::from_config(
        Config::memory(config),
        Config::memory(ProvidersConfig::default()),
        koharu_ml::device(false),
    )?;
    let mut timings = String::new();
    for input in &args[5..] {
        let input = Path::new(input);
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
        let source = fs::read(input)?;
        let decoded = image::load_from_memory(&source)?;
        let mut session = Session::memory().await?;
        let mut page = None;
        let patch = session.snapshot().patch(|edit| {
            let id = edit.add_page(
                PageDraft::new(
                    stem,
                    f64::from(decoded.width()),
                    f64::from(decoded.height()),
                ),
                At::End,
            )?;
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                AssetInput::new(
                    Arc::<[u8]>::from(source.clone()),
                    "image/png",
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
        let page = page.context("page id")?;
        let stages = Arc::new(Mutex::new(Vec::<(String, f64)>::new()));
        let sink = stages.clone();
        let started = Instant::now();
        let snapshot = session.snapshot();
        let mut committer = SessionCommitter(&mut session);
        pipeline
            .execute(
                snapshot,
                Request {
                    operation: Operation::Through {
                        stage: Stage::Inpainting,
                    },
                    scope: Scope::Pages(vec![page]),
                    progress: Some(Arc::new(move |event| {
                        if let Progress::Finished { stage, elapsed, .. } = event {
                            sink.lock()
                                .unwrap()
                                .push((stage.to_string(), elapsed.as_secs_f64()));
                        }
                    })),
                    ..Request::default()
                },
                &mut committer,
            )
            .await?;
        let total = started.elapsed().as_secs_f64();
        let snapshot = session.snapshot();
        for role in ["text-mask", "bubble-text-mask", "artwork-text-mask"] {
            if let Some(asset) = snapshot.asset(page, &AssetRole::new(role)?)? {
                let bytes = snapshot.read_blob(asset.blob).await?;
                fs::write(out.join(format!("{stem}.{label}.{role}.png")), &bytes)?;
            }
        }
        let mut clean = decoded.to_rgba8();
        for entity in snapshot.children(page)?.collect::<Vec<_>>() {
            if snapshot
                .component::<RasterLayer>(entity)?
                .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
                && let Some(asset) = snapshot.asset(entity, &AssetRole::new("source")?)?
            {
                let layer = image::load_from_memory(&snapshot.read_blob(asset.blob).await?)?;
                image::imageops::overlay(&mut clean, &layer.to_rgba8(), 0, 0);
            }
        }
        clean.save(out.join(format!("{stem}.{label}.clean.png")))?;
        let stages = stages.lock().unwrap().clone();
        timings.push_str(&format!(
            "{{\"page\":\"{stem}\",\"label\":\"{label}\",\"total\":{total:.3},\"stages\":{}}}\n",
            serde_json::to_string(&stages)?
        ));
        eprintln!("{stem}: {total:.2}s {stages:?}");
    }
    fs::write(out.join(format!("{label}.timings.jsonl")), timings)?;
    Ok(())
}
