use super::*;

#[test]
fn configuration_ignores_unknown_fields() {
    let config = toml::from_str::<PipelineConfig>("legacy_limit = 1").unwrap();
    assert_eq!(config, PipelineConfig::default());
}

#[tokio::test]
async fn stop_is_a_successful_partial_result() {
    let pipeline = pipeline(Default::default());
    let stop = StopToken::default();
    stop.stop();
    let request = Request {
        stop,
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            request,
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 0);
}

#[tokio::test]
async fn stop_after_a_page_keeps_completed_progress() {
    let translation = TranslationConfig {
        model: koharu_translator::ModelSelection {
            provider: koharu_translator::Provider::OpenAi,
            model: Some("gpt-5.6-luna".to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
            reasoning_efforts: Vec::new(),
            reasoning_always_on: false,
        },
        ..Default::default()
    };
    let pipeline = pipeline(translation);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let patch = session
        .snapshot()
        .patch(|edit| {
            edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let stop = StopToken::default();
    let progress_stop = stop.clone();
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Translation,
        },
        stop,
        progress: Some(std::sync::Arc::new(move |event| {
            if matches!(event, Progress::Skipped { .. }) {
                progress_stop.stop();
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(session.snapshot(), request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 1);
    assert_eq!(report.total, 2);
}

struct RejectCommitter;

fn pipeline(translation: TranslationConfig) -> Pipeline {
    let config = PipelineConfig {
        translation,
        ..PipelineConfig::default()
    };
    Pipeline::from_config(
        koharu_config::Config::memory(config),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        koharu_ml::Device::cpu(),
    )
    .unwrap()
}

#[async_trait::async_trait]
impl Committer for RejectCommitter {
    async fn commit(&mut self, _output: StageOutput) -> anyhow::Result<crate::Committed> {
        anyhow::bail!("stopped execution must not commit")
    }
}

#[test]
fn operations_expand_to_the_supported_workflows() {
    assert_eq!(
        Operation::Through {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Inpainting,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Inpainting],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Decensor
        }
        .stages()
        .unwrap(),
        vec![
            Stage::Detection,
            Stage::CensorDetection,
            Stage::Inpainting,
            Stage::Decensor
        ],
    );
    assert_eq!(
        Operation::Only {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Translation],
    );
    assert_eq!(
        Operation::Stages {
            stages: vec![Stage::Translation, Stage::Detection, Stage::Translation],
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Translation],
    );
}

/// A scene of `count` pages, each with an 8x8 black source image.
async fn pages_with_source(count: usize) -> (koharu_scene::Snapshot, Vec<koharu_scene::EntityId>) {
    let mut source = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(8, 8)
        .write_to(&mut source, image::ImageFormat::Png)
        .unwrap();
    let source = source.into_inner();
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut pages = Vec::new();
    let patch = session
        .snapshot()
        .patch(|edit| {
            for index in 0..count {
                let page = edit.add_page(
                    koharu_scene::PageDraft::new(format!("page {index}"), 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                edit.set_asset(
                    page,
                    &koharu_scene::AssetRole::new("source")?,
                    koharu_scene::AssetInput::new(
                        source.clone(),
                        "image/png",
                        koharu_scene::AssetMetadata {
                            width: Some(8),
                            height: Some(8),
                            attributes: Default::default(),
                        },
                    ),
                )?;
                pages.push(page);
            }
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    (session.snapshot(), pages)
}

/// A stroke painted on `page`, `side` pixels square. The test pages are 8x8,
/// so any other side is rejected by the stage that receives the stroke.
fn painted_mask(page: koharu_scene::EntityId, side: u32, target: CleanupTarget) -> InpaintingMask {
    let mut stroke = image::GrayImage::new(side, side);
    stroke.put_pixel(1, 1, image::Luma([255]));
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(stroke)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    InpaintingMask {
        page,
        target,
        png: png.into_inner().into(),
    }
}

#[tokio::test]
async fn a_painted_stroke_reaches_the_cleanup_stage_its_operation_names() {
    let (snapshot, pages) = pages_with_source(1).await;
    let page = pages[0];
    let pipeline = pipeline(Default::default());

    // The undersized stroke is rejected by the stage that receives it, before
    // any model loads. A stage that never saw the stroke would skip the page
    // for lack of saved masks, and a run that refused the stroke would fail as
    // invalid input instead.
    for (stage, target) in [
        (Stage::Inpainting, CleanupTarget::Fast),
        (Stage::Decensor, CleanupTarget::Decensor),
    ] {
        let request = Request {
            operation: Operation::Only { stage },
            scope: Scope::Region {
                page,
                bounds: Bounds {
                    x: 0.0,
                    y: 0.0,
                    width: 8.0,
                    height: 8.0,
                },
            },
            inpainting_mask: Some(painted_mask(page, 4, target)),
            ..Request::default()
        };

        let error = pipeline
            .execute(snapshot.clone(), request, &mut RejectCommitter)
            .await
            .unwrap_err();

        assert_eq!(
            (error.kind, error.stage),
            (ErrorKind::Processing, Some(stage))
        );
    }
}

#[tokio::test]
async fn a_painted_stroke_needs_its_page_in_scope_and_exactly_one_cleanup_stage() {
    let (snapshot, pages) = pages_with_source(2).await;
    let pipeline = pipeline(Default::default());

    for (operation, painted_on) in [
        // Painted on a page outside the run.
        (
            Operation::Only {
                stage: Stage::Decensor,
            },
            pages[1],
        ),
        // Both cleanup stages would repair the same stroke.
        (Operation::Full, pages[0]),
        // No cleanup stage would repair it.
        (
            Operation::Only {
                stage: Stage::Translation,
            },
            pages[0],
        ),
        // One cleanup stage still cannot consume another target's mask.
        (
            Operation::Only {
                stage: Stage::Inpainting,
            },
            pages[0],
        ),
    ] {
        let request = Request {
            operation,
            scope: Scope::Pages(vec![pages[0]]),
            inpainting_mask: Some(painted_mask(painted_on, 8, CleanupTarget::Decensor)),
            ..Request::default()
        };

        let error = pipeline
            .execute(snapshot.clone(), request, &mut RejectCommitter)
            .await
            .unwrap_err();

        assert_eq!(error.kind, ErrorKind::InvalidInput);
    }
}

#[tokio::test]
async fn a_painted_stroke_is_handed_to_its_own_page_only() {
    let (snapshot, pages) = pages_with_source(2).await;
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Progress>::new()));
    let sink = std::sync::Arc::clone(&events);
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Decensor,
        },
        scope: Scope::Pages(pages.clone()),
        progress: Some(std::sync::Arc::new(move |event| {
            sink.lock().unwrap().push(event);
        })),
        inpainting_mask: Some(painted_mask(pages[1], 4, CleanupTarget::Decensor)),
        ..Request::default()
    };

    let error = pipeline(Default::default())
        .execute(snapshot, request, &mut RejectCommitter)
        .await
        .unwrap_err();

    // The first page has neither a saved censor mask nor a stroke, so it is
    // skipped; had it received the stroke it would have failed instead.
    assert_eq!(
        (error.kind, error.stage),
        (ErrorKind::Processing, Some(Stage::Decensor))
    );
    let skipped = events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|event| match event {
            Progress::Skipped { page, stage } => Some((*page, *stage)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(skipped, [(pages[0], Stage::Decensor)]);
}
