mod backends;
mod censor_detection;
mod cleanup;
mod comfyui;
mod decensor;
mod detection;
mod inpainting;
mod models;
mod ocr;
mod tiling;
mod translation;

#[cfg(test)]
mod test_support;

use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Context as _, Result, ensure};
use async_trait::async_trait;
use image::GrayImage;
use koharu_scene::{Edit, EntityId, Generation, Patch, ProducerId, Snapshot};

pub(crate) use backends::Models;
pub use censor_detection::CensorDetectionConfig;
pub use detection::KoharuLayoutRFDetrSeg2XLConfig;
pub(crate) use models::DECENSOR_PROMPT;
pub use models::{DiffusionCreamPyConfig, Flux2Klein9BConfig, Flux2KleinConfig, RoremMixedConfig};

use crate::{
    Bounds, CleanupTarget, ImageCache, InpaintingMask, PipelineConfig, Stage, TranslationOptions,
};

#[derive(Clone)]
pub(crate) struct StageInput {
    scene: koharu_scene::Snapshot,
    page: EntityId,
    entities: Option<Arc<BTreeSet<EntityId>>>,
    region: Option<Bounds>,
    images: Arc<ImageCache>,
    inpainting_mask: Option<InpaintingMask>,
    translation: Option<Arc<TranslationOptions>>,
    stop: crate::StopToken,
}

impl StageInput {
    pub(crate) fn new(
        scene: Snapshot,
        page: EntityId,
        entities: Option<Arc<BTreeSet<EntityId>>>,
        region: Option<Bounds>,
        images: Arc<ImageCache>,
        inpainting_mask: Option<InpaintingMask>,
        translation: Option<Arc<TranslationOptions>>,
    ) -> Self {
        Self {
            scene,
            page,
            entities,
            region,
            images,
            inpainting_mask,
            translation,
            stop: crate::StopToken::default(),
        }
    }

    /// Lets long-running inference observe the job's stop request.
    #[must_use]
    pub(crate) fn stopped_by(mut self, stop: crate::StopToken) -> Self {
        self.stop = stop;
        self
    }

    pub(crate) fn page(&self) -> EntityId {
        self.page
    }

    fn contains_entity(&self, entity: EntityId) -> Result<bool> {
        crate::scope::contains_entity(
            &self.scene,
            self.page,
            self.entities.as_deref(),
            self.region,
            entity,
        )
    }

    /// The stroke painted for this run with the target it was painted for,
    /// decoded at the page's size and clipped to the requested region like every
    /// saved mask.
    ///
    /// A stroke replaces whatever mask its stage would read from the page; the
    /// two are never merged. Each stage accepts only the targets it repairs, so
    /// a stage that receives another target's stroke must refuse it rather than
    /// repair it with the wrong model.
    fn painted_mask(
        &self,
        (width, height): (u32, u32),
    ) -> Result<Option<(CleanupTarget, GrayImage)>> {
        let Some(painted) = &self.inpainting_mask else {
            return Ok(None);
        };
        let mut mask = image::load_from_memory(&painted.png)
            .context("failed to decode the painted cleanup mask")?
            .into_luma8();
        ensure!(
            mask.dimensions() == (width, height),
            "inpainting mask dimensions do not match page {}",
            self.page
        );
        if let Some(bounds) = self.region {
            tiling::clip_to_region(&mut mask, bounds);
        }
        Ok(Some((painted.target, mask)))
    }
}

#[async_trait]
trait StageProcessor: Send + Sync {
    fn model(&self) -> &'static str;
    /// Whether processing occupies the shared accelerator. Work that only
    /// waits on a remote service must not hold it.
    fn uses_accelerator(&self) -> bool {
        true
    }
    /// Whether the stage has nothing to do for this page. It runs before the
    /// model is loaded, so it may read the page's assets.
    async fn skip(&self, _input: &StageInput) -> Result<bool> {
        Ok(false)
    }
    fn unload(&self) -> bool;
    async fn load(&self) -> Result<()>;
    async fn process(&self, input: StageInput) -> Result<Patch>;
}

/// Rejects the model settings that building the stages would refuse. Saving
/// settings calls this, so a bad value is reported when it is entered instead of
/// when the pipeline reloads and silently keeps running its previous settings.
pub(crate) fn validate(config: &PipelineConfig) -> Result<()> {
    inpainting::validate(&config.inpainting)?;
    backends::validate(backends::Selection::from(&config.decensor()?))
}

pub(crate) struct Stages {
    detection: detection::Processor,
    ocr: ocr::Processor,
    translation: translation::Processor,
    censor_detection: censor_detection::Processor,
    inpainting: inpainting::Processor,
    decensor: decensor::Processor,
}

impl Stages {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
        models: Arc<Models>,
    ) -> Result<Self> {
        // The registry belongs to the pipeline, not to this generation, so
        // stages of different generations that select the same model share its
        // loaded weights.
        Ok(Self {
            detection: detection::Processor::new(config.detection()?, device.clone()),
            ocr: ocr::Processor::new(config.ocr.clone(), device.clone()),
            translation: translation::Processor::new(config.translation.clone(), translator)?,
            censor_detection: censor_detection::Processor::new(
                config.censor_detection.clone(),
                device.clone(),
            )?,
            inpainting: inpainting::Processor::new(
                config.inpainting.clone(),
                device.clone(),
                models.clone(),
            )?,
            decensor: decensor::Processor::new(config.decensor()?, device.clone(), models)?,
        })
    }

    fn processor(&self, stage: Stage) -> &dyn StageProcessor {
        match stage {
            Stage::Detection => &self.detection,
            Stage::Ocr => &self.ocr,
            Stage::Translation => &self.translation,
            Stage::CensorDetection => &self.censor_detection,
            Stage::Inpainting => &self.inpainting,
            Stage::Decensor => &self.decensor,
        }
    }

    pub(crate) fn model(&self, stage: Stage) -> &'static str {
        self.processor(stage).model()
    }

    pub(crate) fn uses_accelerator(&self, stage: Stage) -> bool {
        self.processor(stage).uses_accelerator()
    }

    pub(crate) async fn skip(&self, stage: Stage, input: &StageInput) -> Result<bool> {
        self.processor(stage).skip(input).await
    }

    pub(crate) async fn load(&self, stage: Stage) -> Result<()> {
        self.processor(stage).load().await
    }

    pub(crate) async fn process(&self, stage: Stage, input: StageInput) -> Result<Patch> {
        self.processor(stage).process(input).await
    }

    pub(crate) fn unload(&self, stage: Stage) -> bool {
        self.processor(stage).unload()
    }
}

fn generation(producer: &str, model: &str) -> Result<Generation> {
    let mut generation = Generation::new(ProducerId::new(producer)?);
    generation.model = Some(model.to_owned());
    Ok(generation)
}

fn finish(edit: Edit) -> Result<Patch> {
    edit.finish().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    fn asset(bytes: &'static [u8]) -> koharu_scene::AssetInput {
        koharu_scene::AssetInput::new(
            bytes,
            "image/png",
            koharu_scene::AssetMetadata {
                width: Some(1),
                height: Some(1),
                attributes: std::collections::BTreeMap::new(),
            },
        )
    }

    #[tokio::test]
    async fn translation_and_inpainting_compose_without_weakening_text_guards() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("page", 1.0, 1.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let text = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        setup
            .set(
                text,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user("before".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        setup
            .set_asset(
                page,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"source"),
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let base = session.snapshot();

        let mut text_edit = base.edit();
        text_edit.observe::<koharu_scene::SourceText>(text).unwrap();
        text_edit
            .observe::<koharu_scene::Translation>(text)
            .unwrap();
        text_edit
            .set(
                text,
                &koharu_scene::Translation {
                    text: koharu_scene::Authored::user("after".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let text_patch = text_edit.finish().unwrap();

        let mut image_edit = base.edit();
        image_edit.observe_assets(page).unwrap();
        let cleanup = image_edit
            .add_entity(page, koharu_scene::At::Start)
            .unwrap();
        image_edit
            .set(
                cleanup,
                &koharu_scene::RasterLayer {
                    origin: koharu_scene::Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: koharu_scene::RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
        image_edit
            .set_asset(
                cleanup,
                &koharu_scene::AssetRole::new("source").unwrap(),
                asset(b"clean"),
            )
            .unwrap();
        let image_patch = image_edit.finish().unwrap();

        let image_first = base.preview([&image_patch]).unwrap();
        assert!(text_patch.rebase_on(&image_first).is_ok());
        let text_first = base.preview([&text_patch]).unwrap();
        assert!(image_patch.rebase_on(&text_first).is_ok());

        let changed_source = base
            .patch(|edit| {
                edit.set(
                    text,
                    &koharu_scene::SourceText {
                        text: koharu_scene::Authored::user("changed".to_owned()),
                        language: None,
                    },
                )
            })
            .unwrap();
        let changed_source = base.preview([&changed_source]).unwrap();
        assert!(text_patch.rebase_on(&changed_source).is_err());
    }
}
