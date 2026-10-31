//! Censor detection: finds obscured areas and commits their grayscale mask.
//!
//! Detection settings belong to this stage instead of a generative model
//! profile, so the mask can be tuned and inspected whatever model decensoring
//! later selects.

use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, ensure};
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, ImageFormat, Luma};
use imageproc::{distance_transform::Norm, morphology::dilate};
use koharu_ml::eromanga_seg::Er0mangaSeg;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{StageInput, StageProcessor, finish, generation};
use crate::ModelCell;

const MODEL: &str = "eromanga-seg";
const PRODUCER: &str = "dev.koharu.pipeline.censor-detection";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct CensorDetectionConfig {
    /// Run censor detection as part of the pipeline.
    pub enabled: bool,
    pub detect_mosaic: bool,
    pub threshold: f32,
    pub mask_grow: u32,
    pub mask_blur: f32,
}

impl Default for CensorDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detect_mosaic: true,
            threshold: 0.5,
            mask_grow: 12,
            mask_blur: 4.0,
        }
    }
}

impl CensorDetectionConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.threshold.is_finite() && (0.0..=1.0).contains(&self.threshold),
            "censor detection threshold must be between zero and one"
        );
        ensure!(
            self.mask_grow <= u32::from(u8::MAX),
            "censor mask grow must not exceed 255 pixels"
        );
        ensure!(
            self.mask_blur.is_finite() && self.mask_blur >= 0.0,
            "censor mask blur must be finite and non-negative"
        );
        Ok(())
    }
}

pub(super) struct Processor {
    config: CensorDetectionConfig,
    device: koharu_ml::Device,
    model: ModelCell<Arc<Mutex<Er0mangaSeg>>>,
}

impl Processor {
    pub(super) fn new(config: CensorDetectionConfig, device: koharu_ml::Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            device,
            model: ModelCell::new(),
        })
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        MODEL
    }

    async fn skip(&self, input: &StageInput) -> Result<bool> {
        Ok(!self.config.enabled || input.inpainting_mask.is_some())
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        let device = self.device.clone();
        self.model
            .ensure(|| async move {
                tokio::task::spawn_blocking(move || {
                    Er0mangaSeg::load(device).map(|model| Arc::new(Mutex::new(model)))
                })
                .await
                .context("censor detector load task panicked")?
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let image = input
            .images
            .get(&input.scene, input.page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {} has no source image", input.page))?;
        let model = self
            .model
            .lock()
            .await
            .as_ref()
            .context("censor detector is not loaded")?
            .clone();
        let config = self.config.clone();
        let mut mask = tokio::task::spawn_blocking(move || {
            detect(
                &model,
                &image,
                config.threshold,
                config.detect_mosaic,
                config.mask_grow,
                config.mask_blur,
            )
        })
        .await
        .context("censor detection task panicked")??;
        if let Some(bounds) = input.region {
            let previous = input
                .images
                .get(&input.scene, input.page, "censor-mask")
                .await?
                .map(|image| image.to_luma8());
            if previous
                .as_ref()
                .is_some_and(|previous| previous.dimensions() != mask.dimensions())
            {
                anyhow::bail!(
                    "existing censor mask dimensions do not match page {}",
                    input.page
                );
            }
            for (x, y, pixel) in mask.enumerate_pixels_mut() {
                if f64::from(x + 1) <= bounds.x
                    || f64::from(y + 1) <= bounds.y
                    || f64::from(x) >= bounds.x + bounds.width
                    || f64::from(y) >= bounds.y + bounds.height
                {
                    *pixel = previous
                        .as_ref()
                        .map_or(Luma([0]), |previous| *previous.get_pixel(x, y));
                }
            }
        }
        let (width, height) = mask.dimensions();
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageLuma8(mask).write_to(&mut bytes, ImageFormat::Png)?;
        let mut edit = input.scene.edit_as(generation(PRODUCER, MODEL)?);
        edit.observe_assets(input.page)?;
        edit.set_asset(
            input.page,
            &AssetRole::new("censor-mask")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}

fn detect(
    model: &Mutex<Er0mangaSeg>,
    image: &DynamicImage,
    threshold: f32,
    mosaic: bool,
    grow: u32,
    blur: f32,
) -> Result<GrayImage> {
    let model = model
        .lock()
        .map_err(|_| anyhow!("Er0mangaSeg model lock is poisoned"))?;
    let mut mask = model.detect(image, threshold, mosaic)?;
    if grow > 0 {
        mask = dilate(&mask, Norm::LInf, grow as u8);
    }
    if blur > 0.0 {
        mask = image::imageops::blur(&mask, blur);
    }
    Ok(mask)
}
