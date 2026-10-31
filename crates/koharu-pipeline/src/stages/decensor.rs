//! Censor reconstruction: repaints obscured artwork with its own model.
//!
//! The stage reconstructs the saved `censor-mask` asset, so censored areas can
//! be repaired without rerunning detection or text removal, or exactly the
//! pixels of a stroke painted by hand, which replaces the saved mask for that
//! one run. Text removal is a separate stage with its own model selection; both
//! stages commit generated pixels through the shared cleanup layer.

use std::sync::Arc;

use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView as _, GrayImage};
use koharu_scene::AssetRole;

use super::backends::{Backend, InpaintTask, Models, Selection};
use super::cleanup::{self, CleanupLayer};
use super::tiling;
use super::{StageInput, StageProcessor, finish};
use crate::{CleanupTarget, DecensorModel, ModelCell};

pub(super) struct Processor {
    config: DecensorModel,
    device: koharu_ml::Device,
    models: Arc<Models>,
    backend: Arc<ModelCell<Backend>>,
}

impl Processor {
    pub(super) fn new(
        config: DecensorModel,
        device: koharu_ml::Device,
        models: Arc<Models>,
    ) -> Result<Self> {
        super::backends::validate(Selection::from(&config))?;
        Ok(Self {
            config,
            device,
            models,
            backend: Arc::new(ModelCell::new()),
        })
    }

    async fn run(&self, input: &StageInput, backend: &Backend) -> Result<koharu_scene::Patch> {
        let page = input.page;
        let Prepared { image, layer, mask } = prepare(input).await?;
        let Some(bounds) = tiling::mask_bounds(&mask) else {
            return finish(input.scene.edit());
        };
        let (width, height) = image.dimensions();
        let mask = Arc::new(mask);
        let generated = backend
            .inpaint(InpaintTask {
                image,
                mask: Arc::clone(&mask),
                stop: input.stop.clone(),
            })
            .await?;
        ensure!(
            generated.dimensions() == (width, height),
            "reconstructed image dimensions do not match page {page}"
        );
        cleanup::commit(input, &layer, &generated, &mask, bounds, self.model())
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Selection::from(&self.config).name()
    }

    async fn skip(&self, input: &StageInput) -> Result<bool> {
        // Automatic reconstruction needs the saved censor mask and never
        // overwrites cleanup the user authored. A painted stroke replaces the
        // saved mask, so it needs neither.
        if input.inpainting_mask.is_none()
            && (cleanup::protected(input)?
                || input
                    .scene
                    .asset(input.page, &AssetRole::new("censor-mask")?)?
                    .is_none())
        {
            return Ok(true);
        }
        // Censor detection commits a mask for every page, empty when it finds
        // nothing to reconstruct. Such a page, like a stroke that selects no
        // pixel, needs neither the model nor the accelerator.
        let Some(source) = input.images.get(&input.scene, input.page, "source").await? else {
            return Ok(false);
        };
        let mask = censor_mask(input, source.dimensions()).await?;
        Ok(tiling::mask_bounds(&mask).is_none())
    }

    fn unload(&self) -> bool {
        self.backend.unload()
    }

    async fn load(&self) -> Result<()> {
        self.backend
            .ensure(|| {
                Backend::load(
                    self.device.clone(),
                    Selection::from(&self.config),
                    &self.models,
                )
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let backend = self.backend.lock().await;
        let backend = backend
            .as_ref()
            .ok_or_else(|| anyhow!("decensoring model is not loaded"))?;
        self.run(&input, backend).await
    }
}

struct Prepared {
    image: Arc<DynamicImage>,
    layer: CleanupLayer,
    mask: GrayImage,
}

async fn prepare(input: &StageInput) -> Result<Prepared> {
    let page = input.page;
    let original = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let layer = cleanup::load(input, &original).await?;
    // Reconstruct over the committed artwork, so an earlier pass over the same
    // page is preserved instead of being repainted from the censored source.
    let image = cleanup::composite(&original, &layer);
    let mask = censor_mask(input, image.dimensions()).await?;
    Ok(Prepared { image, layer, mask })
}

/// The mask to reconstruct, clipped to the requested region: the stroke painted
/// for this run when there is one, otherwise the saved censor mask.
///
/// Only a stroke painted for decensoring is reconstructed here; one addressed
/// to another target belongs to the inpainting stage and is refused.
async fn censor_mask(input: &StageInput, (width, height): (u32, u32)) -> Result<GrayImage> {
    if let Some((target, painted)) = input.painted_mask((width, height))? {
        ensure!(
            target == CleanupTarget::Decensor,
            "a {target:?} stroke is repaired by the inpainting stage, not by decensoring"
        );
        return Ok(painted);
    }
    let page = input.page;
    let mut mask = match input.images.get(&input.scene, page, "censor-mask").await? {
        Some(censor) => {
            let censor = tiling::luma(&censor).into_owned();
            if censor.dimensions() != (width, height) {
                bail!("censor-mask dimensions do not match page {page}");
            }
            censor
        }
        None => GrayImage::new(width, height),
    };
    if let Some(bounds) = input.region {
        tiling::clip_to_region(&mut mask, bounds);
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use image::{Luma, Rgb, RgbImage, Rgba, RgbaImage};
    use koharu_scene::{AssetInput, AssetMetadata, Origin};

    use super::*;
    use crate::stages::test_support::{Fixture, encode, painted, stroke};

    fn inpainting() -> crate::stages::inpainting::Processor {
        crate::stages::inpainting::Processor::new(
            crate::InpaintingConfig::default(),
            koharu_ml::Device::cpu(),
            Arc::default(),
        )
        .unwrap()
    }

    /// A page with pending artwork lettering at (2, 2) and censorship at (1, 1).
    async fn pending_cleanup() -> Fixture {
        Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "censor-mask",
                    DynamicImage::ImageLuma8(stroke(&[(1, 1, 255)])),
                ),
                (
                    "artwork-text-mask",
                    DynamicImage::ImageLuma8(stroke(&[(2, 2, 255)])),
                ),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
        )
        .await
    }

    /// Commits `color` at `pixel` as `model` would for `input`.
    async fn commit_pixel(
        fixture: &Fixture,
        input: &StageInput,
        pixel: (u32, u32),
        color: Rgb<u8>,
        model: &str,
    ) -> Fixture {
        let source = input
            .images
            .get(&input.scene, input.page, "source")
            .await
            .unwrap()
            .unwrap();
        let layer = cleanup::load(input, &source).await.unwrap();
        let mask = stroke(&[(pixel.0, pixel.1, 255)]);
        let bounds = tiling::mask_bounds(&mask).unwrap();
        let generated = RgbImage::from_pixel(8, 8, color);
        let patch = cleanup::commit(input, &layer, &generated, &mask, bounds, model).unwrap();
        fixture.after(&patch)
    }

    #[tokio::test]
    async fn automatic_cleanup_stages_rerun_over_each_others_generated_pixels() {
        // A whole pipeline runs inpainting before decensoring; rerunning either
        // stage later must update the shared layer instead of failing on it.
        let decensor = processor();
        let inpainting = inpainting();
        let mut page = pending_cleanup().await;
        let rounds: [(&dyn StageProcessor, u8); 4] = [
            (&inpainting, 40),
            (&decensor, 80),
            (&inpainting, 120),
            (&decensor, 160),
        ];
        for (x, (stage, red)) in (0..).zip(rounds) {
            let input = page.input(None, None);
            assert!(
                !stage.skip(&input).await.unwrap(),
                "{} round {x}",
                stage.model()
            );
            page = commit_pixel(&page, &input, (x, 7), Rgb([red, 0, 0]), stage.model()).await;
        }

        let layers = page.cleanup_layers();
        let [(layer, entity)] = &layers[..] else {
            panic!("both stages must share one cleanup layer");
        };
        assert!(matches!(layer.origin, Origin::Generated(_)));
        assert!(matches!(entity.origin, Origin::Generated(_)));
        let visible = page.visible().await;
        for (x, (_, red)) in (0..).zip(rounds) {
            assert_eq!(visible.get_pixel(x, 7), &Rgba([red, 0, 0, 255]));
        }
    }

    #[tokio::test]
    async fn automatic_cleanup_stages_skip_layer_pixels_the_user_edited() {
        // Pixels committed by a user edit stay the user's even while the layer
        // entity itself is still generated.
        let decensor = processor();
        let inpainting = inpainting();
        let fixture = pending_cleanup().await;
        let input = fixture.input(None, None);
        let generated = commit_pixel(
            &fixture,
            &input,
            (0, 7),
            Rgb([40, 0, 0]),
            inpainting.model(),
        )
        .await;
        let entity = cleanup::layer_entity(&generated.input(None, None))
            .unwrap()
            .unwrap();
        let mut edit = generated.snapshot.edit();
        edit.set_asset(
            entity,
            &AssetRole::new("source").unwrap(),
            AssetInput::new(
                encode(&DynamicImage::ImageRgba8(RgbaImage::new(8, 8))),
                "image/png",
                AssetMetadata {
                    width: Some(8),
                    height: Some(8),
                    attributes: Default::default(),
                },
            ),
        )
        .unwrap();
        let edited = generated.after(&edit.finish().unwrap());

        let automatic = edited.input(None, None);
        assert!(inpainting.skip(&automatic).await.unwrap());
        assert!(decensor.skip(&automatic).await.unwrap());
        // The user may still paint over the page with any brush: the inpainting
        // stage repairs the Fast and Inpainting strokes, this one its own.
        let brushes: [(CleanupTarget, &dyn StageProcessor); 3] = [
            (CleanupTarget::Fast, &inpainting),
            (CleanupTarget::Inpainting, &inpainting),
            (CleanupTarget::Decensor, &decensor),
        ];
        for (target, stage) in brushes {
            let manual = edited.input(Some(painted(target, &[(3, 3, 255)])), None);
            assert!(!stage.skip(&manual).await.unwrap(), "{target:?}");
        }
    }

    fn processor() -> Processor {
        Processor::new(
            DecensorModel::default(),
            koharu_ml::Device::cpu(),
            Arc::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn censor_mask_is_clipped_to_the_region_and_composited_over_cleanup() {
        let mut text = GrayImage::new(8, 8);
        text.put_pixel(2, 3, Luma([255]));
        let mut censor = GrayImage::new(8, 8);
        censor.put_pixel(5, 5, Luma([255]));
        censor.put_pixel(0, 0, Luma([255]));
        let mut cleanup = RgbaImage::new(8, 8);
        cleanup.put_pixel(5, 5, Rgba([25, 50, 75, 255]));
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("text-mask", DynamicImage::ImageLuma8(text)),
                ("censor-mask", DynamicImage::ImageLuma8(censor)),
            ],
            Some((Origin::User, DynamicImage::ImageRgba8(cleanup))),
        )
        .await;
        let input = fixture.input(
            None,
            Some(crate::Bounds {
                x: 2.0,
                y: 3.0,
                width: 4.0,
                height: 3.0,
            }),
        );

        let prepared = prepare(&input).await.unwrap();

        assert_eq!(prepared.mask.get_pixel(5, 5), &Luma([255]));
        assert_eq!(prepared.mask.get_pixel(0, 0), &Luma([0]));
        assert_eq!(prepared.mask.get_pixel(2, 3), &Luma([0]));
        assert_eq!(prepared.image.get_pixel(5, 5), Rgba([25, 50, 75, 255]));
    }

    #[tokio::test]
    async fn decensoring_skips_pages_without_a_censor_mask() {
        let mut censor = GrayImage::new(8, 8);
        censor.put_pixel(1, 1, Luma([255]));
        let without = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let with = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("censor-mask", DynamicImage::ImageLuma8(censor)),
            ],
            None,
        )
        .await;
        let processor = processor();

        assert!(processor.skip(&without.input(None, None)).await.unwrap());
        assert!(!processor.skip(&with.input(None, None)).await.unwrap());
    }

    #[tokio::test]
    async fn a_painted_stroke_needs_no_saved_censor_mask() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let region = Some(crate::Bounds {
            x: 4.0,
            y: 4.0,
            width: 2.0,
            height: 2.0,
        });
        let processor = processor();

        assert!(processor.skip(&fixture.input(None, None)).await.unwrap());
        let selected = fixture.input(Some(painted(CleanupTarget::Decensor, &[(1, 1, 255)])), None);
        assert!(!processor.skip(&selected).await.unwrap());
        // A stroke that selects no pixel needs no model either.
        let fringe = fixture.input(Some(painted(CleanupTarget::Decensor, &[(1, 1, 126)])), None);
        assert!(processor.skip(&fringe).await.unwrap());
        let outside = fixture.input(
            Some(painted(CleanupTarget::Decensor, &[(0, 0, 255)])),
            region,
        );
        assert!(processor.skip(&outside).await.unwrap());
    }

    #[tokio::test]
    async fn a_painted_stroke_replaces_the_saved_censor_mask_inside_the_region() {
        // The saved mask selects other pixels, one of them inside the region:
        // it must not be merged with the stroke.
        let mut cleanup = RgbaImage::new(8, 8);
        cleanup.put_pixel(3, 3, Rgba([25, 50, 75, 255]));
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "censor-mask",
                    DynamicImage::ImageLuma8(stroke(&[(0, 0, 255), (4, 4, 255)])),
                ),
            ],
            Some((Origin::User, DynamicImage::ImageRgba8(cleanup))),
        )
        .await;
        // The region covers x 2..6 and y 2..6: (3, 3) and (5, 5) are inside it,
        // while (1, 1) and (6, 6) lie just past its edges.
        let brush = painted(
            CleanupTarget::Decensor,
            &[(3, 3, 255), (5, 5, 255), (1, 1, 255), (6, 6, 255)],
        );
        let region = Some(crate::Bounds {
            x: 2.0,
            y: 2.0,
            width: 4.0,
            height: 4.0,
        });

        let prepared = prepare(&fixture.input(Some(brush), region)).await.unwrap();

        let selected = prepared
            .mask
            .enumerate_pixels()
            .filter(|(_, _, pixel)| pixel.0[0] != 0)
            .map(|(x, y, _)| (x, y))
            .collect::<Vec<_>>();
        assert_eq!(selected, [(3, 3), (5, 5)]);
        // Committed cleanup is still the context the model sees.
        assert_eq!(prepared.image.get_pixel(3, 3), Rgba([25, 50, 75, 255]));
    }

    #[tokio::test]
    async fn a_painted_stroke_must_match_the_page() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let mut small = GrayImage::new(4, 4);
        small.put_pixel(1, 1, Luma([255]));
        let input = fixture.input(Some((CleanupTarget::Decensor, small)), None);

        assert!(censor_mask(&input, (8, 8)).await.is_err());
        assert!(processor().skip(&input).await.is_err());
    }

    #[tokio::test]
    async fn a_stroke_for_another_target_is_refused_instead_of_reconstructed() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let processor = processor();

        for target in [CleanupTarget::Fast, CleanupTarget::Inpainting] {
            let input = fixture.input(Some(painted(target, &[(1, 1, 255)])), None);

            assert!(censor_mask(&input, (8, 8)).await.is_err(), "{target:?}");
            assert!(prepare(&input).await.is_err(), "{target:?}");
            assert!(processor.skip(&input).await.is_err(), "{target:?}");
        }
    }

    #[tokio::test]
    async fn decensoring_skips_pages_whose_censor_mask_is_empty_or_outside_the_region() {
        let mut faint = GrayImage::new(8, 8);
        faint.put_pixel(1, 1, Luma([126]));
        let mut corner = GrayImage::new(8, 8);
        corner.put_pixel(0, 0, Luma([255]));
        let empty = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "censor-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
        )
        .await;
        let below_threshold = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("censor-mask", DynamicImage::ImageLuma8(faint)),
            ],
            None,
        )
        .await;
        let corner = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("censor-mask", DynamicImage::ImageLuma8(corner)),
            ],
            None,
        )
        .await;
        let region = Some(crate::Bounds {
            x: 4.0,
            y: 4.0,
            width: 2.0,
            height: 2.0,
        });
        let processor = processor();

        assert!(processor.skip(&empty.input(None, None)).await.unwrap());
        assert!(
            processor
                .skip(&below_threshold.input(None, None))
                .await
                .unwrap()
        );
        assert!(processor.skip(&corner.input(None, region)).await.unwrap());
        assert!(!processor.skip(&corner.input(None, None)).await.unwrap());
    }

    #[tokio::test]
    async fn a_painted_stroke_adopts_the_cleanup_layer_and_keeps_every_other_pixel() {
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "censor-mask",
                    DynamicImage::ImageLuma8(stroke(&[(1, 1, 255)])),
                ),
                (
                    "artwork-text-mask",
                    DynamicImage::ImageLuma8(stroke(&[(2, 2, 255)])),
                ),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
        )
        .await;
        let decensor = processor();
        let inpainting = inpainting();

        // The automatic stage commits its reconstruction as generated cleanup.
        let automatic = fixture.input(None, None);
        let prepared = prepare(&automatic).await.unwrap();
        let bounds = tiling::mask_bounds(&prepared.mask).unwrap();
        let red = RgbImage::from_pixel(8, 8, Rgb([200, 0, 0]));
        let patch = cleanup::commit(
            &automatic,
            &prepared.layer,
            &red,
            &prepared.mask,
            bounds,
            decensor.model(),
        )
        .unwrap();
        let generated = fixture.after(&patch);
        let layers = generated.cleanup_layers();
        let [(layer, entity)] = &layers[..] else {
            panic!("the automatic run must commit one cleanup layer");
        };
        assert!(matches!(layer.origin, Origin::Generated(_)));
        assert!(matches!(entity.origin, Origin::Generated(_)));
        let pending = generated.input(None, None);
        assert!(!inpainting.skip(&pending).await.unwrap());

        // A painted stroke is the user's own edit: it adopts that layer instead
        // of adding another, keeps the earlier pixels, and makes automatic text
        // cleanup leave the page alone from then on.
        let manual = generated.input(Some(painted(CleanupTarget::Decensor, &[(5, 6, 255)])), None);
        let prepared = prepare(&manual).await.unwrap();
        let bounds = tiling::mask_bounds(&prepared.mask).unwrap();
        let green = RgbImage::from_pixel(8, 8, Rgb([0, 200, 0]));
        let patch = cleanup::commit(
            &manual,
            &prepared.layer,
            &green,
            &prepared.mask,
            bounds,
            decensor.model(),
        )
        .unwrap();
        let authored = generated.after(&patch);

        let layers = authored.cleanup_layers();
        let [(layer, entity)] = &layers[..] else {
            panic!("a painted stroke must reuse the cleanup layer");
        };
        assert_eq!(layer.origin, Origin::User);
        assert_eq!(entity.origin, Origin::User);
        let visible = authored.visible().await;
        assert_eq!(visible.get_pixel(1, 1), &Rgba([200, 0, 0, 255]));
        assert_eq!(visible.get_pixel(5, 6), &Rgba([0, 200, 0, 255]));
        assert_eq!(visible.get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
        let settled = authored.input(None, None);
        assert!(inpainting.skip(&settled).await.unwrap());
        // The stroke itself is never saved as a censor mask.
        let role = AssetRole::new("censor-mask").unwrap();
        assert_eq!(
            authored.snapshot.asset(authored.page, &role).unwrap(),
            fixture.snapshot.asset(fixture.page, &role).unwrap()
        );
    }
}
