//! Cleanup by route: fast bubble interiors, lettering over artwork, and strokes
//! painted by hand.
//!
//! Detection owns the disjoint lettering masks. This stage never asks an
//! inpainter to locate text or reconstruct censorship. Only selected mask
//! pixels are committed; the artwork pass sees the already cleaned bubbles.
//!
//! A stroke painted by hand replaces those masks for one run and takes exactly
//! the route its target names: the fast bubble model for `Fast`, or FLUX.2 Klein
//! 9B with the manual profile's own settings for `Inpainting`. Reconstructing
//! censored artwork belongs to the decensor stage, which refuses these strokes.

use std::{future::Future, sync::Arc};

use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView as _, GrayImage, RgbImage};
use koharu_scene::{AssetRole, BubbleRegion, Geometry, Region, RegionSpec as _};

use super::backends::{Backend, InpaintTask, Models, Selection};
use super::cleanup::{self, CleanupLayer};
use super::tiling::{self, FlatFillRegion};
use super::{StageInput, StageProcessor, finish};
use crate::{BubbleInpaintingKind, CleanupTarget, InpaintingConfig, InpaintingModel, ModelCell};

/// Names the automatic cleanup in the generation record of its commits.
const MODEL: &str = "bubble-and-artwork-cleanup";

/// A model of this stage together with the pixels it repairs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Route {
    /// The fast model selected in settings: lettering detected inside speech
    /// bubbles, and strokes painted for the `Fast` target.
    Fast,
    /// Lettering detected over artwork.
    Artwork,
    /// Strokes painted for the `Inpainting` target.
    Manual,
}

/// The model each route runs, with the settings it runs with.
struct Routes {
    fast: InpaintingModel,
    artwork: InpaintingModel,
    manual: InpaintingModel,
}

impl Routes {
    fn select(config: &InpaintingConfig) -> Result<Self> {
        let routes = Self {
            fast: match config.bubbles {
                BubbleInpaintingKind::LaMa => InpaintingModel::LaMa {},
                BubbleInpaintingKind::AotInpainting => InpaintingModel::AotInpainting {},
            },
            artwork: config.artwork.selected(),
            // The manual brush always runs FLUX.2 Klein 9B. Its profile is its
            // own: it shares weights with the other routes, never their prompt,
            // sampling options or LoRA.
            manual: InpaintingModel::Flux2Klein9B(config.manual.clone()),
        };
        for model in [&routes.fast, &routes.artwork, &routes.manual] {
            super::backends::validate(Selection::from(model))?;
        }
        Ok(routes)
    }
}

/// Rejects settings that no route of the stage could run with.
pub(super) fn validate(config: &InpaintingConfig) -> Result<()> {
    Routes::select(config).map(|_| ())
}

struct Slot {
    model: InpaintingModel,
    backend: ModelCell<Backend>,
}

impl Slot {
    fn new(model: InpaintingModel) -> Self {
        Self {
            model,
            backend: ModelCell::new(),
        }
    }
}

pub(super) struct Processor {
    fast: Slot,
    artwork: Slot,
    manual: Slot,
    device: koharu_ml::Device,
    models: Arc<Models>,
}

impl Processor {
    pub(super) fn new(
        config: InpaintingConfig,
        device: koharu_ml::Device,
        models: Arc<Models>,
    ) -> Result<Self> {
        let Routes {
            fast,
            artwork,
            manual,
        } = Routes::select(&config)?;
        Ok(Self {
            fast: Slot::new(fast),
            artwork: Slot::new(artwork),
            manual: Slot::new(manual),
            device,
            models,
        })
    }

    fn slot(&self, route: Route) -> &Slot {
        match route {
            Route::Fast => &self.fast,
            Route::Artwork => &self.artwork,
            Route::Manual => &self.manual,
        }
    }

    async fn infer(
        &self,
        route: Route,
        image: Arc<DynamicImage>,
        mask: GrayImage,
        stop: crate::StopToken,
    ) -> Result<RgbImage> {
        let Slot { model, backend } = self.slot(route);
        let selection = Selection::from(model);
        backend
            .ensure(|| Backend::load(self.device.clone(), selection, &self.models))
            .await?;
        let backend = backend.lock().await;
        let backend = backend
            .as_ref()
            .ok_or_else(|| anyhow!("cleanup backend is not loaded"))?;
        backend
            .inpaint(InpaintTask {
                image,
                mask: Arc::new(mask),
                stop,
            })
            .await
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        MODEL
    }

    async fn skip(&self, input: &StageInput) -> Result<bool> {
        // Automatic cleanup leaves a page alone once the user has authored its
        // cleanup; a painted stroke is that authoring.
        if input.inpainting_mask.is_none() && cleanup::protected(input)? {
            return Ok(true);
        }
        let Some(source) = input.images.get(&input.scene, input.page, "source").await? else {
            return Ok(false);
        };
        Ok(masks(input, source.dimensions()).await?.is_empty())
    }

    fn unload(&self) -> bool {
        // `|` rather than `||`: every route unloads, not only the first.
        self.fast.backend.unload() | self.artwork.backend.unload() | self.manual.backend.unload()
    }

    async fn load(&self) -> Result<()> {
        // The input mask decides residency. Loading a generative backend here
        // would allocate it even for a page of flat speech bubbles.
        Ok(())
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let stop = input.stop.clone();
        repair(&input, |route, image, mask| {
            self.infer(route, image, mask, stop.clone())
        })
        .await
    }
}

/// The pixels each route repairs, in the order the routes run so the artwork
/// pass sees the cleaned bubbles. A stroke painted by hand selects one route.
struct Masks(Vec<(Route, GrayImage)>);

impl Masks {
    /// Every pixel some route repairs.
    fn union(&self) -> GrayImage {
        let mut masks = self.0.iter().map(|(_, mask)| mask);
        let mut union = masks.next().cloned().unwrap_or_default();
        for mask in masks {
            for (target, &value) in union.iter_mut().zip(mask.iter()) {
                *target = (*target).max(value);
            }
        }
        union
    }

    fn is_empty(&self) -> bool {
        self.0
            .iter()
            .all(|(_, mask)| tiling::mask_bounds(mask).is_none())
    }

    fn route_mut(&mut self, route: Route) -> Option<&mut GrayImage> {
        self.0
            .iter_mut()
            .find(|(candidate, _)| *candidate == route)
            .map(|(_, mask)| mask)
    }
}

struct Prepared {
    image: Arc<DynamicImage>,
    layer: CleanupLayer,
    masks: Masks,
}

/// Repairs the page's masked pixels, handing each route's mask to `infer`, and
/// commits the result.
///
/// `infer` repairs only the pixels of the mask it receives, whatever model backs
/// it; the commit writes exactly the union of those masks.
async fn repair<Infer, Repaired>(
    input: &StageInput,
    mut infer: Infer,
) -> Result<koharu_scene::Patch>
where
    Infer: FnMut(Route, Arc<DynamicImage>, GrayImage) -> Repaired,
    Repaired: Future<Output = Result<RgbImage>>,
{
    let Prepared {
        image,
        layer,
        mut masks,
    } = prepare(input).await?;
    let mask = masks.union();
    let Some(bounds) = tiling::mask_bounds(&mask) else {
        return finish(input.scene.edit());
    };
    let (width, height) = image.dimensions();
    let mut canvas = image.to_rgb8();
    if input.inpainting_mask.is_none() {
        // A painted stroke can cross a bubble's outline or tail, which a flat
        // fill would erase, so only the model repaints it.
        let regions = flat_fill_regions(input, width, height)?;
        if let Some(fast) = masks.route_mut(Route::Fast) {
            tiling::fill_uniform_regions(&mut canvas, fast, &mask, &regions);
        }
    }
    // Uniform bubbles have already been filled; no model is loaded for them.
    // Every other route loads only its own backend, and only when it has pixels
    // to repair.
    for (route, route_mask) in masks.0 {
        if tiling::mask_bounds(&route_mask).is_none() {
            continue;
        }
        canvas = infer(route, Arc::new(DynamicImage::ImageRgb8(canvas)), route_mask).await?;
    }
    ensure!(
        canvas.dimensions() == (width, height),
        "inpainted image dimensions do not match page {}",
        input.page
    );
    cleanup::commit(input, &layer, &canvas, &mask, bounds, MODEL)
}

async fn prepare(input: &StageInput) -> Result<Prepared> {
    let original = input
        .images
        .get(&input.scene, input.page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {} has no source image", input.page))?;
    let layer = cleanup::load(input, &original).await?;
    let image = if input.inpainting_mask.is_some() {
        cleanup::composite(&original, &layer)
    } else {
        original
    };
    let masks = masks(input, image.dimensions()).await?;
    Ok(Prepared {
        image,
        layer,
        masks,
    })
}

async fn masks(input: &StageInput, (width, height): (u32, u32)) -> Result<Masks> {
    // A painted stroke replaces the detected lettering for this run. The user
    // chose those pixels, not the detector, so neither saved mask is read: the
    // stroke's target names the one route that repairs it.
    if let Some((target, stroke)) = input.painted_mask((width, height))? {
        let route = match target {
            CleanupTarget::Fast => Route::Fast,
            CleanupTarget::Inpainting => Route::Manual,
            CleanupTarget::Decensor => {
                bail!("a decensor stroke is repaired by the decensor stage, not by inpainting")
            }
        };
        return Ok(Masks(vec![(route, stroke)]));
    }
    let decoded = [
        input
            .images
            .get(&input.scene, input.page, "bubble-text-mask")
            .await?,
        input
            .images
            .get(&input.scene, input.page, "artwork-text-mask")
            .await?,
    ];
    if decoded.iter().any(Option::is_none) {
        if input
            .scene
            .asset(input.page, &AssetRole::new("text-mask")?)?
            .is_some()
        {
            bail!("rerun Detection to create separate bubble and artwork lettering masks");
        }
        ensure!(
            decoded.iter().all(Option::is_none),
            "incomplete text masks: rerun Detection"
        );
    }
    let [bubbles, artwork] = [
        ("bubble-text-mask", decoded[0].as_ref()),
        ("artwork-text-mask", decoded[1].as_ref()),
    ]
    .map(|(role, image)| -> Result<GrayImage> {
        let mask = image.map_or_else(|| GrayImage::new(width, height), |image| image.to_luma8());
        ensure!(
            mask.dimensions() == (width, height),
            "{role} dimensions do not match page {}",
            input.page
        );
        Ok(mask)
    });
    let (bubbles, artwork) = (bubbles?, artwork?);
    ensure!(
        !bubbles
            .iter()
            .zip(artwork.iter())
            .any(|(a, b)| *a >= tiling::MASK_THRESHOLD && *b >= tiling::MASK_THRESHOLD),
        "bubble and artwork lettering masks overlap: rerun Detection"
    );
    let mut masks = Masks(vec![(Route::Fast, bubbles), (Route::Artwork, artwork)]);
    if let Some(bounds) = input.region {
        for (_, mask) in &mut masks.0 {
            tiling::clip_to_region(mask, bounds);
        }
    }
    Ok(masks)
}

/// Bubble polygons whose uniform interior is filled without a model.
fn flat_fill_regions(input: &StageInput, width: u32, height: u32) -> Result<Vec<FlatFillRegion>> {
    let mut regions = Vec::new();
    for entity in input.scene.descendants(input.page)? {
        let id = entity.id();
        let is_bubble = input
            .scene
            .component::<Region>(id)?
            .is_some_and(|region| region.kind == BubbleRegion::kind());
        if !is_bubble {
            continue;
        }
        let Some(geometry) = input.scene.component::<Geometry>(id)? else {
            continue;
        };
        let polygon = geometry
            .points
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        if polygon.len() < 3 {
            continue;
        }
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &polygon {
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
        }
        let bounds = [
            left.floor().clamp(0.0, width as f32) as u32,
            top.floor().clamp(0.0, height as f32) as u32,
            right.ceil().clamp(0.0, width as f32) as u32,
            bottom.ceil().clamp(0.0, height as f32) as u32,
        ];
        if bounds[2] > bounds[0] && bounds[3] > bounds[1] {
            regions.push(FlatFillRegion { bounds, polygon });
        }
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use image::{Luma, Rgb, Rgba, RgbaImage};
    use koharu_scene::Origin;

    use super::*;
    use crate::InpaintingKind;
    use crate::stages::test_support::{Fixture, painted, stroke};

    fn white() -> DynamicImage {
        DynamicImage::ImageLuma8(GrayImage::from_pixel(8, 8, Luma([255])))
    }

    fn processor() -> Processor {
        Processor::new(
            InpaintingConfig::default(),
            koharu_ml::Device::cpu(),
            Arc::default(),
        )
        .unwrap()
    }

    /// Each route's stand-in model paints its own colour, so a result shows
    /// which route repaired a pixel.
    fn paint(route: Route) -> [u8; 3] {
        match route {
            Route::Fast => [200, 0, 0],
            Route::Artwork => [0, 200, 0],
            Route::Manual => [0, 0, 200],
        }
    }

    fn opaque(route: Route) -> Rgba<u8> {
        let [red, green, blue] = paint(route);
        Rgba([red, green, blue, 255])
    }

    fn selected(mask: &GrayImage) -> Vec<(u32, u32)> {
        mask.enumerate_pixels()
            .filter(|(_, _, pixel)| pixel.0[0] != 0)
            .map(|(x, y, _)| (x, y))
            .collect()
    }

    /// The pixels each route would repair.
    fn routed(masks: &Masks) -> Vec<(Route, Vec<(u32, u32)>)> {
        masks
            .0
            .iter()
            .map(|(route, mask)| (*route, selected(mask)))
            .collect()
    }

    struct Call {
        route: Route,
        pixels: Vec<(u32, u32)>,
        /// The canvas the route was handed.
        canvas: RgbImage,
    }

    /// Runs the repair with stand-in models. Like the real ones, a stand-in
    /// changes only the pixels of the mask it is handed.
    async fn repaired(input: &StageInput) -> (Result<koharu_scene::Patch>, Vec<Call>) {
        let calls = RefCell::new(Vec::new());
        let patch = repair(input, |route, image, mask| {
            let canvas = image.to_rgb8();
            let mut repaired = canvas.clone();
            for (x, y, pixel) in mask.enumerate_pixels() {
                if pixel.0[0] != 0 {
                    repaired.put_pixel(x, y, Rgb(paint(route)));
                }
            }
            calls.borrow_mut().push(Call {
                route,
                pixels: selected(&mask),
                canvas,
            });
            async move { Ok::<_, anyhow::Error>(repaired) }
        })
        .await;
        (patch, calls.into_inner())
    }

    fn called(calls: &[Call]) -> Vec<(Route, Vec<(u32, u32)>)> {
        calls
            .iter()
            .map(|call| (call.route, call.pixels.clone()))
            .collect()
    }

    /// A page whose saved lettering masks select every pixel of both routes.
    async fn saved_lettering() -> Fixture {
        Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("bubble-text-mask", white()),
                ("artwork-text-mask", white()),
            ],
            None,
        )
        .await
    }

    #[tokio::test]
    async fn a_painted_stroke_replaces_saved_masks_and_takes_the_route_its_target_names() {
        // Both saved masks select the whole page and even overlap, which
        // detected lettering would reject: a stroke never reads them.
        let fixture = saved_lettering().await;

        for (target, route) in [
            (CleanupTarget::Fast, Route::Fast),
            (CleanupTarget::Inpainting, Route::Manual),
        ] {
            let input = fixture.input(Some(painted(target, &[(3, 4, 255)])), None);

            let prepared = prepare(&input).await.unwrap();

            assert_eq!(
                routed(&prepared.masks),
                [(route, vec![(3, 4)])],
                "{target:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_painted_stroke_is_repaired_by_its_route_alone_and_only_inside_its_pixels() {
        for (target, route) in [
            (CleanupTarget::Fast, Route::Fast),
            (CleanupTarget::Inpainting, Route::Manual),
        ] {
            let fixture = saved_lettering().await;
            let input = fixture.input(Some(painted(target, &[(3, 4, 255)])), None);

            let (patch, calls) = repaired(&input).await;

            assert_eq!(called(&calls), [(route, vec![(3, 4)])], "{target:?}");
            let visible = fixture.after(&patch.unwrap()).visible().await;
            for (x, y, pixel) in visible.enumerate_pixels() {
                let expected = if (x, y) == (3, 4) {
                    opaque(route)
                } else {
                    Rgba([0, 0, 0, 255])
                };
                assert_eq!(*pixel, expected, "{target:?} at ({x}, {y})");
            }
        }
    }

    #[tokio::test]
    async fn a_decensor_stroke_is_refused_before_any_route_runs() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let input = fixture.input(Some(painted(CleanupTarget::Decensor, &[(3, 4, 255)])), None);

        let (patch, calls) = repaired(&input).await;

        assert!(patch.is_err());
        assert!(calls.is_empty());
        assert!(masks(&input, (8, 8)).await.is_err());
        assert!(processor().skip(&input).await.is_err());
    }

    #[tokio::test]
    async fn detected_lettering_runs_the_bubble_route_before_the_artwork_route_and_never_the_manual_one()
     {
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(stroke(&[(1, 1, 255)])),
                ),
                (
                    "artwork-text-mask",
                    DynamicImage::ImageLuma8(stroke(&[(5, 5, 255)])),
                ),
            ],
            None,
        )
        .await;

        let (patch, calls) = repaired(&fixture.input(None, None)).await;

        assert_eq!(
            called(&calls),
            [(Route::Fast, vec![(1, 1)]), (Route::Artwork, vec![(5, 5)])]
        );
        // The artwork pass sees the bubble the fast route already cleaned.
        assert_eq!(calls[1].canvas.get_pixel(1, 1), &Rgb(paint(Route::Fast)));
        let visible = fixture.after(&patch.unwrap()).visible().await;
        assert_eq!(visible.get_pixel(1, 1), &opaque(Route::Fast));
        assert_eq!(visible.get_pixel(5, 5), &opaque(Route::Artwork));
        assert_eq!(visible.get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
    }

    #[tokio::test]
    async fn a_painted_stroke_is_clipped_to_the_requested_region() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        // The region covers x 2..6 and y 3..6: its corner pixels are inside and
        // every pixel just past an edge is outside.
        let pixels = [
            (2, 3, 255),
            (5, 5, 255),
            (1, 3, 255),
            (6, 5, 255),
            (2, 2, 255),
            (5, 6, 255),
        ];
        let region = Some(crate::Bounds {
            x: 2.0,
            y: 3.0,
            width: 4.0,
            height: 3.0,
        });

        for (target, route) in [
            (CleanupTarget::Fast, Route::Fast),
            (CleanupTarget::Inpainting, Route::Manual),
        ] {
            let input = fixture.input(Some(painted(target, &pixels)), region);

            let prepared = prepare(&input).await.unwrap();

            assert_eq!(
                routed(&prepared.masks),
                [(route, vec![(2, 3), (5, 5)])],
                "{target:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_painted_stroke_must_match_the_page() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let mut small = GrayImage::new(4, 4);
        small.put_pixel(1, 1, Luma([255]));

        for target in [CleanupTarget::Fast, CleanupTarget::Inpainting] {
            let input = fixture.input(Some((target, small.clone())), None);

            assert!(masks(&input, (8, 8)).await.is_err(), "{target:?}");
            assert!(processor().skip(&input).await.is_err(), "{target:?}");
        }
    }

    #[tokio::test]
    async fn overlapping_routes_and_legacy_masks_require_detection_again() {
        for assets in [
            vec![
                ("bubble-text-mask", white()),
                ("artwork-text-mask", white()),
            ],
            vec![("text-mask", white())],
            vec![("bubble-text-mask", white())],
        ] {
            let mut sources = vec![("source", DynamicImage::new_rgb8(8, 8))];
            sources.extend(assets);
            let fixture = Fixture::new(&sources, None).await;
            assert!(masks(&fixture.input(None, None), (8, 8)).await.is_err());
        }
    }

    #[tokio::test]
    async fn persisted_text_mask_is_clipped_to_the_requested_region() {
        let mut text = GrayImage::new(8, 8);
        text.put_pixel(2, 3, Luma([255]));
        text.put_pixel(0, 0, Luma([255]));
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("artwork-text-mask", DynamicImage::ImageLuma8(text)),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
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

        assert_eq!(
            routed(&prepared.masks),
            [(Route::Fast, vec![]), (Route::Artwork, vec![(2, 3)])]
        );
    }

    #[tokio::test]
    async fn a_painted_stroke_without_selected_pixels_skips_the_stage() {
        let fixture = Fixture::new(&[("source", DynamicImage::new_rgb8(8, 8))], None).await;
        let region = Some(crate::Bounds {
            x: 4.0,
            y: 4.0,
            width: 2.0,
            height: 2.0,
        });
        let processor = processor();

        // Nothing painted, only an antialiased fringe, or a stroke wholly
        // outside the region selects no pixel; the threshold itself does. Every
        // route reads that verdict from its own mask, so each target is checked.
        for target in [CleanupTarget::Fast, CleanupTarget::Inpainting] {
            let ignored = [
                (painted(target, &[]), None),
                (painted(target, &[(1, 1, 126)]), None),
                (painted(target, &[(0, 0, 255)]), region),
            ];
            let selected = [
                (painted(target, &[(1, 1, 127)]), None),
                (painted(target, &[(4, 5, 255)]), region),
            ];
            for (stroke, region) in ignored {
                let input = fixture.input(Some(stroke), region);
                assert!(processor.skip(&input).await.unwrap(), "{target:?}");
            }
            for (stroke, region) in selected {
                let input = fixture.input(Some(stroke), region);
                assert!(!processor.skip(&input).await.unwrap(), "{target:?}");
            }
        }
    }

    #[tokio::test]
    async fn a_manual_stroke_rebuilds_on_committed_cleanup_but_a_rerun_does_not() {
        let mut cleanup = RgbaImage::new(8, 8);
        cleanup.put_pixel(2, 3, Rgba([25, 50, 75, 255]));
        let fixture = Fixture::new(
            &[("source", DynamicImage::new_rgb8(8, 8))],
            Some((Origin::User, DynamicImage::ImageRgba8(cleanup))),
        )
        .await;
        let automatic = prepare(&fixture.input(None, None)).await.unwrap();
        assert_eq!(automatic.image.get_pixel(2, 3), Rgba([0, 0, 0, 255]));

        // Either brush repairs on top of what the page already shows.
        for target in [CleanupTarget::Fast, CleanupTarget::Inpainting] {
            let input = fixture.input(Some(painted(target, &[(2, 3, 255)])), None);

            let manual = prepare(&input).await.unwrap();

            assert_eq!(
                manual.image.get_pixel(2, 3),
                Rgba([25, 50, 75, 255]),
                "{target:?}"
            );
        }
    }

    #[tokio::test]
    async fn automatic_inpainting_skips_committed_cleanup() {
        // Detected lettering is pending, so only the authored cleanup can skip
        // the page.
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "artwork-text-mask",
                    DynamicImage::ImageLuma8(stroke(&[(1, 1, 255)])),
                ),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            Some((Origin::User, DynamicImage::ImageRgba8(RgbaImage::new(8, 8)))),
        )
        .await;
        let automatic = fixture.input(None, None);
        let processor = processor();

        assert!(processor.skip(&automatic).await.unwrap());
        for target in [CleanupTarget::Fast, CleanupTarget::Inpainting] {
            let manual = fixture.input(Some(painted(target, &[(1, 1, 255)])), None);
            assert!(!processor.skip(&manual).await.unwrap(), "{target:?}");
        }
    }

    #[tokio::test]
    async fn automatic_inpainting_runs_without_a_committed_cleanup() {
        let mut text = GrayImage::new(8, 8);
        text.put_pixel(1, 1, Luma([255]));
        let fixture = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("artwork-text-mask", DynamicImage::ImageLuma8(text)),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
        )
        .await;

        assert!(!processor().skip(&fixture.input(None, None)).await.unwrap());
    }

    #[tokio::test]
    async fn automatic_inpainting_skips_pages_without_masked_pixels() {
        let mut faint = GrayImage::new(8, 8);
        faint.put_pixel(1, 1, Luma([126]));
        let mut outside = GrayImage::new(8, 8);
        outside.put_pixel(0, 0, Luma([255]));
        let empty = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                (
                    "artwork-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
        )
        .await;
        let below_threshold = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("artwork-text-mask", DynamicImage::ImageLuma8(faint)),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
            ],
            None,
        )
        .await;
        let outside_region = Fixture::new(
            &[
                ("source", DynamicImage::new_rgb8(8, 8)),
                ("artwork-text-mask", DynamicImage::ImageLuma8(outside)),
                (
                    "bubble-text-mask",
                    DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                ),
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
        assert!(
            processor
                .skip(&outside_region.input(None, region))
                .await
                .unwrap()
        );
        assert!(
            !processor
                .skip(&outside_region.input(None, None))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn a_painted_stroke_adopts_generated_cleanup_and_keeps_every_other_pixel() {
        for (target, route) in [
            (CleanupTarget::Fast, Route::Fast),
            (CleanupTarget::Inpainting, Route::Manual),
        ] {
            let fixture = Fixture::new(
                &[
                    ("source", DynamicImage::new_rgb8(8, 8)),
                    (
                        "artwork-text-mask",
                        DynamicImage::ImageLuma8(stroke(&[(1, 1, 255)])),
                    ),
                    (
                        "bubble-text-mask",
                        DynamicImage::ImageLuma8(GrayImage::new(8, 8)),
                    ),
                ],
                None,
            )
            .await;
            let processor = processor();

            // An automatic run commits generated cleanup that a rerun may
            // replace.
            let (patch, _) = repaired(&fixture.input(None, None)).await;
            let generated = fixture.after(&patch.unwrap());
            let layers = generated.cleanup_layers();
            let [(layer, entity)] = &layers[..] else {
                panic!("the automatic run must commit one cleanup layer");
            };
            assert!(matches!(layer.origin, Origin::Generated(_)));
            assert!(matches!(entity.origin, Origin::Generated(_)));
            let pending = generated.input(None, None);
            assert!(!processor.skip(&pending).await.unwrap());

            // A painted stroke is the user's own edit: it adopts that layer
            // instead of adding another, keeps the earlier pixels, and makes
            // automatic cleanup leave the page alone from then on.
            let manual = generated.input(Some(painted(target, &[(5, 6, 255)])), None);
            let (patch, calls) = repaired(&manual).await;
            let authored = generated.after(&patch.unwrap());

            assert_eq!(called(&calls), [(route, vec![(5, 6)])], "{target:?}");
            let layers = authored.cleanup_layers();
            let [(layer, entity)] = &layers[..] else {
                panic!("a painted stroke must reuse the cleanup layer");
            };
            assert_eq!(layer.origin, Origin::User);
            assert_eq!(entity.origin, Origin::User);
            let visible = authored.visible().await;
            assert_eq!(visible.get_pixel(1, 1), &opaque(Route::Artwork));
            assert_eq!(visible.get_pixel(5, 6), &opaque(route));
            assert_eq!(visible.get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
            assert!(processor.skip(&authored.input(None, None)).await.unwrap());
            // The stroke itself is never saved as a lettering mask.
            for role in ["text-mask", "bubble-text-mask", "artwork-text-mask"] {
                let role = AssetRole::new(role).unwrap();
                assert_eq!(
                    authored.snapshot.asset(authored.page, &role).unwrap(),
                    fixture.snapshot.asset(fixture.page, &role).unwrap()
                );
            }
        }
    }

    #[test]
    fn every_route_runs_the_profile_of_its_own_settings() {
        let mut config = InpaintingConfig::default();
        config.bubbles = BubbleInpaintingKind::AotInpainting;
        config.artwork.model = InpaintingKind::Flux2Klein9B;
        config.artwork.flux2_klein_9b.prompt = "Erase the lettering.".to_owned();
        config.manual.prompt = "Repaint the painted area.".to_owned();
        config.manual.strength = 0.5;
        config.manual.seed = 7;

        let routes = Routes::select(&config).unwrap();

        assert_eq!(routes.fast, InpaintingModel::AotInpainting {});
        assert_eq!(
            routes.artwork,
            InpaintingModel::Flux2Klein9B(config.artwork.flux2_klein_9b.clone())
        );
        assert_eq!(
            routes.manual,
            InpaintingModel::Flux2Klein9B(config.manual.clone())
        );
    }

    #[test]
    fn the_manual_profile_is_validated_even_when_no_other_route_runs_flux() {
        let mut config = InpaintingConfig::default();
        config.artwork.model = InpaintingKind::LaMa;
        assert!(validate(&config).is_ok());

        for strength in [0.0, -0.5, 1.5, f64::NAN] {
            config.manual.strength = strength;
            assert!(validate(&config).is_err(), "strength {strength}");
        }
        config.manual.strength = 1.0;
        config.manual.num_inference_steps = 0;
        assert!(validate(&config).is_err());
    }
}
