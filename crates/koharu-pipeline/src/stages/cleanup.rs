//! The page cleanup layer shared by the inpainting and decensoring stages.
//!
//! Generated pixels are committed as one `Cleanup` raster layer: both stages
//! read the pixels already committed there and write their own result back, so a
//! reconstruction never discards what another stage or the artist painted.

use std::{borrow::Cow, collections::BTreeMap, io::Cursor, sync::Arc};

use anyhow::{Context as _, Result, bail};
use image::{
    DynamicImage, GenericImageView as _, GrayImage, ImageFormat, Pixel as _, RgbImage, Rgba,
    RgbaImage,
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, EntityOrigin, Origin, RasterLayer,
    RasterLayerKind,
};

use super::tiling::MASK_THRESHOLD;
use super::{StageInput, finish, generation};

pub(super) struct CleanupLayer {
    entity: Option<EntityId>,
    /// The decoded layer, shared with the page's image cache instead of copied.
    pixels: Option<Arc<DynamicImage>>,
}

impl CleanupLayer {
    /// A layer that is not attached to any scene entity.
    #[cfg(test)]
    pub(super) fn detached(pixels: DynamicImage) -> Self {
        Self {
            entity: None,
            pixels: Some(Arc::new(pixels)),
        }
    }

    fn rgba(&self) -> Option<Cow<'_, RgbaImage>> {
        let pixels = self.pixels.as_deref()?;
        Some(match pixels.as_rgba8() {
            Some(rgba) => Cow::Borrowed(rgba),
            None => Cow::Owned(pixels.to_rgba8()),
        })
    }
}

/// Inpainting and decensoring rebuild pixels in the page's one cleanup layer, so
/// their automatic commits share a producer: either stage may refresh what the
/// other generated, while the scene still rejects pipeline edits to anything the
/// user authored. The identifier predates decensoring and stays because the
/// generated cleanup of existing projects is stamped with it.
const PRODUCER: &str = "dev.koharu.pipeline.inpainting";

/// The page's cleanup layer entity, if one is committed.
pub(super) fn layer_entity(input: &StageInput) -> Result<Option<EntityId>> {
    for entity in input.scene.children(input.page)? {
        if input
            .scene
            .component::<RasterLayer>(entity)?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
        {
            return Ok(Some(entity));
        }
    }
    Ok(None)
}

/// Whether an automatic run must leave the page's cleanup layer alone: some part
/// of it is not cleanup generated under [`PRODUCER`]. A painted stroke, a brush
/// or eraser edit of the layer pixels, and another producer's output all count.
pub(super) fn protected(input: &StageInput) -> Result<bool> {
    let Some(entity) = layer_entity(input)? else {
        return Ok(false);
    };
    let generated = |origin: &Origin| matches!(origin, Origin::Generated(generation) if generation.producer.as_str() == PRODUCER);
    let owned = input
        .scene
        .component::<EntityOrigin>(entity)?
        .is_some_and(|value| generated(&value.origin))
        && input
            .scene
            .component::<RasterLayer>(entity)?
            .is_some_and(|layer| generated(&layer.origin))
        && input
            .scene
            .asset(entity, &AssetRole::new("source")?)?
            .is_none_or(|asset| generated(&asset.origin));
    Ok(!owned)
}

pub(super) async fn load(input: &StageInput, source: &DynamicImage) -> Result<CleanupLayer> {
    let entity = layer_entity(input)?;
    let pixels = if let Some(entity) = entity {
        input.images.get(&input.scene, entity, "source").await?
    } else {
        None
    };
    if pixels
        .as_ref()
        .is_some_and(|pixels| pixels.dimensions() != source.dimensions())
    {
        bail!("cleanup layer dimensions do not match page {}", input.page);
    }
    Ok(CleanupLayer { entity, pixels })
}

/// Composites committed cleanup pixels over the page source, so a backend sees
/// the artwork it already reconstructed instead of the censored original.
pub(super) fn composite(source: &Arc<DynamicImage>, layer: &CleanupLayer) -> Arc<DynamicImage> {
    let Some(cleanup) = layer.rgba() else {
        return Arc::clone(source);
    };
    let mut composite = source.to_rgba8();
    let pixels: &mut [u8] = &mut composite;
    let (cleanup_pixels, _) = cleanup.as_chunks::<4>();
    let (page_pixels, _) = pixels.as_chunks_mut::<4>();
    for (top, bottom) in cleanup_pixels.iter().zip(page_pixels.iter_mut()) {
        match top[3] {
            0 => {}
            u8::MAX => *bottom = *top,
            _ => {
                // Partial coverage blends like `image::imageops::overlay`.
                let mut blended = Rgba(*bottom);
                blended.blend(&Rgba(*top));
                *bottom = blended.0;
            }
        }
    }
    Arc::new(DynamicImage::ImageRgba8(composite))
}

/// The cleanup layer pixels after `generated` is committed inside `mask`.
///
/// `bounds` is the bounding box of the mask pixels, so only that area is
/// scanned.
pub(super) fn render(
    layer: &CleanupLayer,
    generated: &RgbImage,
    mask: &GrayImage,
    bounds: [u32; 4],
) -> RgbaImage {
    let mut overlay = layer.rgba().map_or_else(
        || RgbaImage::new(generated.width(), generated.height()),
        |existing| existing.into_owned(),
    );
    stamp(&mut overlay, generated, mask, bounds);
    overlay
}

/// Commits `generated` inside `mask` as the page cleanup layer.
///
/// An automatic run is a pipeline edit stamped with `model` that keeps the
/// committed pixels around its mask. A painted stroke is the user's own edit
/// and also adopts the layer, so automatic runs leave the pixels the user chose
/// alone from then on. The committed layer keeps every mask pixel fully opaque.
pub(super) fn commit(
    input: &StageInput,
    layer: &CleanupLayer,
    generated: &RgbImage,
    mask: &GrayImage,
    bounds: [u32; 4],
    model: &str,
) -> Result<koharu_scene::Patch> {
    let manual = input.inpainting_mask.is_some();
    let mut edit = if manual {
        input.scene.edit()
    } else {
        input.scene.edit_as(generation(PRODUCER, model)?)
    };
    let page = input.page;
    let overlay = render(layer, generated, mask, bounds);

    edit.observe_assets(page)?;
    if let Some(entity) = layer.entity {
        edit.observe::<RasterLayer>(entity)?;
        edit.observe_assets(entity)?;
    }
    let cleanup_entity = if let Some(entity) = layer.entity {
        if manual {
            let mut raster = input
                .scene
                .component::<RasterLayer>(entity)?
                .context("cleanup entity has no raster layer component")?;
            let generated = raster.origin != Origin::User
                || input
                    .scene
                    .component::<EntityOrigin>(entity)?
                    .is_some_and(|origin| origin.origin != Origin::User);
            if generated {
                edit.promote_entity_to_user(entity)?;
                raster.origin = Origin::User;
                edit.set(entity, &raster)?;
            }
        }
        entity
    } else {
        let entity = edit.add_entity(page, At::Start)?;
        edit.set(
            entity,
            &RasterLayer {
                origin: Origin::User,
                name: "Cleanup".to_owned(),
                kind: RasterLayerKind::Cleanup,
            },
        )?;
        entity
    };

    let (width, height) = overlay.dimensions();
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;
    edit.set_asset(
        cleanup_entity,
        &AssetRole::new("source")?,
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

/// Writes the generated pixels of every mask pixel inside `bounds` into
/// `overlay`.
///
/// The layer stays opaque over the whole mask: transparent texels next to
/// edited pixels let linear canvas sampling reveal the artwork below.
fn stamp(overlay: &mut RgbaImage, generated: &RgbImage, mask: &GrayImage, bounds: [u32; 4]) {
    let [left, top, right, bottom] = bounds.map(|edge| edge as usize);
    let width = mask.width() as usize;
    let mask: &[u8] = mask;
    let generated: &[u8] = generated;
    let overlay: &mut [u8] = overlay;
    for y in top..bottom {
        let row = y * width;
        for x in left..right {
            if mask[row + x] < MASK_THRESHOLD {
                continue;
            }
            let (from, to) = ((row + x) * 3, (row + x) * 4);
            overlay[to..to + 3].copy_from_slice(&generated[from..from + 3]);
            overlay[to + 3] = u8::MAX;
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{Luma, Rgb};

    use super::*;

    #[test]
    fn composite_matches_overlay_for_every_coverage() {
        let mut page = RgbaImage::new(5, 3);
        for (x, y, pixel) in page.enumerate_pixels_mut() {
            *pixel = Rgba([
                (x * 40) as u8,
                (y * 70) as u8,
                200,
                if x == 4 { 90 } else { 255 },
            ]);
        }
        let mut cleanup = RgbaImage::new(5, 3);
        for (x, y, pixel) in cleanup.enumerate_pixels_mut() {
            *pixel = Rgba([
                10 + x as u8,
                120,
                250 - y as u8,
                [0, 255, 128, 1, 254][x as usize],
            ]);
        }

        // Opaque and translucent pages: the result must equal `overlay` of the
        // page converted to RGBA, whatever the coverage of each cleanup pixel.
        for source in [
            DynamicImage::ImageRgba8(page.clone()),
            DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(page).to_rgb8()),
        ] {
            let layer = CleanupLayer::detached(DynamicImage::ImageRgba8(cleanup.clone()));

            let actual = composite(&Arc::new(source.clone()), &layer).to_rgba8();

            let mut expected = source.to_rgba8();
            image::imageops::overlay(&mut expected, &cleanup, 0, 0);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn stamp_writes_only_mask_pixels_and_keeps_them_opaque() {
        let mut overlay = RgbaImage::new(4, 3);
        overlay.put_pixel(0, 0, Rgba([1, 2, 3, 4]));
        let generated = RgbImage::from_pixel(4, 3, Rgb([9, 8, 7]));
        let mut mask = GrayImage::new(4, 3);
        mask.put_pixel(1, 1, Luma([MASK_THRESHOLD]));
        mask.put_pixel(2, 1, Luma([MASK_THRESHOLD - 1]));
        mask.put_pixel(3, 2, Luma([255]));

        stamp(&mut overlay, &generated, &mask, [1, 1, 4, 3]);

        assert_eq!(overlay.get_pixel(1, 1), &Rgba([9, 8, 7, 255]));
        assert_eq!(overlay.get_pixel(3, 2), &Rgba([9, 8, 7, 255]));
        assert_eq!(overlay.get_pixel(2, 1), &Rgba([0, 0, 0, 0]));
        assert_eq!(overlay.get_pixel(0, 0), &Rgba([1, 2, 3, 4]));
    }

    #[test]
    fn render_preserves_pixels_outside_each_cleanup_mask() {
        let mut existing = RgbaImage::new(4, 3);
        existing.put_pixel(0, 0, Rgba([1, 2, 3, 255]));
        existing.put_pixel(1, 1, Rgba([4, 5, 6, 255]));
        let layer = CleanupLayer::detached(DynamicImage::ImageRgba8(existing));
        let generated = RgbImage::from_pixel(4, 3, Rgb([9, 8, 7]));
        let mut mask = GrayImage::new(4, 3);
        mask.put_pixel(1, 1, Luma([255]));

        let preserved = render(&layer, &generated, &mask, [1, 1, 2, 2]);

        assert_eq!(preserved.get_pixel(0, 0), &Rgba([1, 2, 3, 255]));
        assert_eq!(preserved.get_pixel(1, 1), &Rgba([9, 8, 7, 255]));
    }
}
