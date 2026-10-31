//! Bounded tile crops and masked composition shared by every inpainting backend.
//!
//! BallonsTranslator avoids model inference for a text block when the non-text
//! pixels inside its balloon are nearly uniform, and otherwise sends an enlarged
//! block crop to the inpainter:
//! Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
//! filled first; only the remaining mask is split into bounded model crops.
//!
//! The mask is carried as horizontal runs rather than a label image. Components,
//! crop masks, and tile composition all follow those runs, so their cost scales
//! with the masked area instead of the page area, and every copy moves whole row
//! spans instead of single pixels.

use std::{borrow::Cow, cmp::Ordering, ops::Range, sync::Arc};

use anyhow::{Context as _, Result, ensure};
use image::{DynamicImage, GenericImageView as _, GrayImage, Rgb, RgbImage, imageops::FilterType};

use super::comfyui::ComfyUi;
use super::models::DiffusionCreamPyConfig;
use crate::Bounds;
use crate::StopToken;

const TILE_SIZE: u32 = 512;
const TILE_CONTEXT: u32 = 128;
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;

/// Mask pixels at or above this value belong to the mask.
pub(super) const MASK_THRESHOLD: u8 = 127;

/// A bubble polygon whose uniform interior is filled without a model.
#[derive(Clone, Debug)]
pub(super) struct FlatFillRegion {
    pub(super) bounds: [u32; 4],
    pub(super) polygon: Vec<(f32, f32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InpaintTile {
    components: Vec<u32>,
    core: [u32; 4],
    crop: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaskComponent {
    label: u32,
    bounds: [u32; 4],
}

/// Luma view of a decoded mask that copies only when the pixels are not 8-bit luma.
pub(super) fn luma(image: &DynamicImage) -> Cow<'_, GrayImage> {
    match image.as_luma8() {
        Some(luma) => Cow::Borrowed(luma),
        None => Cow::Owned(image.to_luma8()),
    }
}

/// Bounding box `[left, top, right, bottom)` of the mask pixels, or `None` when
/// no pixel reaches [`MASK_THRESHOLD`].
///
/// Stages use this as their emptiness test, so it agrees with the tiling and
/// the cleanup commit on which pixels count as masked.
pub(super) fn mask_bounds(mask: &GrayImage) -> Option<[u32; 4]> {
    let width = mask.width() as usize;
    if width == 0 {
        return None;
    }
    let mut bounds: Option<[u32; 4]> = None;
    for (y, row) in mask.chunks_exact(width).enumerate() {
        let Some(first) = row.iter().position(|value| *value >= MASK_THRESHOLD) else {
            continue;
        };
        let last = row
            .iter()
            .rposition(|value| *value >= MASK_THRESHOLD)
            .unwrap_or(first);
        let (left, right, y) = (first as u32, last as u32 + 1, y as u32);
        bounds = Some(match bounds {
            Some([min_left, top, max_right, _]) => {
                [min_left.min(left), top, max_right.max(right), y + 1]
            }
            None => [left, y, right, y + 1],
        });
    }
    bounds
}

/// Clears every mask pixel whose cell does not intersect `bounds`.
pub(super) fn clip_to_region(mask: &mut GrayImage, bounds: Bounds) {
    let (width, height) = mask.dimensions();
    if width == 0 {
        return;
    }
    let columns = pixel_range(bounds.x, bounds.width, width);
    let rows = pixel_range(bounds.y, bounds.height, height);
    for (y, row) in mask.chunks_exact_mut(width as usize).enumerate() {
        if rows.contains(&y) {
            row[..columns.start].fill(0);
            row[columns.end..].fill(0);
        } else {
            row.fill(0);
        }
    }
}

/// Pixels `p` below `limit` whose cell `[p, p + 1)` intersects `(origin, origin + extent)`.
fn pixel_range(origin: f64, extent: f64, limit: u32) -> Range<usize> {
    let limit = f64::from(limit);
    let start = origin.floor().clamp(0.0, limit) as usize;
    let end = (origin + extent).ceil().clamp(0.0, limit) as usize;
    start..end.max(start)
}

/// Stops between tiles: a native generation call cannot be interrupted, but a
/// stopped job releases the accelerator after the tile already running.
pub(super) fn inpaint_tiled(
    image: &DynamicImage,
    mask: &GrayImage,
    stop: &StopToken,
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<RgbImage> {
    let mut plan = Inpaint::plan(image, mask)?;
    for index in 0..plan.tiles.len() {
        ensure!(!stop.stopped(), "inpainting was stopped");
        let (crop, crop_mask) = plan.crop(index);
        let generated = inference(&DynamicImage::ImageRgb8(crop), &crop_mask)?;
        plan.commit(index, generated);
    }
    Ok(plan.canvas)
}

pub(super) async fn inpaint_tiled_comfy(
    image: Arc<DynamicImage>,
    mask: Arc<GrayImage>,
    stop: &StopToken,
    client: &ComfyUi,
    config: &DiffusionCreamPyConfig,
) -> Result<RgbImage> {
    // Planning is CPU-bound and must not run on the async executor.
    let mut plan = tokio::task::spawn_blocking(move || Inpaint::plan(&image, &mask))
        .await
        .context("tile planning task panicked")??;
    for index in 0..plan.tiles.len() {
        ensure!(!stop.stopped(), "inpainting was stopped");
        let (crop, crop_mask) = plan.crop(index);
        let generated = client
            .inference(&DynamicImage::ImageRgb8(crop), &crop_mask, config)
            .await?;
        plan.commit(index, generated);
    }
    Ok(plan.canvas)
}

/// A page prepared for masked inference.
///
/// `canvas` starts as the page, receives the uniform fills, and is then
/// overwritten tile by tile, so a later tile's crop already contains the pixels
/// its neighbours generated. Only mask pixels ever change.
struct Inpaint {
    canvas: RgbImage,
    components: Components,
    tiles: Vec<InpaintTile>,
}

impl Inpaint {
    fn plan(image: &DynamicImage, mask: &GrayImage) -> Result<Self> {
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ: image={:?}, mask={:?}",
            image.dimensions(),
            mask.dimensions()
        );

        let canvas = image.to_rgb8();
        let components = Components::new(mask);
        let tiles = plan_tiles(&components, canvas.width(), canvas.height());
        Ok(Self {
            canvas,
            components,
            tiles,
        })
    }

    /// The model inputs of one tile: its crop of the canvas and the binary mask
    /// of every pending pixel inside that crop, including its neighbours' context.
    fn crop(&self, index: usize) -> (RgbImage, GrayImage) {
        let [left, top, right, bottom] = self.tiles[index].crop;
        let (width, height) = (right - left, bottom - top);
        let canvas: &[u8] = &self.canvas;
        let stride = self.canvas.width() as usize * 3;
        let row_length = width as usize * 3;
        let mut pixels = Vec::with_capacity(row_length * height as usize);
        for y in top..bottom {
            let offset = y as usize * stride + left as usize * 3;
            pixels.extend_from_slice(&canvas[offset..offset + row_length]);
        }
        let mut crop_mask = vec![0; width as usize * height as usize];
        for y in top..bottom {
            let row = (y - top) as usize * width as usize;
            for run in self.components.row(y) {
                let (start, end) = (run.start.max(left), run.end.min(right));
                if start < end {
                    crop_mask[row + (start - left) as usize..row + (end - left) as usize]
                        .fill(u8::MAX);
                }
            }
        }
        (
            RgbImage::from_raw(width, height, pixels).expect("crop buffer matches its size"),
            GrayImage::from_raw(width, height, crop_mask).expect("crop mask matches its size"),
        )
    }

    /// Writes the generated crop back over the mask pixels the tile owns.
    fn commit(&mut self, index: usize, generated: DynamicImage) {
        let tile = &self.tiles[index];
        let [left, top, right, bottom] = tile.crop;
        let (width, height) = (right - left, bottom - top);
        let generated = if generated.dimensions() == (width, height) {
            generated.into_rgb8()
        } else {
            generated
                .resize_exact(width, height, FilterType::Lanczos3)
                .into_rgb8()
        };
        let [core_left, core_top, core_right, core_bottom] = tile.core;
        let source: &[u8] = &generated;
        let stride = self.canvas.width() as usize * 3;
        let canvas: &mut [u8] = &mut self.canvas;
        for &label in &tile.components {
            for run in self.components.members(label) {
                if run.y < core_top || run.y >= core_bottom {
                    continue;
                }
                let (start, end) = (run.start.max(core_left), run.end.min(core_right));
                if start >= end {
                    continue;
                }
                let from = ((run.y - top) as usize * width as usize + (start - left) as usize) * 3;
                let to = run.y as usize * stride + start as usize * 3;
                let length = (end - start) as usize * 3;
                canvas[to..to + length].copy_from_slice(&source[from..from + length]);
            }
        }
    }
}

/// Splits the components into bounded tiles, merging neighbours while the
/// merged core still fits one model crop.
fn plan_tiles(components: &Components, width: u32, height: u32) -> Vec<InpaintTile> {
    let mut bounded_tiles: Vec<InpaintTile> = Vec::new();
    let mut split_tiles = Vec::new();

    for label in 1..=components.count() {
        let component = MaskComponent {
            label,
            bounds: components.bounds[label as usize - 1],
        };
        let [left, top, right, bottom] = component.bounds;
        if right - left <= TILE_SIZE && bottom - top <= TILE_SIZE {
            let best = bounded_tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    let bounds = union_bounds(tile.core, component.bounds);
                    ((bounds[2] - bounds[0] <= TILE_SIZE) && (bounds[3] - bounds[1] <= TILE_SIZE))
                        .then_some((bounds_area(bounds) - bounds_area(tile.core), index, bounds))
                })
                .min_by_key(|(growth, index, _)| (*growth, *index));
            if let Some((_, index, bounds)) = best {
                let tile = &mut bounded_tiles[index];
                tile.components.push(component.label);
                tile.core = bounds;
                tile.crop = expand_bounds(bounds, width, height);
            } else {
                bounded_tiles.push(InpaintTile {
                    components: vec![component.label],
                    core: component.bounds,
                    crop: expand_bounds(component.bounds, width, height),
                });
            }
            continue;
        }

        let mut core_top = top;
        while core_top < bottom {
            let core_bottom = core_top.saturating_add(TILE_SIZE).min(bottom);
            let mut core_left = left;
            while core_left < right {
                let core_right = core_left.saturating_add(TILE_SIZE).min(right);
                let core = [core_left, core_top, core_right, core_bottom];
                if let Some(owned_bounds) = components.bounds_in(component.label, core) {
                    split_tiles.push(InpaintTile {
                        components: vec![component.label],
                        core,
                        crop: expand_bounds(owned_bounds, width, height),
                    });
                }
                core_left = core_right;
            }
            core_top = core_bottom;
        }
    }

    bounded_tiles.append(&mut split_tiles);
    bounded_tiles.sort_by_key(|tile| (tile.core[1], tile.core[0], tile.core[3], tile.core[2]));
    bounded_tiles
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    y: u32,
    start: u32,
    end: u32,
}

/// The 8-connected components of a mask, stored as horizontal runs.
///
/// Components are numbered from 1 by the raster position of their first pixel,
/// the numbering an 8-connected pixel labelling produces. Tile merging depends
/// on that order.
struct Components {
    /// Every run, in raster order.
    runs: Vec<Run>,
    /// `row_starts[y]..row_starts[y + 1]` are the runs of row `y`.
    row_starts: Vec<usize>,
    /// Run indices grouped by component, in raster order inside each group.
    members: Vec<u32>,
    /// `member_starts[label - 1]..member_starts[label]` is the group of `label`.
    member_starts: Vec<usize>,
    /// Bounding box of each component, indexed by `label - 1`.
    bounds: Vec<[u32; 4]>,
}

impl Components {
    fn new(mask: &GrayImage) -> Self {
        let (width, height) = mask.dimensions();
        let data: &[u8] = mask;
        let stride = width as usize;
        let mut runs = Vec::new();
        let mut row_starts = Vec::with_capacity(height as usize + 1);
        for y in 0..height {
            row_starts.push(runs.len());
            let row = &data[y as usize * stride..(y as usize + 1) * stride];
            let mut x = 0;
            while let Some(offset) = row[x..].iter().position(|value| *value >= MASK_THRESHOLD) {
                let start = x + offset;
                let length = row[start..]
                    .iter()
                    .position(|value| *value < MASK_THRESHOLD)
                    .unwrap_or(row.len() - start);
                x = start + length;
                runs.push(Run {
                    y,
                    start: start as u32,
                    end: x as u32,
                });
            }
        }
        row_starts.push(runs.len());

        // Runs on adjacent rows are connected when they overlap or meet at a corner.
        let mut parent = (0..runs.len() as u32).collect::<Vec<_>>();
        for y in 1..height as usize {
            let (mut above, above_end) = (row_starts[y - 1], row_starts[y]);
            let (mut current, current_end) = (row_starts[y], row_starts[y + 1]);
            while above < above_end && current < current_end {
                let (upper, lower) = (runs[above], runs[current]);
                if upper.start <= lower.end && lower.start <= upper.end {
                    merge(&mut parent, above as u32, current as u32);
                }
                match upper.end.cmp(&lower.end) {
                    Ordering::Less => above += 1,
                    Ordering::Greater => current += 1,
                    Ordering::Equal => {
                        above += 1;
                        current += 1;
                    }
                }
            }
        }

        // A component's root is its first run in raster order, so labels are
        // assigned in the order roots appear.
        let mut labels = vec![0_u32; runs.len()];
        let mut count = 0_u32;
        for index in 0..runs.len() {
            let representative = root(&mut parent, index as u32) as usize;
            if representative == index {
                count += 1;
                labels[index] = count;
            } else {
                labels[index] = labels[representative];
            }
        }

        let mut member_starts = vec![0_usize; count as usize + 1];
        let mut bounds = vec![[u32::MAX, u32::MAX, 0, 0]; count as usize];
        for (run, &label) in runs.iter().zip(&labels) {
            let component = label as usize - 1;
            member_starts[component + 1] += 1;
            let bounds = &mut bounds[component];
            bounds[0] = bounds[0].min(run.start);
            bounds[1] = bounds[1].min(run.y);
            bounds[2] = bounds[2].max(run.end);
            bounds[3] = bounds[3].max(run.y + 1);
        }
        for component in 0..count as usize {
            member_starts[component + 1] += member_starts[component];
        }
        let mut cursor = member_starts.clone();
        let mut members = vec![0_u32; runs.len()];
        for (index, &label) in labels.iter().enumerate() {
            let slot = &mut cursor[label as usize - 1];
            members[*slot] = index as u32;
            *slot += 1;
        }

        Self {
            runs,
            row_starts,
            members,
            member_starts,
            bounds,
        }
    }

    fn count(&self) -> u32 {
        self.bounds.len() as u32
    }

    fn row(&self, y: u32) -> &[Run] {
        &self.runs[self.row_starts[y as usize]..self.row_starts[y as usize + 1]]
    }

    fn members(&self, label: u32) -> impl Iterator<Item = Run> + '_ {
        let group = self.member_starts[label as usize - 1]..self.member_starts[label as usize];
        self.members[group]
            .iter()
            .map(|&index| self.runs[index as usize])
    }

    /// Bounding box of the component's pixels inside `region`.
    fn bounds_in(
        &self,
        label: u32,
        [region_left, region_top, region_right, region_bottom]: [u32; 4],
    ) -> Option<[u32; 4]> {
        let mut found: Option<[u32; 4]> = None;
        for run in self.members(label) {
            if run.y < region_top || run.y >= region_bottom {
                continue;
            }
            let (start, end) = (run.start.max(region_left), run.end.min(region_right));
            if start >= end {
                continue;
            }
            let bounds = found.get_or_insert([start, run.y, end, run.y + 1]);
            bounds[0] = bounds[0].min(start);
            bounds[1] = bounds[1].min(run.y);
            bounds[2] = bounds[2].max(end);
            bounds[3] = bounds[3].max(run.y + 1);
        }
        found
    }
}

fn root(parent: &mut [u32], mut index: u32) -> u32 {
    while parent[index as usize] != index {
        let grandparent = parent[parent[index as usize] as usize];
        parent[index as usize] = grandparent;
        index = grandparent;
    }
    index
}

/// Joins two sets; the smaller run index stays the root.
fn merge(parent: &mut [u32], left: u32, right: u32) {
    let (left, right) = (root(parent, left), root(parent, right));
    match left.cmp(&right) {
        Ordering::Less => parent[right as usize] = left,
        Ordering::Greater => parent[left as usize] = right,
        Ordering::Equal => {}
    }
}

fn union_bounds(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn bounds_area([left, top, right, bottom]: [u32; 4]) -> u64 {
    u64::from(right - left) * u64::from(bottom - top)
}

fn expand_bounds([left, top, right, bottom]: [u32; 4], width: u32, height: u32) -> [u32; 4] {
    [
        left.saturating_sub(TILE_CONTEXT),
        top.saturating_sub(TILE_CONTEXT),
        right.saturating_add(TILE_CONTEXT).min(width),
        bottom.saturating_add(TILE_CONTEXT).min(height),
    ]
}

/// Fills text pixels inside uniform bubbles and removes them from `pending`.
///
/// Regions are visited in order and each fill is visible to the next region's
/// background sample, exactly as sequential per-pixel filling would be.
pub(super) fn fill_uniform_regions(
    canvas: &mut RgbImage,
    pending: &mut GrayImage,
    text_mask: &GrayImage,
    regions: &[FlatFillRegion],
) {
    let width = canvas.width() as usize;
    let canvas: &mut [u8] = canvas;
    let pending: &mut [u8] = pending;
    let text: &[u8] = text_mask;
    let is_target = |pending: &[u8], index: usize| {
        pending[index] >= MASK_THRESHOLD && text[index] >= MASK_THRESHOLD
    };
    let mut scan = Scanline::default();
    for region in regions {
        let [left, top, right, bottom] = region.bounds;
        // Pending text pixels inside the polygon, as `[y, start, end)` runs.
        let mut targets = Vec::<[u32; 3]>::new();
        for y in top..bottom {
            let row = y as usize * width;
            for &[start, end] in scan.spans(&region.polygon, y, left, right) {
                let (mut x, end) = (start as usize, end as usize);
                while x < end {
                    if !is_target(pending, row + x) {
                        x += 1;
                        continue;
                    }
                    let run_start = x;
                    while x < end && is_target(pending, row + x) {
                        x += 1;
                    }
                    targets.push([y, run_start as u32, x as u32]);
                }
            }
        }
        if targets.is_empty() {
            continue;
        }
        let Some(color) = uniform_region_color(canvas, pending, width, region, &mut scan) else {
            continue;
        };
        for [y, start, end] in targets {
            let row = y as usize * width;
            let (start, end) = (row + start as usize, row + end as usize);
            for pixel in canvas[start * 3..end * 3].as_chunks_mut::<3>().0 {
                *pixel = color.0;
            }
            pending[start..end].fill(0);
        }
    }
}

/// The median background color of a bubble when its interior is uniform.
///
/// The sample is every unmasked pixel inside the polygon farther than
/// [`FLAT_FILL_EDGE_MARGIN`] from its outline.
fn uniform_region_color(
    canvas: &[u8],
    pending: &[u8],
    width: usize,
    region: &FlatFillRegion,
    scan: &mut Scanline,
) -> Option<Rgb<u8>> {
    let [left, top, right, bottom] = region.bounds;
    let band_width = (right - left) as usize;
    let near_edge = polygon_edge_band(region);
    let mut histograms = [[0_u32; 256]; 3];
    let mut count = 0_usize;
    for y in top..bottom {
        let row = y as usize * width;
        let band = (y - top) as usize * band_width;
        for &[start, end] in scan.spans(&region.polygon, y, left, right) {
            for x in start as usize..end as usize {
                if pending[row + x] >= MASK_THRESHOLD || near_edge[band + x - left as usize] {
                    continue;
                }
                let offset = (row + x) * 3;
                for (histogram, value) in histograms.iter_mut().zip(&canvas[offset..offset + 3]) {
                    histogram[*value as usize] += 1;
                }
                count += 1;
            }
        }
    }
    if count < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let medians = histograms
        .each_ref()
        .map(|histogram| median(histogram, count));
    let deviations = std::array::from_fn::<_, 3, _>(|channel| {
        standard_deviation(&histograms[channel], count, f64::from(medians[channel]))
    });
    let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
    let channel_spread = (deviations
        .iter()
        .map(|deviation| (deviation - mean_deviation).powi(2))
        .sum::<f64>()
        / deviations.len() as f64)
        .sqrt();
    let threshold = if channel_spread > 1.0 { 7.0 } else { 10.0 };
    (deviations.iter().copied().fold(0.0, f64::max) < threshold).then_some(Rgb(medians))
}

/// Scanline evaluation of the even-odd rule: the polygon is intersected with a
/// row once instead of being tested for every pixel.
#[derive(Default)]
struct Scanline {
    crossings: Vec<f32>,
    spans: Vec<[u32; 2]>,
}

impl Scanline {
    /// Half-open ranges within `[left, right)` of row `y` whose pixel centres
    /// `(x + 0.5, y + 0.5)` lie inside `polygon`.
    ///
    /// A centre is inside when an odd number of edge crossings are at or left of
    /// it. The crossings use the same arithmetic as a per-pixel ray test, so
    /// the result is identical to testing each pixel.
    fn spans(&mut self, polygon: &[(f32, f32)], y: u32, left: u32, right: u32) -> &[[u32; 2]] {
        let row = y as f32 + 0.5;
        self.crossings.clear();
        let mut previous = polygon[polygon.len() - 1];
        for &current in polygon {
            if (current.1 > row) != (previous.1 > row) {
                self.crossings.push(
                    (previous.0 - current.0) * (row - current.1) / (previous.1 - current.1)
                        + current.0,
                );
            }
            previous = current;
        }
        self.crossings.sort_unstable_by(f32::total_cmp);
        self.spans.clear();
        for pair in self.crossings.as_chunks::<2>().0 {
            let start = first_pixel_not_before(pair[0], left, right);
            let end = first_pixel_not_before(pair[1], left, right);
            if start < end {
                self.spans.push([start, end]);
            }
        }
        &self.spans
    }
}

/// The first pixel in `[left, right]` whose centre is not left of `crossing`.
fn first_pixel_not_before(crossing: f32, left: u32, right: u32) -> u32 {
    let guess = (crossing - 0.5).ceil();
    let estimate = if guess <= left as f32 {
        left
    } else if guess >= right as f32 {
        right
    } else {
        guess as u32
    };
    let mut x = estimate.max(left);
    // The estimate can be one pixel off after rounding; settle on the exact boundary.
    while x > left && (x - 1) as f32 + 0.5 >= crossing {
        x -= 1;
    }
    while x < right && (x as f32 + 0.5) < crossing {
        x += 1;
    }
    x
}

/// Which pixel centres of the region's bounds lie within
/// [`FLAT_FILL_EDGE_MARGIN`] of a polygon edge.
///
/// Each edge only visits the pixels near it, using the exact distance a
/// per-pixel test against every edge would compute.
fn polygon_edge_band(region: &FlatFillRegion) -> Vec<bool> {
    let [left, top, right, bottom] = region.bounds;
    let width = (right - left) as usize;
    let mut band = vec![false; width * (bottom - top) as usize];
    // The extra pixel keeps rounding in the bounds below from clipping a candidate.
    let reach = FLAT_FILL_EDGE_MARGIN + 1.0;
    let mut start = region.polygon[region.polygon.len() - 1];
    for &end in &region.polygon {
        let rows = (start.1.min(end.1) - reach - 0.5).ceil().max(top as f32) as u32
            ..((start.1.max(end.1) + reach - 0.5).floor() + 1.0).min(bottom as f32) as u32;
        for y in rows {
            let centre_y = y as f32 + 0.5;
            let Some((low, high)) = segment_x_range(start, end, centre_y - reach, centre_y + reach)
            else {
                continue;
            };
            let columns = (low - reach - 0.5).floor().max(left as f32) as u32
                ..((high + reach - 0.5).ceil() + 1.0).min(right as f32) as u32;
            for x in columns {
                let centre = (x as f32 + 0.5, centre_y);
                if segment_distance_squared(centre, start, end)
                    < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
                {
                    band[(y - top) as usize * width + (x - left) as usize] = true;
                }
            }
        }
        start = end;
    }
    band
}

/// Horizontal extent of the part of a segment with `y` in `[y_low, y_high]`.
fn segment_x_range(
    first: (f32, f32),
    second: (f32, f32),
    y_low: f32,
    y_high: f32,
) -> Option<(f32, f32)> {
    let (upper, lower) = if first.1 <= second.1 {
        (first, second)
    } else {
        (second, first)
    };
    if lower.1 < y_low || upper.1 > y_high {
        return None;
    }
    if upper.1 == lower.1 {
        return Some((upper.0.min(lower.0), upper.0.max(lower.0)));
    }
    let x_at = |y: f32| {
        upper.0 + (lower.0 - upper.0) * ((y - upper.1) / (lower.1 - upper.1)).clamp(0.0, 1.0)
    };
    let (x_start, x_end) = (x_at(y_low), x_at(y_high));
    Some((x_start.min(x_end), x_start.max(x_end)))
}

fn segment_distance_squared(point: (f32, f32), start: (f32, f32), end: (f32, f32)) -> f32 {
    let segment = (end.0 - start.0, end.1 - start.1);
    let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
    let projection = (if length_squared > 0.0 {
        ((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1) / length_squared
    } else {
        0.0
    })
    .clamp(0.0, 1.0);
    let closest = (
        start.0 + segment.0 * projection,
        start.1 + segment.1 * projection,
    );
    let dx = point.0 - closest.0;
    let dy = point.1 - closest.1;
    dx * dx + dy * dy
}

/// Median of the sampled values, as the sorted middle (or middle pair) would give.
fn median(histogram: &[u32; 256], count: usize) -> u8 {
    let middle = count / 2;
    if count.is_multiple_of(2) {
        ((u16::from(nth_smallest(histogram, middle - 1))
            + u16::from(nth_smallest(histogram, middle)))
            / 2) as u8
    } else {
        nth_smallest(histogram, middle)
    }
}

fn nth_smallest(histogram: &[u32; 256], index: usize) -> u8 {
    let mut seen = 0_usize;
    for (value, &occurrences) in histogram.iter().enumerate() {
        seen += occurrences as usize;
        if seen > index {
            return value as u8;
        }
    }
    u8::MAX
}

/// Standard deviation around `center`. Every term is an exact integer in `f64`,
/// so the sum does not depend on the order the samples were gathered in.
fn standard_deviation(histogram: &[u32; 256], count: usize, center: f64) -> f64 {
    (histogram
        .iter()
        .enumerate()
        .map(|(value, &occurrences)| f64::from(occurrences) * (value as f64 - center).powi(2))
        .sum::<f64>()
        / count as f64)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use image::Luma;
    use imageproc::region_labelling::{Connectivity, connected_components};

    use super::*;

    fn rectangle_region([left, top, right, bottom]: [u32; 4]) -> FlatFillRegion {
        FlatFillRegion {
            bounds: [left, top, right, bottom],
            polygon: vec![
                (left as f32, top as f32),
                (right as f32, top as f32),
                (right as f32, bottom as f32),
                (left as f32, bottom as f32),
            ],
        }
    }

    /// Deterministic pseudo-random bytes for fixtures.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        fn below(&mut self, limit: u32) -> u32 {
            self.next() % limit
        }
    }

    fn random_mask(noise: &mut Noise, width: u32, height: u32, blobs: u32) -> GrayImage {
        let mut mask = GrayImage::new(width, height);
        for _ in 0..blobs {
            let (x, y) = (noise.below(width), noise.below(height));
            let (w, h) = (1 + noise.below(40), 1 + noise.below(40));
            let level = if noise.below(4) == 0 { 90 } else { 255 };
            for yy in y..(y + h).min(height) {
                for xx in x..(x + w).min(width) {
                    mask.put_pixel(xx, yy, Luma([level]));
                }
            }
        }
        mask
    }

    /// The original per-pixel ray test.
    fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
        let mut inside = false;
        let mut previous = polygon[polygon.len() - 1];
        for &current in polygon {
            if (current.1 > point.1) != (previous.1 > point.1) {
                let intersection_x = (previous.0 - current.0) * (point.1 - current.1)
                    / (previous.1 - current.1)
                    + current.0;
                if point.0 < intersection_x {
                    inside = !inside;
                }
            }
            previous = current;
        }
        inside
    }

    /// The original distance to the nearest edge, evaluated for every edge.
    fn polygon_edge_distance_squared(point: (f32, f32), polygon: &[(f32, f32)]) -> f32 {
        let mut minimum = f32::INFINITY;
        let mut start = polygon[polygon.len() - 1];
        for &end in polygon {
            let segment = (end.0 - start.0, end.1 - start.1);
            let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
            let projection = if length_squared > 0.0 {
                ((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1) / length_squared
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            let closest = (
                start.0 + segment.0 * projection,
                start.1 + segment.1 * projection,
            );
            let dx = point.0 - closest.0;
            let dy = point.1 - closest.1;
            minimum = minimum.min(dx * dx + dy * dy);
            start = end;
        }
        minimum
    }

    fn star_polygon(noise: &mut Noise, centre: (f32, f32), points: u32) -> Vec<(f32, f32)> {
        (0..points)
            .map(|index| {
                let angle = index as f32 / points as f32 * std::f32::consts::TAU;
                let radius = 30.0 + noise.below(2000) as f32 / 100.0;
                (
                    centre.0 + radius * angle.cos() * 1.3,
                    centre.1 + radius * angle.sin(),
                )
            })
            .collect()
    }

    fn polygon_region(polygon: Vec<(f32, f32)>, width: u32, height: u32) -> FlatFillRegion {
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for &(x, y) in &polygon {
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
        }
        FlatFillRegion {
            bounds: [
                bounds[0].floor().clamp(0.0, width as f32) as u32,
                bounds[1].floor().clamp(0.0, height as f32) as u32,
                bounds[2].ceil().clamp(0.0, width as f32) as u32,
                bounds[3].ceil().clamp(0.0, height as f32) as u32,
            ],
            polygon,
        }
    }

    #[test]
    fn uniform_mask_background_is_filled_without_inference() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([240, 241, 242]));
        let mut mask = GrayImage::new(96, 96);
        for y in 40..56 {
            for x in 32..64 {
                image.put_pixel(x, y, Rgb([20, 20, 20]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let text = mask.clone();
        fill_uniform_regions(
            &mut image,
            &mut mask,
            &text,
            &[rectangle_region([0, 0, 96, 96])],
        );
        assert!(mask.iter().all(|&value| value == 0));
        assert_eq!(image.get_pixel(48, 48), &Rgb([240, 241, 242]));
    }

    #[test]
    fn uniform_bubbles_are_filled_independently_inside_one_textured_tile() {
        let mut image = RgbImage::new(200, 100);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([20, 80, 140])
            } else {
                Rgb([220, 80, 120])
            };
        }
        for y in 10..90 {
            for x in 10..90 {
                image.put_pixel(x, y, Rgb([245, 245, 245]));
            }
            for x in 110..190 {
                image.put_pixel(x, y, Rgb([250, 240, 220]));
            }
        }
        let mut mask = GrayImage::new(200, 100);
        for y in 40..60 {
            for x in 30..60 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
            for x in 135..165 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let text = mask.clone();
        fill_uniform_regions(
            &mut image,
            &mut mask,
            &text,
            &[
                rectangle_region([10, 10, 90, 90]),
                rectangle_region([110, 10, 190, 90]),
            ],
        );
        assert!(mask.iter().all(|&value| value == 0));
        assert_eq!(image.get_pixel(45, 50), &Rgb([245, 245, 245]));
        assert_eq!(image.get_pixel(150, 50), &Rgb([250, 240, 220]));
    }

    #[test]
    fn connected_mask_crossing_both_page_grid_axes_is_inferred_once() {
        let image = RgbImage::from_pixel(900, 900, Rgb([20, 40, 60]));
        let mut mask = GrayImage::new(900, 900);
        for y in 500..525 {
            for x in 500..525 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &StopToken::default(),
            |tile, tile_mask| {
                calls += 1;
                assert_eq!(
                    tile_mask.pixels().filter(|pixel| pixel[0] >= 127).count(),
                    25 * 25
                );
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap();

        assert_eq!(calls, 1);
        assert_eq!(output.get_pixel(500, 500), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(524, 524), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(499, 499), &Rgb([20, 40, 60]));
    }

    #[test]
    fn oversized_component_tiles_mask_their_continuation_context() {
        let mut mask = GrayImage::new(1000, 300);
        for y in 100..120 {
            for x in 50..850 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let image = DynamicImage::ImageRgb8(RgbImage::new(1000, 300));

        let plan = Inpaint::plan(&image, &mask).unwrap();

        assert_eq!(plan.tiles.len(), 2);
        for (index, tile) in plan.tiles.iter().enumerate() {
            assert!(tile.crop[2] - tile.crop[0] <= TILE_SIZE + TILE_CONTEXT * 2);
            assert!(tile.crop[3] - tile.crop[1] <= TILE_SIZE + TILE_CONTEXT * 2);
            let label = tile.components[0];
            let [left, top, right, bottom] = tile.core;
            let owned = plan
                .components
                .members(label)
                .filter(|run| run.y >= top && run.y < bottom)
                .map(|run| run.end.min(right).saturating_sub(run.start.max(left)) as usize)
                .sum::<usize>();
            let masked = plan
                .crop(index)
                .1
                .pixels()
                .filter(|pixel| pixel[0] >= 127)
                .count();
            assert!(masked > owned);
        }
    }

    #[test]
    fn textured_mask_regions_are_inferred_as_bounded_tiles() {
        let mut image = RgbImage::new(1200, 700);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([0, 40, 80])
            } else {
                Rgb([255, 215, 175])
            };
        }
        let original = image.clone();
        let mut mask = GrayImage::new(1200, 700);
        mask.put_pixel(10, 10, Luma([u8::MAX]));
        mask.put_pixel(1100, 600, Luma([u8::MAX]));
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &StopToken::default(),
            |tile, _| {
                calls += 1;
                assert!(tile.width() <= TILE_CONTEXT * 2 + 1);
                assert!(tile.height() <= TILE_CONTEXT * 2 + 1);
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap();

        assert_eq!(calls, 2);
        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(1100, 600), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(500, 300), original.get_pixel(500, 300));
    }

    #[test]
    fn later_tiles_see_the_pixels_earlier_tiles_generated() {
        let image = RgbImage::from_pixel(900, 120, Rgb([9, 9, 9]));
        let mut mask = GrayImage::new(900, 120);
        // Two 300 wide blocks whose union is wider than one tile core, so each
        // gets its own call, but whose crops overlap across the gap.
        for y in 50..60 {
            for x in (100..400).chain(450..750) {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut seen = Vec::new();

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &StopToken::default(),
            |tile, _| {
                seen.push(tile.to_rgb8());
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([seen.len() as u8 * 50, 0, 0]),
                )))
            },
        )
        .unwrap();

        assert_eq!(seen.len(), 2);
        assert_eq!(output.get_pixel(250, 55), &Rgb([50, 0, 0]));
        assert_eq!(output.get_pixel(600, 55), &Rgb([100, 0, 0]));
        assert_eq!(output.get_pixel(420, 55), &Rgb([9, 9, 9]));
        // The second crop starts at x = 322 and reaches back over the first
        // block: it must show the first tile's output there, and the original
        // pixels in the gap between the blocks.
        assert_eq!(seen[1].get_pixel(350 - 322, 55), &Rgb([50, 0, 0]));
        assert_eq!(seen[1].get_pixel(420 - 322, 55), &Rgb([9, 9, 9]));
    }

    #[test]
    fn mismatched_generated_size_is_resampled_to_the_crop() {
        let image = RgbImage::from_pixel(64, 64, Rgb([5, 5, 5]));
        let mut mask = GrayImage::new(64, 64);
        mask.put_pixel(32, 32, Luma([u8::MAX]));

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &StopToken::default(),
            |_, _| {
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    8,
                    8,
                    Rgb([77, 77, 77]),
                )))
            },
        )
        .unwrap();

        assert!(output.get_pixel(32, 32)[0].abs_diff(77) <= 1);
        assert_eq!(output.get_pixel(31, 32), &Rgb([5, 5, 5]));
    }

    #[test]
    fn components_match_pixel_labelling() {
        let mut noise = Noise(7);
        for blobs in [0, 1, 3, 25, 120] {
            let mask = random_mask(&mut noise, 160, 120, blobs);
            let binary = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
                Luma([(mask.get_pixel(x, y)[0] >= MASK_THRESHOLD) as u8 * u8::MAX])
            });
            let reference = connected_components(&binary, Connectivity::Eight, Luma([0]));

            let components = Components::new(&mask);

            let mut covered = 0;
            for label in 1..=components.count() {
                let mut bounds = [u32::MAX, u32::MAX, 0, 0];
                for run in components.members(label) {
                    for x in run.start..run.end {
                        assert_eq!(reference.get_pixel(x, run.y)[0], label);
                        covered += 1;
                    }
                    bounds[0] = bounds[0].min(run.start);
                    bounds[1] = bounds[1].min(run.y);
                    bounds[2] = bounds[2].max(run.end);
                    bounds[3] = bounds[3].max(run.y + 1);
                }
                assert_eq!(components.bounds[label as usize - 1], bounds);
            }
            let foreground = reference.pixels().filter(|pixel| pixel[0] != 0).count();
            assert_eq!(covered, foreground);
            assert_eq!(
                components.count(),
                reference.pixels().map(|pixel| pixel[0]).max().unwrap_or(0)
            );
        }
    }

    #[test]
    fn component_bounds_in_matches_a_pixel_scan() {
        let mut noise = Noise(11);
        let mask = random_mask(&mut noise, 200, 150, 60);
        let components = Components::new(&mask);
        for label in 1..=components.count() {
            for _ in 0..8 {
                let left = noise.below(200);
                let top = noise.below(150);
                let region = [
                    left,
                    top,
                    (left + 1 + noise.below(120)).min(200),
                    (top + 1 + noise.below(120)).min(150),
                ];
                let mut expected: Option<[u32; 4]> = None;
                for run in components.members(label) {
                    for x in run.start..run.end {
                        if x >= region[0]
                            && x < region[2]
                            && run.y >= region[1]
                            && run.y < region[3]
                        {
                            let b = expected.get_or_insert([x, run.y, x + 1, run.y + 1]);
                            b[0] = b[0].min(x);
                            b[1] = b[1].min(run.y);
                            b[2] = b[2].max(x + 1);
                            b[3] = b[3].max(run.y + 1);
                        }
                    }
                }
                assert_eq!(components.bounds_in(label, region), expected);
            }
        }
    }

    #[test]
    fn scanline_spans_match_the_per_pixel_ray_test() {
        let mut noise = Noise(3);
        let mut scan = Scanline::default();
        for _ in 0..12 {
            let centre = (
                90.0 + noise.below(40) as f32 * 0.37,
                80.0 + noise.below(40) as f32 * 0.61,
            );
            let points = 3 + noise.below(40);
            let polygon = star_polygon(&mut noise, centre, points);
            let region = polygon_region(polygon, 200, 160);
            let [left, top, right, bottom] = region.bounds;
            for y in top..bottom {
                let spans = scan.spans(&region.polygon, y, left, right).to_vec();
                for x in left..right {
                    let expected =
                        point_in_polygon((x as f32 + 0.5, y as f32 + 0.5), &region.polygon);
                    let actual = spans.iter().any(|[start, end]| x >= *start && x < *end);
                    assert_eq!(actual, expected, "pixel ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn edge_band_matches_the_distance_to_every_edge() {
        let mut noise = Noise(5);
        for _ in 0..8 {
            let centre = (
                90.0 + noise.below(40) as f32 * 0.29,
                80.0 + noise.below(40) as f32 * 0.53,
            );
            let points = 3 + noise.below(30);
            let polygon = star_polygon(&mut noise, centre, points);
            let region = polygon_region(polygon, 200, 160);
            let [left, top, right, bottom] = region.bounds;
            let band = polygon_edge_band(&region);
            for y in top..bottom {
                for x in left..right {
                    let near = polygon_edge_distance_squared(
                        (x as f32 + 0.5, y as f32 + 0.5),
                        &region.polygon,
                    ) < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN;
                    assert_eq!(
                        band[(y - top) as usize * (right - left) as usize + (x - left) as usize],
                        near
                    );
                }
            }
        }
    }

    /// The original sample statistics: gather, sort, and take the middle.
    fn sorted_uniform_color(
        image: &RgbImage,
        mask: &GrayImage,
        region: &FlatFillRegion,
    ) -> Option<Rgb<u8>> {
        let [left, top, right, bottom] = region.bounds;
        let mut channels = [Vec::new(), Vec::new(), Vec::new()];
        for y in top..bottom {
            for x in left..right {
                let point = (x as f32 + 0.5, y as f32 + 0.5);
                if mask.get_pixel(x, y)[0] >= 127
                    || !point_in_polygon(point, &region.polygon)
                    || polygon_edge_distance_squared(point, &region.polygon)
                        < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
                {
                    continue;
                }
                let pixel = image.get_pixel(x, y);
                for channel in 0..3 {
                    channels[channel].push(pixel[channel]);
                }
            }
        }
        if channels[0].len() < UNIFORM_BACKGROUND_MIN_PIXELS {
            return None;
        }
        let medians = channels.each_mut().map(|values| {
            values.sort_unstable();
            let middle = values.len() / 2;
            if values.len().is_multiple_of(2) {
                ((u16::from(values[middle - 1]) + u16::from(values[middle])) / 2) as u8
            } else {
                values[middle]
            }
        });
        let deviations = std::array::from_fn::<_, 3, _>(|channel| {
            (channels[channel]
                .iter()
                .map(|value| (f64::from(*value) - f64::from(medians[channel])).powi(2))
                .sum::<f64>()
                / channels[channel].len() as f64)
                .sqrt()
        });
        let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
        let channel_spread = (deviations
            .iter()
            .map(|deviation| (deviation - mean_deviation).powi(2))
            .sum::<f64>()
            / deviations.len() as f64)
            .sqrt();
        let threshold = if channel_spread > 1.0 { 7.0 } else { 10.0 };
        (deviations.iter().copied().fold(0.0, f64::max) < threshold).then_some(Rgb(medians))
    }

    #[test]
    fn uniform_color_matches_the_sorted_sample_statistics() {
        let mut noise = Noise(13);
        let mut scan = Scanline::default();
        for amplitude in [0, 2, 5, 9, 14, 30] {
            let mut image = RgbImage::new(200, 160);
            for pixel in image.pixels_mut() {
                let jitter = |noise: &mut Noise| (200 + noise.below(amplitude + 1)) as u8;
                *pixel = Rgb([jitter(&mut noise), jitter(&mut noise), jitter(&mut noise)]);
            }
            let mask = random_mask(&mut noise, 200, 160, 20);
            let polygon = star_polygon(&mut noise, (100.0, 80.0), 24);
            let region = polygon_region(polygon, 200, 160);

            let expected = sorted_uniform_color(&image, &mask, &region);
            let actual = uniform_region_color(&image, &mask, 200, &region, &mut scan);

            assert_eq!(actual, expected, "amplitude {amplitude}");
        }
    }

    #[test]
    fn fill_matches_pixel_by_pixel_filling_with_overlapping_regions() {
        let mut noise = Noise(17);
        let (width, height) = (220_u32, 170_u32);
        let mut image = RgbImage::new(width, height);
        for pixel in image.pixels_mut() {
            *pixel = Rgb([230 + noise.below(3) as u8, 231, 232]);
        }
        let mut mask = GrayImage::new(width, height);
        for _ in 0..14 {
            let (x, y) = (20 + noise.below(150), 20 + noise.below(110));
            for yy in y..y + 12 {
                for xx in x..x + 25 {
                    mask.put_pixel(xx, yy, Luma([255]));
                    image.put_pixel(xx, yy, Rgb([10, 10, 10]));
                }
            }
        }
        let regions = [
            polygon_region(star_polygon(&mut noise, (80.0, 80.0), 18), width, height),
            polygon_region(star_polygon(&mut noise, (120.0, 90.0), 22), width, height),
            polygon_region(star_polygon(&mut noise, (60.0, 60.0), 9), width, height),
        ];

        let mut actual_canvas = image.clone();
        let mut actual_pending = mask.clone();
        fill_uniform_regions(&mut actual_canvas, &mut actual_pending, &mask, &regions);

        // Reference: the original per-pixel algorithm.
        let mut expected_canvas = image.clone();
        let mut expected_pending = mask.clone();
        for region in &regions {
            let [left, top, right, bottom] = region.bounds;
            let mut targets = Vec::new();
            for y in top..bottom {
                for x in left..right {
                    if expected_pending.get_pixel(x, y)[0] >= 127
                        && mask.get_pixel(x, y)[0] >= 127
                        && point_in_polygon((x as f32 + 0.5, y as f32 + 0.5), &region.polygon)
                    {
                        targets.push((x, y));
                    }
                }
            }
            if targets.is_empty() {
                continue;
            }
            let Some(color) = sorted_uniform_color(&expected_canvas, &expected_pending, region)
            else {
                continue;
            };
            for (x, y) in targets {
                expected_canvas.put_pixel(x, y, color);
                expected_pending.put_pixel(x, y, Luma([0]));
            }
        }

        assert_eq!(actual_canvas, expected_canvas);
        assert_eq!(actual_pending, expected_pending);
        assert!(
            actual_pending.as_raw() != mask.as_raw(),
            "the fixture must exercise at least one fill"
        );
    }

    #[test]
    fn region_clip_keeps_exactly_the_cells_that_intersect_the_bounds() {
        let mut noise = Noise(19);
        for _ in 0..40 {
            let bounds = Bounds {
                x: noise.below(2400) as f64 / 100.0 - 5.0,
                y: noise.below(2400) as f64 / 100.0 - 5.0,
                width: 0.25 + noise.below(1800) as f64 / 100.0,
                height: 0.25 + noise.below(1800) as f64 / 100.0,
            };
            let mut mask = GrayImage::from_pixel(23, 19, Luma([255]));

            clip_to_region(&mut mask, bounds);

            for (x, y, pixel) in mask.enumerate_pixels() {
                let outside = f64::from(x + 1) <= bounds.x
                    || f64::from(y + 1) <= bounds.y
                    || f64::from(x) >= bounds.x + bounds.width
                    || f64::from(y) >= bounds.y + bounds.height;
                assert_eq!(pixel[0] == 0, outside, "pixel ({x}, {y}) for {bounds:?}");
            }
        }
    }

    #[test]
    fn mask_bounds_only_counts_pixels_at_the_threshold() {
        let mut mask = GrayImage::new(10, 8);
        assert_eq!(mask_bounds(&mask), None);
        mask.put_pixel(4, 3, Luma([MASK_THRESHOLD - 1]));
        assert_eq!(mask_bounds(&mask), None);
        mask.put_pixel(6, 2, Luma([MASK_THRESHOLD]));
        mask.put_pixel(1, 5, Luma([255]));
        assert_eq!(mask_bounds(&mask), Some([1, 2, 7, 6]));
    }
}
