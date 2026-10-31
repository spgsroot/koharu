//! White-stroke censorship candidates for manga pages.
//!
//! Some releases hide anatomy with strokes painted in paper white instead of
//! black bars or mosaic. The Er0mangaSeg checkpoint scores none of them, and a
//! white stroke lying on white paper is invisible by construction. What stays
//! observable is where a stroke interrupts ink: a straight, constant-width,
//! pure-white band whose long edges end tone dots or line work. Natural white
//! shapes with the same silhouette (hair gaps, speed-line gaps, panel gutters,
//! lettering rows) are far more common than strokes, so one band proves
//! nothing; a mask is reported only for a stack of such bands.
//!
//! This is Koharu's own heuristic and is not part of any ported network.
//! Pixel sizes below are stated for a page whose longest side is
//! [`REFERENCE_EXTENT`] and scale linearly with the page.
//!
//! 1. Large white regions (speech bubbles, paper margins, bright skin) are
//!    located with a distance transform; bands near them are never strokes.
//! 2. For [`ORIENTATIONS`] directions the page is rotated so the candidate
//!    direction is horizontal, and the white mask is opened with a thin
//!    rectangle. What survives and is not near a large region is a band: a
//!    long, thin, straight run of white. Rotating once per direction keeps
//!    every later measurement axis-aligned.
//! 3. A band is kept as a strip only when
//!    - it is not wider than a brush stroke and its long edges are straight,
//!    - both long edges meet ink that is textured (broken, neither a
//!      continuous outline nor empty paper), and
//!    - the page around it carries ink, which separates strokes from the
//!      white gaps between sparse speed lines.
//! 4. A stroke lying between two directions yields strips in both, and a
//!    stroke seen in one direction yields only one. Strips that cover the same
//!    area are merged into a bar; strips that merely touch or cross are not.
//! 5. Three or more bars that lie side by side, roughly parallel and a few
//!    widths apart form a stack and are reported. Isolated bars, pairs, rows
//!    packed closer than three widths (lettering) and bars far longer than a
//!    stroke never are.
//!
//! A white stroke that is not part of a stack, one drawn entirely over paper,
//! and one over tone that the source has resampled into flat grey are not
//! detected. The mask is meant to be merged into the censor mask and refined
//! by hand where it misses.

use std::collections::HashMap;

use image::{GrayImage, RgbImage};
use imageproc::{
    distance_transform::{Norm, euclidean_squared_distance_transform},
    morphology::dilate,
};
use rayon::prelude::*;

/// Longest page side for which the pixel lengths below are stated.
const REFERENCE_EXTENT: f64 = 1808.0;
/// Pages smaller than this on a side (after downscaling) carry no stack.
const MIN_EXTENT: usize = 48;

/// Every channel of a paper pixel is at least this bright.
const WHITE_LEVEL: u8 = 240;
/// Luma below which a pixel counts as ink.
const INK_LUMA: u32 = 190;

/// A white region whose inscribed circle reaches this radius is paper, not a
/// stroke (strokes are at most [`MAX_THICKNESS`] wide).
const LARGE_RADIUS: f64 = 18.0;
/// Bands closer than this to a large region are its rim, not strokes.
const LARGE_MARGIN: f64 = 6.0;

/// Directions tried, evenly spread over 180 degrees. The step bounds how far a
/// stroke can sit from the nearest direction, and the rectangle below fits a
/// thin stroke only up to that tilt.
const ORIENTATIONS: usize = 24;
const RECT_LENGTH: f64 = 48.0;
const RECT_THICKNESS: f64 = 10.0;
/// Shortest band, measured along its direction.
const MIN_LENGTH: f64 = 56.0;
/// Widest band: the median thickness along its length.
const MAX_THICKNESS: f64 = 32.0;

/// How far from a band's edge to look for the first ink pixel.
const FLANK_REACH: f64 = 6.0;
/// Rows sampled outward from that first ink pixel.
const FLANK_DEPTH: f64 = 3.0;
/// Longest unbroken stretch of ink along an edge, as a share of the band.
/// Outlines and neighbouring panel borders run alongside a band for most of
/// its length; tone and cut line work do not.
const FLANK_MAX_RUN: f64 = 0.4;
/// Share of sampled flank pixels that are ink, on each side.
const FLANK_INK: std::ops::RangeInclusive<f64> = 0.15..=0.75;
/// Ink/paper changes per column along an edge, at least.
const FLANK_MIN_TRANSITIONS: f64 = 0.08;
/// The page band from this distance to [`CONTEXT_TO`] beyond each edge must
/// carry ink on average. Gaps between sparse speed lines are bordered by thin
/// lines and surrounded by paper.
const CONTEXT_FROM: f64 = 4.0;
const CONTEXT_TO: f64 = 16.0;
const MIN_CONTEXT_INK: f64 = 0.15;
/// Largest 80th-percentile deviation of an edge from its best straight line.
/// Hand-drawn gaps wander; a brushed stroke does not.
const MAX_EDGE_DEVIATION: f64 = 3.0;

/// Share of the smaller strip that another must cover to be the same bar.
const MIN_OVERLAP: f64 = 0.5;
/// Merged strips are bars when their equal-moment rectangle is within these
/// bounds. The longest bar is a share of the page's longest side.
const MIN_BAR_WIDTH: f64 = 8.0;
const MAX_BAR_WIDTH: f64 = 30.0;
const MIN_ELONGATION: f64 = 3.0;
const MAX_BAR_LENGTH: f64 = 0.30;

const MIN_STACK_BARS: usize = 3;
/// Centre separation across the bars' axes, in bar widths. The lower bound
/// rejects rows of lettering; the upper bound keeps stacks local.
const MIN_SEPARATION: f64 = 3.0;
const MAX_SEPARATION: f64 = 8.0;
/// Centre offset along the axes as a share of the mean bar length.
const MAX_OFFSET: f64 = 0.6;
/// Strokes of one object fan out. The tolerated angle is the one at which
/// two bars drift apart by [`DIVERGENCE`] widths over the shorter bar,
/// clamped to these degrees.
const MIN_ANGLE_TOLERANCE: f64 = 12.0;
const MAX_ANGLE_TOLERANCE: f64 = 30.0;
const DIVERGENCE: f64 = 3.5;
const WIDTH_RATIO: std::ops::RangeInclusive<f64> = 0.5..=2.0;

/// Growth of the reported mask, which covers the strokes' anti-aliased rim.
const GROW: f64 = 2.0;

const WHITE: u8 = 1;
const INK: u8 = 2;
const LARGE: u8 = 4;
/// Canvas pixels outside the page: paper that no stroke can be part of.
const OUTSIDE: u8 = WHITE | LARGE;

/// Marks the strokes of every white-stroke stack on `image`.
///
/// The mask has the page's size and holds 255 on stroke pixels and 0
/// elsewhere. It is empty when the page carries no stack.
pub(super) fn white_bar_mask(image: &RgbImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut mask = GrayImage::new(width, height);
    let longest = width.max(height) as f64;
    let factor = ((longest / REFERENCE_EXTENT).round() as usize).max(1);

    let mut planes = Planes::new(image, factor);
    if planes.width.min(planes.height) < MIN_EXTENT {
        return mask;
    }
    planes.mark_large_regions();
    let limits = Limits::new(planes.scale);

    // Collected in direction order, so the merge below is deterministic.
    let per_direction: Vec<Vec<Vec<u32>>> = (0..ORIENTATIONS)
        .into_par_iter()
        .map(|index| {
            directional_strips(&planes, 180.0 * index as f64 / ORIENTATIONS as f64, &limits)
        })
        .collect();
    let strips: Vec<Vec<u32>> = per_direction.into_iter().flatten().collect();
    let bars = select_bars(
        merge_strips(&strips, planes.width * planes.height),
        &planes,
        &limits,
    );
    let stacks = find_stacks(&bars);
    if stacks.is_empty() {
        return mask;
    }

    let page_width = width as usize;
    let raw: &mut [u8] = &mut mask;
    for bar in stacks.iter().flatten().map(|&index| &bars[index]) {
        for &pixel in &bar.pixels {
            let (x, y) = (pixel as usize % planes.width, pixel as usize / planes.width);
            for row in (y * factor)..((y + 1) * factor).min(height as usize) {
                let start = row * page_width;
                let end = ((x + 1) * factor).min(page_width);
                raw[start + x * factor..start + end].fill(255);
            }
        }
    }
    let grow = (GROW * longest / REFERENCE_EXTENT)
        .round()
        .clamp(1.0, 255.0) as u8;
    dilate(&mask, Norm::LInf, grow)
}

/// The page at the analysis resolution: one flag byte per pixel.
struct Planes {
    flags: Vec<u8>,
    width: usize,
    height: usize,
    /// Analysis extent over [`REFERENCE_EXTENT`].
    scale: f64,
}

impl Planes {
    /// Pools every `factor` x `factor` block: paper only when all of it is
    /// paper, ink when any of it is ink.
    fn new(image: &RgbImage, factor: usize) -> Self {
        let (page_width, page_height) = (image.width() as usize, image.height() as usize);
        let (width, height) = (page_width.div_ceil(factor), page_height.div_ceil(factor));
        let raw = image.as_raw();
        let mut flags = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut white = true;
                let mut ink = false;
                for row in (y * factor)..((y + 1) * factor).min(page_height) {
                    for column in (x * factor)..((x + 1) * factor).min(page_width) {
                        let at = (row * page_width + column) * 3;
                        let [red, green, blue] = [raw[at], raw[at + 1], raw[at + 2]];
                        white &= red.min(green).min(blue) >= WHITE_LEVEL;
                        let luma =
                            299 * u32::from(red) + 587 * u32::from(green) + 114 * u32::from(blue);
                        ink |= luma / 1000 < INK_LUMA;
                    }
                }
                flags[y * width + x] = (u8::from(white) * WHITE) | (u8::from(ink) * INK);
            }
        }
        let scale = width.max(height) as f64 / REFERENCE_EXTENT;
        Self {
            flags,
            width,
            height,
            scale,
        }
    }

    /// Sets [`LARGE`] on every pixel within the rim of a large white region.
    fn mark_large_regions(&mut self) {
        let (width, height) = (self.width as u32, self.height as u32);
        let non_white: Vec<u8> = self
            .flags
            .iter()
            .map(|&flags| if flags & WHITE == 0 { 255 } else { 0 })
            .collect();
        let non_white = GrayImage::from_raw(width, height, non_white)
            .expect("the buffer holds one byte per pixel");
        let clearance = euclidean_squared_distance_transform(&non_white);

        let radius = LARGE_RADIUS * self.scale;
        let core: Vec<u8> = clearance
            .as_raw()
            .iter()
            .map(|&squared| if squared >= radius * radius { 255 } else { 0 })
            .collect();
        if core.iter().all(|&cell| cell == 0) {
            return;
        }
        let core =
            GrayImage::from_raw(width, height, core).expect("the buffer holds one byte per pixel");
        let distance = euclidean_squared_distance_transform(&core);
        let reach = (LARGE_RADIUS + LARGE_MARGIN) * self.scale;
        for (flags, &squared) in self.flags.iter_mut().zip(distance.as_raw()) {
            if squared <= reach * reach {
                *flags |= LARGE;
            }
        }
    }
}

/// Pixel limits derived from the analysis scale.
struct Limits {
    rect_length: usize,
    rect_thickness: usize,
    min_length: usize,
    max_thickness: usize,
    reach: usize,
    depth: usize,
    context_from: usize,
    context_to: usize,
    deviation: f64,
}

impl Limits {
    fn new(scale: f64) -> Self {
        let scaled = |length: f64, minimum: usize| ((length * scale).round() as usize).max(minimum);
        let context_from = scaled(CONTEXT_FROM, 2);
        Self {
            rect_length: scaled(RECT_LENGTH, 3),
            rect_thickness: scaled(RECT_THICKNESS, 3),
            min_length: scaled(MIN_LENGTH, 4),
            max_thickness: scaled(MAX_THICKNESS, 4),
            reach: scaled(FLANK_REACH, 3),
            depth: scaled(FLANK_DEPTH, 2),
            context_from,
            context_to: scaled(CONTEXT_TO, 0).max(context_from + 4),
            deviation: MAX_EDGE_DEVIATION * scale,
        }
    }
}

/// The page rotated by a fixed angle onto a canvas that holds all of it.
struct Frame {
    cos: f64,
    sin: f64,
    width: usize,
    height: usize,
    source_centre: (f64, f64),
    centre: (f64, f64),
}

impl Frame {
    fn new(source_width: usize, source_height: usize, degrees: f64) -> Self {
        let (sin, cos) = degrees.to_radians().sin_cos();
        let (source_width, source_height) = (source_width as f64, source_height as f64);
        let width = (source_width * cos.abs() + source_height * sin.abs()).ceil() as usize + 2;
        let height = (source_width * sin.abs() + source_height * cos.abs()).ceil() as usize + 2;
        Self {
            cos,
            sin,
            width,
            height,
            source_centre: ((source_width - 1.0) / 2.0, (source_height - 1.0) / 2.0),
            centre: ((width as f64 - 1.0) / 2.0, (height as f64 - 1.0) / 2.0),
        }
    }

    fn to_source(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.centre.0, y - self.centre.1);
        (
            dx * self.cos - dy * self.sin + self.source_centre.0,
            dx * self.sin + dy * self.cos + self.source_centre.1,
        )
    }

    fn to_canvas(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.source_centre.0, y - self.source_centre.1);
        (
            dx * self.cos + dy * self.sin + self.centre.0,
            -dx * self.sin + dy * self.cos + self.centre.1,
        )
    }

    /// Nearest-neighbour rotation of the flags; pixels off the page are
    /// [`OUTSIDE`].
    fn rotate(&self, planes: &Planes) -> Vec<u8> {
        let mut canvas = vec![OUTSIDE; self.width * self.height];
        for (y, row) in canvas.chunks_exact_mut(self.width).enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let (source_x, source_y) = self.to_source(x as f64, y as f64);
                let (source_x, source_y) = ((source_x + 0.5).floor(), (source_y + 0.5).floor());
                if source_x >= 0.0
                    && source_y >= 0.0
                    && (source_x as usize) < planes.width
                    && (source_y as usize) < planes.height
                {
                    *cell = planes.flags[source_y as usize * planes.width + source_x as usize];
                }
            }
        }
        canvas
    }

    /// Carries numbered regions on the canvas (`labels`, 1 to `count`, 0 for
    /// none) back to the page grid as one pixel list per region.
    fn project(&self, labels: &[u16], count: usize, width: usize, height: usize) -> Vec<Vec<u32>> {
        let mut regions: Vec<Vec<u32>> = vec![Vec::new(); count];
        for y in 0..height {
            for x in 0..width {
                let (canvas_x, canvas_y) = self.to_canvas(x as f64, y as f64);
                let (canvas_x, canvas_y) = ((canvas_x + 0.5).floor(), (canvas_y + 0.5).floor());
                if canvas_x >= 0.0
                    && canvas_y >= 0.0
                    && (canvas_x as usize) < self.width
                    && (canvas_y as usize) < self.height
                {
                    let label = labels[canvas_y as usize * self.width + canvas_x as usize];
                    if label != 0 {
                        regions[usize::from(label) - 1].push((y * width + x) as u32);
                    }
                }
            }
        }
        regions
    }
}

/// The strips whose direction is `degrees`, each as the page pixels it
/// covers.
fn directional_strips(planes: &Planes, degrees: f64, limits: &Limits) -> Vec<Vec<u32>> {
    let frame = Frame::new(planes.width, planes.height, degrees);
    let canvas = frame.rotate(planes);

    let mut candidates: Vec<u8> = canvas.iter().map(|&flags| flags & WHITE).collect();
    open_rectangle(
        &mut candidates,
        frame.width,
        limits.rect_length,
        limits.rect_thickness,
    );
    for (cell, &flags) in candidates.iter_mut().zip(&canvas) {
        if flags & LARGE != 0 {
            *cell = 0;
        }
    }

    let mut labels: Vec<u16> = Vec::new();
    let mut count = 0usize;
    each_component(&mut candidates, frame.width, frame.height, |pixels| {
        if count < usize::from(u16::MAX)
            && limits.accepts(&canvas, frame.width, frame.height, pixels)
        {
            if labels.is_empty() {
                labels = vec![0u16; frame.width * frame.height];
            }
            count += 1;
            for &pixel in pixels {
                labels[pixel as usize] = count as u16;
            }
        }
    });
    if count == 0 {
        return Vec::new();
    }
    frame.project(&labels, count, planes.width, planes.height)
}

/// Morphological opening of the 0/1 `cells` by a `length` x `thickness`
/// rectangle. Cells beyond the canvas edge count as set when eroding, so a
/// band touching the edge is not shortened by it.
fn open_rectangle(cells: &mut [u8], width: usize, length: usize, thickness: usize) {
    const FAR: usize = usize::MAX / 2;
    // Erosion: a cell survives when the run of set cells from it onward is
    // long enough.
    for row in cells.chunks_exact_mut(width) {
        let mut run = FAR;
        for cell in row.iter_mut().rev() {
            run = if *cell != 0 { run.saturating_add(1) } else { 0 };
            *cell = u8::from(run >= length);
        }
    }
    let mut runs = vec![FAR; width];
    for row in cells.chunks_exact_mut(width).rev() {
        for (cell, run) in row.iter_mut().zip(runs.iter_mut()) {
            *run = if *cell != 0 { run.saturating_add(1) } else { 0 };
            *cell = u8::from(*run >= thickness);
        }
    }
    // Dilation by the same rectangle: a cell is set when a survivor lies
    // within the preceding window.
    for row in cells.chunks_exact_mut(width) {
        let mut since = FAR;
        for cell in row.iter_mut() {
            since = if *cell != 0 {
                0
            } else {
                since.saturating_add(1)
            };
            *cell = u8::from(since < length);
        }
    }
    let mut gaps = vec![FAR; width];
    for row in cells.chunks_exact_mut(width) {
        for (cell, gap) in row.iter_mut().zip(gaps.iter_mut()) {
            *gap = if *cell != 0 { 0 } else { gap.saturating_add(1) };
            *cell = u8::from(*gap < thickness);
        }
    }
}

/// Calls `visit` with the pixel indices of every eight-connected component of
/// cells equal to 1, and leaves those cells at 2.
fn each_component(cells: &mut [u8], width: usize, height: usize, mut visit: impl FnMut(&[u32])) {
    let mut stack: Vec<u32> = Vec::new();
    let mut pixels: Vec<u32> = Vec::new();
    for start in 0..cells.len() {
        if cells[start] != 1 {
            continue;
        }
        cells[start] = 2;
        stack.push(start as u32);
        pixels.clear();
        while let Some(index) = stack.pop() {
            pixels.push(index);
            let (x, y) = (index as usize % width, index as usize / width);
            for neighbour_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for neighbour_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbour = neighbour_y * width + neighbour_x;
                    if cells[neighbour] == 1 {
                        cells[neighbour] = 2;
                        stack.push(neighbour as u32);
                    }
                }
            }
        }
        visit(&pixels);
    }
}

/// A candidate band measured column by column on its frame's canvas.
struct Strip {
    /// Canvas column of every column the band occupies, with the topmost and
    /// bottommost band row in it.
    columns: Vec<usize>,
    top: Vec<usize>,
    bottom: Vec<usize>,
    /// Columns from the band's first to its last, occupied or not.
    span: usize,
    /// Rows from the band's first to its last.
    rows: usize,
}

impl Strip {
    fn measure(pixels: &[u32], width: usize) -> Self {
        let (mut left, mut right) = (usize::MAX, 0);
        let (mut first, mut last) = (usize::MAX, 0);
        for &pixel in pixels {
            let (x, y) = (pixel as usize % width, pixel as usize / width);
            left = left.min(x);
            right = right.max(x);
            first = first.min(y);
            last = last.max(y);
        }
        let span = right - left + 1;
        let mut top = vec![usize::MAX; span];
        let mut bottom = vec![0usize; span];
        for &pixel in pixels {
            let (x, y) = (pixel as usize % width - left, pixel as usize / width);
            top[x] = top[x].min(y);
            bottom[x] = bottom[x].max(y);
        }
        let mut strip = Self {
            columns: Vec::new(),
            top: Vec::new(),
            bottom: Vec::new(),
            span,
            rows: last - first + 1,
        };
        for offset in 0..span {
            if top[offset] != usize::MAX {
                strip.columns.push(left + offset);
                strip.top.push(top[offset]);
                strip.bottom.push(bottom[offset]);
            }
        }
        strip
    }

    fn median_thickness(&self) -> usize {
        let mut thickness: Vec<usize> = self
            .top
            .iter()
            .zip(&self.bottom)
            .map(|(top, bottom)| bottom - top + 1)
            .collect();
        thickness.sort_unstable();
        thickness[thickness.len() / 2]
    }
}

/// What lies just beyond one long edge of a band.
struct Flank {
    ink: f64,
    longest_run: f64,
    transitions: f64,
    context: f64,
}

impl Limits {
    /// Whether the band made of `pixels` is a stroke candidate.
    fn accepts(&self, canvas: &[u8], width: usize, height: usize, pixels: &[u32]) -> bool {
        if pixels.len() < self.min_length * self.rect_thickness {
            return false;
        }
        let strip = Strip::measure(pixels, width);
        if strip.span < self.min_length
            || strip.rows < self.rect_thickness
            || strip.median_thickness() > self.max_thickness
        {
            return false;
        }
        let upper = self.flank(canvas, width, height, &strip, true);
        let lower = self.flank(canvas, width, height, &strip, false);
        textured(&upper)
            && textured(&lower)
            && 0.5 * (upper.context + lower.context) >= MIN_CONTEXT_INK
            && edge_deviation(&strip.columns, &strip.top)
                .max(edge_deviation(&strip.columns, &strip.bottom))
                <= self.deviation
    }

    /// Samples the page beyond the upper or lower edge of `strip`, starting at
    /// the first pixel that is not paper within reach of the band.
    fn flank(
        &self,
        canvas: &[u8],
        width: usize,
        height: usize,
        strip: &Strip,
        upper: bool,
    ) -> Flank {
        let edges = if upper { &strip.top } else { &strip.bottom };
        let step: isize = if upper { -1 } else { 1 };
        let last_row = height as isize - 1;
        let count = strip.columns.len();
        let (mut ink, mut context) = (0usize, 0usize);
        let (mut previous, mut run, mut longest, mut transitions) = (false, 0usize, 0usize, 0usize);
        for (index, (&x, &edge)) in strip.columns.iter().zip(edges).enumerate() {
            let mut boundary = edge as isize + step * (self.reach as isize + 1);
            for distance in 1..=(self.reach as isize + 1) {
                let row = edge as isize + step * distance;
                if row < 0 || row > last_row {
                    boundary = row.clamp(0, last_row);
                    break;
                }
                if canvas[row as usize * width + x] & WHITE == 0 {
                    boundary = row;
                    break;
                }
            }
            let is_ink = |offset: usize| {
                let row = (boundary + step * offset as isize).clamp(0, last_row);
                canvas[row as usize * width + x] & INK != 0
            };
            let mut any = false;
            for offset in 0..self.depth {
                let sampled = is_ink(offset);
                ink += usize::from(sampled);
                any |= sampled;
            }
            for offset in self.context_from..self.context_to {
                context += usize::from(is_ink(offset));
            }
            if index > 0 && any != previous {
                transitions += 1;
            }
            run = if any { run + 1 } else { 0 };
            longest = longest.max(run);
            previous = any;
        }
        Flank {
            ink: ink as f64 / (self.depth * count) as f64,
            longest_run: longest as f64 / count as f64,
            transitions: transitions as f64 / count as f64,
            context: context as f64 / ((self.context_to - self.context_from) * count) as f64,
        }
    }
}

fn textured(flank: &Flank) -> bool {
    flank.longest_run <= FLANK_MAX_RUN
        && FLANK_INK.contains(&flank.ink)
        && flank.transitions >= FLANK_MIN_TRANSITIONS
}

/// 80th percentile of the distance of `edge` from its least-squares line.
fn edge_deviation(columns: &[usize], edge: &[usize]) -> f64 {
    let count = columns.len() as f64;
    let mean_x = columns.iter().map(|&x| x as f64).sum::<f64>() / count;
    let mean_y = edge.iter().map(|&y| y as f64).sum::<f64>() / count;
    let (mut variance, mut covariance) = (0.0, 0.0);
    for (&x, &y) in columns.iter().zip(edge) {
        let dx = x as f64 - mean_x;
        variance += dx * dx;
        covariance += dx * (y as f64 - mean_y);
    }
    let slope = if variance > 0.0 {
        covariance / variance
    } else {
        0.0
    };
    let mut residuals: Vec<f64> = columns
        .iter()
        .zip(edge)
        .map(|(&x, &y)| (y as f64 - mean_y - slope * (x as f64 - mean_x)).abs())
        .collect();
    residuals.sort_by(f64::total_cmp);
    residuals[((residuals.len() - 1) as f64 * 0.8).round() as usize]
}

/// Groups strips that cover the same page area. Each group is the sorted,
/// deduplicated union of its strips' pixels.
fn merge_strips(strips: &[Vec<u32>], page_len: usize) -> Vec<Vec<u32>> {
    const NONE: u32 = u32::MAX;

    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }

    // The first strip to cover a pixel owns it; later strips count their
    // pixels per owner.
    let mut owner = vec![NONE; page_len];
    let mut parent: Vec<usize> = (0..strips.len()).collect();
    for (index, pixels) in strips.iter().enumerate() {
        let mut shared: HashMap<u32, usize> = HashMap::new();
        for &pixel in pixels {
            let slot = &mut owner[pixel as usize];
            if *slot == NONE {
                *slot = index as u32;
            } else {
                *shared.entry(*slot).or_default() += 1;
            }
        }
        for (other, count) in shared {
            let smaller = pixels.len().min(strips[other as usize].len());
            if count as f64 >= MIN_OVERLAP * smaller as f64 {
                let (first, second) = (root(&mut parent, index), root(&mut parent, other as usize));
                parent[first] = second;
            }
        }
    }

    let mut group_of_leader = vec![usize::MAX; strips.len()];
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for (index, pixels) in strips.iter().enumerate() {
        let leader = root(&mut parent, index);
        if group_of_leader[leader] == usize::MAX {
            group_of_leader[leader] = groups.len();
            groups.push(Vec::new());
        }
        groups[group_of_leader[leader]].extend_from_slice(pixels);
    }
    for group in &mut groups {
        group.sort_unstable();
        group.dedup();
    }
    groups
}

/// Merged strips, measured as a rectangle of equal second moments.
struct Bar {
    centre: [f64; 2],
    /// Unit vector along the bar.
    axis: [f64; 2],
    length: f64,
    width: f64,
    pixels: Vec<u32>,
}

impl Bar {
    fn measure(pixels: Vec<u32>, width: usize) -> Self {
        let count = pixels.len() as f64;
        let at = |pixel: u32| {
            (
                (pixel as usize % width) as f64,
                (pixel as usize / width) as f64,
            )
        };
        let (mut sum_x, mut sum_y) = (0.0, 0.0);
        for &pixel in &pixels {
            let (x, y) = at(pixel);
            sum_x += x;
            sum_y += y;
        }
        let centre = [sum_x / count, sum_y / count];
        let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
        for &pixel in &pixels {
            let (x, y) = at(pixel);
            let (dx, dy) = (x - centre[0], y - centre[1]);
            xx += dx * dx;
            xy += dx * dy;
            yy += dy * dy;
        }
        let (xx, xy, yy) = (xx / count, xy / count, yy / count);
        let mean = 0.5 * (xx + yy);
        let spread = (0.25 * (xx - yy) * (xx - yy) + xy * xy).sqrt();
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        Self {
            centre,
            axis: [angle.cos(), angle.sin()],
            length: (12.0 * (mean + spread)).sqrt(),
            width: (12.0 * (mean - spread).max(1e-9)).sqrt(),
            pixels,
        }
    }
}

/// The merged strips that could be brush strokes.
fn select_bars(groups: Vec<Vec<u32>>, planes: &Planes, limits: &Limits) -> Vec<Bar> {
    let longest = planes.width.max(planes.height) as f64;
    let min_area = limits.min_length * limits.rect_thickness;
    let widths = MIN_BAR_WIDTH * planes.scale..=MAX_BAR_WIDTH * planes.scale;
    groups
        .into_iter()
        .filter(|pixels| pixels.len() >= min_area)
        .map(|pixels| Bar::measure(pixels, planes.width))
        .filter(|bar| {
            bar.length >= MIN_LENGTH * planes.scale
                && bar.length <= MAX_BAR_LENGTH * longest
                && widths.contains(&bar.width)
                && bar.length >= MIN_ELONGATION * bar.width
        })
        .collect()
}

/// Largest angle between two bars that can still be strokes of one stack.
fn angle_tolerance(first: &Bar, second: &Bar) -> f64 {
    let width = 0.5 * (first.width + second.width);
    let shorter = first.length.min(second.length);
    (DIVERGENCE * width / shorter)
        .atan()
        .to_degrees()
        .clamp(MIN_ANGLE_TOLERANCE, MAX_ANGLE_TOLERANCE)
}

/// How two bars sit relative to each other, along their mean axis.
struct Pair {
    axis: [f64; 2],
    normal: [f64; 2],
    along: f64,
    across: f64,
    degrees: f64,
}

impl Pair {
    fn new(first: &Bar, second: &Bar) -> Self {
        let dot = first.axis[0] * second.axis[0] + first.axis[1] * second.axis[1];
        let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
        let mut axis = [
            first.axis[0] + sign * second.axis[0],
            first.axis[1] + sign * second.axis[1],
        ];
        let norm = axis[0].hypot(axis[1]);
        if norm < 1e-9 {
            axis = first.axis;
        } else {
            axis = [axis[0] / norm, axis[1] / norm];
        }
        let normal = [-axis[1], axis[0]];
        let offset = [
            second.centre[0] - first.centre[0],
            second.centre[1] - first.centre[1],
        ];
        Self {
            axis,
            normal,
            along: (offset[0] * axis[0] + offset[1] * axis[1]).abs(),
            across: (offset[0] * normal[0] + offset[1] * normal[1]).abs(),
            degrees: dot.abs().min(1.0).acos().to_degrees(),
        }
    }
}

/// Whether `first` and `second` are neighbours in a stack: roughly parallel
/// and alike in width, overlapping along their axes, a few widths apart, and
/// with no other parallel bar between them.
fn neighbours(bars: &[Bar], first: usize, second: usize) -> bool {
    let (a, b) = (&bars[first], &bars[second]);
    let pair = Pair::new(a, b);
    let width = 0.5 * (a.width + b.width);
    let length = 0.5 * (a.length + b.length);
    if pair.degrees > angle_tolerance(a, b)
        || !WIDTH_RATIO.contains(&(a.width / b.width))
        || pair.along > MAX_OFFSET * length
        || pair.across < MIN_SEPARATION * width
        || pair.across > MAX_SEPARATION * width
    {
        return false;
    }
    let project = |point: [f64; 2]| point[0] * pair.normal[0] + point[1] * pair.normal[1];
    let (a_at, b_at) = (project(a.centre), project(b.centre));
    let (near, far) = (a_at.min(b_at), a_at.max(b_at));
    let middle = [
        0.5 * (a.centre[0] + b.centre[0]),
        0.5 * (a.centre[1] + b.centre[1]),
    ];
    !bars.iter().enumerate().any(|(index, other)| {
        if index == first || index == second {
            return false;
        }
        let at = project(other.centre);
        if at <= near + 0.5 * width || at >= far - 0.5 * width {
            return false;
        }
        let along = ((other.centre[0] - middle[0]) * pair.axis[0]
            + (other.centre[1] - middle[1]) * pair.axis[1])
            .abs();
        along <= MAX_OFFSET * length && Pair::new(a, other).degrees <= angle_tolerance(a, other)
    })
}

/// Connected groups of neighbouring bars with at least [`MIN_STACK_BARS`]
/// members, as indices into `bars`.
fn find_stacks(bars: &[Bar]) -> Vec<Vec<usize>> {
    let mut adjacent = vec![Vec::new(); bars.len()];
    for first in 0..bars.len() {
        for second in first + 1..bars.len() {
            if neighbours(bars, first, second) {
                adjacent[first].push(second);
                adjacent[second].push(first);
            }
        }
    }
    let mut seen = vec![false; bars.len()];
    let mut stacks = Vec::new();
    for start in 0..bars.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut members = Vec::new();
        let mut pending = vec![start];
        while let Some(bar) = pending.pop() {
            members.push(bar);
            for &next in &adjacent[bar] {
                if !seen[next] {
                    seen[next] = true;
                    pending.push(next);
                }
            }
        }
        if members.len() >= MIN_STACK_BARS {
            stacks.push(members);
        }
    }
    stacks
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 900;

    /// A painted white stroke: its centre, size and tilt.
    struct Stroke {
        x: f64,
        y: f64,
        length: f64,
        thickness: f64,
        degrees: f64,
    }

    impl Stroke {
        /// Position of a pixel relative to the stroke: along it, across it.
        fn relative(&self, x: u32, y: u32) -> (f64, f64) {
            let (sin, cos) = self.degrees.to_radians().sin_cos();
            let (dx, dy) = (f64::from(x) - self.x, f64::from(y) - self.y);
            (dx * cos + dy * sin, -dx * sin + dy * cos)
        }
    }

    fn paper() -> RgbImage {
        RgbImage::from_pixel(WIDTH, HEIGHT, Rgb([255; 3]))
    }

    fn solid_ink(image: &mut RgbImage, left: u32, top: u32, right: u32, bottom: u32) {
        for y in top..bottom {
            for x in left..right {
                image.put_pixel(x, y, Rgb([0; 3]));
            }
        }
    }

    /// Halftone dots: 2 x 2 ink on a 3 px grid.
    fn tone(image: &mut RgbImage, left: u32, top: u32, right: u32, bottom: u32) {
        for y in top..bottom {
            for x in left..right {
                if x % 3 < 2 && y % 3 < 2 {
                    image.put_pixel(x, y, Rgb([0; 3]));
                }
            }
        }
    }

    fn paint(image: &mut RgbImage, stroke: &Stroke) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let (along, across) = stroke.relative(x, y);
                if along.abs() <= stroke.length / 2.0 && across.abs() <= stroke.thickness / 2.0 {
                    image.put_pixel(x, y, Rgb([255; 3]));
                }
            }
        }
    }

    /// Halftone with `count` tilted white strokes 60 px apart across it.
    fn ladder(count: usize) -> (RgbImage, Vec<Stroke>) {
        let mut image = paper();
        tone(&mut image, 120, 200, 520, 700);
        let tilts = [0.0, 5.0, -4.0];
        let strokes: Vec<Stroke> = (0..count)
            .map(|index| Stroke {
                x: 320.0,
                y: 450.0 + 60.0 * (index as f64 - (count as f64 - 1.0) / 2.0),
                length: 160.0,
                thickness: 10.0,
                degrees: tilts[index % tilts.len()],
            })
            .collect();
        for stroke in &strokes {
            paint(&mut image, stroke);
        }
        (image, strokes)
    }

    fn marked(mask: &GrayImage) -> usize {
        mask.pixels().filter(|pixel| pixel.0[0] > 0).count()
    }

    #[test]
    fn a_stack_of_white_strokes_across_tone_is_marked_and_nothing_else() {
        let (image, strokes) = ladder(3);
        let mask = white_bar_mask(&image);
        assert_eq!(mask.dimensions(), image.dimensions());

        for stroke in &strokes {
            let (mut core, mut hit) = (0usize, 0usize);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let (along, across) = stroke.relative(x, y);
                    if along.abs() <= stroke.length / 2.0 - 20.0 && across.abs() <= 2.0 {
                        core += 1;
                        hit += usize::from(mask.get_pixel(x, y).0[0] > 0);
                    }
                }
            }
            assert!(hit * 10 >= core * 9, "{hit} of {core} stroke pixels marked");
        }

        for (x, y, pixel) in mask.enumerate_pixels() {
            if pixel.0[0] == 0 {
                continue;
            }
            let near = strokes.iter().any(|stroke| {
                let (along, across) = stroke.relative(x, y);
                let gap_along = (along.abs() - stroke.length / 2.0).max(0.0);
                let gap_across = (across.abs() - stroke.thickness / 2.0).max(0.0);
                gap_along.hypot(gap_across) <= 4.0
            });
            assert!(near, "stray mask pixel at ({x}, {y})");
        }
    }

    #[test]
    fn two_strokes_are_not_a_stack() {
        let (image, _) = ladder(2);
        assert_eq!(marked(&white_bar_mask(&image)), 0);
    }

    #[test]
    fn bubbles_gutters_and_lettering_rows_are_left_alone() {
        // Two toned panels with borders and a gutter, and an outlined bubble.
        let mut panels = paper();
        for (top, bottom) in [(20, 24), (400, 404), (440, 444), (880, 884)] {
            solid_ink(&mut panels, 20, top, 620, bottom);
        }
        tone(&mut panels, 30, 30, 610, 390);
        tone(&mut panels, 30, 452, 610, 870);
        for y in 90..310 {
            for x in 200..440 {
                let (dx, dy) = ((f64::from(x) - 320.0) / 90.0, (f64::from(y) - 200.0) / 60.0);
                let radius = dx * dx + dy * dy;
                if (1.0..1.07).contains(&radius) {
                    panels.put_pixel(x, y, Rgb([0; 3]));
                } else if radius < 1.0 {
                    panels.put_pixel(x, y, Rgb([255; 3]));
                }
            }
        }
        assert_eq!(marked(&white_bar_mask(&panels)), 0);

        // Eight rows of glyph-sized ink blocks on a 30 px pitch.
        let mut lettering = paper();
        for row in 0..8 {
            for column in (200..400).step_by(18) {
                let top = 200 + row * 30;
                solid_ink(&mut lettering, column, top, column + 12, top + 18);
            }
        }
        assert_eq!(marked(&white_bar_mask(&lettering)), 0);
    }

    #[test]
    fn blank_and_tiny_pages_have_empty_masks() {
        assert_eq!(marked(&white_bar_mask(&paper())), 0);
        let tiny = RgbImage::from_pixel(30, 20, Rgb([255; 3]));
        let mask = white_bar_mask(&tiny);
        assert_eq!(mask.dimensions(), (30, 20));
        assert_eq!(marked(&mask), 0);
    }
}
