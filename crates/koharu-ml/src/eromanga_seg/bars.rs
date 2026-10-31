//! Black censorship-bar refinement for Er0mangaSeg scores.
//!
//! This is Koharu's own post-processing, not part of the ported network. The
//! checkpoint scores a quarter-resolution logit grid that is interpolated back
//! onto the page, so confident pixels often cover only part of a bar while
//! thick black panel borders score as highly as real bars. Thresholding that
//! map and keeping every dark pixel therefore loses partly scored bars and
//! either needs a strict bar-shape test or masks borders.
//!
//! The refinement works on three score levels per pixel and only on dark ink
//! the model supports:
//!
//! 1. Hysteresis: dark pixels with weak support join the confident ink they
//!    touch, so a bar scored only around its centre is recovered to its ends.
//! 2. An opening with a small disk detaches strokes thinner than a bar (hair
//!    lines, outlines) before shapes are measured.
//! 3. Stacks: elongated components that lie side by side, roughly parallel and
//!    a few bar widths apart, form a stack. Three or more are censorship.
//!    Fragments of one collinear border and borders far apart never qualify.
//! 4. Completion: rings, ladders and caps drawn from touching bars fail the
//!    elongation test. Components of comparable stroke width within a few bar
//!    widths of an accepted stack join it.
//! 5. The accepted core regrows over the ink the opening removed and over its
//!    anti-aliased rim, so the mask covers whole bars.

/// Pixels the model scored above the weak threshold.
pub(super) const SUPPORTED: u8 = 1;
/// Pixels the model scored above the caller's threshold.
pub(super) const CONFIDENT: u8 = 2;

const SUPPORT_RATIO: f32 = 0.5;
const SUPPORT_FLOOR: f32 = 0.2;

const DARK_LUMA: u32 = 80;
const HALO_LUMA: u32 = 128;
const HALO_RADIUS: usize = 2;
const MIN_CONFIDENT_PIXELS: usize = 16;

const MIN_COMPONENT_AREA: u32 = 20;
const MIN_THICKNESS: f64 = 2.0;
const MIN_ELONGATION: f64 = 2.2;

// Opening radius grows with the page so the same stroke thickness ratio is
// removed at every resolution; bars stay far wider than the disk.
const OPENING_EXTENT: f64 = 900.0;
const MAX_OPENING_RADIUS: usize = 4;

const MIN_STACK_BARS: usize = 3;
// Combined bar length required of a stack on a page whose longest side is
// REFERENCE_EXTENT; scaled linearly (within 0.5..=2.0) for other pages.
const MIN_STACK_LENGTH: f64 = 128.0;
const REFERENCE_EXTENT: f64 = 1800.0;
// Centre separation across the bar axes, in bar widths. The lower bound
// rejects collinear fragments of a single border.
const MIN_SEPARATION: f64 = 0.6;
const MAX_SEPARATION: f64 = 5.0;
// Centre offset along the axes, as a fraction of the mean bar length.
const MAX_OFFSET: f64 = 0.75;
// Neighbouring bars may diverge as a fan. The tolerance is the angle at which
// two bars drift apart by DIVERGENCE widths over the shorter bar's length,
// kept between these degrees.
const MIN_ANGLE_TOLERANCE: f64 = 12.0;
const MAX_ANGLE_TOLERANCE: f64 = 30.0;
const DIVERGENCE: f64 = 3.5;

// Completion limits, relative to the accepted stack.
const ATTACH_GAP: f64 = 4.0;
const ATTACH_LENGTH: f64 = 2.5;
const ATTACH_AREA: f64 = 4.0;
const WIDTH_BAND: (f64, f64) = (0.5, 2.0);
// Junctions widen a stroke by at most this factor (a perpendicular crossing
// reaches about 1.41); anything wider is a solid region, not a bar network.
const JUNCTION_BULGE: f64 = 1.6;

/// Weak threshold derived from the caller's confident threshold.
pub(super) fn support_threshold(threshold: f32) -> f32 {
    (threshold * SUPPORT_RATIO)
        .max(SUPPORT_FLOOR)
        .min(threshold)
}

/// Mask pixels (`0` or `255`) for the window that contains every accepted bar.
pub(super) struct Placement {
    pub(super) left: usize,
    pub(super) top: usize,
    pub(super) width: usize,
    pub(super) pixels: Vec<u8>,
}

impl Placement {
    /// Raises `page` (row-major, `page_width` columns) to this mask's pixels.
    pub(super) fn merge_into(&self, page: &mut [u8], page_width: usize) {
        for (row, source) in self.pixels.chunks_exact(self.width).enumerate() {
            let start = (self.top + row) * page_width + self.left;
            for (target, &value) in page[start..start + self.width].iter_mut().zip(source) {
                *target = (*target).max(value);
            }
        }
    }
}

/// Refines per-pixel scores into a censorship-bar mask.
///
/// `rgb` is the original page (`width * height * 3` bytes) and `levels` holds
/// [`SUPPORTED`] / [`CONFIDENT`] scores for every pixel (0 elsewhere).
/// Returns `None` when the page carries no bar stack.
pub(super) fn bar_mask(
    rgb: &[u8],
    width: usize,
    height: usize,
    levels: &[u8],
) -> Option<Placement> {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(levels.len(), width * height);
    let page = Page { rgb, width };
    let window = score_window(levels, width, height)?;
    let (w, h) = (window.width, window.height);
    let area = w * h;

    // Dark pixels the model supports, and the confident seeds among them.
    let mut supported = vec![0_u8; area];
    let mut confident = Vec::new();
    for y in 0..h {
        let row = &levels[(window.top + y) * width + window.left..][..w];
        for (x, &level) in row.iter().enumerate() {
            if level < SUPPORTED || page.luma(window.left + x, window.top + y) > DARK_LUMA {
                continue;
            }
            supported[y * w + x] = 1;
            if level >= CONFIDENT {
                confident.push(y * w + x);
            }
        }
    }
    if confident.len() < MIN_CONFIDENT_PIXELS {
        return None;
    }

    // Hysteresis: keep supported ink components that hold enough seeds.
    let (ink_labels, ink_components) = label_components(&supported, w, h);
    let mut seeds = vec![0_usize; ink_components.len()];
    for &index in &confident {
        seeds[ink_labels[index] as usize] += 1;
    }
    if !seeds.iter().any(|&count| count >= MIN_CONFIDENT_PIXELS) {
        return None;
    }
    let ink: Vec<u8> = ink_labels
        .iter()
        .map(|&label| u8::from(label != 0 && seeds[label as usize] >= MIN_CONFIDENT_PIXELS))
        .collect();
    drop(ink_labels);

    let extent = width.max(height);
    let radius = opening_radius(extent);
    let core = open(&ink, w, h, &disk(radius));
    let (labels, components) = label_components(&core, w, h);
    let shapes: Vec<Shape> = components
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, component)| component.area >= MIN_COMPONENT_AREA)
        .map(|(label, component)| Shape::new(label as u32, component))
        .collect();

    let bars: Vec<usize> = (0..shapes.len())
        .filter(|&index| shapes[index].is_bar())
        .collect();
    let minimum_length = MIN_STACK_LENGTH * (extent as f64 / REFERENCE_EXTENT).clamp(0.5, 2.0);
    let stacks = find_stacks(&shapes, &bars, minimum_length);
    if stacks.is_empty() {
        return None;
    }

    let mut accepted = vec![false; components.len()];
    for stack in &stacks {
        for &index in stack {
            accepted[shapes[index].label as usize] = true;
        }
    }
    for stack in &stacks {
        attach_neighbours(&labels, w, h, &shapes, stack, &mut accepted);
    }

    // Regrow the accepted core over the ink the opening removed, then over
    // the anti-aliased rim around it.
    let core_pixels: Vec<usize> = (0..area)
        .filter(|&index| accepted[labels[index] as usize])
        .collect();
    let mut painted = vec![0_u8; area];
    stamp(
        &core_pixels,
        w,
        h,
        &disk(radius + 1),
        &mut painted,
        |target| ink[target] != 0,
    );
    let restored: Vec<usize> = (0..area).filter(|&index| painted[index] != 0).collect();
    stamp(
        &restored,
        w,
        h,
        &disk(HALO_RADIUS),
        &mut painted,
        |target| page.luma(window.left + target % w, window.top + target / w) <= HALO_LUMA,
    );
    for value in &mut painted {
        *value *= 255;
    }
    Some(Placement {
        left: window.left,
        top: window.top,
        width: w,
        pixels: painted,
    })
}

struct Page<'a> {
    rgb: &'a [u8],
    width: usize,
}

impl Page<'_> {
    /// BT.601 luma with the integer arithmetic the pipeline has always used.
    fn luma(&self, x: usize, y: usize) -> u32 {
        let offset = (y * self.width + x) * 3;
        (u32::from(self.rgb[offset]) * 299
            + u32::from(self.rgb[offset + 1]) * 587
            + u32::from(self.rgb[offset + 2]) * 114)
            / 1000
    }
}

#[derive(Clone, Copy)]
struct Window {
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

/// Smallest window containing every scored pixel.
fn score_window(levels: &[u8], width: usize, height: usize) -> Option<Window> {
    debug_assert_eq!(levels.len(), width * height);
    let mut bounds: Option<(usize, usize, usize, usize)> = None;
    for (y, row) in levels.chunks_exact(width).enumerate() {
        let Some(first) = row.iter().position(|&level| level != 0) else {
            continue;
        };
        let last = row.iter().rposition(|&level| level != 0).unwrap_or(first);
        bounds = Some(match bounds {
            None => (first, y, last + 1, y + 1),
            Some((left, top, right, _)) => (left.min(first), top, right.max(last + 1), y + 1),
        });
    }
    bounds.map(|(left, top, right, bottom)| Window {
        left,
        top,
        width: right - left,
        height: bottom - top,
    })
}

fn opening_radius(extent: usize) -> usize {
    ((extent as f64 / OPENING_EXTENT).round() as usize).clamp(1, MAX_OPENING_RADIUS)
}

/// Offsets of a digital disk: `dx² + dy² <= r(r + 1)`.
fn disk(radius: usize) -> Vec<(isize, isize)> {
    let radius = radius as isize;
    let limit = radius * (radius + 1);
    let mut offsets = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= limit {
                offsets.push((dx, dy));
            }
        }
    }
    offsets
}

/// Opening by `offsets`; pixels outside the window are background.
fn open(mask: &[u8], width: usize, height: usize, offsets: &[(isize, isize)]) -> Vec<u8> {
    let mut centres = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] == 0 {
                continue;
            }
            let fits = offsets.iter().all(|&(dx, dy)| {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                nx >= 0
                    && ny >= 0
                    && (nx as usize) < width
                    && (ny as usize) < height
                    && mask[ny as usize * width + nx as usize] != 0
            });
            if fits {
                centres.push(y * width + x);
            }
        }
    }
    let mut opened = vec![0_u8; mask.len()];
    stamp(&centres, width, height, offsets, &mut opened, |_| true);
    opened
}

/// Sets every in-window pixel within `offsets` of a source pixel that `allow`
/// accepts and that is still unset.
fn stamp(
    sources: &[usize],
    width: usize,
    height: usize,
    offsets: &[(isize, isize)],
    painted: &mut [u8],
    mut allow: impl FnMut(usize) -> bool,
) {
    for &source in sources {
        let (x, y) = ((source % width) as isize, (source / width) as isize);
        for &(dx, dy) in offsets {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || (nx as usize) >= width || (ny as usize) >= height {
                continue;
            }
            let target = ny as usize * width + nx as usize;
            if painted[target] == 0 && allow(target) {
                painted[target] = 1;
            }
        }
    }
}

#[derive(Clone)]
struct Component {
    area: u32,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    sum_x: i64,
    sum_y: i64,
    sum_xx: i64,
    sum_yy: i64,
    sum_xy: i64,
}

impl Component {
    fn new() -> Self {
        Self {
            area: 0,
            min_x: usize::MAX,
            min_y: usize::MAX,
            max_x: 0,
            max_y: 0,
            sum_x: 0,
            sum_y: 0,
            sum_xx: 0,
            sum_yy: 0,
            sum_xy: 0,
        }
    }

    fn add(&mut self, x: usize, y: usize) {
        self.area += 1;
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        let (x, y) = (x as i64, y as i64);
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }
}

/// Eight-connected components; label 0 is background and component 0 a stub.
fn label_components(mask: &[u8], width: usize, height: usize) -> (Vec<u32>, Vec<Component>) {
    let mut labels = vec![0_u32; mask.len()];
    let mut components = vec![Component::new()];
    let mut pending = Vec::new();
    for start in 0..mask.len() {
        if mask[start] == 0 || labels[start] != 0 {
            continue;
        }
        let label = components.len() as u32;
        let mut component = Component::new();
        labels[start] = label;
        pending.push(start);
        while let Some(index) = pending.pop() {
            let (x, y) = (index % width, index / width);
            component.add(x, y);
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbour = ny * width + nx;
                    if mask[neighbour] != 0 && labels[neighbour] == 0 {
                        labels[neighbour] = label;
                        pending.push(neighbour);
                    }
                }
            }
        }
        components.push(component);
    }
    (labels, components)
}

/// A component measured as a rectangle of equal second moments.
struct Shape {
    label: u32,
    area: u32,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    center: [f64; 2],
    length: f64,
    thickness: f64,
    cos: f64,
    sin: f64,
}

impl Shape {
    fn new(label: u32, component: &Component) -> Self {
        let area = f64::from(component.area);
        let center = [component.sum_x as f64 / area, component.sum_y as f64 / area];
        let covariance_xx = component.sum_xx as f64 / area - center[0] * center[0];
        let covariance_yy = component.sum_yy as f64 / area - center[1] * center[1];
        let covariance_xy = component.sum_xy as f64 / area - center[0] * center[1];
        let trace = covariance_xx + covariance_yy;
        let spread = ((covariance_xx - covariance_yy).powi(2) + 4.0 * covariance_xy.powi(2)).sqrt();
        let major = ((trace + spread) / 2.0).max(0.0);
        let minor = ((trace - spread) / 2.0).max(0.0);
        let angle = 0.5 * (2.0 * covariance_xy).atan2(covariance_xx - covariance_yy);
        let (sin, cos) = angle.sin_cos();
        Self {
            label,
            area: component.area,
            min_x: component.min_x,
            min_y: component.min_y,
            max_x: component.max_x,
            max_y: component.max_y,
            center,
            length: (12.0 * major).sqrt(),
            thickness: (12.0 * minor).sqrt(),
            cos,
            sin,
        }
    }

    fn is_bar(&self) -> bool {
        self.thickness >= MIN_THICKNESS && self.length >= MIN_ELONGATION * self.thickness
    }
}

/// Whether two bars sit next to each other as members of one stack.
fn side_by_side(first: &Shape, second: &Shape) -> bool {
    let alignment = first.cos * second.cos + first.sin * second.sin;
    let sign = if alignment >= 0.0 { 1.0 } else { -1.0 };
    let (axis_x, axis_y) = (first.cos + sign * second.cos, first.sin + sign * second.sin);
    let norm = axis_x.hypot(axis_y);
    if norm < 1e-9 {
        return false;
    }
    let (axis_x, axis_y) = (axis_x / norm, axis_y / norm);
    let dx = second.center[0] - first.center[0];
    let dy = second.center[1] - first.center[1];
    let along = (dx * axis_x + dy * axis_y).abs();
    let across = (-dx * axis_y + dy * axis_x).abs();
    let thickness = first.thickness.max(second.thickness);
    let shorter = first.length.min(second.length);
    let tolerance = (DIVERGENCE * thickness / shorter).min(1.0).asin().clamp(
        MIN_ANGLE_TOLERANCE.to_radians(),
        MAX_ANGLE_TOLERANCE.to_radians(),
    );
    alignment.abs().min(1.0).acos() <= tolerance
        && across >= MIN_SEPARATION * thickness
        && across <= MAX_SEPARATION * thickness
        && along <= MAX_OFFSET * (first.length + second.length) / 2.0
}

/// Chains of bars linked by [`side_by_side`] that are long and numerous
/// enough to be censorship. Members are indices into `shapes`.
fn find_stacks(shapes: &[Shape], bars: &[usize], minimum_length: f64) -> Vec<Vec<usize>> {
    let mut visited = vec![false; bars.len()];
    let mut stacks = Vec::new();
    for start in 0..bars.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut members = Vec::new();
        let mut pending = vec![start];
        while let Some(current) = pending.pop() {
            members.push(bars[current]);
            for other in 0..bars.len() {
                if !visited[other] && side_by_side(&shapes[bars[current]], &shapes[bars[other]]) {
                    visited[other] = true;
                    pending.push(other);
                }
            }
        }
        let length: f64 = members.iter().map(|&index| shapes[index].length).sum();
        if members.len() >= MIN_STACK_BARS && length >= minimum_length {
            stacks.push(members);
        }
    }
    stacks
}

/// Accepts components that continue a stack: bars of comparable width close to
/// it, and bar networks (rings, ladders, caps) whose strokes match its width.
fn attach_neighbours(
    labels: &[u32],
    width: usize,
    height: usize,
    shapes: &[Shape],
    stack: &[usize],
    accepted: &mut [bool],
) {
    let members: Vec<&Shape> = stack.iter().map(|&index| &shapes[index]).collect();
    let stack_width = median(members.iter().map(|shape| shape.thickness).collect());
    let stack_length = median(members.iter().map(|shape| shape.length).collect());
    let stack_area: f64 = members.iter().map(|shape| f64::from(shape.area)).sum();
    let gap = ATTACH_GAP * stack_width;
    let reach = gap.ceil() as usize;
    let narrowest = WIDTH_BAND.0 * stack_width / 2.0;
    let widest = WIDTH_BAND.1 * stack_width / 2.0;

    let mut member = vec![false; accepted.len()];
    let mut extent = (usize::MAX, usize::MAX, 0_usize, 0_usize);
    for shape in &members {
        member[shape.label as usize] = true;
        extent = (
            extent.0.min(shape.min_x),
            extent.1.min(shape.min_y),
            extent.2.max(shape.max_x),
            extent.3.max(shape.max_y),
        );
    }

    for candidate in shapes {
        if accepted[candidate.label as usize] || box_gap(candidate, extent) > reach {
            continue;
        }
        let half_width = if candidate.is_bar() {
            if candidate.length > ATTACH_LENGTH * stack_length {
                continue;
            }
            candidate.thickness / 2.0
        } else {
            if f64::from(candidate.area) > ATTACH_AREA * stack_area {
                continue;
            }
            let (ridge, bulge) = stroke_profile(labels, width, candidate);
            if bulge > JUNCTION_BULGE * ridge {
                continue;
            }
            ridge
        };
        if half_width < narrowest || half_width > widest {
            continue;
        }
        let close = nearest_squared(labels, width, height, candidate, &member, reach)
            .is_some_and(|distance| f64::from(distance) <= gap * gap);
        if close {
            accepted[candidate.label as usize] = true;
        }
    }
}

/// Chebyshev gap between a shape's bounding box and `extent` (left, top,
/// right, bottom, inclusive).
fn box_gap(shape: &Shape, extent: (usize, usize, usize, usize)) -> usize {
    let horizontal = shape
        .min_x
        .saturating_sub(extent.2)
        .max(extent.0.saturating_sub(shape.max_x));
    let vertical = shape
        .min_y
        .saturating_sub(extent.3)
        .max(extent.1.saturating_sub(shape.max_y));
    horizontal.max(vertical)
}

/// Smallest squared distance from `candidate` to any pixel of a member
/// component, or `None` when no member pixel lies within `reach`.
fn nearest_squared(
    labels: &[u32],
    width: usize,
    height: usize,
    candidate: &Shape,
    member: &[bool],
    reach: usize,
) -> Option<u32> {
    let left = candidate.min_x.saturating_sub(reach);
    let top = candidate.min_y.saturating_sub(reach);
    let right = (candidate.max_x + reach + 1).min(width);
    let bottom = (candidate.max_y + reach + 1).min(height);
    let (region_width, region_height) = (right - left, bottom - top);
    let mut seeds = vec![0_u8; region_width * region_height];
    let mut any = false;
    for y in 0..region_height {
        let row = &labels[(top + y) * width + left..][..region_width];
        for (x, &label) in row.iter().enumerate() {
            if label != 0 && member[label as usize] {
                seeds[y * region_width + x] = 1;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    let distance = squared_distance(&seeds, region_width, region_height);
    let mut nearest: Option<u32> = None;
    for y in 0..region_height {
        let row = &labels[(top + y) * width + left..][..region_width];
        for (x, &label) in row.iter().enumerate() {
            if label == candidate.label {
                let squared = distance[y * region_width + x];
                nearest = Some(nearest.map_or(squared, |current| current.min(squared)));
            }
        }
    }
    nearest
}

/// Median half-width of a component's strokes and its widest inscribed radius.
fn stroke_profile(labels: &[u32], width: usize, shape: &Shape) -> (f64, f64) {
    let inner_width = shape.max_x - shape.min_x + 1;
    let inner_height = shape.max_y - shape.min_y + 1;
    let (local_width, local_height) = (inner_width + 2, inner_height + 2);
    let mut outside = vec![1_u8; local_width * local_height];
    for y in 0..inner_height {
        let row = &labels[(shape.min_y + y) * width + shape.min_x..][..inner_width];
        for (x, &label) in row.iter().enumerate() {
            if label == shape.label {
                outside[(y + 1) * local_width + x + 1] = 0;
            }
        }
    }
    let distance = squared_distance(&outside, local_width, local_height);
    let mut widest = 0_u32;
    let mut ridge = Vec::new();
    for y in 1..=inner_height {
        for x in 1..=inner_width {
            let index = y * local_width + x;
            let squared = distance[index];
            if outside[index] != 0 {
                continue;
            }
            widest = widest.max(squared);
            let is_ridge = squared >= 3
                && (0..3).all(|dy| {
                    (0..3).all(|dx| distance[(y + dy - 1) * local_width + x + dx - 1] <= squared)
                });
            if is_ridge {
                ridge.push(f64::from(squared).sqrt());
            }
        }
    }
    (median(ridge), f64::from(widest).sqrt())
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

const FAR: i64 = 1 << 40;

/// Exact squared Euclidean distance from every pixel to the nearest seed
/// (Felzenszwalb & Huttenlocher, 2012). Saturates when there is no seed.
fn squared_distance(seeds: &[u8], width: usize, height: usize) -> Vec<u32> {
    let mut grid: Vec<i64> = seeds
        .iter()
        .map(|&seed| if seed != 0 { 0 } else { FAR })
        .collect();
    let longest = width.max(height);
    let mut line = vec![0_i64; longest];
    let mut transformed = vec![0_i64; longest];
    let mut parabolas = vec![0_usize; longest];
    let mut boundaries = vec![0_f64; longest + 1];
    for x in 0..width {
        for (y, value) in line[..height].iter_mut().enumerate() {
            *value = grid[y * width + x];
        }
        transform_line(
            &line[..height],
            &mut transformed[..height],
            &mut parabolas,
            &mut boundaries,
        );
        for (y, &value) in transformed[..height].iter().enumerate() {
            grid[y * width + x] = value;
        }
    }
    for row in grid.chunks_exact_mut(width) {
        line[..width].copy_from_slice(row);
        transform_line(
            &line[..width],
            &mut transformed[..width],
            &mut parabolas,
            &mut boundaries,
        );
        row.copy_from_slice(&transformed[..width]);
    }
    grid.into_iter()
        .map(|distance| distance.min(i64::from(u32::MAX)) as u32)
        .collect()
}

/// One-dimensional lower envelope of parabolas rooted at `values`.
fn transform_line(
    values: &[i64],
    distances: &mut [i64],
    parabolas: &mut [usize],
    boundaries: &mut [f64],
) {
    let mut last = 0;
    parabolas[0] = 0;
    boundaries[0] = f64::NEG_INFINITY;
    boundaries[1] = f64::INFINITY;
    for q in 1..values.len() {
        let crossing = loop {
            let p = parabolas[last];
            let crossing = ((values[q] + (q * q) as i64) - (values[p] + (p * p) as i64)) as f64
                / (2 * (q - p)) as f64;
            if crossing > boundaries[last] {
                break crossing;
            }
            last -= 1;
        };
        last += 1;
        parabolas[last] = q;
        boundaries[last] = crossing;
        boundaries[last + 1] = f64::INFINITY;
    }
    last = 0;
    for (q, distance) in distances.iter_mut().enumerate() {
        while boundaries[last + 1] < q as f64 {
            last += 1;
        }
        let offset = q as i64 - parabolas[last] as i64;
        *distance = offset * offset + values[parabolas[last]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rod {
        cx: f64,
        cy: f64,
        length: f64,
        thickness: f64,
        degrees: f64,
    }

    fn rod(cx: f64, cy: f64, length: f64, thickness: f64, degrees: f64) -> Rod {
        Rod {
            cx,
            cy,
            length,
            thickness,
            degrees,
        }
    }

    struct Canvas {
        width: usize,
        height: usize,
        rgb: Vec<u8>,
        levels: Vec<u8>,
    }

    impl Canvas {
        fn new(width: usize, height: usize) -> Self {
            Self {
                width,
                height,
                rgb: vec![255; width * height * 3],
                levels: vec![0; width * height],
            }
        }

        fn bounds(&self, rod: Rod, margin: f64) -> (usize, usize, usize, usize) {
            let extent = (rod.length + rod.thickness) / 2.0 + margin + 2.0;
            (
                (rod.cx - extent).floor().max(0.0) as usize,
                (rod.cy - extent).floor().max(0.0) as usize,
                ((rod.cx + extent).ceil() as usize).min(self.width),
                ((rod.cy + extent).ceil() as usize).min(self.height),
            )
        }

        /// Paints an anti-aliased black bar with 4x4 supersampling.
        fn bar(&mut self, rod: Rod) {
            let (sin, cos) = rod.degrees.to_radians().sin_cos();
            let (x0, y0, x1, y1) = self.bounds(rod, 0.0);
            for y in y0..y1 {
                for x in x0..x1 {
                    let mut covered = 0_u32;
                    for sy in 0..4 {
                        for sx in 0..4 {
                            let dx = x as f64 + (f64::from(sx) + 0.5) / 4.0 - rod.cx;
                            let dy = y as f64 + (f64::from(sy) + 0.5) / 4.0 - rod.cy;
                            let along = dx * cos + dy * sin;
                            let across = -dx * sin + dy * cos;
                            if along.abs() <= rod.length / 2.0
                                && across.abs() <= rod.thickness / 2.0
                            {
                                covered += 1;
                            }
                        }
                    }
                    if covered > 0 {
                        let luma = (255.0 * (1.0 - f64::from(covered) / 16.0)).round() as u8;
                        let offset = (y * self.width + x) * 3;
                        for channel in 0..3 {
                            self.rgb[offset + channel] = self.rgb[offset + channel].min(luma);
                        }
                    }
                }
            }
        }

        fn block(&mut self, left: usize, top: usize, right: usize, bottom: usize) {
            for y in top..bottom {
                for x in left..right {
                    let offset = (y * self.width + x) * 3;
                    self.rgb[offset..offset + 3].fill(0);
                }
            }
        }

        /// Raises the score of every pixel centre inside the grown rod.
        fn score(&mut self, rod: Rod, level: u8, margin: f64) {
            let (sin, cos) = rod.degrees.to_radians().sin_cos();
            let (x0, y0, x1, y1) = self.bounds(rod, margin);
            for y in y0..y1 {
                for x in x0..x1 {
                    let dx = x as f64 + 0.5 - rod.cx;
                    let dy = y as f64 + 0.5 - rod.cy;
                    let along = dx * cos + dy * sin;
                    let across = -dx * sin + dy * cos;
                    if along.abs() <= rod.length / 2.0 + margin
                        && across.abs() <= rod.thickness / 2.0 + margin
                    {
                        let index = y * self.width + x;
                        self.levels[index] = self.levels[index].max(level);
                    }
                }
            }
        }

        /// A bar plus the blurry, centre-weighted scores the checkpoint gives it.
        fn predicted_bar(&mut self, rod: Rod) {
            self.bar(rod);
            self.score(
                Rod {
                    length: rod.length * 0.8,
                    ..rod
                },
                CONFIDENT,
                2.0,
            );
            self.score(rod, SUPPORTED, 6.0);
        }

        fn mask(&self) -> Vec<u8> {
            let mut page = vec![0_u8; self.width * self.height];
            if let Some(placement) = bar_mask(&self.rgb, self.width, self.height, &self.levels) {
                placement.merge_into(&mut page, self.width);
            }
            page
        }

        fn set(&self, mask: &[u8], x: usize, y: usize) -> bool {
            mask[y * self.width + x] != 0
        }
    }

    fn stack_of(canvas: &mut Canvas, count: usize, length: f64, thickness: f64, pitch: f64) {
        for index in 0..count {
            canvas.predicted_bar(rod(
                canvas.width as f64 / 2.0,
                60.0 + index as f64 * pitch,
                length,
                thickness,
                0.0,
            ));
        }
    }

    #[test]
    fn squared_distance_matches_exhaustive_search() {
        let (width, height) = (23, 17);
        let seeds: Vec<u8> = (0..width * height)
            .map(|index| u8::from((index * 7 + index / 5) % 31 == 0))
            .collect();
        let distance = squared_distance(&seeds, width, height);
        for y in 0..height {
            for x in 0..width {
                let expected = (0..height)
                    .flat_map(|sy| (0..width).map(move |sx| (sx, sy)))
                    .filter(|&(sx, sy)| seeds[sy * width + sx] != 0)
                    .map(|(sx, sy)| {
                        let (dx, dy) = (x.abs_diff(sx), y.abs_diff(sy));
                        (dx * dx + dy * dy) as u32
                    })
                    .min()
                    .expect("the pattern has seeds");
                assert_eq!(distance[y * width + x], expected, "at ({x}, {y})");
            }
        }
        assert!(
            squared_distance(&vec![0; 12], 4, 3)
                .iter()
                .all(|&value| value >= u32::MAX / 2)
        );
    }

    #[test]
    fn support_threshold_tracks_the_confident_threshold() {
        assert_eq!(support_threshold(0.6), 0.3);
        assert_eq!(support_threshold(0.5), 0.25);
        assert_eq!(support_threshold(0.3), 0.2);
        assert_eq!(support_threshold(0.1), 0.1);
        assert_eq!(support_threshold(0.0), 0.0);
    }

    #[test]
    fn rotated_anti_aliased_stack_is_retained_with_its_rim() {
        let mut canvas = Canvas::new(1400, 420);
        let (sin, cos) = 25_f64.to_radians().sin_cos();
        let centers: Vec<(f64, f64)> = (0..5)
            .map(|index| {
                let offset = (index as f64 - 2.0) * 38.0;
                (700.0 - offset * sin, 210.0 + offset * cos)
            })
            .collect();
        for &(cx, cy) in &centers {
            canvas.predicted_bar(rod(cx, cy, 120.0, 14.0, 25.0));
        }
        let mask = canvas.mask();

        for &(cx, cy) in &centers {
            assert!(canvas.set(&mask, cx.round() as usize, cy.round() as usize));
        }
        assert!(!canvas.set(&mask, 5, 5));
        // Between two bars: a 24 px gap is far from every rim.
        assert!(!canvas.set(
            &mask,
            (700.0 - 19.0 * sin) as usize,
            (210.0 + 19.0 * cos) as usize
        ));
        // Anti-aliased edge pixels (luma 81..=128) are covered, light ones are not.
        for y in 0..canvas.height {
            for x in 0..canvas.width {
                let luma = canvas.rgb[(y * canvas.width + x) * 3];
                if luma > 80 && luma <= 128 {
                    assert!(canvas.set(&mask, x, y), "rim pixel ({x}, {y}) luma {luma}");
                }
                if luma > 200 {
                    assert!(
                        !canvas.set(&mask, x, y),
                        "light pixel ({x}, {y}) luma {luma}"
                    );
                }
            }
        }
    }

    #[test]
    fn fan_with_wide_angle_steps_is_one_stack() {
        let mut canvas = Canvas::new(520, 520);
        let bars: Vec<(f64, f64)> = (0..7)
            .map(|index| {
                let phi = (200.0 + index as f64 * 12.0).to_radians();
                (260.0 + 150.0 * phi.cos(), 260.0 + 150.0 * phi.sin())
            })
            .collect();
        for (index, &(cx, cy)) in bars.iter().enumerate() {
            canvas.predicted_bar(rod(cx, cy, 100.0, 13.0, 8.0 + index as f64 * 17.0));
        }
        let mask = canvas.mask();
        for &(cx, cy) in &bars {
            assert!(canvas.set(&mask, cx.round() as usize, cy.round() as usize));
        }
        assert!(!canvas.set(&mask, 5, 5));
    }

    #[test]
    fn rings_and_ladders_join_the_stack_they_touch() {
        let mut canvas = Canvas::new(520, 600);
        stack_of(&mut canvas, 4, 160.0, 18.0, 44.0);
        let corners = [
            (160.0, 270.0),
            (360.0, 270.0),
            (400.0, 360.0),
            (360.0, 450.0),
            (160.0, 450.0),
            (120.0, 360.0),
        ];
        for index in 0..corners.len() {
            let (x0, y0) = corners[index];
            let (x1, y1) = corners[(index + 1) % corners.len()];
            canvas.predicted_bar(rod(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                (x1 - x0).hypot(y1 - y0),
                22.0,
                (y1 - y0).atan2(x1 - x0).to_degrees(),
            ));
        }
        canvas.predicted_bar(rod(260.0, 360.0, 260.0, 20.0, 0.0));
        let mask = canvas.mask();

        for (x, y) in [
            (260, 60),
            (260, 192),
            (260, 270),
            (400, 360),
            (260, 360),
            (122, 360),
        ] {
            assert!(canvas.set(&mask, x, y), "({x}, {y})");
        }
        assert!(!canvas.set(&mask, 260, 315), "the ring's hole stays open");
        assert!(!canvas.set(&mask, 5, 590));
    }

    #[test]
    fn a_ring_without_a_stack_is_not_a_bar_network() {
        let mut canvas = Canvas::new(520, 400);
        let corners = [
            (160.0, 100.0),
            (360.0, 100.0),
            (360.0, 300.0),
            (160.0, 300.0),
        ];
        for index in 0..corners.len() {
            let (x0, y0) = corners[index];
            let (x1, y1) = corners[(index + 1) % corners.len()];
            canvas.predicted_bar(rod(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                (x1 - x0).hypot(y1 - y0) + 22.0,
                22.0,
                (y1 - y0).atan2(x1 - x0).to_degrees(),
            ));
        }
        assert!(canvas.mask().iter().all(|&value| value == 0));
    }

    #[test]
    fn weakly_scored_bar_ends_grow_from_confident_centres() {
        let mut canvas = Canvas::new(420, 300);
        for index in 0..4 {
            let bar = rod(210.0, 60.0 + index as f64 * 50.0, 220.0, 16.0, 0.0);
            canvas.bar(bar);
            canvas.score(
                Rod {
                    length: 66.0,
                    ..bar
                },
                CONFIDENT,
                2.0,
            );
            canvas.score(bar, SUPPORTED, 6.0);
        }
        let mask = canvas.mask();
        for index in 0..4 {
            let y = 60 + index * 50;
            assert!(canvas.set(&mask, 104, y), "left end of bar {index}");
            assert!(canvas.set(&mask, 210, y), "centre of bar {index}");
            assert!(canvas.set(&mask, 316, y), "right end of bar {index}");
        }

        // Without weak support the confident centres alone stay as scored.
        let mut canvas = Canvas::new(420, 300);
        for index in 0..4 {
            let bar = rod(210.0, 60.0 + index as f64 * 50.0, 220.0, 16.0, 0.0);
            canvas.bar(bar);
            canvas.score(
                Rod {
                    length: 66.0,
                    ..bar
                },
                CONFIDENT,
                2.0,
            );
        }
        let mask = canvas.mask();
        assert!(canvas.set(&mask, 210, 60));
        assert!(!canvas.set(&mask, 104, 60));
    }

    #[test]
    fn thin_strokes_crossing_a_stack_neither_join_bars_nor_enter_the_mask() {
        // A 1400 px page opens with a radius of two, which removes 2 px hair.
        let mut canvas = Canvas::new(1400, 320);
        for index in 0..3 {
            canvas.predicted_bar(rod(300.0, 80.0 + index as f64 * 60.0, 220.0, 18.0, 0.0));
        }
        canvas.block(299, 30, 301, 250);
        canvas.score(rod(300.0, 140.0, 220.0, 2.0, 90.0), SUPPORTED, 3.0);
        let mask = canvas.mask();
        for y in [80, 140, 200] {
            assert!(canvas.set(&mask, 200, y));
        }
        for y in [40, 110, 170, 240] {
            assert!(!canvas.set(&mask, 300, y), "stroke at y={y}");
        }
    }

    #[test]
    fn borders_frames_and_blocks_are_not_censorship() {
        // One long diagonal gutter plus a heavy rectangular frame.
        let mut canvas = Canvas::new(600, 600);
        canvas.predicted_bar(rod(300.0, 300.0, 420.0, 14.0, 40.0));
        for (cx, cy, length, degrees) in [
            (300.0, 66.0, 480.0, 0.0),
            (300.0, 534.0, 480.0, 0.0),
            (66.0, 300.0, 480.0, 90.0),
            (534.0, 300.0, 480.0, 90.0),
        ] {
            canvas.predicted_bar(rod(cx, cy, length, 12.0, degrees));
        }
        assert!(canvas.mask().iter().all(|&value| value == 0));

        // A solid black block is not a bar however confidently it scores.
        let mut canvas = Canvas::new(300, 300);
        canvas.block(90, 90, 210, 210);
        canvas.score(rod(150.0, 150.0, 130.0, 130.0, 0.0), CONFIDENT, 2.0);
        canvas.score(rod(150.0, 150.0, 130.0, 130.0, 0.0), SUPPORTED, 6.0);
        assert!(canvas.mask().iter().all(|&value| value == 0));

        // Two bars, or three far apart, are neither stacks.
        let mut canvas = Canvas::new(420, 300);
        stack_of(&mut canvas, 2, 200.0, 16.0, 50.0);
        assert!(canvas.mask().iter().all(|&value| value == 0));
        let mut canvas = Canvas::new(420, 800);
        stack_of(&mut canvas, 3, 200.0, 16.0, 220.0);
        assert!(canvas.mask().iter().all(|&value| value == 0));
    }

    #[test]
    fn collinear_fragments_of_one_border_are_not_a_stack() {
        // Four pieces of a single 58 degree gutter (a page the previous
        // "parallel and near" rule masked as a bar group).
        let mut canvas = Canvas::new(900, 900);
        let (sin, cos) = 58_f64.to_radians().sin_cos();
        for (offset, length) in [
            (-340.0, 230.0),
            (-120.0, 212.0),
            (100.0, 161.0),
            (250.0, 137.0),
        ] {
            canvas.predicted_bar(rod(
                450.0 + offset * cos,
                450.0 + offset * sin,
                length,
                14.0,
                58.0,
            ));
        }
        assert!(canvas.mask().iter().all(|&value| value == 0));
    }

    #[test]
    fn unsupported_or_barely_scored_ink_is_ignored() {
        let mut canvas = Canvas::new(420, 300);
        for index in 0..4 {
            canvas.bar(rod(210.0, 60.0 + index as f64 * 50.0, 220.0, 16.0, 0.0));
        }
        assert!(canvas.mask().iter().all(|&value| value == 0));

        for index in 0..4 {
            canvas.score(
                rod(210.0, 60.0 + index as f64 * 50.0, 220.0, 16.0, 0.0),
                SUPPORTED,
                6.0,
            );
        }
        assert!(
            canvas.mask().iter().all(|&value| value == 0),
            "weak support alone"
        );

        // Thin parallel lines never survive the opening.
        let mut canvas = Canvas::new(300, 300);
        for index in 0..5 {
            canvas.predicted_bar(rod(150.0, 80.0 + index as f64 * 10.0, 160.0, 2.0, 0.0));
        }
        assert!(canvas.mask().iter().all(|&value| value == 0));
    }

    #[test]
    fn page_size_scales_the_required_stack_length() {
        // Three 40 px bars total 120 px: a 640 px page asks for 64 px of bar,
        // a 3600 px page for 256 px. Bars stay wider than the opening disk.
        let mut small = Canvas::new(640, 200);
        stack_of(&mut small, 3, 40.0, 12.0, 36.0);
        assert!(small.set(&small.mask(), 320, 60));

        let mut large = Canvas::new(3600, 200);
        stack_of(&mut large, 3, 40.0, 12.0, 36.0);
        assert!(large.mask().iter().all(|&value| value == 0));
    }

    #[test]
    fn placement_merge_keeps_the_larger_value() {
        let placement = Placement {
            left: 1,
            top: 1,
            width: 2,
            pixels: vec![255, 0, 0, 255],
        };
        let mut page = vec![0_u8; 16];
        page[5] = 7;
        page[6] = 7;
        placement.merge_into(&mut page, 4);
        assert_eq!(page[5], 255);
        assert_eq!(page[6], 7);
        assert_eq!(page[9], 0);
        assert_eq!(page[10], 255);
        assert_eq!(page[0], 0);
    }
}
