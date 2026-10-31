//! Automatic censorship-mask detection for manga inpainting.
//!
//! Bar segmentation uses Er0mangaSeg's Apache-2.0 ConvNeXt-T checkpoint on the
//! original page. Inverting the page makes white backgrounds look like dark
//! censorship and creates false-positive masks. The checkpoint's scores become
//! a mask through [`bars`], which is Koharu's own geometry rather than part of
//! the ported network.
//! Mosaic detection ports the feature-map and period heuristics from
//! `comfyui-fast-mosaic-detector` (MIT, Takahiro Yano). Koharu adds one check
//! of its own: a lattice vertex must sit among flat tiles at full resolution,
//! which keeps halftone dot screens from passing for mosaics.
//! Paper-white stroke stacks use the separate, conservative geometry in
//! [`white_bars`]; the checkpoint itself is never run on an inverted page.

use std::{borrow::Cow, path::PathBuf};

use crate::backend::TryIntoDevice;
use anyhow::{Context as _, Result, anyhow, ensure};
use image::{DynamicImage, GrayImage, Luma, RgbImage, imageops::FilterType};
use imageproc::region_labelling::{Connectivity, connected_components};
use koharu_runtime::Store;
use koharu_torch::{CModule, Device, Kind, Tensor, no_grad};

mod bars;
mod white_bars;

const MODEL_REVISION: &str = "2f1d5210459c751ed716b4887be906dffbb4f034";
const MODEL_SIZE: u32 = 1024;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

#[derive(Debug)]
pub struct Er0mangaSeg {
    model: CModule,
    device: Device,
    mean: Tensor,
    std: Tensor,
}

impl Er0mangaSeg {
    pub fn model_path() -> PathBuf {
        Store::root()
            .join("eromanga-seg")
            .join(MODEL_REVISION)
            .join("model.pt")
    }

    pub fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let path = Self::model_path();
        ensure!(
            path.is_file(),
            "Er0mangaSeg model is not installed at {}; run scripts/export_eromangaseg.py",
            path.display()
        );
        let mut model = CModule::load_on_device(&path, device)
            .with_context(|| format!("failed to load Er0mangaSeg model from {}", path.display()))?;
        model.f_set_eval()?;
        let statistic =
            |values: &[f32; 3]| Tensor::from_slice(values).view([3, 1, 1]).to_device(device);
        Ok(Self {
            model,
            device,
            mean: statistic(&MEAN),
            std: statistic(&STD),
        })
    }

    pub fn detect(
        &self,
        image: &DynamicImage,
        threshold: f32,
        detect_mosaic: bool,
    ) -> Result<GrayImage> {
        ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "censor detection threshold must be between zero and one"
        );
        let rgb = match image {
            DynamicImage::ImageRgb8(rgb) => Cow::Borrowed(rgb),
            other => Cow::Owned(other.to_rgb8()),
        };
        let (width, height) = rgb.dimensions();
        ensure!(
            width > 0 && height > 0,
            "cannot detect censorship in an empty image"
        );

        // Host heuristics run while the accelerator scores the black bars.
        let (placement, mosaic, white) = std::thread::scope(|scope| {
            let mosaic = detect_mosaic.then(|| scope.spawn(|| detect_mosaic_mask(&rgb)));
            let white = scope.spawn(|| white_bars::white_bar_mask(&rgb));
            let placement = self.score_levels(&rgb, threshold).map(|levels| {
                bars::bar_mask(rgb.as_raw(), width as usize, height as usize, &levels)
            });
            (placement, mosaic.map(|handle| handle.join()), white.join())
        });
        let mut mask = match mosaic {
            Some(mosaic) => mosaic.map_err(|_| anyhow!("mosaic detection panicked"))?,
            None => GrayImage::new(width, height),
        };
        if let Some(placement) = placement? {
            placement.merge_into(&mut mask, width as usize);
        }
        let white = white.map_err(|_| anyhow!("white-stroke detection panicked"))?;
        for (target, stroke) in mask.pixels_mut().zip(white.pixels()) {
            target[0] = target[0].max(stroke[0]);
        }
        Ok(mask)
    }

    /// Scores every page pixel with the checkpoint: 0 below the weak
    /// threshold, [`bars::SUPPORTED`] above it, [`bars::CONFIDENT`] above
    /// `threshold`. Thresholding stays on the accelerator so only one byte per
    /// pixel crosses to the host.
    fn score_levels(&self, rgb: &RgbImage, threshold: f32) -> Result<Vec<u8>> {
        let (width, height) = rgb.dimensions();
        let scale = f64::from(MODEL_SIZE) / f64::from(width.max(height));
        let resized_width = (f64::from(width) * scale).round().max(1.0) as u32;
        let resized_height = (f64::from(height) * scale).round().max(1.0) as u32;
        let resized =
            image::imageops::resize(rgb, resized_width, resized_height, FilterType::Triangle);
        let input = self.model_input(&resized);

        let probability = no_grad(|| -> Result<Tensor> {
            let logits = self.model.forward_ts(&[input])?;
            let shape = logits.size();
            ensure!(
                shape.len() == 4 && shape[0] == 1 && shape[1] == 2,
                "unexpected Er0mangaSeg output shape {shape:?}"
            );
            let probability = logits.softmax(1, Kind::Float).select(1, 1);
            let probability = probability
                .unsqueeze(1)
                .upsample_bilinear2d(
                    [i64::from(MODEL_SIZE), i64::from(MODEL_SIZE)],
                    false,
                    None::<f64>,
                    None::<f64>,
                )
                .squeeze_dim(1);
            Ok(probability.squeeze_dim(0))
        })?;
        let probability = probability
            .narrow(0, 0, i64::from(resized_height))
            .narrow(1, 0, i64::from(resized_width))
            .unsqueeze(0)
            .unsqueeze(0)
            .upsample_bilinear2d(
                [i64::from(height), i64::from(width)],
                false,
                None::<f64>,
                None::<f64>,
            )
            .squeeze();
        let supported = bars::support_threshold(threshold);
        let levels = probability.ge(f64::from(supported)).to_kind(Kind::Uint8)
            + probability.ge(f64::from(threshold)).to_kind(Kind::Uint8);
        let levels = Vec::<u8>::try_from(&levels.to_device(Device::Cpu).view([-1]))?;
        ensure!(
            levels.len() == width as usize * height as usize,
            "Er0mangaSeg returned an invalid mask length"
        );
        Ok(levels)
    }

    /// Normalizes on the accelerator and pads to the checkpoint's square
    /// input. Zero padding after normalization is the mean colour, as in the
    /// upstream test pipeline.
    fn model_input(&self, resized: &RgbImage) -> Tensor {
        let (width, height) = resized.dimensions();
        let pixels = Tensor::from_slice(resized.as_raw())
            .view([i64::from(height), i64::from(width), 3])
            .to_device(self.device)
            .permute([2, 0, 1])
            .to_kind(Kind::Float);
        ((pixels - &self.mean) / &self.std)
            .constant_pad_nd([
                0,
                i64::from(MODEL_SIZE - width),
                0,
                i64::from(MODEL_SIZE - height),
            ])
            .unsqueeze(0)
            .contiguous()
    }
}

#[derive(Clone, Copy)]
struct MosaicConfig {
    hue_low: f32,
    hue_high: f32,
    saturation_threshold: f32,
    gradient_threshold: f32,
    ratio_threshold: f32,
    length_min: usize,
    length_max: usize,
    band_height: usize,
    band_half_width: usize,
    /// Side of the window that must be flat for a pixel to count as tile interior.
    flat_window: usize,
    /// Largest per-channel spread inside that window; lossy sources wander by a few levels.
    flat_tolerance: u8,
    /// Radius around a lattice vertex over which tile interior is counted.
    flat_radius: usize,
    /// Smallest share of tile interior around a vertex.
    flat_fraction: f32,
}

impl Default for MosaicConfig {
    fn default() -> Self {
        Self {
            hue_low: 35.0,
            hue_high: 160.0,
            saturation_threshold: 25.0,
            gradient_threshold: 2.5,
            ratio_threshold: 2.74,
            length_min: 4,
            length_max: 25,
            band_height: 1,
            band_half_width: 15,
            flat_window: 4,
            flat_tolerance: 20,
            flat_radius: 10,
            flat_fraction: 0.12,
        }
    }
}

pub fn detect_mosaic_mask(image: &RgbImage) -> GrayImage {
    let config = MosaicConfig::default();
    let width = (image.width() / 2).max(1) as usize;
    let height = (image.height() / 2).max(1) as usize;
    if width < 9 || height < 9 {
        return GrayImage::new(image.width(), image.height());
    }
    // The half-resolution grid keeps its source coordinates so lattice vertices
    // can be checked against the full-resolution page.
    let columns = sample_positions(image.width() as usize, width);
    let rows = sample_positions(image.height() as usize, height);
    let small = RgbImage::from_fn(width as u32, height as u32, |x, y| {
        *image.get_pixel(columns[x as usize] as u32, rows[y as usize] as u32)
    });

    let mut value = vec![0.0_f32; width * height];
    let mut eligible = vec![false; width * height];
    for (index, pixel) in small.pixels().enumerate() {
        let (hue, saturation, brightness) = rgb_to_opencv_hsv(pixel.0);
        value[index] = brightness;
        eligible[index] = hue <= config.hue_low
            || hue >= config.hue_high
            || saturation <= config.saturation_threshold;
    }

    let mut horizontal = vec![0.0_f32; width * height];
    let mut vertical = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if y + 1 < height {
                horizontal[index] = (value[index] - value[index + width]).abs();
            }
            if x + 1 < width {
                vertical[index] = (value[index] - value[index + 1]).abs();
            }
        }
    }
    let horizontal = integral_image(&horizontal, width, height);
    let vertical = integral_image(&vertical, width, height);
    let mut feature = vec![false; width * height];
    let band = config.band_height;
    let half = config.band_half_width;
    for y in (2 + band)..height.saturating_sub(3 + band) {
        for x in (2 + band)..width.saturating_sub(3 + band) {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half + 1).min(width);
            let y0 = y.saturating_sub(half);
            let y1 = (y + half + 1).min(height);
            let val1h = rect_average(&horizontal, width, x0, x1, y - 2 - band, y - 2);
            let val2h = rect_average(&horizontal, width, x0, x1, y + 3, y + 3 + band);
            let val3h = rect_average(&horizontal, width, x0, x1, y, y + 1);
            let val1v = rect_average(&vertical, width, x - 2 - band, x - 2, y0, y1);
            let val2v = rect_average(&vertical, width, x + 3, x + 3 + band, y0, y1);
            let val3v = rect_average(&vertical, width, x, x + 1, y0, y1);
            let val12 = val1h + val2h + val1v + val2v + 1.0;
            let val3hv = val3h * val3v;
            let ratio = 16.0 * val3hv / (val12 * val12 + f32::EPSILON);
            feature[y * width + x] = eligible[y * width + x]
                && val3hv > config.gradient_threshold.powi(2)
                && ratio > config.ratio_threshold.powi(2);
        }
    }

    retain_vertices_among_flat_tiles(&mut feature, image, &columns, &rows, config);

    let Some(period) = estimate_period(&feature, width, height, config) else {
        return GrayImage::new(image.width(), image.height());
    };
    let radius = period.div_ceil(2);
    let mut dilated = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            if !feature[y * width + x] || !repeated_feature(&feature, width, height, x, y, period) {
                continue;
            }
            let top = y.saturating_sub(radius);
            let bottom = (y + radius + 1).min(height);
            let left = x.saturating_sub(radius);
            let right = (x + radius + 1).min(width);
            for target_y in top..bottom {
                dilated[target_y * width + left..target_y * width + right].fill(true);
            }
        }
    }

    let mut small_mask = GrayImage::new(width as u32, height as u32);
    for (index, pixel) in small_mask.pixels_mut().enumerate() {
        if dilated[index] {
            *pixel = Luma([255]);
        }
    }
    let mask = image::imageops::resize(
        &small_mask,
        image.width(),
        image.height(),
        FilterType::Nearest,
    );
    if mostly_monochrome(image) {
        retain_large_mosaic_regions(&mask)
    } else {
        mask
    }
}

fn mostly_monochrome(image: &RgbImage) -> bool {
    let chromatic = image
        .pixels()
        .filter(|pixel| {
            let maximum = pixel.0.into_iter().max().unwrap_or(0);
            let minimum = pixel.0.into_iter().min().unwrap_or(0);
            maximum.abs_diff(minimum) > 8
        })
        .count();
    chromatic.saturating_mul(100) < image.width() as usize * image.height() as usize
}

fn retain_large_mosaic_regions(mask: &GrayImage) -> GrayImage {
    let labels = connected_components(mask, Connectivity::Eight, Luma([0]));
    let max_label = labels.pixels().map(|pixel| pixel.0[0]).max().unwrap_or(0) as usize;
    let mut areas = vec![0_u32; max_label + 1];
    for pixel in labels.pixels() {
        areas[pixel.0[0] as usize] += 1;
    }
    let minimum_area = mask
        .width()
        .saturating_mul(mask.height())
        .div_ceil(400)
        .max(64);
    GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        let label = labels.get_pixel(x, y).0[0] as usize;
        Luma([if label > 0 && areas[label] >= minimum_area {
            255
        } else {
            0
        }])
    })
}

/// Full-resolution source of every cell of the half-resolution grid: the pixel
/// `imageops::resize` samples for `FilterType::Nearest`, kept explicit so
/// vertices can be mapped back onto the page.
fn sample_positions(full: usize, small: usize) -> Vec<usize> {
    let ratio = full as f32 / small as f32;
    (0..small)
        .map(|index| (((index as f32 + 0.5) * ratio).floor() as usize).min(full - 1))
        .collect()
}

/// Drops lattice-vertex candidates that do not sit among flat tiles.
///
/// A pixel mosaic repaints every tile in a single colour, so the page around a
/// real vertex is mostly flat. A dot screen is not: ink and paper alternate
/// inside every lattice cell, so no flat window is wider than a dot. Sampling
/// such a screen on the half-resolution grid still aliases the dots into a
/// dense lattice of vertices, which would outvote real tiles in the period
/// histogram and satisfy the repetition test on its own, so candidates are
/// verified before either runs. This is a deliberate divergence from upstream,
/// which accepts any periodic edge lattice.
fn retain_vertices_among_flat_tiles(
    feature: &mut [bool],
    image: &RgbImage,
    columns: &[usize],
    rows: &[usize],
    config: MosaicConfig,
) {
    if !feature.contains(&true) {
        return;
    }
    let full_width = image.width() as usize;
    let full_height = image.height() as usize;
    let flat = flat_integral(image, config);
    for (index, hit) in feature.iter_mut().enumerate() {
        if !*hit {
            continue;
        }
        let x = columns[index % columns.len()];
        let y = rows[index / columns.len()];
        let density = rect_average(
            &flat,
            full_width,
            x.saturating_sub(config.flat_radius),
            (x + config.flat_radius + 1).min(full_width),
            y.saturating_sub(config.flat_radius),
            (y + config.flat_radius + 1).min(full_height),
        );
        *hit = density >= config.flat_fraction;
    }
}

/// Integral image of the pixels whose `flat_window`-square neighbourhood,
/// clamped at the page edge, varies by at most `flat_tolerance` in every
/// channel.
fn flat_integral(image: &RgbImage, config: MosaicConfig) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let raw = image.as_raw();
    let before = config.flat_window / 2;
    let after = config.flat_window - before - 1;
    let stride = width + 1;
    let mut integral = vec![0.0_f32; stride * (height + 1)];
    for y in 0..height {
        let rows = y.saturating_sub(before)..=(y + after).min(height - 1);
        let mut row_total = 0.0_f32;
        for x in 0..width {
            let columns = x.saturating_sub(before)..=(x + after).min(width - 1);
            let mut minimum = [u8::MAX; 3];
            let mut maximum = [u8::MIN; 3];
            let mut is_flat = true;
            'window: for row in rows.clone() {
                for column in columns.clone() {
                    let offset = (row * width + column) * 3;
                    for (channel, &sample) in raw[offset..offset + 3].iter().enumerate() {
                        minimum[channel] = minimum[channel].min(sample);
                        maximum[channel] = maximum[channel].max(sample);
                        if maximum[channel] - minimum[channel] > config.flat_tolerance {
                            is_flat = false;
                            break 'window;
                        }
                    }
                }
            }
            if is_flat {
                row_total += 1.0;
            }
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row_total;
        }
    }
    integral
}

fn rgb_to_opencv_hsv([red, green, blue]: [u8; 3]) -> (f32, f32, f32) {
    let red = f32::from(red) / 255.0;
    let green = f32::from(green) / 255.0;
    let blue = f32::from(blue) / 255.0;
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let hue = if delta <= f32::EPSILON {
        0.0
    } else if maximum == red {
        60.0 * ((green - blue) / delta).rem_euclid(6.0)
    } else if maximum == green {
        60.0 * ((blue - red) / delta + 2.0)
    } else {
        60.0 * ((red - green) / delta + 4.0)
    } / 2.0;
    let saturation = if maximum <= f32::EPSILON {
        0.0
    } else {
        delta / maximum * 255.0
    };
    (hue, saturation, maximum * 255.0)
}

fn integral_image(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let stride = width + 1;
    let mut integral = vec![0.0_f32; stride * (height + 1)];
    for y in 0..height {
        let mut row = 0.0;
        for x in 0..width {
            row += values[y * width + x];
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
        }
    }
    integral
}

fn rect_average(
    integral: &[f32],
    width: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> f32 {
    if right <= left || bottom <= top {
        return 0.0;
    }
    let stride = width + 1;
    let sum = integral[bottom * stride + right]
        - integral[top * stride + right]
        - integral[bottom * stride + left]
        + integral[top * stride + left];
    sum / ((right - left) * (bottom - top)) as f32
}

fn repeated_feature(
    feature: &[bool],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    period: usize,
) -> bool {
    let matching = |horizontal: bool| {
        [-1_i64, 1].iter().any(|direction| {
            (-1_i64..=1).any(|slack| {
                let offset = *direction * period as i64 + slack;
                let target = if horizontal {
                    (x as i64 + offset, y as i64)
                } else {
                    (x as i64, y as i64 + offset)
                };
                target.0 >= 0
                    && target.1 >= 0
                    && target.0 < width as i64
                    && target.1 < height as i64
                    && feature[target.1 as usize * width + target.0 as usize]
            })
        })
    };
    matching(true) && matching(false)
}

fn estimate_period(
    feature: &[bool],
    width: usize,
    height: usize,
    config: MosaicConfig,
) -> Option<usize> {
    let mut horizontal = vec![0_u32; config.length_max + 1];
    let mut vertical = vec![0_u32; config.length_max + 1];
    for y in 0..height.saturating_sub(config.length_max) {
        for x in 0..width.saturating_sub(config.length_max) {
            if !feature[y * width + x] {
                continue;
            }
            // Upstream discards clusters of adjacent gradient hits before
            // looking for repeated orthogonal, tile-sized spacing.
            if !(1..config.length_min).any(|offset| feature[y * width + x + offset])
                && let Some(distance) = (config.length_min..=config.length_max)
                    .find(|distance| feature[y * width + x + distance])
            {
                horizontal[distance] += 1;
            }
            if !(1..config.length_min).any(|offset| feature[(y + offset) * width + x])
                && let Some(distance) = (config.length_min..=config.length_max)
                    .find(|distance| feature[(y + distance) * width + x])
            {
                vertical[distance] += 1;
            }
        }
    }
    let sum_x: u32 = horizontal.iter().sum();
    let sum_y: u32 = vertical.iter().sum();
    let (x_period, x_hits) = (config.length_min..=config.length_max)
        .map(|period| (period, horizontal[period]))
        .max_by_key(|&(_, hits)| hits)?;
    let (y_period, y_hits) = (config.length_min..=config.length_max)
        .map(|period| (period, vertical[period]))
        .max_by_key(|&(_, hits)| hits)?;
    if sum_x <= 2
        || sum_y <= 2
        || x_hits <= 3
        || y_hits <= 3
        || x_hits as f32 / sum_x as f32 <= 0.1
        || y_hits as f32 / sum_y as f32 <= 0.1
        || x_period.abs_diff(y_period) > 2
    {
        return None;
    }
    Some((x_period + y_period) / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_pixel_grid_and_rejects_flat_image() {
        let mut pixelated = RgbImage::from_pixel(192, 192, image::Rgb([220, 170, 150]));
        for tile_y in 5..19 {
            for tile_x in 5..19 {
                let value = if (tile_x + tile_y) % 2 == 0 { 40 } else { 235 };
                for y in tile_y * 8..(tile_y + 1) * 8 {
                    for x in tile_x * 8..(tile_x + 1) * 8 {
                        pixelated.put_pixel(x, y, image::Rgb([value, value, value]));
                    }
                }
            }
        }
        let detected = detect_mosaic_mask(&pixelated);
        assert!(detected.get_pixel(96, 96).0[0] > 0);

        let flat = RgbImage::from_pixel(192, 192, image::Rgb([128, 128, 128]));
        assert!(
            detect_mosaic_mask(&flat)
                .as_raw()
                .iter()
                .all(|value| *value == 0)
        );
    }

    /// Soft 45-degree dot screen, like a scanned screentone: dots sit on a
    /// 4.2 px diamond lattice (8.4 px along rows) and fade from ink to paper
    /// between radii `core` and `edge`, both in tenths of a pixel.
    fn dot_screen(width: u32, height: u32, core: i32, edge: i32) -> RgbImage {
        const STEP: i32 = 42;
        let span = edge * edge - core * core;
        RgbImage::from_fn(width, height, |x, y| {
            let (u, v) = (10 * x as i32 + 5, 10 * y as i32 + 5);
            let mut shade = 255_i32;
            for column in u.div_euclid(STEP)..=u.div_euclid(STEP) + 1 {
                for row in v.div_euclid(STEP)..=v.div_euclid(STEP) + 1 {
                    if (column + row) % 2 != 0 {
                        continue;
                    }
                    let distance = (u - STEP * column).pow(2) + (v - STEP * row).pow(2);
                    shade = shade.min((distance - core * core).clamp(0, span) * 255 / span);
                }
            }
            image::Rgb([shade as u8; 3])
        })
    }

    /// A page with a dot screen on its left half and a tile mosaic on the
    /// right, one deterministic shade in `low..low + span` per tile.
    fn mosaic_beside_screen(tile: u32, tiles: u32, low: u32, span: u32) -> RgbImage {
        let mut page = RgbImage::from_pixel(384, 256, image::Rgb([255, 255, 255]));
        for (x, y, pixel) in dot_screen(192, 256, 15, 26).enumerate_pixels() {
            page.put_pixel(x, y, *pixel);
        }
        let mut state = 12_345_u32;
        for tile_y in 0..tiles {
            for tile_x in 0..tiles {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
                let shade = (low + (state >> 16) % span) as u8;
                for y in 56 + tile_y * tile..56 + (tile_y + 1) * tile {
                    for x in 224 + tile_x * tile..224 + (tile_x + 1) * tile {
                        page.put_pixel(x, y, image::Rgb([shade; 3]));
                    }
                }
            }
        }
        page
    }

    #[test]
    fn rejects_halftone_dot_screens() {
        for (core, edge) in [(10, 22), (12, 24), (15, 26), (18, 28)] {
            let screen = dot_screen(256, 256, core, edge);
            assert!(
                detect_mosaic_mask(&screen)
                    .as_raw()
                    .iter()
                    .all(|value| *value == 0),
                "dot screen with core {core} and edge {edge} was taken for a mosaic"
            );
        }
    }

    #[test]
    fn keeps_pixel_mosaic_beside_halftone() {
        for (tile, tiles, low, span) in [(12, 11, 100, 60), (16, 8, 90, 100)] {
            let mask = detect_mosaic_mask(&mosaic_beside_screen(tile, tiles, low, span));
            let middle = tile * tiles / 2;
            assert!(
                mask.get_pixel(224 + middle, 56 + middle).0[0] > 0,
                "mosaic of {tile} px tiles was not detected beside the screen"
            );
            assert!(
                (0..256).all(|y| (0..192).all(|x| mask.get_pixel(x, y).0[0] == 0)),
                "screen beside the mosaic of {tile} px tiles was masked"
            );
        }
    }

    #[test]
    fn bar_refinement_keeps_stacked_bars_and_rejects_isolated_dark_shapes() {
        let mut image = RgbImage::from_pixel(512, 512, image::Rgb([255, 255, 255]));
        for top in [50, 75, 100] {
            for y in top..top + 8 {
                for x in 40..120 {
                    image.put_pixel(x, y, image::Rgb([0, 0, 0]));
                }
            }
        }
        for y in 400..408 {
            for x in 300..380 {
                image.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        for y in 50..70 {
            for x in 300..320 {
                image.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }

        let levels = vec![bars::CONFIDENT; 512 * 512];
        let mut page = vec![0_u8; 512 * 512];
        bars::bar_mask(image.as_raw(), 512, 512, &levels)
            .expect("three stacked bars are censorship")
            .merge_into(&mut page, 512);

        for y in [53, 78, 103] {
            assert_eq!(page[y * 512 + 80], 255);
        }
        assert_eq!(page[403 * 512 + 340], 0);
        assert_eq!(page[60 * 512 + 310], 0);
    }
}
