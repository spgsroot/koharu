//! Lettering routes use instance masks, never bounding-box fills.
//! Artwork keeps disconnected glyphs disconnected; only bubble text is closed.

use std::collections::VecDeque;

use image::{GrayImage, Luma};
use imageproc::{
    distance_transform::Norm,
    morphology::{close, dilate, erode},
};

use super::{
    ImageSize, KoharuLayoutDetection, KoharuLayoutMask, expand_bounds, mask_bounds, stamp_mask,
    union_bounds, valid_mask,
};

pub(super) struct Masks {
    pub bubbles: GrayImage,
    pub artwork: GrayImage,
}

pub(super) fn build(detections: &[KoharuLayoutDetection], size: ImageSize) -> Masks {
    let mut result = Masks {
        bubbles: GrayImage::new(size.width, size.height),
        artwork: GrayImage::new(size.width, size.height),
    };
    if size.width == 0 || size.height == 0 {
        return result;
    }
    let radius = |amount: f32| {
        ((size.width.max(size.height) as f32 / 1024.0) * amount)
            .round()
            .clamp(1.0, 255.0) as u8
    };
    let bubble_grow = radius(6.0);
    let artwork_grow = radius(2.0);
    let margin = radius(4.0);
    let bubbles = detections
        .iter()
        .filter(|d| d.label == "bubble" && usable(&d.mask))
        .collect::<Vec<_>>();
    let mut dialogue = vec![Vec::new(); bubbles.len()];
    let structures = detections
        .iter()
        .filter(|d| matches!(d.label.as_str(), "bubble" | "panel") && usable(&d.mask))
        .map(|d| &d.mask)
        .collect::<Vec<_>>();
    for text in detections
        .iter()
        .filter(|d| matches!(d.label.as_str(), "text" | "onomatopoeia") && usable(&d.mask))
    {
        let owner = (text.label == "text")
            .then(|| {
                super::smallest_containing_bubble(
                    bubbles
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (i, &b.mask, b.area)),
                    &text.mask,
                    text.bbox,
                )
            })
            .flatten();
        if let Some(owner) = owner {
            dialogue[owner].push(&text.mask);
        } else {
            let bounds = expand_bounds(
                mask_bounds(&text.mask, size),
                u32::from(artwork_grow) + u32::from(margin),
                size,
            );
            let [left, top, right, bottom] = bounds;
            if left >= right || top >= bottom {
                continue;
            }
            let mut ink = GrayImage::new(right - left, bottom - top);
            stamp_mask(&mut ink, &text.mask, left, top);
            fill_enclosed_background(&mut ink);
            let mut grown = dilate(&ink, Norm::L2, artwork_grow);
            // Structural bands constrain added pixels, not genuine crossing SFX.
            // The margin halo keeps local erosion equivalent to page erosion.
            for structure in &structures {
                let shape_bounds = mask_bounds(structure, size);
                if shape_bounds[0] >= right
                    || shape_bounds[2] <= left
                    || shape_bounds[1] >= bottom
                    || shape_bounds[3] <= top
                {
                    continue;
                }
                let mut shape = GrayImage::new(right - left, bottom - top);
                stamp_mask(&mut shape, structure, left, top);
                let outer = dilate(&shape, Norm::L2, margin);
                let inner = erode(&shape, Norm::L2, margin);
                for (((target, &source), &outside), &inside) in grown
                    .iter_mut()
                    .zip(ink.iter())
                    .zip(outer.iter())
                    .zip(inner.iter())
                {
                    if source == 0 && outside != 0 && inside == 0 {
                        *target = 0;
                    }
                }
            }
            paste(&mut result.artwork, &grown, left, top);
        }
    }
    for (bubble, texts) in bubbles.iter().zip(dialogue) {
        if texts.is_empty() {
            continue;
        }
        let mut bounds = mask_bounds(&bubble.mask, size);
        for text in &texts {
            bounds = union_bounds(bounds, mask_bounds(text, size));
        }
        let [left, top, right, bottom] =
            expand_bounds(bounds, u32::from(bubble_grow) * 3 + u32::from(margin), size);
        if left >= right || top >= bottom {
            continue;
        }
        let mut interior = GrayImage::new(right - left, bottom - top);
        stamp_mask(&mut interior, &bubble.mask, left, top);
        let interior = erode(&interior, Norm::L2, margin);
        let mut ink = GrayImage::new(right - left, bottom - top);
        for text in texts {
            stamp_mask(&mut ink, text, left, top);
        }
        let mut grown = close(&dilate(&ink, Norm::L2, bubble_grow), Norm::L2, bubble_grow);
        for (target, &inside) in grown.iter_mut().zip(interior.iter()) {
            if inside == 0 {
                *target = 0;
            }
        }
        paste(&mut result.bubbles, &grown, left, top);
    }
    // A duplicate text/SFX detection belongs to artwork, never both passes.
    for (bubble, &artwork) in result.bubbles.iter_mut().zip(result.artwork.iter()) {
        if artwork != 0 {
            *bubble = 0;
        }
    }
    result
}

// Outlined SFX include an opaque glyph interior, not just the detected ink.
// Flooding exterior background fills holes without bridging separate glyphs.
fn fill_enclosed_background(mask: &mut GrayImage) {
    let (width, height) = (mask.width() as usize, mask.height() as usize);
    if width == 0 || height == 0 {
        return;
    }
    let mut outside = vec![false; width * height];
    let mut queue = VecDeque::new();
    let visit = |index: usize, outside: &mut [bool], queue: &mut VecDeque<usize>| {
        if !outside[index] && mask.as_raw()[index] == 0 {
            outside[index] = true;
            queue.push_back(index);
        }
    };
    for x in 0..width {
        visit(x, &mut outside, &mut queue);
        visit((height - 1) * width + x, &mut outside, &mut queue);
    }
    for y in 0..height {
        visit(y * width, &mut outside, &mut queue);
        visit(y * width + width - 1, &mut outside, &mut queue);
    }
    while let Some(index) = queue.pop_front() {
        let (x, y) = (index % width, index / width);
        if x > 0 {
            visit(index - 1, &mut outside, &mut queue);
        }
        if x + 1 < width {
            visit(index + 1, &mut outside, &mut queue);
        }
        if y > 0 {
            visit(index - width, &mut outside, &mut queue);
        }
        if y + 1 < height {
            visit(index + width, &mut outside, &mut queue);
        }
    }
    for (pixel, exterior) in mask.iter_mut().zip(outside) {
        if !exterior {
            *pixel = 255;
        }
    }
}

fn usable(mask: &KoharuLayoutMask) -> bool {
    valid_mask(mask) && mask.width > 0 && mask.height > 0 && mask.pixels.iter().any(|&v| v != 0)
}

fn paste(target: &mut GrayImage, mask: &GrayImage, left: u32, top: u32) {
    for (x, y, value) in mask.enumerate_pixels() {
        if value[0] != 0 {
            target.put_pixel(left + x, top + y, Luma([255]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(label: &str, [x, y, width, height]: [u32; 4]) -> KoharuLayoutDetection {
        KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 1.0,
            bbox: [x as f32, y as f32, (x + width) as f32, (y + height) as f32],
            area: width * height,
            mask: KoharuLayoutMask {
                x,
                y,
                width,
                height,
                pixels: vec![255; (width * height) as usize],
            },
        }
    }

    #[test]
    fn outlined_sfx_fills_its_interior_without_filling_exterior_gaps() {
        let mut text = detection("onomatopoeia", [10, 10, 38, 18]);
        text.mask.pixels.fill(0);
        for y in 0..15 {
            for x in 0..15 {
                if x < 3 || x >= 12 || y < 3 || y >= 12 {
                    text.mask.pixels[y * 38 + x] = 255;
                }
            }
        }
        for y in 0..15 {
            for x in 30..35 {
                text.mask.pixels[y * 38 + x] = 255;
            }
        }
        let masks = build(
            &[text],
            ImageSize {
                width: 80,
                height: 80,
            },
        );
        assert_eq!(masks.artwork.get_pixel(17, 17)[0], 255);
        assert_eq!(masks.artwork.get_pixel(35, 17)[0], 0);
        assert_eq!(masks.artwork.get_pixel(42, 17)[0], 255);
    }

    #[test]
    fn dialogue_and_crossing_sfx_have_disjoint_routes_and_protected_borders() {
        let masks = build(
            &[
                detection("bubble", [100, 20, 200, 80]),
                detection("text", [105, 40, 50, 20]),
                detection("onomatopoeia", [280, 40, 50, 20]),
                detection("text", [500, 40, 20, 20]),
            ],
            ImageSize {
                width: 1024,
                height: 128,
            },
        );
        assert_eq!(masks.bubbles.get_pixel(120, 50)[0], 255);
        assert_eq!(masks.artwork.get_pixel(120, 50)[0], 0);
        assert_eq!(masks.bubbles.get_pixel(100, 50)[0], 0);
        assert_eq!(masks.artwork.get_pixel(310, 50)[0], 255);
        assert_eq!(masks.artwork.get_pixel(510, 50)[0], 255);
        assert!(
            !masks
                .bubbles
                .iter()
                .zip(masks.artwork.iter())
                .any(|(&a, &b)| a != 0 && b != 0)
        );
    }

    #[test]
    fn irregular_bubble_bbox_does_not_claim_artwork_in_its_corners() {
        let mut bubble = detection("bubble", [10, 10, 20, 20]);
        for y in 0..20 {
            for x in 0..20 {
                if (x as i32 - 10).abs() + (y as i32 - 10).abs() > 9 {
                    bubble.mask.pixels[y * 20 + x] = 0;
                }
            }
        }
        let masks = build(
            &[bubble, detection("text", [11, 11, 3, 3])],
            ImageSize {
                width: 64,
                height: 64,
            },
        );
        assert_eq!(masks.artwork.get_pixel(12, 12)[0], 255);
        assert!(masks.bubbles.iter().all(|&v| v == 0));
    }

    #[test]
    fn artwork_growth_does_not_close_gaps_or_eat_neighbouring_borders() {
        let masks = build(
            &[
                detection("bubble", [100, 20, 200, 80]),
                detection("onomatopoeia", [302, 40, 10, 20]),
                detection("onomatopoeia", [320, 40, 10, 20]),
            ],
            ImageSize {
                width: 1024,
                height: 128,
            },
        );
        assert_eq!(masks.artwork.get_pixel(300, 50)[0], 0);
        assert_eq!(masks.artwork.get_pixel(302, 50)[0], 255);
        assert_eq!(masks.artwork.get_pixel(316, 50)[0], 0);
        assert_eq!(masks.artwork.get_pixel(331, 50)[0], 255);
        assert_eq!(masks.artwork.get_pixel(332, 50)[0], 0);
    }

    #[test]
    fn onomatopoeia_inside_bubble_takes_priority_over_duplicate_text() {
        let masks = build(
            &[
                detection("bubble", [10, 10, 50, 50]),
                detection("text", [20, 20, 15, 15]),
                detection("onomatopoeia", [25, 25, 15, 15]),
            ],
            ImageSize {
                width: 80,
                height: 80,
            },
        );
        assert_eq!(masks.bubbles.get_pixel(22, 22)[0], 255);
        assert_eq!(masks.bubbles.get_pixel(30, 30)[0], 0);
        assert_eq!(masks.artwork.get_pixel(30, 30)[0], 255);
    }
}
