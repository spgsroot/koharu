use std::sync::Arc;

use anyhow::{Context as _, anyhow, bail};
use image::{GrayImage, ImageEncoder as _, codecs::png::PngEncoder};
use koharu_desktop::{CanvasState, Desktop, Frame, TransformFrame};
use koharu_pipeline::CleanupTarget;
use koharu_rasterizer::ResourceId;
use koharu_scene::{EntityId, Revision};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{
    AppHandle, Cef, Manager as _, State,
    ipc::{Channel, IpcResponse},
};

use super::{
    ChannelExt as _, Error,
    lifecycle::ProjectChannel,
    processing,
    processing::{JobChannel, JobId, Processing},
    project::{CurrentProject, Page, Project, RasterStrokeMode},
};
use crate::history::HistoryName;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
pub struct PaintBrush {
    pub diameter: f32,
    pub color: [u8; 4],
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct LayerCommit {
    pub revision: Revision,
    pub layer: EntityId,
}

#[derive(Type)]
#[specta(transparent)]
pub(crate) struct CanvasBytes(#[specta(type = Vec<u8>)] Vec<u8>);

#[derive(Clone, Copy, Deserialize, Type)]
#[specta(transparent)]
pub(crate) struct CanvasGeneration(#[specta(type = f64)] u64);

#[derive(Clone, Debug, Serialize, Type)]
pub struct CanvasPagePreparation {
    pub revision: Revision,
    pub page: Page,
}

impl IpcResponse for CanvasBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Default)]
pub(crate) struct CanvasChannel {
    pub(crate) channel: Mutex<Option<Channel<CanvasState>>>,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_manifest(
    generation: CanvasGeneration,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    Ok(CanvasBytes(desktop.frame_manifest_bytes(generation.0)?))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_resource(
    generation: CanvasGeneration,
    resource: String,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    let resource = resource
        .parse::<ResourceId>()
        .context("canvas resource id is invalid")?;
    Ok(CanvasBytes(
        desktop.frame_resource_bytes(generation.0, resource)?,
    ))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn prepare_canvas_page(
    page: EntityId,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
) -> Result<Option<CanvasPagePreparation>, Error> {
    let (snapshot, prepared_page) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        let snapshot = project.snapshot();
        let prepared_page = Project::page(&snapshot, page)?;
        (snapshot, prepared_page)
    };
    let revision = snapshot.revision();
    Ok(desktop
        .prepare_page(&snapshot, page)
        .await?
        .then_some(CanvasPagePreparation {
            revision,
            page: prepared_page,
        }))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_page_manifest(
    page: EntityId,
    revision: Revision,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    Ok(CanvasBytes(desktop.page_manifest_bytes(page, revision)?))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_page_resource(
    page: EntityId,
    revision: Revision,
    resource: String,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    let resource = resource
        .parse::<ResourceId>()
        .context("canvas resource id is invalid")?;
    Ok(CanvasBytes(
        desktop.page_resource_bytes(page, revision, resource)?,
    ))
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "point_text_added",
    skip_all,
    fields(origin = "user", point_count = 1_u64)
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn add_point_text(
    point: Point,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<LayerCommit, Error> {
    let (commit, page, info, layer) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, layer) = project.add_point_text(page, point).await?;
        project.record_named(HistoryName::AddText, None, None, &commit);
        (commit, project.active_page(), project.info(), layer)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    project_channel.channel.publish(Some(info));
    Ok(LayerCommit {
        revision: commit.revision,
        layer,
    })
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "text_box_added",
    skip_all,
    fields(origin = "user", entity_count = 1_u64)
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn add_text_box(
    frame: Frame,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<LayerCommit, Error> {
    let (commit, page, info, layer) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, layer) = project.add_text_box(page, frame).await?;
        project.record_named(HistoryName::AddText, None, None, &commit);
        (commit, project.active_page(), project.info(), layer)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    project_channel.channel.publish(Some(info));
    Ok(LayerCommit {
        revision: commit.revision,
        layer,
    })
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "paint_committed",
    skip_all,
    fields(origin = "user", point_count = points.len(), size = f64::from(brush.diameter)),
)]
#[tauri::command]
#[specta::specta]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn commit_paint(
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    brush: PaintBrush,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<LayerCommit, Error> {
    commit_raster_stroke(
        expected_revision,
        layer,
        points,
        brush.diameter,
        brush.color,
        RasterStrokeMode::Paint,
        &desktop,
        &project,
        &canvas_channel,
        &project_channel,
    )
    .await
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "erase_committed",
    skip_all,
    fields(origin = "user", point_count = points.len(), size = f64::from(diameter)),
)]
#[tauri::command]
#[specta::specta]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn commit_erase(
    expected_revision: Revision,
    layer: EntityId,
    points: Vec<Point>,
    diameter: f32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<LayerCommit, Error> {
    commit_raster_stroke(
        expected_revision,
        Some(layer),
        points,
        diameter,
        [0; 4],
        RasterStrokeMode::Erase,
        &desktop,
        &project,
        &canvas_channel,
        &project_channel,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn commit_raster_stroke(
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    diameter: f32,
    color: [u8; 4],
    mode: RasterStrokeMode,
    desktop: &Desktop,
    project: &CurrentProject,
    canvas_channel: &CanvasChannel,
    project_channel: &ProjectChannel,
) -> Result<LayerCommit, Error> {
    let (commit, page, info, element) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        ensure_revision(project.snapshot().revision(), expected_revision)?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, element) = project
            .apply_raster_stroke(
                page,
                layer,
                mode,
                color,
                diameter,
                points
                    .into_iter()
                    .map(|point| koharu_scene::Point {
                        x: point.x,
                        y: point.y,
                    })
                    .collect(),
            )
            .await?;
        let name = match mode {
            RasterStrokeMode::Paint => HistoryName::Brush,
            RasterStrokeMode::Erase => HistoryName::Erase,
        };
        project.record_named(name, None, None, &commit);
        (commit, project.active_page(), project.info(), element)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    project_channel.channel.publish(Some(info));
    Ok(LayerCommit {
        revision: commit.revision,
        layer: element,
    })
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "transform_committed",
    skip_all,
    fields(origin = "user", entity_count = elements.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_transform(
    expected_revision: Revision,
    elements: Vec<TransformFrame>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<Option<Revision>, Error> {
    let geometries = desktop.transform_geometries(expected_revision, &elements)?;
    if geometries.is_empty() {
        return Ok(None);
    }
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        ensure_revision(project.snapshot().revision(), expected_revision)?;
        let commit = project.set_geometries(geometries).await?;
        project.record_named(HistoryName::Transform, None, None, &commit);
        (commit, project.active_page(), project.info())
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    project_channel.channel.publish(Some(info));
    Ok(Some(commit.revision))
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "inpaint_requested",
    skip_all,
    fields(origin = "user", point_count = points.len(), size = f64::from(diameter)),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_inpaint(
    page: EntityId,
    expected_revision: Revision,
    points: Vec<Point>,
    diameter: f32,
    target: CleanupTarget,
    handle: AppHandle<Cef>,
    project: State<'_, CurrentProject>,
) -> Result<Option<JobId>, Error> {
    if !diameter.is_finite() || diameter <= 0.0 || points.is_empty() {
        return Err(anyhow!(
            "an inpaint stroke requires a positive diameter and at least one point"
        )
        .into());
    }
    if points
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err(anyhow!("inpaint stroke points must be finite").into());
    }
    let (width, height, session, pipeline) = {
        let current = project.project.lock().await;
        let current = current.as_ref().context("no project is open")?;
        let snapshot = current.snapshot();
        if expected_revision > snapshot.revision() {
            return Err(anyhow!("the stroke refers to a future project revision").into());
        }
        let page_value = snapshot.page(page)?.page()?;
        (
            page_value.width.round() as u32,
            page_value.height.round() as u32,
            handle.state::<Processing>().session(),
            handle.state::<koharu_pipeline::Pipeline>().capture()?,
        )
    };
    // Preparation may take different times for long and short strokes. Admit
    // them in submission order rather than in PNG-worker completion order.
    let queue = handle.state::<Processing>();
    let _submission = queue.submission.lock().await;
    let (png, bounds) =
        tokio::task::spawn_blocking(move || encode_mask(width, height, &points, diameter))
            .await
            .context("inpaint mask worker stopped unexpectedly")??;
    Ok(Some(
        processing::process_with_inpainting_mask(
            handle.clone(),
            koharu_pipeline::Scope::Region { page, bounds },
            koharu_pipeline::Operation::Only {
                stage: target.stage(),
            },
            handle.state::<CurrentProject>(),
            handle.state::<Processing>(),
            handle.state::<JobChannel>(),
            koharu_pipeline::InpaintingMask {
                page,
                target,
                png: Arc::from(png),
            },
            session,
            pipeline,
        )
        .await?,
    ))
}

/// A gesture started on `expected` still applies after other jobs committed:
/// it names its own layers and page coordinates, so only a revision this
/// process has never seen is rejected. Dropping the gesture instead would
/// silently lose user work whenever a pipeline commits mid-stroke.
fn ensure_revision(actual: Revision, expected: Revision) -> anyhow::Result<()> {
    if expected > actual {
        bail!("canvas edit refers to future revision {expected}; the project is at {actual}");
    }
    Ok(())
}

fn encode_mask(
    width: u32,
    height: u32,
    points: &[Point],
    diameter: f32,
) -> anyhow::Result<(Vec<u8>, koharu_pipeline::Bounds)> {
    if width == 0 || height == 0 {
        bail!("page dimensions must be positive");
    }
    let mut image = GrayImage::new(width, height);
    let radius = f64::from(diameter) * 0.5;
    let mut dirty = None::<[u32; 4]>;
    for (start, end) in points
        .iter()
        .zip(points.iter().skip(1))
        .chain(points.last().map(|point| (point, point)))
    {
        let left = (start.x.min(end.x) - radius - 0.5).floor().max(0.0) as u32;
        let top = (start.y.min(end.y) - radius - 0.5).floor().max(0.0) as u32;
        let right = (start.x.max(end.x) + radius + 0.5)
            .ceil()
            .min(f64::from(width)) as u32;
        let bottom = (start.y.max(end.y) + radius + 0.5)
            .ceil()
            .min(f64::from(height)) as u32;
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length_squared = dx.mul_add(dx, dy * dy);
        for y in top..bottom {
            for x in left..right {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let progress = if length_squared <= f64::EPSILON {
                    0.0
                } else {
                    (((px - start.x) * dx + (py - start.y) * dy) / length_squared).clamp(0.0, 1.0)
                };
                let nearest_x = start.x + progress * dx;
                let nearest_y = start.y + progress * dy;
                let distance = (px - nearest_x).hypot(py - nearest_y);
                let coverage = ((radius + 0.5 - distance).clamp(0.0, 1.0) * 255.0).round() as u8;
                if coverage == 0 {
                    continue;
                }
                let pixel = image.get_pixel_mut(x, y);
                pixel.0[0] = pixel.0[0].max(coverage);
                dirty = Some(match dirty {
                    Some([min_x, min_y, max_x, max_y]) => {
                        [min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y)]
                    }
                    None => [x, y, x, y],
                });
            }
        }
    }
    let [left, top, right, bottom] = dirty.context("inpaint stroke does not intersect the page")?;
    let mut png = Vec::new();
    PngEncoder::new(&mut png).write_image(
        image.as_raw(),
        width,
        height,
        image::ExtendedColorType::L8,
    )?;
    Ok((
        png,
        koharu_pipeline::Bounds {
            x: f64::from(left),
            y: f64::from(top),
            width: f64::from(right - left + 1),
            height: f64::from(bottom - top + 1),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(png: &[u8]) -> GrayImage {
        image::load_from_memory(png).unwrap().to_luma8()
    }

    /// `[left, top, right, bottom]` of every painted pixel.
    fn painted_extent(mask: &GrayImage) -> [u32; 4] {
        mask.enumerate_pixels()
            .filter(|(_, _, pixel)| pixel.0[0] > 0)
            .fold(
                [u32::MAX, u32::MAX, 0, 0],
                |[left, top, right, bottom], (x, y, _)| {
                    [left.min(x), top.min(y), right.max(x), bottom.max(y)]
                },
            )
    }

    #[test]
    fn a_stroke_past_the_page_edge_is_clipped_and_its_region_covers_every_painted_pixel() {
        let off_the_left = [Point { x: -5.0, y: 3.5 }, Point { x: 2.0, y: 3.5 }];
        let (png, bounds) = encode_mask(8, 8, &off_the_left, 2.0).unwrap();
        let mask = decode(&png);

        assert_eq!(mask.dimensions(), (8, 8));
        assert_eq!(mask.get_pixel(0, 3).0[0], 255);
        assert_eq!(mask.get_pixel(7, 3).0[0], 0);
        let [left, top, right, bottom] = painted_extent(&mask);
        assert_eq!(bounds.x, 0.0);
        assert_eq!(
            (bounds.x, bounds.y, bounds.width, bounds.height),
            (
                f64::from(left),
                f64::from(top),
                f64::from(right - left + 1),
                f64::from(bottom - top + 1)
            )
        );

        // A brush wider than the page paints all of it and never reaches past it.
        let (png, bounds) = encode_mask(8, 6, &[Point { x: 4.0, y: 3.0 }], 100.0).unwrap();
        assert_eq!(decode(&png).dimensions(), (8, 6));
        assert_eq!(
            (bounds.x, bounds.y, bounds.width, bounds.height),
            (0.0, 0.0, 8.0, 6.0)
        );
    }

    #[test]
    fn a_stroke_that_misses_the_page_paints_nothing() {
        for point in [
            Point { x: 40.0, y: 40.0 },
            Point { x: -40.0, y: -40.0 },
            Point { x: 4.0, y: 40.0 },
        ] {
            assert!(encode_mask(8, 8, &[point], 4.0).is_err());
        }
        assert!(encode_mask(0, 8, &[Point { x: 1.0, y: 1.0 }], 4.0).is_err());
    }
}
