use anyhow::Context as _;
use koharu_desktop::{CanvasState, Desktop};
use koharu_scene::EntityId;
use serde::Deserialize;
use specta::Type;
use tauri::State;

use super::{
    ChannelExt as _, Error,
    canvas::{CanvasChannel, Point},
    lifecycle::ProjectChannel,
    project::{CurrentProject, LayerClipboard, Page, Project, Typography},
};
use crate::history::HistoryName;
pub(crate) async fn synchronize_canvas(
    desktop: &Desktop,
    commit: &koharu_scene::Commit,
    page: Option<EntityId>,
) -> anyhow::Result<CanvasState> {
    desktop.synchronize(&commit.snapshot, page, commit).await?;
    Ok(desktop.canvas_state())
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct GeometryUpdate {
    pub layer: EntityId,
    pub points: Option<Vec<Point>>,
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct TypographyUpdate {
    pub layer: EntityId,
    pub typography: Typography,
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "page_renamed",
    skip_all,
    fields(origin = "user", character_count = label.chars().count(), empty = label.is_empty()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn rename_page(
    page: EntityId,
    label: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.rename_page(page, label).await?;
        project.record_named(HistoryName::RenamePage, None, None, &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "pages_deleted",
    skip_all,
    fields(origin = "user", entity_count = pages.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_pages(
    pages: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_pages(pages).await?;
        project.record_named(HistoryName::DeletePages, None, None, &commit);
        project.reconcile_page();
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "page_moved",
    skip_all,
    fields(origin = "user", page_number = index + 1),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn move_page(
    page: EntityId,
    index: u32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_page(page, index as usize).await?;
        project.record_named(HistoryName::MovePage, None, None, &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "source_text_edited",
    skip_all,
    fields(origin = "user", character_count = text.chars().count(), empty = text.is_empty()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn set_source_text(
    layer: EntityId,
    text: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_source_text(layer, text).await?;
        project.record_named(HistoryName::SourceText, None, Some(layer), &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "translation_edited",
    skip_all,
    fields(
        origin = "user",
        character_count = text.as_ref().map_or(0, |text| text.chars().count()),
        empty = text.as_ref().is_none_or(String::is_empty),
    ),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn set_translation(
    layer: EntityId,
    text: Option<String>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_translation(layer, text).await?;
        project.record_named(HistoryName::Translation, None, Some(layer), &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "typography_edited",
    skip_all,
    fields(origin = "user", entity_count = updates.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn set_typography(
    updates: Vec<TypographyUpdate>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let merge = (updates.len() == 1).then(|| updates[0].layer);
        let commit = project.set_typography(updates).await?;
        project.record_named(HistoryName::Typography, None, merge, &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "geometry_edited",
    skip_all,
    fields(origin = "user", entity_count = updates.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn set_geometry(
    updates: Vec<GeometryUpdate>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_geometry(updates).await?;
        project.record_named(HistoryName::Geometry, None, None, &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "visibility_edited",
    skip_all,
    fields(origin = "user", entity_count = layers.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn set_visibility(
    layers: Vec<EntityId>,
    visible: Option<bool>,
    opacity: Option<f32>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let name = if visible.is_some() {
            HistoryName::ToggleLayer
        } else {
            HistoryName::Opacity
        };
        let merge = (opacity.is_some() && layers.len() == 1).then(|| layers[0]);
        let commit = project.set_visibility(layers, visible, opacity).await?;
        project.record_named(name, None, merge, &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "layers_deleted",
    skip_all,
    fields(origin = "user", entity_count = layers.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_layers(
    layers: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_layers(layers).await?;
        project.record_named(HistoryName::DeleteLayers, None, None, &commit);
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

/// The layers last copied in the editor. It survives project switches so text
/// can move between projects; raster layers are checked against the target
/// page's size when pasted.
#[derive(Default)]
pub(crate) struct LayerClipboardState {
    copied: tokio::sync::Mutex<Option<LayerClipboard>>,
}

/// What the editor needs to place a later paste: copies pasted onto their
/// source page are offset so they do not hide the originals.
#[derive(Clone, Debug, serde::Serialize, Type)]
pub struct CopiedLayers {
    pub page: EntityId,
    pub count: u32,
}

/// Copies the selected layers; a zero count leaves the previous clipboard
/// untouched.
#[tracing::instrument(
    target = "koharu_metrics",
    name = "layers_copied",
    skip_all,
    fields(origin = "user", entity_count = layers.len()),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn copy_layers(
    layers: Vec<EntityId>,
    project: State<'_, CurrentProject>,
    clipboard: State<'_, LayerClipboardState>,
) -> Result<CopiedLayers, Error> {
    let copied = {
        let project = project.project.lock().await;
        project
            .as_ref()
            .context("no project is open")?
            .copy_layers(layers)
            .await?
    };
    let result = CopiedLayers {
        page: copied.page,
        count: u32::try_from(copied.layers.len()).unwrap_or(u32::MAX),
    };
    if result.count > 0 {
        *clipboard.copied.lock().await = Some(copied);
    }
    Ok(result)
}

/// Pastes the copied layers onto `page`, shifted by `offset`, as one history
/// step. Returns the new top-level layers so the editor can select them.
#[tracing::instrument(
    target = "koharu_metrics",
    name = "layers_pasted",
    skip_all,
    fields(origin = "user")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn paste_layers(
    page: EntityId,
    offset: Point,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    clipboard: State<'_, LayerClipboardState>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<Vec<EntityId>, Error> {
    let copied = clipboard
        .copied
        .lock()
        .await
        .clone()
        .context("no layers have been copied")?;
    let (commit, active, info, pasted) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let (commit, pasted) = project.paste_layers(page, copied, offset).await?;
        project.record_named(HistoryName::PasteLayers, None, None, &commit);
        (commit, project.active_page(), project.info(), pasted)
    };
    let canvas = synchronize_canvas(&desktop, &commit, active).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(pasted)
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "layer_moved",
    skip_all,
    fields(origin = "user", entity_count = 1_u64)
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn move_layer(
    layer: EntityId,
    parent: EntityId,
    index: u32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<Page, Error> {
    let (commit, page, info, view) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_layer(layer, parent, index as usize).await?;
        project.record_named(HistoryName::MoveLayer, None, None, &commit);
        let page = project.active_page().context("no active page")?;
        let view = Project::page(&commit.snapshot, page)?;
        (commit, page, project.info(), view)
    };
    desktop
        .synchronize(&commit.snapshot, Some(page), &commit)
        .await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    project_channel.channel.publish(Some(info));
    Ok(view)
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "undo",
    skip_all,
    fields(origin = "user")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn undo(
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.undo().await?;
        project.reconcile_page();
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "redo",
    skip_all,
    fields(origin = "user")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn redo(
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
) -> Result<(), Error> {
    let (commit, page, info) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.redo().await?;
        project.reconcile_page();
        (commit, project.active_page(), project.info())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    project_channel.channel.publish(Some(info));
    Ok(())
}
