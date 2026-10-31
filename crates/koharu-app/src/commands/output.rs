use anyhow::{Context as _, Result};
use futures::{StreamExt as _, TryStreamExt as _, stream};
use image::{
    ExtendedColorType, ImageEncoder as _,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_psd::{PsdExportOptions, export_page};
use koharu_rasterizer::{Raster, RasterOptions, Rasterizer};
use koharu_renderer::{Frame, Renderer};
use koharu_scene::{AssetRole, EntityId, Snapshot};
use serde::Deserialize;
use specta::Type;
use std::{path::PathBuf, sync::Arc};
use tauri::{AppHandle, Cef, Manager as _, State, WebviewWindow, ipc::IpcResponse};

use super::{Error, project::CurrentProject};
use koharu_desktop::Desktop;

const THUMBNAIL_EDGE: u32 = 128;

fn encode_png(image: &image::RgbaImage, writer: impl std::io::Write) -> image::ImageResult<()> {
    PngEncoder::new_with_quality(writer, CompressionType::Fast, FilterType::Adaptive).write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        ExtendedColorType::Rgba8,
    )
}

#[derive(Type)]
#[specta(transparent)]
pub(crate) struct ThumbnailBytes(#[specta(type = Vec<u8>)] Vec<u8>);

impl IpcResponse for ThumbnailBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Psd,
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "export",
    skip_all,
    fields(origin = "user", format = ?format),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn export_pages(
    window: WebviewWindow<Cef>,
    pages: Vec<EntityId>,
    format: ExportFormat,
    handle: AppHandle<Cef>,
) -> std::result::Result<(), Error> {
    let Some(directory) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .pick_folder()
        .await
        .map(|directory| directory.path().to_owned())
    else {
        return Ok(());
    };
    export_pages_to_directory(&handle, pages, format, directory).await?;
    Ok(())
}

pub(crate) async fn export_pages_to_directory(
    handle: &AppHandle<Cef>,
    pages: Vec<EntityId>,
    format: ExportFormat,
    directory: PathBuf,
) -> Result<Vec<PathBuf>> {
    let snapshot = {
        let current = handle.state::<CurrentProject>();
        let project = current.project.lock().await;
        project.as_ref().context("no project is open")?.snapshot()
    };
    let pages = if pages.is_empty() {
        snapshot.pages().map(|page| page.id()).collect()
    } else {
        pages
    };
    if pages.is_empty() {
        anyhow::bail!("there are no pages to export");
    }
    tokio::fs::create_dir_all(&directory).await?;
    let desktop = handle.state::<Desktop>();
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let jobs = pages
        .into_iter()
        .enumerate()
        .map(|(index, page_id)| {
            let page = snapshot.page(page_id)?.page()?;
            let name = page
                .label
                .trim()
                .trim_end_matches(|character: char| character == '.' || character.is_whitespace());
            let name = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
            let name = name
                .chars()
                .map(|character| {
                    if matches!(
                        character,
                        '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
                    ) {
                        '_'
                    } else {
                        character
                    }
                })
                .collect::<String>();
            let stem = format!(
                "{:04}_{}",
                index + 1,
                if name.is_empty() { "page" } else { &name }
            );
            Ok::<_, anyhow::Error>((page_id, stem))
        })
        .collect::<Result<Vec<_>>>()?;
    stream::iter(jobs)
        .map(|(page_id, stem)| {
            let renderer = renderer.clone();
            let rasterizer = Arc::clone(&rasterizer);
            let snapshot = snapshot.clone();
            let directory = directory.clone();
            async move {
                let frame = renderer.render(&snapshot, page_id).await?;
                let path = match format {
                    ExportFormat::Png => {
                        let image =
                            rasterize(Arc::clone(&rasterizer), &frame, RasterOptions::default())
                                .await?
                                .image;
                        let path = directory.join(format!("{stem}.png"));
                        let output = path.clone();
                        tokio::task::spawn_blocking(move || -> Result<()> {
                            // Publish via tempfile+rename so a crash mid-export
                            // never leaves a truncated page behind.
                            let mut file = tempfile::NamedTempFile::new_in(
                                output.parent().unwrap_or(std::path::Path::new(".")),
                            )?;
                            encode_png(&image, file.as_file_mut())?;
                            file.as_file().sync_all()?;
                            file.persist(&output).map_err(|error| error.error)?;
                            Ok(())
                        })
                        .await
                        .context("PNG export worker stopped unexpectedly")??;
                        path
                    }
                    ExportFormat::Psd => {
                        let bytes = export_page(
                            Arc::clone(&rasterizer),
                            &snapshot,
                            &frame,
                            &PsdExportOptions::default(),
                        )
                        .await?;
                        let path = directory.join(format!("{stem}.psd"));
                        let output = path.clone();
                        tokio::task::spawn_blocking(move || -> Result<()> {
                            let mut file = tempfile::NamedTempFile::new_in(
                                output.parent().unwrap_or(std::path::Path::new(".")),
                            )?;
                            std::io::Write::write_all(file.as_file_mut(), &bytes)?;
                            file.as_file().sync_all()?;
                            file.persist(&output).map_err(|error| error.error)?;
                            Ok(())
                        })
                        .await
                        .context("PSD export worker stopped unexpectedly")??;
                        path
                    }
                };
                tracing::info!(
                    target: "koharu_metrics",
                    metric = "page_exported",
                    format = ?format,
                );
                Ok::<_, anyhow::Error>(path)
            }
        })
        .buffer_unordered(4)
        .try_collect::<Vec<_>>()
        .await
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_thumbnail(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> std::result::Result<ThumbnailBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    snapshot.page(page)?;
    let blob = snapshot
        .asset(page, &AssetRole::new("source")?)?
        .with_context(|| format!("page {page} has no source image"))?
        .blob;
    let bytes = snapshot.read_blob(blob).await?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let image = image::load_from_memory(&bytes).context("failed to decode source image")?;
        if image.width() == 0 || image.height() == 0 {
            return Err(anyhow::anyhow!("source image is empty"));
        }
        let image = image.thumbnail(THUMBNAIL_EDGE, THUMBNAIL_EDGE).to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok(encoder.encode(80.0).to_vec())
    })
    .await
    .context("thumbnail worker stopped unexpectedly")??;
    Ok(ThumbnailBytes(bytes))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_censor_mask(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> std::result::Result<ThumbnailBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    snapshot.page(page)?;
    let blob = snapshot
        .asset(page, &AssetRole::new("censor-mask")?)?
        .with_context(|| format!("page {page} has no detected censor mask"))?
        .blob;
    let bytes = snapshot.read_blob(blob).await?;
    let tinted = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let mask = image::load_from_memory(&bytes)?.to_luma8();
        let overlay = image::RgbaImage::from_fn(mask.width(), mask.height(), |x, y| {
            image::Rgba([
                255,
                48,
                48,
                (u16::from(mask.get_pixel(x, y)[0]) * 2 / 3) as u8,
            ])
        });
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, image::ImageFormat::Png)?;
        Ok(bytes.into_inner())
    })
    .await
    .context("censor mask preview task panicked")??;
    Ok(ThumbnailBytes(tinted))
}

pub(crate) async fn rendered_preview(
    renderer: &Renderer,
    rasterizer: Arc<Rasterizer>,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Vec<u8>> {
    snapshot.page(page)?;
    let frame = renderer.render(snapshot, page).await?;
    let image = rasterize(rasterizer, &frame, RasterOptions::default())
        .await?
        .image;
    tokio::task::spawn_blocking(move || {
        let image = image::DynamicImage::ImageRgba8(image)
            .resize(1024, 1024, image::imageops::FilterType::Lanczos3)
            .to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok::<_, anyhow::Error>(encoder.encode(85.0).to_vec())
    })
    .await
    .context("preview encode worker stopped unexpectedly")?
}

async fn rasterize(
    rasterizer: Arc<Rasterizer>,
    frame: &Frame,
    options: RasterOptions,
) -> Result<Raster> {
    let frame = frame.raster_frame()?;
    tokio::task::spawn_blocking(move || rasterizer.rasterize(&frame, options))
        .await
        .context("rasterizer worker stopped unexpectedly")?
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use image::{ImageEncoder as _, Rgba, RgbaImage};

    use super::{CompressionType, FilterType, PngEncoder, encode_png};

    #[test]
    fn fast_png_export_preserves_best_compression_pixels() {
        let image = RgbaImage::from_fn(37, 23, |x, y| {
            Rgba([
                (x * 7 + y * 3) as u8,
                (x * 11 + y * 5) as u8,
                (x ^ y) as u8,
                (255 - x - y) as u8,
            ])
        });
        let encode_best = || {
            let mut bytes = Vec::new();
            PngEncoder::new_with_quality(&mut bytes, CompressionType::Best, FilterType::Adaptive)
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )
                .unwrap();
            bytes
        };
        let mut fast = Vec::new();
        encode_png(&image, &mut fast).unwrap();

        let best_pixels = image::load_from_memory(&encode_best()).unwrap().to_rgba8();
        let fast_pixels = image::load_from_memory(&fast).unwrap().to_rgba8();
        assert_eq!(fast_pixels, best_pixels);
    }
}
