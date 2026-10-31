use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::PathBuf,
};

use anyhow::{Context as _, Result, bail};
use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_desktop::Frame;
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, BlobId, Commit, EntityId, EntityOrigin,
    Geometry as SceneGeometry, Group as SceneGroup, Origin, PageDraft, Point as ScenePoint,
    Presents, RasterLayer as SceneRasterLayer, RasterLayerKind, Region as SceneRegion,
    RemovePolicy, Revision, Session, Snapshot, SourceText as SceneSourceText,
    TextGroup as SceneTextGroup, TextLayout as SceneTextLayout, TextLayoutKind,
    TextRole as SceneTextRole, Translation as SceneTranslation, Typography as SceneTypography,
    Visibility as SceneVisibility,
};
use serde::{Deserialize, Serialize};
use specta::Type;
use tokio::sync::Mutex;

use super::{
    canvas::Point,
    editing::{GeometryUpdate, TypographyUpdate},
};
use crate::history::{DEFAULT_CAPACITY, History, HistoryName, HistoryView};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RasterStrokeMode {
    Paint,
    Erase,
}

/// The outcome of committing pipeline output onto the latest document.
pub(crate) enum Rebased {
    Applied(Commit),
    /// Nothing was left to change after rebasing.
    Unchanged,
    /// A document edit conflicted with the output, which was discarded.
    Superseded,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectInfo {
    /// Fresh for every opening, including a reopening of the same project file.
    pub session: EntityId,
    pub name: String,
    pub revision: Revision,
    pub active_page: Option<EntityId>,
    pub can_undo: bool,
    pub can_redo: bool,
    pub history: HistoryView,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectSummary {
    pub name: String,
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct PageSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct PageSummary {
    pub id: EntityId,
    pub label: String,
    pub size: PageSize,
    pub source_asset: Option<String>,
    #[specta(type = f64)]
    pub layer_count: usize,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Page {
    pub id: EntityId,
    pub label: String,
    pub size: PageSize,
    pub censor_mask_asset: Option<String>,
    pub layers: Vec<Layer>,
    pub regions: Vec<AnalysisRegion>,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Layer {
    Group {
        id: EntityId,
        parent: Option<EntityId>,
        visibility: LayerVisibility,
        name: String,
        role: Option<GroupRole>,
    },
    Text {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Option<Geometry>,
        visibility: LayerVisibility,
        content: Box<TextContent>,
        typography: Option<Typography>,
        layout: TextLayoutKind,
        automatic_region: Option<EntityId>,
    },
    Raster {
        id: EntityId,
        parent: Option<EntityId>,
        visibility: LayerVisibility,
        image: Option<String>,
        name: String,
        kind: RasterLayerKind,
    },
    Image {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Geometry,
        visibility: LayerVisibility,
        image: String,
    },
    Artwork {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Geometry,
        visibility: LayerVisibility,
        image: String,
    },
}

/// Layers copied in the editor. Kept natively so raster pixels never travel
/// through the webview; pasting validates them against the target page.
#[derive(Clone, Debug)]
pub struct LayerClipboard {
    pub page: EntityId,
    pub layers: Vec<ClipboardLayer>,
}

#[derive(Clone, Debug)]
pub enum ClipboardLayer {
    Group {
        name: String,
        visibility: LayerVisibility,
        children: Vec<ClipboardLayer>,
    },
    Text {
        geometry: Option<Geometry>,
        visibility: LayerVisibility,
        content: ClipboardTextContent,
        typography: Option<Typography>,
        layout: TextLayoutKind,
    },
    Raster {
        visibility: LayerVisibility,
        image: Option<ClipboardRasterImage>,
        name: String,
        kind: RasterLayerKind,
    },
}

#[derive(Clone, Debug)]
pub struct ClipboardTextContent {
    pub source: Option<SourceText>,
    pub translation: Option<Translation>,
    pub role: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ClipboardRasterImage {
    pub bytes: Vec<u8>,
    pub media_type: String,
    pub metadata: AssetMetadata,
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum GroupRole {
    Text,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct TextContent {
    pub id: EntityId,
    pub source: Option<SourceText>,
    pub translation: Option<Translation>,
    pub role: Option<String>,
    pub source_region: Option<EntityId>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct AnalysisRegion {
    pub id: EntityId,
    pub parent: Option<EntityId>,
    pub geometry: Geometry,
    pub kind: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct Geometry {
    pub points: Vec<Point>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Type)]
pub struct LayerVisibility {
    pub visible: bool,
    pub opacity: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct SourceText {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct Translation {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, serde::Deserialize, Serialize, Type)]
pub struct Typography {
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub font_style: Option<koharu_scene::FontStyle>,
    pub size: Option<f32>,
    pub auto_fit: bool,
    pub color: Option<[u8; 4]>,
    pub stroke_color: Option<[u8; 4]>,
    pub stroke_width: Option<f32>,
    pub alignment: Option<koharu_scene::TextAlignment>,
    pub writing_mode: Option<koharu_scene::WritingMode>,
}

pub(crate) struct CurrentProject {
    pub(crate) project: Mutex<Option<Project>>,
}

#[derive(Clone)]
pub(crate) struct ProjectLibrary {
    root: PathBuf,
}

/// Relocates the project library (and the orders store kept in it) for
/// portable installs and isolated profiles.
const PROJECTS_DIR_ENV: &str = "KOHARU_PROJECTS_DIR";

impl ProjectLibrary {
    pub(crate) fn new() -> Result<Self> {
        let root = match std::env::var_os(PROJECTS_DIR_ENV).filter(|value| !value.is_empty()) {
            Some(directory) => {
                let directory = PathBuf::from(directory);
                anyhow::ensure!(
                    directory.is_absolute(),
                    "{PROJECTS_DIR_ENV} must be an absolute path: {}",
                    directory.display()
                );
                directory
            }
            None => dirs::document_dir()
                .context("the Documents directory is unavailable")?
                .join("Koharu"),
        };
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create {}", root.display()))?;
        Ok(Self { root })
    }

    /// Root directory containing every `.khrproj` (and the orders store).
    pub(crate) fn root(&self) -> &std::path::Path {
        &self.root
    }

    pub(crate) fn list(&self) -> Result<Vec<ProjectSummary>> {
        let mut projects = std::fs::read_dir(&self.root)
            .with_context(|| format!("failed to read {}", self.root.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .filter_map(|entry| {
                let path = entry.path();
                let is_project_directory = path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("khrproj"))
                    && (path.join("state-a.khr").is_file() || path.join("state-b.khr").is_file());
                if !is_project_directory {
                    return None;
                }
                let last_used = ["state-a.khr", "state-b.khr"]
                    .into_iter()
                    .filter_map(|file| std::fs::metadata(path.join(file)).ok()?.modified().ok())
                    .max()
                    .unwrap_or(std::time::UNIX_EPOCH);
                Some((
                    last_used,
                    ProjectSummary {
                        name: path.file_stem()?.to_str()?.to_owned(),
                    },
                ))
            })
            .collect::<Vec<_>>();
        projects.sort_unstable_by(|(left_used, left), (right_used, right)| {
            right_used
                .cmp(left_used)
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
        });
        Ok(projects.into_iter().map(|(_, project)| project).collect())
    }

    pub(crate) async fn create(&self, name: &str) -> Result<Project> {
        let (name, path) = self.resolve(name)?;
        Project::create(name, path).await
    }

    pub(crate) async fn open(&self, name: &str) -> Result<Project> {
        let (name, path) = self.resolve(name)?;
        Project::open(name, path).await
    }

    pub(crate) fn delete(&self, name: &str) -> Result<()> {
        let (_, path) = self.resolve(name)?;
        if !path.is_dir() {
            bail!("project {name:?} does not exist");
        }
        std::fs::remove_dir_all(&path)
            .with_context(|| format!("failed to delete {}", path.display()))
    }

    fn resolve(&self, name: &str) -> Result<(String, PathBuf)> {
        let name = validate_project_name(name)?;
        Ok((name.clone(), self.root.join(format!("{name}.khrproj"))))
    }
}

pub(crate) struct Project {
    pub(crate) session: Session,
    instance: EntityId,
    pub(crate) name: String,
    pub(crate) active_page: Option<EntityId>,
    pub(crate) history: History,
}

impl Project {
    pub(crate) async fn create(name: String, path: PathBuf) -> Result<Self> {
        let session = Session::create(&path)
            .await
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self::new(session, name))
    }

    pub(crate) async fn open(name: String, path: PathBuf) -> Result<Self> {
        let session = Session::open(&path)
            .await
            .with_context(|| format!("failed to open {}", path.display()))?;
        Ok(Self::new(session, name))
    }

    fn new(session: Session, name: String) -> Self {
        let active_page = session.snapshot().pages().next().map(|page| page.id());
        let history = History::new(session.snapshot(), active_page, DEFAULT_CAPACITY);
        Self {
            session,
            instance: EntityId::new(),
            name,
            active_page,
            history,
        }
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        self.session.snapshot()
    }

    pub(crate) fn revision(&self) -> Revision {
        self.snapshot().revision()
    }

    pub(crate) fn active_page(&self) -> Option<EntityId> {
        self.active_page
    }

    pub(crate) fn select_page(&mut self, page: EntityId) -> Result<()> {
        self.snapshot().page(page)?;
        self.active_page = Some(page);
        Ok(())
    }

    pub(crate) fn reconcile_page(&mut self) {
        let snapshot = self.snapshot();
        if self
            .active_page
            .is_none_or(|page| snapshot.page(page).is_err())
        {
            self.active_page = snapshot.pages().next().map(|page| page.id());
        }
    }

    pub(crate) fn info(&self) -> ProjectInfo {
        ProjectInfo {
            session: self.instance,
            name: self.name.clone(),
            revision: self.revision(),
            active_page: self.active_page,
            can_undo: self.history.can_undo(),
            can_redo: self.history.can_redo(),
            history: self.history.view(),
        }
    }

    pub(crate) fn pages(snapshot: &Snapshot) -> Result<Vec<PageSummary>> {
        snapshot
            .pages()
            .map(|page| {
                let value = page.page()?;
                let source_asset = Self::asset_id(snapshot, page.id(), "source")?;
                let layer_count = snapshot
                    .descendants(page.id())?
                    .map(|entity| Self::is_content_layer(snapshot, entity.id()))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|present| *present)
                    .count()
                    + usize::from(source_asset.is_some());
                Ok(PageSummary {
                    id: page.id(),
                    label: value.label,
                    size: PageSize {
                        width: value.width,
                        height: value.height,
                    },
                    source_asset,
                    layer_count,
                })
            })
            .collect()
    }

    pub(crate) async fn rename_page(&mut self, page: EntityId, label: String) -> Result<Commit> {
        let snapshot = self.snapshot();
        let current = snapshot.page(page)?.page()?;
        let patch = snapshot.patch(|edit| {
            edit.set_page(page, PageDraft::new(label, current.width, current.height))
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn delete_pages(&mut self, pages: Vec<EntityId>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let pages = Self::unique_roots(&snapshot, pages)?;
        let patch = snapshot.patch(|edit| {
            for page in pages {
                edit.remove_entity(page, RemovePolicy::Cascade)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn move_page(&mut self, page: EntityId, index: usize) -> Result<Commit> {
        let snapshot = self.snapshot();
        let siblings = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let at = Self::placement(&siblings, page, index);
        let patch = snapshot.patch(|edit| edit.move_entity(page, None, at))?;
        self.commit(patch).await
    }

    pub(crate) async fn add_point_text(
        &mut self,
        page: EntityId,
        point: Point,
    ) -> Result<(Commit, EntityId)> {
        if !point.x.is_finite() || !point.y.is_finite() {
            bail!("text position must contain finite coordinates");
        }
        let value = self.snapshot().page(page)?.page()?;
        if point.x < 0.0 || point.y < 0.0 || point.x >= value.width || point.y >= value.height {
            bail!("text position must be inside the page");
        }
        self.add_text(
            page,
            Frame {
                x: point.x as f32,
                y: point.y as f32,
                width: (value.width - point.x).clamp(1.0, 320.0) as f32,
                height: (value.height - point.y).clamp(1.0, 120.0) as f32,
                angle_degrees: 0.0,
            },
            TextLayoutKind::Point,
        )
        .await
    }

    pub(crate) async fn add_text_box(
        &mut self,
        page: EntityId,
        frame: Frame,
    ) -> Result<(Commit, EntityId)> {
        self.add_text(page, frame, TextLayoutKind::Paragraph).await
    }

    async fn add_text(
        &mut self,
        page: EntityId,
        frame: Frame,
        kind: TextLayoutKind,
    ) -> Result<(Commit, EntityId)> {
        let snapshot = self.snapshot();
        let geometry = Self::geometry_from_frame(frame)?;
        let mut layer = None;
        let patch = snapshot.patch(|edit| {
            let content = edit.add_text_content(page, At::End)?;
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(String::new()),
                    language: None,
                },
            )?;
            let added_layer = edit.add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind,
                },
            )?;
            layer = Some(added_layer);
            edit.set(added_layer, &geometry)?;
            edit.set(
                added_layer,
                &SceneTypography {
                    origin: Origin::User,
                    preferred_font: None,
                    font_weight: None,
                    font_style: None,
                    size: None,
                    auto_fit: true,
                    color: None,
                    stroke_color: None,
                    stroke_width: None,
                    alignment: match kind {
                        TextLayoutKind::Point => Some(koharu_scene::TextAlignment::Start),
                        TextLayoutKind::Paragraph => None,
                    },
                    writing_mode: None,
                    extensions: Default::default(),
                },
            )?;
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            layer.expect("text layer was added while building the patch"),
        ))
    }

    pub(crate) async fn set_source_text(
        &mut self,
        layer: EntityId,
        text: String,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let previous = snapshot.component::<SceneSourceText>(content)?;
        let changed = previous
            .as_ref()
            .is_none_or(|source| source.text.value != text);
        let language = previous.and_then(|source| source.language);
        let invalidate_translation = changed
            && snapshot
                .component::<SceneTranslation>(content)?
                .is_some_and(|translation| matches!(translation.text.origin, Origin::Generated(_)));
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(text),
                    language,
                },
            )?;
            if invalidate_translation {
                edit.remove::<SceneTranslation>(content)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_translation(
        &mut self,
        layer: EntityId,
        text: Option<String>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let language = snapshot
            .component::<SceneTranslation>(content)?
            .and_then(|translation| translation.language);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            match text {
                Some(text) => edit.set(
                    content,
                    &SceneTranslation {
                        text: Authored::user(text),
                        language,
                    },
                ),
                None => edit.remove::<SceneTranslation>(content),
            }
        })?;
        self.commit(patch).await
    }

    /// Applies many translation updates as a single patch so a translation
    /// import lands atomically and records exactly one history state.
    pub(crate) async fn set_translations_batch(
        &mut self,
        updates: Vec<(EntityId, Option<String>)>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|(layer, text)| {
                let content = Self::text_content(&snapshot, layer)?;
                let language = snapshot
                    .component::<SceneTranslation>(content)?
                    .and_then(|translation| translation.language);
                Ok((layer, content, language, text))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (layer, content, language, text) in updates {
                edit.promote_entity_to_user(layer)?;
                edit.promote_entity_to_user(content)?;
                match text {
                    Some(text) => edit.set(
                        content,
                        &SceneTranslation {
                            text: Authored::user(text),
                            language,
                        },
                    ),
                    None => edit.remove::<SceneTranslation>(content),
                }?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    /// Every translatable segment in page order, keyed by its layer entity id
    /// (the `segment_id` of the translation exchange format).
    pub(crate) fn translation_segments(
        snapshot: &Snapshot,
    ) -> Result<Vec<koharu_orders::SegmentRow>> {
        let mut segments = Vec::new();
        for page in snapshot.pages() {
            let label = page.page()?.label;
            Self::collect_segments(snapshot, page.id(), &label, &mut segments)?;
        }
        Ok(segments)
    }

    fn collect_segments(
        snapshot: &Snapshot,
        parent: EntityId,
        page_label: &str,
        segments: &mut Vec<koharu_orders::SegmentRow>,
    ) -> Result<()> {
        for child in snapshot.children(parent)? {
            if !Self::is_layer(snapshot, child)? {
                continue;
            }
            if snapshot.component::<SceneGroup>(child)?.is_some() {
                Self::collect_segments(snapshot, child, page_label, segments)?;
                continue;
            }
            if snapshot.component::<SceneTextLayout>(child)?.is_none() {
                continue;
            }
            let content = snapshot.text_layer(child)?.content()?;
            let source = content
                .source()?
                .map(|source| source.text.value)
                .unwrap_or_default();
            let translation = content
                .translation()?
                .map(|translation| translation.text.value);
            // Segments without any text yet carry nothing to translate.
            if source.is_empty() && translation.is_none() {
                continue;
            }
            segments.push(koharu_orders::SegmentRow {
                id: child.to_string(),
                page: page_label.to_owned(),
                source,
                translation,
            });
        }
        Ok(())
    }

    /// Computed page counters for the orders dashboard; never user input.
    /// A page is prepared once it carries source text (detection/OCR ran) and
    /// translated once every one of its segments has a translation.
    pub(crate) fn page_counts(snapshot: &Snapshot) -> Result<koharu_orders::PageCounts> {
        let mut counts = koharu_orders::PageCounts::default();
        for page in snapshot.pages() {
            counts.total += 1;
            let mut segments = Vec::new();
            Self::collect_segments(snapshot, page.id(), &page.page()?.label, &mut segments)?;
            if segments.iter().any(|segment| !segment.source.is_empty()) {
                counts.prepared += 1;
            }
            if !segments.is_empty()
                && segments.iter().all(|segment| {
                    segment
                        .translation
                        .as_deref()
                        .is_some_and(|text| !text.is_empty())
                })
            {
                counts.translated += 1;
            }
        }
        Ok(counts)
    }

    pub(crate) async fn set_typography(
        &mut self,
        updates: Vec<TypographyUpdate>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|update| {
                let content = Self::text_content(&snapshot, update.layer)?;
                Ok((update, content))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (update, content) in updates {
                edit.promote_entity_to_user(update.layer)?;
                edit.promote_entity_to_user(content)?;
                edit.set(
                    update.layer,
                    &SceneTypography {
                        origin: Origin::User,
                        preferred_font: update.typography.preferred_font,
                        font_weight: update.typography.font_weight,
                        font_style: update.typography.font_style,
                        size: update.typography.size.filter(|size| *size > 0.0),
                        auto_fit: update.typography.auto_fit,
                        color: update.typography.color,
                        stroke_color: update.typography.stroke_color,
                        stroke_width: update.typography.stroke_width,
                        alignment: update.typography.alignment,
                        writing_mode: update.typography.writing_mode,
                        extensions: Default::default(),
                    },
                )?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_geometry(&mut self, updates: Vec<GeometryUpdate>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|update| {
                if snapshot
                    .component::<SceneTextLayout>(update.layer)?
                    .is_none()
                {
                    bail!("only text layers can change geometry");
                }
                let content = Self::text_content(&snapshot, update.layer)?;
                if update.points.is_none()
                    && snapshot
                        .text_layer(update.layer)?
                        .automatic_target()?
                        .is_none()
                {
                    bail!("only automatically placed text can reset its geometry");
                }
                Ok((update, content))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (update, content) in updates {
                edit.promote_entity_to_user(update.layer)?;
                edit.promote_entity_to_user(content)?;
                match update.points {
                    Some(points) => edit.set(
                        update.layer,
                        &SceneGeometry {
                            origin: Origin::User,
                            points: points
                                .into_iter()
                                .map(|point| ScenePoint {
                                    x: point.x,
                                    y: point.y,
                                })
                                .collect(),
                        },
                    )?,
                    None => edit.remove::<SceneGeometry>(update.layer)?,
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_visibility(
        &mut self,
        layers: Vec<EntityId>,
        visible: Option<bool>,
        opacity: Option<f32>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let patch = snapshot.patch(|edit| {
            for layer in layers {
                let mut value =
                    snapshot
                        .component::<SceneVisibility>(layer)?
                        .unwrap_or(SceneVisibility {
                            origin: Origin::User,
                            visible: true,
                            opacity: 1.0,
                        });
                if let Some(visible) = visible {
                    value.visible = visible;
                }
                if let Some(opacity) = opacity {
                    value.opacity = opacity;
                }
                value.origin = Origin::User;
                edit.set(layer, &value)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn copy_layers(&self, layers: Vec<EntityId>) -> Result<LayerClipboard> {
        let snapshot = self.snapshot();
        let mut expanded = Vec::new();
        for layer in layers {
            if snapshot.component::<SceneTextGroup>(layer)?.is_some() {
                expanded.extend(snapshot.children(layer)?);
            } else {
                expanded.push(layer);
            }
        }
        let layers = Self::unique_roots(&snapshot, expanded)?;
        let Some(first) = layers.first().copied() else {
            return Ok(LayerClipboard {
                page: self.active_page.context("the project has no active page")?,
                layers: Vec::new(),
            });
        };
        let page = Self::owning_page(&snapshot, first)?;
        if layers
            .iter()
            .copied()
            .map(|layer| Self::owning_page(&snapshot, layer))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .any(|candidate| candidate != page)
        {
            bail!("layers from different pages cannot be copied together");
        }
        let source = AssetRole::new("source")?;
        let mut blobs = HashSet::new();
        for layer in &layers {
            Self::collect_clipboard_blobs(&snapshot, *layer, &source, &mut blobs)?;
        }
        let mut images = HashMap::new();
        for blob in blobs {
            images.insert(blob, snapshot.read_blob(blob).await?.to_vec());
        }
        let layers = layers
            .into_iter()
            .map(|layer| Self::clipboard_layer(&snapshot, layer, &source, &images))
            .collect::<Result<Vec<_>>>()?;
        Ok(LayerClipboard { page, layers })
    }

    pub(crate) async fn paste_layers(
        &mut self,
        page: EntityId,
        clipboard: LayerClipboard,
        offset: Point,
    ) -> Result<(Commit, Vec<EntityId>)> {
        if !offset.x.is_finite() || !offset.y.is_finite() {
            bail!("paste offset must contain finite coordinates");
        }
        if clipboard.layers.is_empty() {
            bail!("the layer clipboard is empty");
        }
        let snapshot = self.snapshot();
        let page_value = snapshot.page(page)?.page()?;
        for layer in &clipboard.layers {
            Self::validate_clipboard_layer(layer, page_value.width, page_value.height)?;
        }
        let mut pasted = Vec::new();
        let patch = snapshot.patch(|edit| {
            for layer in &clipboard.layers {
                pasted.push(Self::paste_clipboard_layer(
                    &snapshot, edit, page, None, layer, offset,
                )?);
            }
            Ok(())
        })?;
        Ok((self.commit(patch).await?, pasted))
    }

    pub(crate) async fn delete_layers(&mut self, layers: Vec<EntityId>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let mut expanded = Vec::new();
        for layer in layers {
            if snapshot.component::<SceneTextGroup>(layer)?.is_some() {
                expanded.extend(snapshot.children(layer)?);
            } else {
                expanded.push(layer);
            }
        }
        let layers = Self::unique_roots(&snapshot, expanded)?;
        let mut orphaned_contents = Vec::new();
        for layer in &layers {
            let Some(relation) = snapshot.relation_from::<Presents>(*layer)? else {
                continue;
            };
            let content = relation.value().target;
            if snapshot.relations_to_as::<Presents>(content).count() == 1 {
                orphaned_contents.push(content);
            }
        }
        let patch = snapshot.patch(|edit| {
            for layer in layers {
                if snapshot.page(layer).is_ok() {
                    return Err(koharu_scene::Error::Invalid(
                        "delete pages with delete_pages".to_owned(),
                    ));
                }
                edit.remove_entity(layer, RemovePolicy::Cascade)?;
            }
            for content in orphaned_contents {
                if snapshot.entity(content).is_ok() {
                    edit.remove_entity(content, RemovePolicy::Cascade)?;
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    #[tracing::instrument(level = "info", skip_all, fields(layer = %layer, parent = %parent))]
    pub(crate) async fn move_layer(
        &mut self,
        layer: EntityId,
        parent: EntityId,
        index: usize,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let siblings = snapshot
            .children(parent)?
            .map(|candidate| Ok(Self::is_layer(&snapshot, candidate)?.then_some(candidate)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let at = Self::placement(&siblings, layer, index);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            if snapshot.component::<SceneTextGroup>(parent)?.is_some() {
                edit.promote_entity_to_user(parent)?;
            }
            edit.move_entity(layer, Some(parent), at)
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_geometries(
        &mut self,
        geometries: impl IntoIterator<Item = (EntityId, SceneGeometry)>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let geometries = geometries
            .into_iter()
            .map(|(element, geometry)| {
                if snapshot.component::<SceneTextLayout>(element)?.is_none() {
                    bail!("only text layers can change geometry");
                }
                let content = Self::text_content(&snapshot, element)?;
                Ok((element, geometry, content))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (element, mut geometry, content) in geometries {
                edit.promote_entity_to_user(element)?;
                edit.promote_entity_to_user(content)?;
                geometry.origin = Origin::User;
                edit.set(element, &geometry)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn apply_raster_stroke(
        &mut self,
        page: EntityId,
        layer: Option<EntityId>,
        mode: RasterStrokeMode,
        color: [u8; 4],
        diameter: f32,
        points: Vec<ScenePoint>,
    ) -> Result<(Commit, EntityId)> {
        if !diameter.is_finite() || diameter <= 0.0 || points.is_empty() {
            bail!("a raster stroke requires a positive diameter and at least one point");
        }
        if points
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            bail!("raster stroke points must be finite");
        }
        let snapshot = self.snapshot();
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        let mut raster_layer = None;
        let mut promote_layer = false;
        let mut image = if let Some(layer) = layer {
            if snapshot.parent(layer)? != Some(page) {
                bail!("the raster target must be a layer on the active page");
            }
            let target = snapshot
                .component::<SceneRasterLayer>(layer)?
                .context("the raster target must be a layer on the active page")?;
            promote_layer = target.origin != Origin::User
                || snapshot
                    .component::<EntityOrigin>(layer)?
                    .is_some_and(|origin| origin.origin != Origin::User);
            raster_layer = Some(target);
            match snapshot.asset(layer, &AssetRole::new("source")?)? {
                Some(asset) => {
                    let bytes = snapshot.read_blob(asset.blob).await?;
                    image::load_from_memory(&bytes)?.to_rgba8()
                }
                None => RgbaImage::new(width, height),
            }
        } else {
            if mode == RasterStrokeMode::Erase {
                bail!("eraser requires a raster layer target");
            }
            RgbaImage::new(width, height)
        };
        if image.dimensions() != (width, height) {
            bail!("raster layer dimensions must match the page");
        }
        rasterize_stroke(&mut image, mode, color, diameter, &points);
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
        let source = AssetRole::new("source")?;
        let name = format!(
            "Paint {}",
            snapshot
                .children(page)?
                .filter(|entity| {
                    snapshot
                        .component::<SceneRasterLayer>(*entity)
                        .ok()
                        .flatten()
                        .is_some_and(|layer| layer.kind == RasterLayerKind::Paint)
                })
                .count()
                + 1
        );
        let mut committed_layer = layer;
        let patch = snapshot.patch(|edit| {
            let layer = if let Some(layer) = committed_layer {
                if promote_layer {
                    edit.promote_entity_to_user(layer)?;
                    let raster = raster_layer
                        .as_mut()
                        .expect("an existing raster target has a layer component");
                    raster.origin = Origin::User;
                    edit.set(layer, raster)?;
                }
                layer
            } else {
                let at = snapshot
                    .page(page)?
                    .text_group()?
                    .map_or(At::End, |group| At::Before(group.id()));
                let layer = edit.add_entity(page, at)?;
                edit.set(
                    layer,
                    &SceneRasterLayer {
                        origin: Origin::User,
                        name,
                        kind: RasterLayerKind::Paint,
                    },
                )?;
                committed_layer = Some(layer);
                layer
            };
            edit.set_asset(
                layer,
                &source,
                AssetInput::new(
                    bytes.into_inner(),
                    "image/png",
                    AssetMetadata {
                        width: Some(width),
                        height: Some(height),
                        attributes: Default::default(),
                    },
                ),
            )
        })?;
        Ok((
            self.commit(patch).await?,
            committed_layer.expect("raster layer was selected or added while building the patch"),
        ))
    }

    pub(crate) async fn undo(&mut self) -> Result<Commit> {
        self.step(false).await?.context("nothing to undo")
    }

    pub(crate) async fn redo(&mut self) -> Result<Commit> {
        self.step(true).await?.context("nothing to redo")
    }

    async fn step(&mut self, forward: bool) -> Result<Option<Commit>> {
        let (commit, page) = match forward {
            true => self.history.step_forward(&mut self.session).await?,
            false => self.history.step_back(&mut self.session).await?,
        }
        .with_context(|| match forward {
            true => "nothing to redo",
            false => "nothing to undo",
        })?;
        self.active_page = page;
        self.reconcile_page();
        Ok(Some(commit))
    }
    /// Reverts and removes the newest history state after a coupled external
    /// transaction fails.
    pub(crate) async fn rollback_history_tip(&mut self) -> Result<Commit> {
        let (commit, page) = self.history.rollback_tip(&mut self.session).await?;
        self.active_page = page;
        self.reconcile_page();
        Ok(commit)
    }

    /// Moves history to an arbitrary panel entry. Restores the active page
    /// recorded on the target state when it still exists.
    pub(crate) async fn jump_to(&mut self, target: usize) -> Result<Option<Commit>> {
        let Some((commit, page)) = self.history.jump(&mut self.session, target).await? else {
            return Ok(None);
        };
        self.active_page = page;
        self.reconcile_page();
        Ok(Some(commit))
    }

    /// Records a commit under a semantic name; the name is part of the call
    /// signature so no commit can land in history anonymously.
    pub(crate) fn record_named(
        &mut self,
        name: HistoryName,
        detail: Option<String>,
        merge: Option<EntityId>,
        commit: &Commit,
    ) -> bool {
        self.history.record(
            &mut self.session,
            name,
            detail,
            merge,
            commit,
            self.active_page,
        )
    }

    pub(crate) fn clear_history(&mut self) {
        self.history.clear(&mut self.session);
    }

    /// Pins the current state as a named snapshot (Photoshop-style).
    pub(crate) fn create_snapshot(&mut self, name: Option<String>) -> u32 {
        self.history.create_snapshot(self.session.snapshot(), name)
    }

    /// Restores a snapshot as a new history state; undoing that state steps
    /// back to whatever preceded the restore.
    pub(crate) async fn restore_snapshot(&mut self, id: u32) -> Result<Commit> {
        let (snapshot, name) = self
            .history
            .snapshot(id)
            .map(|(snapshot, name)| (snapshot.clone(), name.to_owned()))
            .context("no such history snapshot")?;
        let commit = self.session.restore(&snapshot).await?;
        self.history
            .record_restore(&mut self.session, Some(name), &commit, self.active_page);
        self.reconcile_page();
        Ok(commit)
    }

    pub(crate) fn delete_snapshot(&mut self, id: u32) -> bool {
        self.history.delete_snapshot(id)
    }

    pub(crate) async fn commit_rebased(&mut self, patch: koharu_scene::Patch) -> Result<Rebased> {
        let current = self.snapshot();
        let patch = match patch.rebase_on(&current) {
            Ok(patch) => patch,
            Err(
                error @ (koharu_scene::Error::PatchConflict(_)
                | koharu_scene::Error::EntityNotFound(_)
                | koharu_scene::Error::RelationNotFound(_)),
            ) => {
                tracing::debug!(%error, "pipeline output was superseded by a document edit");
                return Ok(Rebased::Superseded);
            }
            Err(error) => return Err(error.into()),
        };
        if patch.is_empty() {
            return Ok(Rebased::Unchanged);
        }
        Ok(Rebased::Applied(self.commit(patch).await?))
    }

    async fn commit(&mut self, patch: koharu_scene::Patch) -> Result<Commit> {
        Ok(self.session.commit(patch).await?)
    }

    pub(crate) fn page(snapshot: &Snapshot, page: EntityId) -> Result<Page> {
        let value = snapshot.page(page)?.page()?;
        let source = Self::asset_id(snapshot, page, "source")?;
        let mut layers = Vec::new();
        Self::collect_layer_views(snapshot, page, &mut layers)?;
        if let Some(image) = source.clone() {
            layers.insert(
                0,
                Layer::Artwork {
                    id: page,
                    parent: None,
                    geometry: Geometry {
                        points: vec![
                            Point { x: 0.0, y: 0.0 },
                            Point {
                                x: value.width,
                                y: 0.0,
                            },
                            Point {
                                x: value.width,
                                y: value.height,
                            },
                            Point {
                                x: 0.0,
                                y: value.height,
                            },
                        ],
                    },
                    visibility: LayerVisibility {
                        visible: true,
                        opacity: 1.0,
                    },
                    image,
                },
            );
        }
        let censor_mask_asset = Self::asset_id(snapshot, page, "censor-mask")?;
        let regions = snapshot
            .descendants(page)?
            .map(|entity| {
                let id = entity.id();
                snapshot
                    .component::<SceneRegion>(id)?
                    .map(|region| Self::analysis_region(snapshot, id, region))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Page {
            id: page,
            label: value.label,
            size: PageSize {
                width: value.width,
                height: value.height,
            },
            censor_mask_asset,
            layers,
            regions,
        })
    }

    fn layer_view(snapshot: &Snapshot, layer: EntityId) -> Result<Layer> {
        let parent = snapshot.parent(layer)?;
        let visibility = Self::layer_visibility(snapshot, layer)?;
        if let Some(group) = snapshot.component::<SceneGroup>(layer)? {
            return Ok(Layer::Group {
                id: layer,
                parent,
                visibility,
                name: group.name,
                role: snapshot
                    .component::<SceneTextGroup>(layer)?
                    .map(|_| GroupRole::Text),
            });
        }
        if let Some(layout) = snapshot.component::<SceneTextLayout>(layer)? {
            let text_layer = snapshot.text_layer(layer)?;
            let content = text_layer.content()?;
            let source = content.source()?.map(|source| SourceText {
                text: source.text.value,
                language: source.language.map(|language| language.to_string()),
            });
            let translation = content.translation()?.map(|translation| Translation {
                text: translation.text.value,
                language: translation.language.map(|language| language.to_string()),
            });
            let role = content.role()?.map(|role| role.role);
            let source_region = content.source_region()?.map(|region| region.id());
            let automatic_region = text_layer.automatic_target()?.map(|region| region.id());
            let typography = text_layer.typography()?.map(Self::typography_view);
            return Ok(Layer::Text {
                id: layer,
                parent,
                geometry: snapshot
                    .component::<SceneGeometry>(layer)?
                    .map(Self::geometry_view),
                visibility,
                content: Box::new(TextContent {
                    id: content.id(),
                    source,
                    translation,
                    role,
                    source_region,
                }),
                typography,
                layout: layout.kind,
                automatic_region,
            });
        }
        if let Some(raster) = snapshot.component::<SceneRasterLayer>(layer)? {
            return Ok(Layer::Raster {
                id: layer,
                parent,
                visibility,
                image: Self::asset_id(snapshot, layer, "source")?,
                name: raster.name,
                kind: raster.kind,
            });
        }
        let geometry = snapshot
            .component::<SceneGeometry>(layer)?
            .map(Self::geometry_view)
            .context("image layer has no geometry")?;
        let image = Self::asset_id(snapshot, layer, "source")?
            .context("image layer has no source asset")?;
        Ok(Layer::Image {
            id: layer,
            parent,
            geometry,
            visibility,
            image,
        })
    }

    fn analysis_region(
        snapshot: &Snapshot,
        id: EntityId,
        region: SceneRegion,
    ) -> Result<AnalysisRegion> {
        let geometry = snapshot
            .component::<SceneGeometry>(id)?
            .map(Self::geometry_view)
            .context("analysis region has no geometry")?;
        Ok(AnalysisRegion {
            id,
            parent: snapshot.parent(id)?,
            geometry,
            kind: region.kind.as_str().to_owned(),
            label: region.label,
        })
    }

    fn layer_visibility(snapshot: &Snapshot, layer: EntityId) -> Result<LayerVisibility> {
        Ok(snapshot.component::<SceneVisibility>(layer)?.map_or(
            LayerVisibility {
                visible: true,
                opacity: 1.0,
            },
            |visibility| LayerVisibility {
                visible: visibility.visible,
                opacity: visibility.opacity,
            },
        ))
    }

    fn typography_view(typography: SceneTypography) -> Typography {
        let writing_mode = if matches!(&typography.origin, Origin::User) {
            typography.writing_mode
        } else {
            None
        };
        Typography {
            preferred_font: typography.preferred_font,
            font_weight: typography.font_weight,
            font_style: typography.font_style,
            size: typography.size,
            auto_fit: typography.auto_fit,
            color: typography.color,
            stroke_color: typography.stroke_color,
            stroke_width: typography.stroke_width,
            alignment: typography.alignment,
            writing_mode,
        }
    }

    fn geometry_view(geometry: SceneGeometry) -> Geometry {
        Geometry {
            points: geometry
                .points
                .into_iter()
                .map(|point| Point {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
        }
    }

    fn asset_id(snapshot: &Snapshot, entity: EntityId, role: &str) -> Result<Option<String>> {
        Ok(snapshot
            .asset(entity, &AssetRole::new(role)?)?
            .map(|asset| asset.blob.to_string()))
    }

    fn is_layer(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
        Ok(snapshot.component::<SceneGroup>(entity)?.is_some()
            || snapshot.component::<SceneTextLayout>(entity)?.is_some()
            || snapshot.component::<SceneRasterLayer>(entity)?.is_some()
            || (snapshot.component::<SceneGeometry>(entity)?.is_some()
                && snapshot
                    .asset(entity, &AssetRole::new("source")?)?
                    .is_some()))
    }

    fn is_content_layer(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
        Ok(
            snapshot.component::<SceneGroup>(entity)?.is_none()
                && Self::is_layer(snapshot, entity)?,
        )
    }

    fn collect_layer_views(
        snapshot: &Snapshot,
        parent: EntityId,
        layers: &mut Vec<Layer>,
    ) -> Result<()> {
        for child in snapshot.children(parent)? {
            if !Self::is_layer(snapshot, child)? {
                continue;
            }
            let group = snapshot.component::<SceneGroup>(child)?.is_some();
            layers.push(Self::layer_view(snapshot, child)?);
            if group {
                Self::collect_layer_views(snapshot, child, layers)?;
            }
        }
        Ok(())
    }

    fn owning_page(snapshot: &Snapshot, mut entity: EntityId) -> Result<EntityId> {
        while let Some(parent) = snapshot.parent(entity)? {
            entity = parent;
        }
        Ok(entity)
    }

    fn collect_clipboard_blobs(
        snapshot: &Snapshot,
        layer: EntityId,
        source: &AssetRole,
        blobs: &mut HashSet<BlobId>,
    ) -> Result<()> {
        if snapshot.component::<SceneGroup>(layer)?.is_some() {
            for child in snapshot.children(layer)? {
                if Self::is_layer(snapshot, child)? {
                    Self::collect_clipboard_blobs(snapshot, child, source, blobs)?;
                }
            }
        } else if snapshot.component::<SceneRasterLayer>(layer)?.is_some()
            && let Some(asset) = snapshot.asset(layer, source)?
        {
            blobs.insert(asset.blob);
        }
        Ok(())
    }

    fn clipboard_layer(
        snapshot: &Snapshot,
        layer: EntityId,
        source: &AssetRole,
        images: &HashMap<BlobId, Vec<u8>>,
    ) -> Result<ClipboardLayer> {
        let visibility = Self::layer_visibility(snapshot, layer)?;
        if let Some(group) = snapshot.component::<SceneGroup>(layer)? {
            let children = snapshot
                .children(layer)?
                .filter(|child| Self::is_layer(snapshot, *child).unwrap_or(false))
                .map(|child| Self::clipboard_layer(snapshot, child, source, images))
                .collect::<Result<Vec<_>>>()?;
            return Ok(ClipboardLayer::Group {
                name: group.name,
                visibility,
                children,
            });
        }
        if let Some(layout) = snapshot.component::<SceneTextLayout>(layer)? {
            let text = snapshot.text_layer(layer)?;
            let content = text.content()?;
            return Ok(ClipboardLayer::Text {
                geometry: text.frame()?.map(Self::geometry_view),
                visibility,
                content: ClipboardTextContent {
                    source: content.source()?.map(|source| SourceText {
                        text: source.text.value,
                        language: source.language.map(|language| language.to_string()),
                    }),
                    translation: content.translation()?.map(|translation| Translation {
                        text: translation.text.value,
                        language: translation.language.map(|language| language.to_string()),
                    }),
                    role: content.role()?.map(|role| role.role),
                },
                typography: text.typography()?.map(Self::typography_view),
                layout: layout.kind,
            });
        }
        if let Some(raster) = snapshot.component::<SceneRasterLayer>(layer)? {
            let image = snapshot
                .asset(layer, source)?
                .map(|asset| {
                    anyhow::Ok(ClipboardRasterImage {
                        bytes: images
                            .get(&asset.blob)
                            .context("clipboard image was not loaded")?
                            .clone(),
                        media_type: asset.media_type,
                        metadata: asset.metadata,
                    })
                })
                .transpose()?;
            return Ok(ClipboardLayer::Raster {
                visibility,
                image,
                name: raster.name,
                kind: raster.kind,
            });
        }
        bail!("only groups, text layers and raster paint layers can be copied")
    }

    fn validate_clipboard_layer(
        layer: &ClipboardLayer,
        page_width: f64,
        page_height: f64,
    ) -> Result<()> {
        match layer {
            ClipboardLayer::Group { children, .. } => {
                for child in children {
                    Self::validate_clipboard_layer(child, page_width, page_height)?;
                }
            }
            ClipboardLayer::Raster {
                image: Some(image), ..
            } if image.metadata.width != Some(page_width.round() as u32)
                || image.metadata.height != Some(page_height.round() as u32) =>
            {
                bail!("raster layers can only be pasted onto pages with matching dimensions");
            }
            _ => {}
        }
        Ok(())
    }

    fn paste_clipboard_layer(
        snapshot: &Snapshot,
        edit: &mut koharu_scene::Edit,
        page: EntityId,
        parent: Option<EntityId>,
        layer: &ClipboardLayer,
        offset: Point,
    ) -> koharu_scene::Result<EntityId> {
        let visibility = |visibility: LayerVisibility| SceneVisibility {
            origin: Origin::User,
            visible: visibility.visible,
            opacity: visibility.opacity,
        };
        match layer {
            ClipboardLayer::Group {
                name,
                visibility: layer_visibility,
                children,
                ..
            } => {
                let group = edit.add_entity(parent.unwrap_or(page), At::End)?;
                edit.set(
                    group,
                    &SceneGroup {
                        origin: Origin::User,
                        name: name.clone(),
                    },
                )?;
                edit.set(group, &visibility(*layer_visibility))?;
                for child in children {
                    Self::paste_clipboard_layer(snapshot, edit, page, Some(group), child, offset)?;
                }
                Ok(group)
            }
            ClipboardLayer::Text {
                geometry,
                visibility: layer_visibility,
                content,
                typography,
                layout,
                ..
            } => {
                if parent.is_some() {
                    return Err(koharu_scene::Error::Invalid(
                        "text layers must belong to the page text group".to_owned(),
                    ));
                }
                let text_content = edit.add_text_content(page, At::End)?;
                if let Some(source) = &content.source {
                    edit.set(
                        text_content,
                        &SceneSourceText {
                            text: Authored::user(source.text.clone()),
                            language: source.language.as_deref().map(str::parse).transpose()?,
                        },
                    )?;
                }
                if let Some(translation) = &content.translation {
                    edit.set(
                        text_content,
                        &SceneTranslation {
                            text: Authored::user(translation.text.clone()),
                            language: translation
                                .language
                                .as_deref()
                                .map(str::parse)
                                .transpose()?,
                        },
                    )?;
                }
                if let Some(role) = &content.role {
                    edit.set(
                        text_content,
                        &SceneTextRole {
                            origin: Origin::User,
                            role: role.clone(),
                        },
                    )?;
                }
                let text_layer = edit.add_text_layer(
                    page,
                    At::End,
                    text_content,
                    &SceneTextLayout {
                        origin: Origin::User,
                        kind: *layout,
                    },
                )?;
                if let Some(geometry) = geometry {
                    edit.set(
                        text_layer,
                        &SceneGeometry {
                            origin: Origin::User,
                            points: geometry
                                .points
                                .iter()
                                .map(|point| ScenePoint {
                                    x: point.x + offset.x,
                                    y: point.y + offset.y,
                                })
                                .collect(),
                        },
                    )?;
                }
                if let Some(typography) = typography {
                    edit.set(
                        text_layer,
                        &SceneTypography {
                            origin: Origin::User,
                            preferred_font: typography.preferred_font.clone(),
                            font_weight: typography.font_weight,
                            font_style: typography.font_style,
                            size: typography.size.filter(|size| *size > 0.0),
                            auto_fit: typography.auto_fit,
                            color: typography.color,
                            stroke_color: typography.stroke_color,
                            stroke_width: typography.stroke_width,
                            alignment: typography.alignment,
                            writing_mode: typography.writing_mode,
                            extensions: Default::default(),
                        },
                    )?;
                }
                edit.set(text_layer, &visibility(*layer_visibility))?;
                Ok(text_layer)
            }
            ClipboardLayer::Raster {
                visibility: layer_visibility,
                image,
                name,
                kind,
                ..
            } => {
                let at = if parent.is_none() {
                    snapshot
                        .page(page)?
                        .text_group()?
                        .map_or(At::End, |group| At::Before(group.id()))
                } else {
                    At::End
                };
                let raster = edit.add_entity(parent.unwrap_or(page), at)?;
                edit.set(
                    raster,
                    &SceneRasterLayer {
                        origin: Origin::User,
                        name: name.clone(),
                        kind: *kind,
                    },
                )?;
                edit.set(raster, &visibility(*layer_visibility))?;
                if let Some(image) = image {
                    edit.set_asset(
                        raster,
                        &AssetRole::new("source")?,
                        AssetInput::new(
                            image.bytes.clone(),
                            image.media_type.clone(),
                            image.metadata.clone(),
                        ),
                    )?;
                }
                Ok(raster)
            }
        }
    }

    fn text_content(snapshot: &Snapshot, layer: EntityId) -> Result<EntityId> {
        snapshot
            .text_layer(layer)?
            .content()
            .map(|content| content.id())
            .map_err(Into::into)
    }

    fn placement(siblings: &[EntityId], moving: EntityId, index: usize) -> At {
        siblings
            .iter()
            .copied()
            .filter(|entity| *entity != moving)
            .nth(index)
            .map_or(At::End, At::Before)
    }

    fn unique_roots(snapshot: &Snapshot, entities: Vec<EntityId>) -> Result<Vec<EntityId>> {
        let selected = entities.into_iter().collect::<HashSet<_>>();
        let mut roots = Vec::new();
        for entity in selected.iter().copied() {
            let mut parent = snapshot.parent(entity)?;
            let mut nested = false;
            while let Some(value) = parent {
                if selected.contains(&value) {
                    nested = true;
                    break;
                }
                parent = snapshot.parent(value)?;
            }
            if !nested {
                roots.push(entity);
            }
        }
        roots.sort_unstable();
        Ok(roots)
    }

    fn geometry_from_frame(frame: Frame) -> Result<SceneGeometry> {
        if !frame.x.is_finite()
            || !frame.y.is_finite()
            || !frame.width.is_finite()
            || !frame.height.is_finite()
            || !frame.angle_degrees.is_finite()
            || frame.width <= 0.0
            || frame.height <= 0.0
        {
            bail!("frame must contain finite coordinates and positive dimensions");
        }
        let center_x = f64::from(frame.x + frame.width * 0.5);
        let center_y = f64::from(frame.y + frame.height * 0.5);
        let half_width = f64::from(frame.width) * 0.5;
        let half_height = f64::from(frame.height) * 0.5;
        let (sin, cos) = f64::from(frame.angle_degrees).to_radians().sin_cos();
        Ok(SceneGeometry {
            origin: Origin::User,
            points: [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            ]
            .map(|(x, y)| ScenePoint {
                x: center_x + x * cos - y * sin,
                y: center_y + x * sin + y * cos,
            })
            .into(),
        })
    }
}

fn validate_project_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        bail!("project name cannot be empty");
    }
    if name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|character| character.is_control() || r#"<>:"/\|?*"#.contains(character))
    {
        bail!("project name contains characters that cannot be used in a file name");
    }
    let stem = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|number| {
                matches!(number, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
            })
    {
        bail!("project name is reserved by Windows");
    }
    Ok(name.to_owned())
}

fn rasterize_stroke(
    image: &mut RgbaImage,
    mode: RasterStrokeMode,
    color: [u8; 4],
    diameter: f32,
    points: &[ScenePoint],
) {
    let radius = f64::from(diameter) * 0.5;
    for (start, end) in points
        .iter()
        .zip(points.iter().skip(1))
        .chain(points.last().map(|point| (point, point)))
    {
        let left = (start.x.min(end.x) - radius - 0.5).floor().max(0.0) as u32;
        let top = (start.y.min(end.y) - radius - 0.5).floor().max(0.0) as u32;
        let right = (start.x.max(end.x) + radius + 0.5)
            .ceil()
            .min(f64::from(image.width())) as u32;
        let bottom = (start.y.max(end.y) + radius + 0.5)
            .ceil()
            .min(f64::from(image.height())) as u32;
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length_squared = dx * dx + dy * dy;
        for y in top..bottom {
            for x in left..right {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let t = if length_squared == 0.0 {
                    0.0
                } else {
                    (((px - start.x) * dx + (py - start.y) * dy) / length_squared).clamp(0.0, 1.0)
                };
                let distance =
                    ((px - (start.x + t * dx)).powi(2) + (py - (start.y + t * dy)).powi(2)).sqrt();
                let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0) as f32;
                if coverage == 0.0 {
                    continue;
                }
                let pixel = image.get_pixel_mut(x, y);
                match mode {
                    RasterStrokeMode::Paint => {
                        let source_alpha = f32::from(color[3]) / 255.0 * coverage;
                        let destination_alpha = f32::from(pixel[3]) / 255.0;
                        let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
                        if output_alpha > 0.0 {
                            for channel in 0..3 {
                                let source = f32::from(color[channel]) / 255.0;
                                let destination = f32::from(pixel[channel]) / 255.0;
                                pixel[channel] = (((source * source_alpha
                                    + destination * destination_alpha * (1.0 - source_alpha))
                                    / output_alpha)
                                    * 255.0)
                                    .round() as u8;
                            }
                        }
                        pixel[3] = (output_alpha * 255.0).round() as u8;
                    }
                    RasterStrokeMode::Erase => {
                        pixel[3] = (f32::from(pixel[3]) * (1.0 - coverage)).round() as u8;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{Generation, ProducerId, WritingMode};

    use super::*;

    #[test]
    fn only_user_authored_direction_is_projected_as_an_override() {
        let mut typography = SceneTypography {
            origin: Origin::Generated(Generation::new(
                ProducerId::new("dev.koharu.pipeline.detection").expect("valid producer"),
            )),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: None,
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(WritingMode::Vertical),
            extensions: Default::default(),
        };

        assert_eq!(
            Project::typography_view(typography.clone()).writing_mode,
            None
        );
        typography.origin = Origin::User;
        assert_eq!(
            Project::typography_view(typography).writing_mode,
            Some(WritingMode::Vertical)
        );
    }

    #[tokio::test]
    async fn pipeline_commit_rebases_or_yields_to_the_manual_edit() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let pipeline_page = setup
            .add_page(PageDraft::new("pipeline", 100.0, 100.0), At::End)
            .unwrap();
        let manual_page = setup
            .add_page(PageDraft::new("manual", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let base = project.snapshot();
        let pipeline = base
            .patch(|edit| {
                edit.set_page(
                    pipeline_page,
                    PageDraft::new("pipeline result", 100.0, 100.0),
                )
            })
            .unwrap();
        let manual = base
            .patch(|edit| edit.set_page(manual_page, PageDraft::new("manual edit", 100.0, 100.0)))
            .unwrap();
        project.commit(manual).await.unwrap();

        let Rebased::Applied(commit) = project.commit_rebased(pipeline).await.unwrap() else {
            panic!("a non-conflicting pipeline result must apply");
        };
        assert_eq!(
            commit
                .snapshot
                .page(pipeline_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "pipeline result"
        );
        assert_eq!(
            commit
                .snapshot
                .page(manual_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "manual edit"
        );

        let base = project.snapshot();
        let pipeline = base
            .patch(|edit| {
                edit.set_page(
                    pipeline_page,
                    PageDraft::new("stale pipeline", 100.0, 100.0),
                )
            })
            .unwrap();
        let manual = base
            .patch(|edit| {
                edit.set_page(pipeline_page, PageDraft::new("latest manual", 100.0, 100.0))
            })
            .unwrap();
        project.commit(manual).await.unwrap();

        assert!(matches!(
            project.commit_rebased(pipeline).await.unwrap(),
            Rebased::Superseded
        ));
        assert_eq!(
            project
                .snapshot()
                .page(pipeline_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "latest manual"
        );
    }

    #[test]
    fn raster_strokes_are_continuous_and_erasable() {
        let mut image = RgbaImage::new(32, 16);
        let points = [
            ScenePoint { x: 4.0, y: 8.0 },
            ScenePoint { x: 28.0, y: 8.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [210, 40, 20, 255],
            5.0,
            &points,
        );
        for x in 4..28 {
            assert_eq!(image.get_pixel(x, 8).0, [210, 40, 20, 255]);
        }

        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Erase,
            [0, 0, 0, 0],
            5.0,
            &[ScenePoint { x: 16.0, y: 8.0 }],
        );
        assert_eq!(image.get_pixel(16, 8)[3], 0);
        assert_eq!(image.get_pixel(4, 8)[3], 255);

        let mut white = RgbaImage::new(4, 4);
        rasterize_stroke(
            &mut white,
            RasterStrokeMode::Paint,
            [255, 255, 255, 255],
            3.0,
            &[ScenePoint { x: 2.0, y: 2.0 }],
        );
        assert_eq!(white.get_pixel(2, 2).0, [255, 255, 255, 255]);
    }

    #[tokio::test]
    async fn changed_source_invalidates_generated_translation_but_keeps_manual_translation() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let (_, layer) = project
            .add_point_text(page, Point { x: 10.0, y: 10.0 })
            .await
            .unwrap();
        project
            .set_source_text(layer, "original".into())
            .await
            .unwrap();
        let content = Project::text_content(&project.snapshot(), layer).unwrap();
        let generation =
            Generation::new(ProducerId::new("dev.koharu.pipeline.translation").unwrap());
        let mut translated = project.snapshot().edit_as(generation.clone());
        translated
            .set(
                content,
                &SceneTranslation {
                    text: Authored::generated("generated".into(), generation),
                    language: None,
                },
            )
            .unwrap();
        project.commit(translated.finish().unwrap()).await.unwrap();
        project
            .set_source_text(layer, "original".into())
            .await
            .unwrap();
        assert_eq!(
            project
                .snapshot()
                .component::<SceneTranslation>(content)
                .unwrap()
                .unwrap()
                .text
                .value,
            "generated"
        );
        project
            .set_source_text(layer, "corrected".into())
            .await
            .unwrap();
        assert!(
            project
                .snapshot()
                .component::<SceneTranslation>(content)
                .unwrap()
                .is_none()
        );
        project
            .set_translation(layer, Some("manual".into()))
            .await
            .unwrap();
        project
            .set_source_text(layer, "another correction".into())
            .await
            .unwrap();
        assert_eq!(
            project
                .snapshot()
                .component::<SceneTranslation>(content)
                .unwrap()
                .unwrap()
                .text
                .value,
            "manual"
        );
    }

    #[tokio::test]
    async fn pasted_text_keeps_its_content_and_is_offset_only_on_its_source_page() {
        fn texts(project: &Project, page: EntityId) -> Vec<(EntityId, String, String, f64)> {
            Project::page(&project.snapshot(), page)
                .unwrap()
                .layers
                .into_iter()
                .filter_map(|layer| match layer {
                    Layer::Text {
                        id,
                        geometry,
                        content,
                        ..
                    } => Some((
                        id,
                        content.source?.text,
                        content.translation?.text,
                        geometry?.points.first()?.x,
                    )),
                    _ => None,
                })
                .collect()
        }

        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let first = setup
            .add_page(PageDraft::new("first", 100.0, 100.0), At::End)
            .unwrap();
        let second = setup
            .add_page(PageDraft::new("second", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let (_, layer) = project
            .add_point_text(first, Point { x: 10.0, y: 10.0 })
            .await
            .unwrap();
        project
            .set_source_text(layer, "source".into())
            .await
            .unwrap();
        project
            .set_translation(layer, Some("перевод".into()))
            .await
            .unwrap();
        let [(_, _, _, original_x)] = texts(&project, first)[..] else {
            panic!("the first page has one text layer");
        };

        let copied = project.copy_layers(vec![layer]).await.unwrap();
        assert_eq!(copied.page, first);
        project
            .paste_layers(second, copied.clone(), Point { x: 0.0, y: 0.0 })
            .await
            .unwrap();
        let [(pasted, ref source, ref translation, x)] = texts(&project, second)[..] else {
            panic!("the second page received one text layer");
        };
        assert_ne!(pasted, layer);
        assert_eq!(
            (source.as_str(), translation.as_str()),
            ("source", "перевод")
        );
        assert_eq!(x, original_x);

        let before = project.snapshot().revision();
        let (commit, _) = project
            .paste_layers(first, copied, Point { x: 16.0, y: 16.0 })
            .await
            .unwrap();
        assert_eq!(commit.changes.from, before, "a paste is one commit");
        let pasted_on_source = texts(&project, first);
        assert_eq!(pasted_on_source.len(), 2);
        assert!(
            pasted_on_source
                .iter()
                .any(|(_, _, _, x)| *x == original_x + 16.0)
        );
    }
}
