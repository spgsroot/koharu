//! Administrative orders dashboard commands: CRUD, stages, notes, activity
//! and the `.xlsx`/`.csv` translation exchange. Orders live in a standalone
//! SQLite store; projects stay editor documents. A parsed import file never
//! touches the project — only `apply_translation_import` commits, atomically,
//! as exactly one history state.

use std::path::{Path as StdPath, PathBuf};

use anyhow::{Context as _, Result, bail};
use koharu_desktop::Desktop;
use koharu_orders::{
    ActivityEntry, ImportMeta, Lifecycle, NewNote, NewOrder, NoteKind, OrderDetail, OrderFilter,
    OrderListItem, OrderNote, OrderPatch, OrdersSummary, StageName, StageState, Store,
    TranslationImport,
};
use koharu_scene::EntityId;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{Cef, State, WebviewWindow};

use super::{
    ChannelExt as _, Error,
    canvas::CanvasChannel,
    editing::synchronize_canvas,
    lifecycle::ProjectChannel,
    project::{CurrentProject, Project, ProjectLibrary},
};
use crate::history::HistoryName;

/// Managed orders state: the store, the local actor, and the directory where
/// uploaded translation files are kept as checksum-addressed attachments.
pub(crate) struct Orders {
    pub(crate) store: Store,
    actor: String,
    files: PathBuf,
}

impl Orders {
    pub(crate) fn new(root: &StdPath) -> Result<Self> {
        let store =
            Store::open(&root.join("orders.sqlite3")).context("failed to open the orders store")?;
        let actor = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "local".into());
        Ok(Self {
            store,
            actor,
            files: root.join("order-files"),
        })
    }

    /// Local user name stamped onto every audited action.
    pub(crate) fn actor(&self) -> &str {
        &self.actor
    }

    /// Import attachment path; one file per attempt, never overwritten.
    fn import_file(&self, order_id: &str, import_id: &str, filename: &str) -> PathBuf {
        let filename = filename
            .chars()
            .map(|character| {
                if character.is_alphanumeric() || matches!(character, '.' | '-' | '_') {
                    character
                } else {
                    '_'
                }
            })
            .collect::<String>();
        self.files
            .join(order_id)
            .join(format!("{import_id}-{filename}"))
    }

    fn detail(&self, library: &ProjectLibrary, order_id: &str) -> Result<OrderDetail, Error> {
        let detail = self
            .store
            .order_detail_available(order_id, &available_projects(library))?;
        Ok(detail.context("order not found")?)
    }
}

fn available_projects(library: &ProjectLibrary) -> std::collections::HashSet<String> {
    library
        .list()
        .map(|projects| projects.into_iter().map(|project| project.name).collect())
        .unwrap_or_default()
}

/// Local-day bounds from the client (the UI owns the time zone). Falls back
/// to the UTC day when absent.
#[derive(Clone, Copy, Debug, Deserialize, Type)]
pub struct DayBounds {
    #[specta(type = f64)]
    pub start: i64,
    #[specta(type = f64)]
    pub end: i64,
}

fn clock(today: Option<DayBounds>) -> koharu_orders::Clock {
    let now = koharu_orders::now();
    let (start, end) = today
        .map(|bounds| (bounds.start, bounds.end))
        .unwrap_or_else(|| {
            let millis_of_day = now.rem_euclid(86_400_000);
            let start = now - millis_of_day;
            (start, start + 86_400_000)
        });
    koharu_orders::Clock::new(now, start, end)
}

/// Syncs computed page counters of the open project into its linked order.
pub(crate) fn sync_progress(orders: &Orders, project: &Project) {
    if let Ok(Some(order_id)) = orders.store.order_for_project(&project.name)
        && let Ok(counts) = Project::page_counts(&project.snapshot())
    {
        let _ = orders.store.set_progress(&order_id, counts);
    }
}

/// Read-through freshness: whenever the open project backs an order, its
/// counters are recomputed from the live snapshot before serving dashboard
/// reads — so undo/redo/history jumps and pipeline commits never drift.
async fn sync_open_project(current: &State<'_, CurrentProject>, orders: &State<'_, Orders>) {
    let project = current.project.lock().await;
    if let Some(project) = project.as_ref() {
        sync_progress(orders, project);
    }
}

/// Locks the open project and proves it is the one linked to the order.
async fn lock_order_project<'a>(
    orders: &Orders,
    order_id: &str,
    current: &'a CurrentProject,
) -> Result<tokio::sync::MappedMutexGuard<'a, Project>> {
    let order = orders
        .store
        .get_order(order_id)?
        .with_context(|| format!("order {order_id} not found"))?;
    let project_name = order
        .project_name
        .clone()
        .context("the order has no linked project")?;
    let guard = current.project.lock().await;
    tokio::sync::MutexGuard::try_map(guard, |project: &mut Option<Project>| {
        project
            .as_mut()
            .filter(|project| project.name == project_name)
    })
    .map_err(|_| {
        anyhow::anyhow!(
            "open the order's project \"{project_name}\" before exchanging translations"
        )
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_orders(
    filter: OrderFilter,
    today: Option<DayBounds>,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
    current: State<'_, CurrentProject>,
) -> std::result::Result<Vec<OrderListItem>, Error> {
    sync_open_project(&current, &orders).await;
    let mut filter = filter;
    if filter.view == koharu_orders::OrderView::Mine && filter.actor.is_none() {
        filter.actor = Some(orders.actor.clone());
    }
    Ok(orders
        .store
        .list_orders(&filter, &clock(today), &available_projects(&library))?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn orders_summary(
    today: Option<DayBounds>,
    orders: State<'_, Orders>,
    current: State<'_, CurrentProject>,
) -> std::result::Result<OrdersSummary, Error> {
    sync_open_project(&current, &orders).await;
    Ok(orders.store.summary(&clock(today))?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn orders_actor(orders: State<'_, Orders>) -> std::result::Result<String, Error> {
    Ok(orders.actor.clone())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_order(
    id: String,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
    current: State<'_, CurrentProject>,
) -> std::result::Result<OrderDetail, Error> {
    sync_open_project(&current, &orders).await;
    orders.detail(&library, &id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn create_order(
    input: NewOrder,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    let detail = orders.store.create_order(input, &orders.actor)?;
    orders.detail(&library, &detail.item.order.id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn update_order(
    id: String,
    patch: OrderPatch,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    orders.store.update_order(&id, patch, &orders.actor)?;
    orders.detail(&library, &id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_order_lifecycle(
    id: String,
    lifecycle: Lifecycle,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    orders.store.set_lifecycle(&id, lifecycle, &orders.actor)?;
    orders.detail(&library, &id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn attach_order_project(
    id: String,
    project_name: Option<String>,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
    current: State<'_, CurrentProject>,
) -> std::result::Result<OrderDetail, Error> {
    orders
        .store
        .attach_project(&id, project_name, &orders.actor)?;
    // When the freshly linked project is already open, sync its counters now.
    let project = current.project.lock().await;
    if let Some(project) = project.as_ref() {
        sync_progress(&orders, project);
    }
    orders.detail(&library, &id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_order_stage(
    id: String,
    stage: StageName,
    state: StageState,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    // Open blockers freeze the stage route; resolving notes unfreezes it.
    if orders.store.open_blockers(&id)? > 0 {
        return Err(anyhow::anyhow!(
            "the order has unresolved blocker notes; resolve them before moving stages"
        )
        .into());
    }
    orders.store.set_stage(&id, stage, state, &orders.actor)?;
    orders.detail(&library, &id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn add_order_note(
    order_id: String,
    input: NewNote,
    block_stage: bool,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    let input = NewNote {
        author: orders.actor.clone(),
        ..input
    };
    orders.store.add_note(&order_id, input)?;
    // A confirmed blocker marks the current stage as blocked.
    if block_stage && let Some(detail) = orders.store.order_detail(&order_id)? {
        orders.store.set_stage(
            &order_id,
            detail.item.current_stage,
            StageState::Blocked,
            &orders.actor,
        )?;
    }
    orders.detail(&library, &order_id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn resolve_order_note(
    note_id: String,
    resolved: bool,
    orders: State<'_, Orders>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    let order_id = orders
        .store
        .set_note_resolved(&note_id, resolved, &orders.actor)?;
    // Closing the last blocker unblocks the stage; it never auto-completes.
    if resolved
        && orders.store.open_blockers(&order_id)? == 0
        && let Some(detail) = orders.store.order_detail(&order_id)?
        && detail.item.current_stage_state == StageState::Blocked
    {
        orders.store.set_stage(
            &order_id,
            detail.item.current_stage,
            StageState::InProgress,
            &orders.actor,
        )?;
    }
    orders.detail(&library, &order_id)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_order_note(
    note_id: String,
    orders: State<'_, Orders>,
) -> std::result::Result<(), Error> {
    Ok(orders.store.delete_note(&note_id, &orders.actor)?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_order_notes(
    order_id: String,
    orders: State<'_, Orders>,
) -> std::result::Result<Vec<OrderNote>, Error> {
    Ok(orders.store.list_notes(&order_id)?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_order_activity(
    order_id: String,
    orders: State<'_, Orders>,
) -> std::result::Result<Vec<ActivityEntry>, Error> {
    Ok(orders.store.list_activity(&order_id, 200)?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_order_imports(
    order_id: String,
    orders: State<'_, Orders>,
) -> std::result::Result<Vec<ImportMeta>, Error> {
    Ok(orders
        .store
        .list_imports(&order_id)?
        .into_iter()
        .map(ImportMeta::from)
        .collect())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_translation_import(
    id: String,
    orders: State<'_, Orders>,
) -> std::result::Result<TranslationImport, Error> {
    let import = orders.store.import(&id)?;
    Ok(import.with_context(|| format!("import {id} not found"))?)
}

/// `.xlsx` is the primary exchange format; `.csv` the lossless backup.
#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TemplateFormat {
    Xlsx,
    Csv,
}

impl TemplateFormat {
    fn extension(self) -> &'static str {
        match self {
            Self::Xlsx => "xlsx",
            Self::Csv => "csv",
        }
    }
}

/// Result of picking and parsing a translation file.
#[derive(Clone, Debug, Serialize, Type)]
pub struct ImportBeginResult {
    pub import: TranslationImport,
    /// The identical file was already uploaded for this order.
    pub duplicate: bool,
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "translation_template_exported",
    skip_all,
    fields(origin = "user", format = ?format),
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn export_translation_template(
    order_id: String,
    format: TemplateFormat,
    window: WebviewWindow<Cef>,
    orders: State<'_, Orders>,
    current: State<'_, CurrentProject>,
) -> std::result::Result<Option<String>, Error> {
    let order = orders
        .store
        .get_order(&order_id)?
        .context("order not found")?;
    let bytes = {
        let project = lock_order_project(&orders, &order_id, &current).await?;
        let snapshot = project.snapshot();
        let segments = Project::translation_segments(&snapshot)?;
        if segments.is_empty() {
            return Err(anyhow::anyhow!(
                "the project has no translatable segments yet; run detection and OCR first"
            )
            .into());
        }
        match format {
            TemplateFormat::Xlsx => koharu_orders::template_xlsx(&order.number, &segments)?,
            TemplateFormat::Csv => koharu_orders::template_csv(&order.number, &segments)?,
        }
    };
    let Some(target) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .set_file_name(format!(
            "{}-translations.{}",
            order.number,
            format.extension()
        ))
        .add_filter("Translation file", &[format.extension()])
        .save_file()
        .await
    else {
        return Ok(None);
    };
    write_atomic(target.path(), &bytes)?;
    Ok(Some(target.path().display().to_string()))
}

/// Largest accepted translation upload. Real files are well under 1 MiB.
const MAX_IMPORT_BYTES: u64 = 64 * 1024 * 1024;

#[tracing::instrument(
    target = "koharu_metrics",
    name = "translation_import_started",
    skip_all,
    fields(origin = "user")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn begin_translation_import(
    order_id: String,
    window: WebviewWindow<Cef>,
    orders: State<'_, Orders>,
    current: State<'_, CurrentProject>,
) -> std::result::Result<Option<ImportBeginResult>, Error> {
    let order = orders
        .store
        .get_order(&order_id)?
        .context("order not found")?;
    let Some(picked) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .add_filter("Translation file", &["xlsx", "csv"])
        .pick_file()
        .await
    else {
        return Ok(None);
    };
    let path = picked.path().to_owned();
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("translations.xlsx")
        .to_owned();
    let metadata = std::fs::metadata(&path)?;
    if metadata.len() > MAX_IMPORT_BYTES {
        return Err(anyhow::anyhow!("translation file exceeds 64 MiB").into());
    }
    let bytes = std::fs::read(&path)?;
    let checksum = koharu_orders::checksum(&bytes);
    // Repeated upload of the identical file is recognized, never duplicated.
    if let Some(existing) = orders.store.find_import_by_checksum(&order_id, &checksum)? {
        return Ok(Some(ImportBeginResult {
            import: existing,
            duplicate: true,
        }));
    }
    let report = {
        let project = lock_order_project(&orders, &order_id, &current).await?;
        let segments = Project::translation_segments(&project.snapshot())?;
        koharu_orders::parse_import(&order.number, &filename, &bytes, &segments)
    };
    let import =
        orders
            .store
            .create_import(&order_id, filename.clone(), checksum, report, &orders.actor)?;
    // Keep the original file as a checksum-addressed order attachment.
    let target = orders.import_file(&order_id, &import.id, &filename);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_atomic(&target, &bytes)?;
    Ok(Some(ImportBeginResult {
        import,
        duplicate: false,
    }))
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "translation_import_applied",
    skip_all,
    fields(origin = "user")
)]
#[tauri::command]
#[specta::specta]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn apply_translation_import(
    import_id: String,
    desktop: State<'_, Desktop>,
    orders: State<'_, Orders>,
    current: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
    project_channel: State<'_, ProjectChannel>,
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<OrderDetail, Error> {
    let import = orders
        .store
        .import(&import_id)?
        .context("import not found")?;
    if !import.state.pending() {
        return Err(
            anyhow::anyhow!("import {} and cannot be applied", import.state.as_str()).into(),
        );
    }
    if import.errors > 0 {
        return Err(anyhow::anyhow!(
            "the import has blocking errors; fix the file and upload it again"
        )
        .into());
    }
    if orders.store.open_blockers(&import.order_id)? > 0 {
        return Err(anyhow::anyhow!(
            "the order has unresolved blocker notes; resolve them before applying translations"
        )
        .into());
    }
    let order_id = import.order_id.clone();
    let notes = import
        .report
        .rows
        .iter()
        .filter_map(|row| {
            row.translator_note.as_ref().map(|body| NewNote {
                kind: NoteKind::Internal,
                body: body.clone(),
                author: import.actor.clone(),
                page_id: None,
                segment_id: Some(row.segment_id.clone()),
                pinned: false,
            })
        })
        .collect();
    let outcome = apply_import_commit(&orders, &current, &import, notes).await;
    let applied = match outcome {
        Ok(applied) => applied,
        Err(error) => {
            let _ = orders.store.record_import_failure(
                &import_id,
                &orders.actor,
                &format!("{error:#}"),
            );
            return Err(error.into());
        }
    };

    // Publish canvas/project updates only when a commit actually landed (every
    // row unchanged since the preview is a valid no-op apply).
    if let Some((commit, page, info)) = applied {
        canvas_channel
            .channel
            .publish(synchronize_canvas(&desktop, &commit, page).await?);
        project_channel.channel.publish(Some(info));
    }
    let project = current.project.lock().await;
    if let Some(project) = project.as_ref() {
        sync_progress(&orders, project);
    }
    drop(project);
    orders.detail(&library, &order_id)
}

/// Applies the project patch and administrative transaction as one observable
/// operation. A bookkeeping failure removes the new history state and restores
/// the project before the caller can publish either side.
async fn apply_import_commit(
    orders: &State<'_, Orders>,
    current: &State<'_, CurrentProject>,
    import: &TranslationImport,
    notes: Vec<NewNote>,
) -> Result<
    Option<(
        koharu_scene::Commit,
        Option<EntityId>,
        super::project::ProjectInfo,
    )>,
> {
    let mut project = lock_order_project(orders, &import.order_id, current).await?;
    // Drift check: the project must still match the previewed sources.
    let segments = Project::translation_segments(&project.snapshot())?;
    let current_by_id: std::collections::HashMap<&str, &koharu_orders::SegmentRow> = segments
        .iter()
        .map(|segment| (segment.id.as_str(), segment))
        .collect();
    if current_by_id.len() != import.report.rows.len() {
        bail!("the project segment set changed since the preview; re-upload the translation file");
    }
    let mut updates = Vec::new();
    for row in &import.report.rows {
        let Some(segment) = current_by_id.get(row.segment_id.as_str()) else {
            bail!(
                "segment of row {} no longer exists; re-upload the translation file",
                row.row
            );
        };
        if segment.page != row.page {
            bail!(
                "page of row {} changed since the preview; re-upload the translation file",
                row.row
            );
        }
        if segment.source != row.source_text {
            bail!(
                "source text of row {} changed since the preview; re-upload the translation file",
                row.row
            );
        }
        if segment.translation != row.previous_translation
            && segment.translation.as_deref() != Some(row.translation.as_str())
        {
            bail!(
                "translation of row {} changed since the preview; re-upload the translation file",
                row.row
            );
        }
        if segment.translation.as_deref() == Some(row.translation.as_str()) {
            continue; // unchanged since the preview: nothing to write
        }
        let layer: EntityId = row
            .segment_id
            .parse()
            .with_context(|| format!("invalid segment id in row {}", row.row))?;
        updates.push((layer, Some(row.translation.clone())));
    }
    if updates.is_empty() {
        orders
            .store
            .finalize_import(&import.id, notes, &orders.actor)?;
        return Ok(None);
    }
    let snapshot = project.create_snapshot(Some("Before translation import".into()));
    let commit = project.set_translations_batch(updates).await?;
    project.record_named(HistoryName::ImportTranslations, None, None, &commit);
    if let Err(error) = orders
        .store
        .finalize_import(&import.id, notes, &orders.actor)
    {
        match project.rollback_history_tip().await {
            Ok(_) => {
                project.delete_snapshot(snapshot);
                bail!(
                    "administrative import transaction failed; project changes were rolled back: {error}"
                );
            }
            Err(rollback) => {
                bail!(
                    "administrative import transaction failed ({error}); project rollback also failed ({rollback:#})"
                );
            }
        }
    }
    let info = project.info();
    let page = project.active_page();
    Ok(Some((commit, page, info)))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn discard_translation_import(
    import_id: String,
    orders: State<'_, Orders>,
) -> std::result::Result<(), Error> {
    let import = orders
        .store
        .import(&import_id)?
        .context("import not found")?;
    orders.store.discard_import(&import_id, &orders.actor)?;
    let prefix = format!("{import_id}-");
    let directory = orders.files.join(&import.order_id);
    if let Ok(entries) = std::fs::read_dir(&directory) {
        for entry in entries.flatten() {
            if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix))
            {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn export_import_report(
    import_id: String,
    window: WebviewWindow<Cef>,
    orders: State<'_, Orders>,
) -> std::result::Result<Option<String>, Error> {
    let import = orders
        .store
        .import(&import_id)?
        .context("import not found")?;
    let order = orders
        .store
        .get_order(&import.order_id)?
        .context("order not found")?;
    let bytes = koharu_orders::report_csv(&import.report)?;
    let Some(target) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .set_file_name(format!("{}-import-report.csv", order.number))
        .add_filter("CSV report", &["csv"])
        .save_file()
        .await
    else {
        return Ok(None);
    };
    write_atomic(target.path(), &bytes)?;
    Ok(Some(target.path().display().to_string()))
}

/// Tempfile+rename publication for order-produced files; a crash must never
/// leave a partially written template, upload copy, or report behind.
fn write_atomic(path: &StdPath, bytes: &[u8]) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(StdPath::new(".")))?;
    std::io::Write::write_all(file.as_file_mut(), bytes)?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}
