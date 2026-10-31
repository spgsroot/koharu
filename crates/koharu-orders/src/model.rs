//! Order domain model shared by the store, the exchange format, and the UI.

use serde::{Deserialize, Serialize};
use specta::Type;

/// Unix epoch milliseconds; bridges to TypeScript `number`.
pub type Millis = i64;

/// Current time in [`Millis`].
pub fn now() -> Millis {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as Millis)
        .unwrap_or_default()
}

fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

fn millis<S: serde::Serializer>(value: &Millis, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(*value as f64)
}

fn opt_millis<S: serde::Serializer>(
    value: &Option<Millis>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => serializer.serialize_f64(*value as f64),
        None => serializer.serialize_none(),
    }
}

/// Administrative lifecycle of an order, independent from production stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Lifecycle {
    #[default]
    Active,
    OnHold,
    Completed,
    Delivered,
    Cancelled,
}

impl Lifecycle {
    pub const ALL: [Self; 5] = [
        Self::Active,
        Self::OnHold,
        Self::Completed,
        Self::Delivered,
        Self::Cancelled,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::OnHold => "on_hold",
            Self::Completed => "completed",
            Self::Delivered => "delivered",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Client-visible urgency of an order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    #[default]
    Normal,
    High,
    Urgent,
}

impl Priority {
    pub const ALL: [Self; 3] = [Self::Normal, Self::High, Self::Urgent];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::High => "high",
            Self::Urgent => "urgent",
        }
    }
}

/// Fixed production route. The order of [`StageName::ALL`] defines the pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum StageName {
    Intake,
    Preparation,
    Translation,
    Editing,
    Typesetting,
    Qa,
    Delivery,
}

impl StageName {
    pub const ALL: [Self; 7] = [
        Self::Intake,
        Self::Preparation,
        Self::Translation,
        Self::Editing,
        Self::Typesetting,
        Self::Qa,
        Self::Delivery,
    ];

    /// Stages that must be done before an order is ready to deliver.
    pub const BEFORE_DELIVERY: [Self; 6] = [
        Self::Intake,
        Self::Preparation,
        Self::Translation,
        Self::Editing,
        Self::Typesetting,
        Self::Qa,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Intake => "intake",
            Self::Preparation => "preparation",
            Self::Translation => "translation",
            Self::Editing => "editing",
            Self::Typesetting => "typesetting",
            Self::Qa => "qa",
            Self::Delivery => "delivery",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|stage| stage.as_str() == value)
    }
}

/// Per-stage production state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    #[default]
    NotStarted,
    InProgress,
    Blocked,
    Done,
}

impl StageState {
    pub const ALL: [Self; 4] = [
        Self::NotStarted,
        Self::InProgress,
        Self::Blocked,
        Self::Done,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::InProgress => "in_progress",
            Self::Blocked => "blocked",
            Self::Done => "done",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct StageInfo {
    pub stage: StageName,
    pub state: StageState,
    #[serde(serialize_with = "opt_millis")]
    #[specta(type = Option<f64>)]
    pub started_at: Option<Millis>,
    #[serde(serialize_with = "opt_millis")]
    #[specta(type = Option<f64>)]
    pub completed_at: Option<Millis>,
}

/// Kind of an order note. Blockers pin to the top and gate attention.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum NoteKind {
    #[default]
    Internal,
    Client,
    Blocker,
    Qa,
}

impl NoteKind {
    pub const ALL: [Self; 4] = [Self::Internal, Self::Client, Self::Blocker, Self::Qa];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Internal => "internal",
            Self::Client => "client",
            Self::Blocker => "blocker",
            Self::Qa => "qa",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == value)
    }
}

/// Lifecycle of one translation import attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ImportState {
    /// Parsed; warnings only or clean. Awaiting the admin's decision.
    Ready,
    /// Reserved for warnings-only previews; ready and needs_attention both apply.
    NeedsAttention,
    /// File-level or row-level errors; apply is blocked until a fixed file is uploaded.
    Failed,
    /// Applied to the project atomically.
    Applied,
}

impl ImportState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NeedsAttention => "needs_attention",
            Self::Failed => "failed",
            Self::Applied => "applied",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        [
            Self::Ready,
            Self::NeedsAttention,
            Self::Failed,
            Self::Applied,
        ]
        .into_iter()
        .find(|state| state.as_str() == value)
    }

    /// True while the import awaits a human decision.
    pub const fn pending(self) -> bool {
        matches!(self, Self::Ready | Self::NeedsAttention)
    }
}

/// The administrative order record. Document content never lives here.
#[derive(Clone, Debug, Serialize, Type)]
pub struct Order {
    pub id: String,
    pub number: String,
    pub title: String,
    pub client_name: String,
    pub project_name: Option<String>,
    pub source_language: String,
    pub target_language: String,
    pub lifecycle: Lifecycle,
    pub priority: Priority,
    pub assignee: Option<String>,
    #[serde(serialize_with = "opt_millis")]
    #[specta(type = Option<f64>)]
    pub deadline_at: Option<Millis>,
    pub brief: String,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub created_at: Millis,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub updated_at: Millis,
    /// denormalized counters synced from the linked project; never user input
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub pages_total: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub pages_prepared: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub pages_translated: i64,
}

impl Order {
    pub(crate) fn new(input: NewOrder, number: String, at: Millis) -> Self {
        Self {
            id: new_id(),
            number,
            title: input.title,
            client_name: input.client_name,
            project_name: None,
            source_language: input.source_language,
            target_language: input.target_language,
            lifecycle: Lifecycle::Active,
            priority: input.priority,
            assignee: input.assignee.filter(|name| !name.trim().is_empty()),
            deadline_at: input.deadline_at,
            brief: input.brief,
            created_at: at,
            updated_at: at,
            pages_total: 0,
            pages_prepared: 0,
            pages_translated: 0,
        }
    }
}

/// Input for creating an order.
#[derive(Clone, Debug, Default, Deserialize, Type)]
pub struct NewOrder {
    /// Explicit client-facing number; generated as `ORD-YYYY-NNNN` when omitted.
    pub number: Option<String>,
    pub title: String,
    #[serde(default)]
    pub client_name: String,
    #[serde(default)]
    pub source_language: String,
    #[serde(default)]
    pub target_language: String,
    #[serde(default)]
    pub priority: Priority,
    pub assignee: Option<String>,
    #[specta(type = Option<f64>)]
    pub deadline_at: Option<Millis>,
    #[serde(default)]
    pub brief: String,
}

/// Editable order metadata. `None` fields stay unchanged; for optional
/// columns the double option distinguishes "leave" from "clear".
#[derive(Clone, Debug, Default, Deserialize, Type)]
pub struct OrderPatch {
    pub number: Option<String>,
    pub title: Option<String>,
    pub client_name: Option<String>,
    pub source_language: Option<String>,
    pub target_language: Option<String>,
    pub priority: Option<Priority>,
    pub assignee: Option<Option<String>>,
    #[specta(type = Option<Option<f64>>)]
    pub deadline_at: Option<Option<Millis>>,
    pub brief: Option<String>,
}

/// A journal entry; notes are append-only apart from resolution/deletion.
#[derive(Clone, Debug, Serialize, Type)]
pub struct OrderNote {
    pub id: String,
    pub order_id: String,
    pub kind: NoteKind,
    pub body: String,
    pub author: String,
    /// Layer entity of the referenced page's segment, serialized UUID.
    pub page_id: Option<String>,
    pub segment_id: Option<String>,
    pub pinned: bool,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub created_at: Millis,
    #[serde(serialize_with = "opt_millis")]
    #[specta(type = Option<f64>)]
    pub resolved_at: Option<Millis>,
    pub resolved_by: Option<String>,
}

impl OrderNote {
    pub(crate) fn new(order_id: &str, input: NewNote, at: Millis) -> Self {
        Self {
            id: new_id(),
            order_id: order_id.to_owned(),
            kind: input.kind,
            body: input.body,
            author: input.author,
            page_id: input.page_id,
            segment_id: input.segment_id,
            pinned: input.pinned,
            created_at: at,
            resolved_at: None,
            resolved_by: None,
        }
    }
}

/// Input for adding a note.
#[derive(Clone, Debug, Deserialize, Type)]
pub struct NewNote {
    #[serde(default)]
    pub kind: NoteKind,
    pub body: String,
    pub author: String,
    pub page_id: Option<String>,
    pub segment_id: Option<String>,
    #[serde(default)]
    pub pinned: bool,
}

/// Severity of an import validation message.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum MessageSeverity {
    /// Apply remains possible.
    Warning,
    /// Apply is blocked while any error exists.
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct ImportMessage {
    pub severity: MessageSeverity,
    /// Stable key the client localizes, e.g. `unknown_segment`.
    pub kind: String,
    pub detail: String,
}

impl ImportMessage {
    pub fn warning(kind: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            severity: MessageSeverity::Warning,
            kind: kind.into(),
            detail: detail.into(),
        }
    }

    pub fn error(kind: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            severity: MessageSeverity::Error,
            kind: kind.into(),
            detail: detail.into(),
        }
    }
}

/// How a row's translation relates to the current project state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum RowChange {
    /// No translation existed before.
    New,
    /// Differs from the current translation; apply overwrites.
    Changed,
    /// Identical to the current translation; apply is a no-op for this row.
    Unchanged,
}

/// One validated spreadsheet row.
#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct ImportRow {
    /// 1-based spreadsheet row number (header is row 1).
    #[specta(type = f64)]
    pub row: u32,
    pub page: String,
    pub segment_id: String,
    pub source_text: String,
    pub translation: String,
    /// Translation value observed when the file was validated. Apply rejects
    /// a row when the project changed to a different value in the meantime.
    pub previous_translation: Option<String>,
    pub translator_note: Option<String>,
    pub review_status: Option<String>,
    pub change: RowChange,
    pub messages: Vec<ImportMessage>,
}

impl ImportRow {
    pub fn has_error(&self) -> bool {
        self.messages
            .iter()
            .any(|message| message.severity == MessageSeverity::Error)
    }
}

/// Persisted validation result of an import attempt.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Type)]
pub struct ImportReport {
    /// File-level problems (headers, encoding, order mismatch).
    pub file_messages: Vec<ImportMessage>,
    pub rows: Vec<ImportRow>,
}

/// One translation import attempt, persisted so previews survive restarts.
#[derive(Clone, Debug, Serialize, Type)]
pub struct TranslationImport {
    pub id: String,
    pub order_id: String,
    pub filename: String,
    /// blake3 hex of the uploaded file; deduplicates repeat uploads.
    pub checksum: String,
    pub state: ImportState,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub total: i64,
    /// Rows that will (or did) apply: total minus rows with errors.
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub accepted: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub warnings: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub errors: i64,
    pub actor: String,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub created_at: Millis,
    #[serde(serialize_with = "opt_millis")]
    #[specta(type = Option<f64>)]
    pub applied_at: Option<Millis>,
    pub report: ImportReport,
}

/// Import list row without the full validation report (report travels only
/// through the detail/preview endpoints).
#[derive(Clone, Debug, Serialize, Type)]
pub struct ImportMeta {
    pub id: String,
    pub order_id: String,
    pub filename: String,
    pub checksum: String,
    pub state: ImportState,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub total: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub accepted: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub warnings: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub errors: i64,
    pub actor: String,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub created_at: Millis,
    #[serde(serialize_with = "opt_millis")]
    #[specta(type = Option<f64>)]
    pub applied_at: Option<Millis>,
}

impl From<TranslationImport> for ImportMeta {
    fn from(import: TranslationImport) -> Self {
        Self {
            id: import.id,
            order_id: import.order_id,
            filename: import.filename,
            checksum: import.checksum,
            state: import.state,
            total: import.total,
            accepted: import.accepted,
            warnings: import.warnings,
            errors: import.errors,
            actor: import.actor,
            created_at: import.created_at,
            applied_at: import.applied_at,
        }
    }
}

/// Audit trail entry. `kind` is a stable machine string; `payload` is a flat
/// details map (values rendered as strings — the trail is for humans).
#[derive(Clone, Debug, Serialize, Type)]
pub struct ActivityEntry {
    pub id: String,
    pub order_id: String,
    pub kind: String,
    pub actor: String,
    /// Structured context, flattened via [`details`].
    pub payload: std::collections::HashMap<String, String>,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub created_at: Millis,
}

impl ActivityEntry {
    pub(crate) fn new(
        order_id: &str,
        kind: impl Into<String>,
        actor: &str,
        payload: serde_json::Value,
        at: Millis,
    ) -> Self {
        Self {
            id: new_id(),
            order_id: order_id.to_owned(),
            kind: kind.into(),
            actor: actor.to_owned(),
            payload: details(payload),
            created_at: at,
        }
    }
}

/// Flattens a JSON object into a display-ready details map; nested values keep
/// their JSON text so nothing is silently dropped.
pub(crate) fn details(payload: serde_json::Value) -> std::collections::HashMap<String, String> {
    payload
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        serde_json::Value::String(text) => text.clone(),
                        other => other.to_string(),
                    };
                    (key.clone(), value)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Computed page counters of the linked project.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Type)]
pub struct PageCounts {
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub total: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub prepared: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub translated: i64,
}

/// Why an order shows up in the attention view; the client localizes kinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AttentionReason {
    Overdue,
    DueSoon,
    Blocked,
    ImportFailed,
    ImportUnreviewed,
    PipelineFailed,
    ReadyToDeliver,
    Stale,
}

/// Saved views of the orders dashboard.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum OrderView {
    /// Everything needing action right now; the default landing view.
    #[default]
    Attention,
    /// Assigned to the local actor.
    Mine,
    /// Deadline today.
    Today,
    All,
}

/// Query filters for the orders table; every field ANDs.
#[derive(Clone, Debug, Default, Deserialize, Type)]
pub struct OrderFilter {
    #[serde(default)]
    pub view: OrderView,
    pub search: Option<String>,
    pub lifecycle: Option<Vec<Lifecycle>>,
    pub stage: Option<StageName>,
    pub assignee: Option<String>,
    pub client: Option<String>,
    pub has_blocker: Option<bool>,
    pub import_state: Option<ImportState>,
    #[specta(type = Option<f64>)]
    pub deadline_from: Option<Millis>,
    #[specta(type = Option<f64>)]
    pub deadline_to: Option<Millis>,
    /// Actor for the `mine` view; resolved by the caller.
    pub actor: Option<String>,
}

/// An order row enriched with computed state for the dashboard table.
#[derive(Clone, Debug, Serialize, Type)]
pub struct OrderListItem {
    #[serde(flatten)]
    pub order: Order,
    /// First stage not yet done; `done` orders keep the delivery stage.
    pub current_stage: StageName,
    pub current_stage_state: StageState,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub open_blockers: i64,
    pub last_import_state: Option<ImportState>,
    pub project_available: bool,
    pub attention: Vec<AttentionReason>,
}

/// Full order detail: the list fields plus the stage route.
#[derive(Clone, Debug, Serialize, Type)]
pub struct OrderDetail {
    #[serde(flatten)]
    pub item: OrderListItem,
    pub stages: Vec<StageInfo>,
}

/// Actionable dashboard summary; every chip doubles as a filter.
#[derive(Clone, Debug, Default, Serialize, Type)]
pub struct OrdersSummary {
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub overdue: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub due_today: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub blocked: i64,
    #[serde(serialize_with = "millis")]
    #[specta(type = f64)]
    pub ready: i64,
}

/// Time bounds derived from "now" for attention/summary computation.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    pub now: Millis,
    /// now + 24h: due-soon horizon.
    pub soon: Millis,
    /// Local-today start and end for the "today" view/chip.
    pub today_start: Millis,
    pub today_end: Millis,
    /// Orders without updates before this are stale.
    pub stale_before: Millis,
}

impl Clock {
    /// Bounds from explicit values; callers compute the local-day edges.
    pub fn new(now: Millis, today_start: Millis, today_end: Millis) -> Self {
        Self {
            now,
            soon: now + 24 * 60 * 60 * 1000,
            today_start,
            today_end,
            stale_before: now - 7 * 24 * 60 * 60 * 1000,
        }
    }
}
