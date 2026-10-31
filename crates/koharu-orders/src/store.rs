//! SQLite-backed administrative store. One file per app, WAL, foreign keys on.
//!
//! All administrative state lives here and never inside `.khrproj` documents:
//! deleting a project directory leaves its order intact (project_available is
//! computed by the caller), and document undo/redo never touches order state.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};

use crate::{
    Millis, OrderFilter,
    error::{Error, Result},
    model::*,
};

const SCHEMA_VERSION: u32 = 1;

const SCHEMA: &str = "
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    number TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    client_name TEXT NOT NULL DEFAULT '',
    project_name TEXT UNIQUE,
    source_language TEXT NOT NULL DEFAULT '',
    target_language TEXT NOT NULL DEFAULT '',
    lifecycle TEXT NOT NULL DEFAULT 'active',
    priority TEXT NOT NULL DEFAULT 'normal',
    assignee TEXT,
    deadline_at INTEGER,
    brief TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    pages_total INTEGER NOT NULL DEFAULT 0,
    pages_prepared INTEGER NOT NULL DEFAULT 0,
    pages_translated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE order_stages (
    order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'not_started',
    started_at INTEGER,
    completed_at INTEGER,
    PRIMARY KEY (order_id, stage)
);
CREATE INDEX order_stages_order ON order_stages(order_id, state);
CREATE TABLE order_notes (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    body TEXT NOT NULL,
    author TEXT NOT NULL,
    page_id TEXT,
    segment_id TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    resolved_at INTEGER,
    resolved_by TEXT
);
CREATE INDEX order_notes_order ON order_notes(order_id, created_at);
CREATE INDEX order_notes_blockers ON order_notes(order_id, kind, resolved_at);
CREATE TABLE translation_imports (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    checksum TEXT NOT NULL,
    state TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    warnings INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    actor TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    applied_at INTEGER,
    report TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX translation_imports_order ON translation_imports(order_id, created_at);
CREATE INDEX translation_imports_checksum ON translation_imports(order_id, checksum);
CREATE TABLE order_activity (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX order_activity_order ON order_activity(order_id, created_at);
CREATE INDEX order_activity_kind ON order_activity(order_id, kind, created_at);
";

/// Thread-safe orders store. Blocking SQLite calls are milliseconds-scale
/// local I/O; desktop commands take the mutex directly instead of a pool.
#[derive(Clone)]
pub struct Store {
    db: Arc<Mutex<Connection>>,
}

impl Store {
    /// Opens (creating when missing) the store at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|source| Error::io("create store directory", source))?;
        }
        let connection = Connection::open(path)
            .map_err(|source| Error::open(path.display().to_string(), source))?;
        Self::init(connection)
    }

    /// In-memory store for tests.
    pub fn open_in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory().map_err(Error::sql)?;
        Self::init(connection)
    }

    fn init(connection: Connection) -> Result<Self> {
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(Error::sql)?;
        connection
            .pragma_update(None, "foreign_keys", "ON")
            .map_err(Error::sql)?;
        connection
            .pragma_update(None, "busy_timeout", 5000)
            .map_err(Error::sql)?;
        let version: u32 = connection
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .map_err(Error::sql)?;
        if version > SCHEMA_VERSION {
            return Err(Error::Migration(format!(
                "store schema {version} is newer than supported {SCHEMA_VERSION}"
            )));
        }
        if version < SCHEMA_VERSION {
            connection.execute_batch(SCHEMA).map_err(Error::sql)?;
            connection
                .pragma_update(None, "user_version", SCHEMA_VERSION)
                .map_err(Error::sql)?;
        }
        Ok(Self {
            db: Arc::new(Mutex::new(connection)),
        })
    }

    // ---- orders -------------------------------------------------------

    /// Creates an order with all seven production stages at `not_started`.
    pub fn create_order(&self, input: NewOrder, actor: &str) -> Result<OrderDetail> {
        let at = now();
        let number = match input.number.as_deref().map(str::trim) {
            Some(number) if !number.is_empty() => number.to_owned(),
            _ => self.next_number(at)?,
        };
        if input.title.trim().is_empty() {
            return Err(Error::Input("order title must not be empty".into()));
        }
        let order = Order::new(input, number, at);
        let mut db = self.db.lock();
        let transaction = db.transaction().map_err(Error::sql)?;
        insert_order(&transaction, &order)?;
        for stage in StageName::ALL {
            transaction
                .execute(
                    "INSERT INTO order_stages (order_id, stage) VALUES (?1, ?2)",
                    params![order.id, stage.as_str()],
                )
                .map_err(Error::sql)?;
        }
        insert_activity(
            &transaction,
            &ActivityEntry::new(
                &order.id,
                "order_created",
                actor,
                serde_json::json!({ "number": order.number }),
                at,
            ),
        )?;
        transaction.commit().map_err(Error::sql)?;
        drop(db);
        self.order_detail(&order.id)?
            .ok_or_else(|| Error::NotFound(order.id.clone()))
    }

    /// Next `ORD-YYYY-NNNN` for the order's creation year (UTC).
    fn next_number(&self, at: Millis) -> Result<String> {
        let year = time::OffsetDateTime::from_unix_timestamp(at / 1000)
            .map_err(|error| Error::Migration(error.to_string()))?
            .year();
        let db = self.db.lock();
        let mut next = 1u32;
        let pattern = format!("ORD-{year}-%");
        let mut statement = db
            .prepare("SELECT number FROM orders WHERE number LIKE ?1")
            .map_err(Error::sql)?;
        let existing = statement
            .query_map(params![pattern], |row| row.get::<_, String>(0))
            .map_err(Error::sql)?;
        for number in existing.flatten() {
            if let Some(sequence) = number
                .strip_prefix(&format!("ORD-{year}-"))
                .and_then(|rest| rest.parse::<u32>().ok())
            {
                next = next.max(sequence + 1);
            }
        }
        Ok(format!("ORD-{year}-{next:04}"))
    }

    /// Applies a metadata patch; every changed field lands in the audit trail.
    pub fn update_order(&self, id: &str, patch: OrderPatch, actor: &str) -> Result<Order> {
        let mut order = self
            .get_order(id)?
            .ok_or_else(|| Error::NotFound(id.into()))?;
        let changes = apply_order_patch(&mut order, patch)?;
        if changes.is_empty() {
            return Ok(order);
        }
        order.updated_at = now();
        let db = self.db.lock();
        update_order_row(&db, &order)?;
        insert_activity(
            &db,
            &ActivityEntry::new(
                id,
                "order_updated",
                actor,
                serde_json::Value::Object(changes),
                order.updated_at,
            ),
        )?;
        Ok(order)
    }
    /// Creates or refreshes a client-supplied order and links its project in
    /// one transaction. Administrative-only fields absent from `patch` remain
    /// untouched on an existing order.
    pub fn sync_linked_order(
        &self,
        number: &str,
        project_name: &str,
        patch: OrderPatch,
        actor: &str,
    ) -> Result<Order> {
        let number = number.trim();
        let project_name = project_name.trim();
        if number.is_empty() {
            return Err(Error::Input("order number must not be empty".into()));
        }
        if project_name.is_empty() {
            return Err(Error::Input("project name must not be empty".into()));
        }
        if patch
            .number
            .as_deref()
            .is_some_and(|candidate| candidate.trim() != number)
        {
            return Err(Error::Input(
                "order patch number does not match the lookup number".into(),
            ));
        }

        let at = now();
        let mut db = self.db.lock();
        let transaction = db.transaction().map_err(Error::sql)?;
        let existing = transaction
            .query_row(
                "SELECT id, number, title, client_name, project_name, source_language,
                        target_language, lifecycle, priority, assignee, deadline_at, brief,
                        created_at, updated_at, pages_total, pages_prepared, pages_translated
                 FROM orders WHERE number = ?1",
                params![number],
                map_order,
            )
            .optional()
            .map_err(Error::sql)?;

        let order = match existing {
            Some(mut order) => {
                let previous_project = order.project_name.clone();
                let mut changes = apply_order_patch(&mut order, patch)?;
                if order.project_name.as_deref() != Some(project_name) {
                    changes.insert(
                        "project_name".into(),
                        serde_json::json!({
                            "from": order.project_name,
                            "to": project_name,
                        }),
                    );
                    order.project_name = Some(project_name.to_owned());
                }
                if !changes.is_empty() {
                    order.updated_at = at;
                    update_order_row(&transaction, &order)?;
                    let project_changed = previous_project != order.project_name;
                    insert_activity(
                        &transaction,
                        &ActivityEntry::new(
                            &order.id,
                            "order_updated",
                            actor,
                            serde_json::Value::Object(changes),
                            at,
                        ),
                    )?;
                    if project_changed {
                        insert_activity(
                            &transaction,
                            &ActivityEntry::new(
                                &order.id,
                                "project_attached",
                                actor,
                                serde_json::json!({
                                    "from": previous_project,
                                    "to": order.project_name,
                                }),
                                at,
                            ),
                        )?;
                    }
                }
                order
            }
            None => {
                let title = patch.title.unwrap_or_default();
                if title.trim().is_empty() {
                    return Err(Error::Input("order title must not be empty".into()));
                }
                let mut order = Order::new(
                    NewOrder {
                        number: Some(number.to_owned()),
                        title,
                        client_name: patch.client_name.unwrap_or_default(),
                        source_language: patch.source_language.unwrap_or_default(),
                        target_language: patch.target_language.unwrap_or_default(),
                        priority: patch.priority.unwrap_or_default(),
                        assignee: patch.assignee.flatten(),
                        deadline_at: patch.deadline_at.flatten(),
                        brief: patch.brief.unwrap_or_default(),
                    },
                    number.to_owned(),
                    at,
                );
                order.project_name = Some(project_name.to_owned());
                insert_order(&transaction, &order)?;
                for stage in StageName::ALL {
                    transaction
                        .execute(
                            "INSERT INTO order_stages (order_id, stage) VALUES (?1, ?2)",
                            params![order.id, stage.as_str()],
                        )
                        .map_err(Error::sql)?;
                }
                insert_activity(
                    &transaction,
                    &ActivityEntry::new(
                        &order.id,
                        "order_created",
                        actor,
                        serde_json::json!({ "number": order.number }),
                        at,
                    ),
                )?;
                insert_activity(
                    &transaction,
                    &ActivityEntry::new(
                        &order.id,
                        "project_attached",
                        actor,
                        serde_json::json!({ "from": null, "to": order.project_name }),
                        at,
                    ),
                )?;
                set_stage(
                    &transaction,
                    &order.id,
                    StageName::Intake,
                    StageState::Done,
                    actor,
                    at,
                )?;
                set_stage(
                    &transaction,
                    &order.id,
                    StageName::Preparation,
                    StageState::InProgress,
                    actor,
                    at,
                )?;
                order
            }
        };
        transaction.commit().map_err(Error::sql)?;
        Ok(order)
    }

    /// Sets the administrative lifecycle, recorded as `lifecycle` activity.
    pub fn set_lifecycle(&self, id: &str, lifecycle: Lifecycle, actor: &str) -> Result<Order> {
        let mut order = self
            .get_order(id)?
            .ok_or_else(|| Error::NotFound(id.into()))?;
        if order.lifecycle == lifecycle {
            return Ok(order);
        }
        let previous = order.lifecycle;
        order.lifecycle = lifecycle;
        order.updated_at = now();
        let db = self.db.lock();
        update_order_row(&db, &order)?;
        insert_activity(
            &db,
            &ActivityEntry::new(
                id,
                "lifecycle_changed",
                actor,
                serde_json::json!({ "from": previous, "to": lifecycle }),
                order.updated_at,
            ),
        )?;
        Ok(order)
    }

    /// Attaches (or with `None` detaches) the linked Koharu project. A name
    /// already linked elsewhere is rejected, keeping the 0..1 relation.
    pub fn attach_project(
        &self,
        id: &str,
        project_name: Option<String>,
        actor: &str,
    ) -> Result<Order> {
        let mut order = self
            .get_order(id)?
            .ok_or_else(|| Error::NotFound(id.into()))?;
        let project_name = project_name.filter(|name| !name.trim().is_empty());
        if order.project_name == project_name {
            return Ok(order);
        }
        let previous = order.project_name.take();
        order.project_name = project_name;
        order.updated_at = now();
        let db = self.db.lock();
        update_order_row(&db, &order).map_err(|error| match error {
            Error::Conflict(message) => Error::Conflict(format!(
                "project is already linked to another order: {message}"
            )),
            other => other,
        })?;
        let kind = if order.project_name.is_some() {
            "project_attached"
        } else {
            "project_detached"
        };
        insert_activity(
            &db,
            &ActivityEntry::new(
                id,
                kind,
                actor,
                serde_json::json!({ "from": previous, "to": order.project_name }),
                order.updated_at,
            ),
        )?;
        Ok(order)
    }

    /// Sets one stage's state, maintaining started/completed timestamps.
    pub fn set_stage(
        &self,
        id: &str,
        stage: StageName,
        state: StageState,
        actor: &str,
    ) -> Result<()> {
        set_stage(&self.db.lock(), id, stage, state, actor, now())
    }

    /// Persists page counters computed from the linked project's snapshot.
    pub fn set_progress(&self, id: &str, counts: PageCounts) -> Result<()> {
        let db = self.db.lock();
        let changed = db
            .execute(
                "UPDATE orders SET pages_total = ?2, pages_prepared = ?3, pages_translated = ?4,
                 updated_at = ?5
                 WHERE id = ?1 AND (pages_total != ?2 OR pages_prepared != ?3 OR pages_translated != ?4)",
                params![id, counts.total, counts.prepared, counts.translated, now()],
            )
            .map_err(Error::sql)?;
        let _ = changed;
        Ok(())
    }

    /// Order id linked to a project name, if any.
    pub fn order_for_project(&self, project_name: &str) -> Result<Option<String>> {
        let db = self.db.lock();
        db.query_row(
            "SELECT id FROM orders WHERE project_name = ?1",
            params![project_name],
            |row| row.get(0),
        )
        .optional()
        .map_err(Error::sql)
    }

    pub fn get_order(&self, id: &str) -> Result<Option<Order>> {
        let db = self.db.lock();
        get_order(&db, id)
    }
    /// Looks up an order by its unique client-facing number.
    pub fn get_order_by_number(&self, number: &str) -> Result<Option<Order>> {
        let db = self.db.lock();
        db.query_row(
            "SELECT id, number, title, client_name, project_name, source_language, target_language,
                    lifecycle, priority, assignee, deadline_at, brief, created_at, updated_at,
                    pages_total, pages_prepared, pages_translated
             FROM orders WHERE number = ?1",
            params![number],
            map_order,
        )
        .optional()
        .map_err(Error::sql)
    }

    /// Dashboard table rows: order facts plus computed attention state.
    /// `available_projects` marks whether the linked project directory exists.
    pub fn list_orders(
        &self,
        filter: &OrderFilter,
        clock: &Clock,
        available_projects: &HashSet<String>,
    ) -> Result<Vec<OrderListItem>> {
        let scan = self.scan()?;
        let mut items: Vec<OrderListItem> = scan
            .orders
            .iter()
            .map(|order| self.enrich(&scan, order, clock, available_projects))
            .filter(|item| matches_filter(item, filter))
            .collect();
        items.sort_by(|left, right| {
            right
                .order
                .updated_at
                .cmp(&left.order.updated_at)
                .then_with(|| left.order.number.cmp(&right.order.number))
        });
        Ok(items)
    }

    /// Counts for the four actionable dashboard chips.
    pub fn summary(&self, clock: &Clock) -> Result<OrdersSummary> {
        let scan = self.scan()?;
        let available = HashSet::new();
        let mut summary = OrdersSummary::default();
        for order in &scan.orders {
            if order.lifecycle != Lifecycle::Active {
                continue;
            }
            let item = self.enrich(&scan, order, clock, &available);
            for reason in &item.attention {
                match reason {
                    AttentionReason::Overdue => summary.overdue += 1,
                    AttentionReason::Blocked => summary.blocked += 1,
                    AttentionReason::ReadyToDeliver => summary.ready += 1,
                    _ => {}
                }
            }
            if order
                .deadline_at
                .is_some_and(|deadline| deadline >= clock.today_start && deadline < clock.today_end)
            {
                summary.due_today += 1;
            }
        }
        Ok(summary)
    }

    /// One order with stage route and computed flags.
    pub fn order_detail(&self, id: &str) -> Result<Option<OrderDetail>> {
        self.order_detail_available(id, &HashSet::new())
    }

    /// Same as [`Store::order_detail`] with project availability resolved.
    pub fn order_detail_available(
        &self,
        id: &str,
        available_projects: &HashSet<String>,
    ) -> Result<Option<OrderDetail>> {
        let scan = self.scan()?;
        let Some(order) = scan.orders.iter().find(|order| order.id == id) else {
            return Ok(None);
        };
        let clock = Clock::new(now(), 0, Millis::MAX);
        let item = self.enrich(&scan, order, &clock, available_projects);
        Ok(Some(OrderDetail {
            item,
            stages: scan.stages.get(id).cloned().unwrap_or_default(),
        }))
    }

    // ---- notes ---------------------------------------------------------

    pub fn add_note(&self, order_id: &str, input: NewNote) -> Result<OrderNote> {
        let db = self.db.lock();
        if !order_exists(&db, order_id)? {
            return Err(Error::NotFound(order_id.into()));
        }
        insert_note(&db, order_id, input, now())
    }

    /// Marks a note resolved (or reopens with `resolved = false`); returns
    /// the owning order id.
    pub fn set_note_resolved(&self, note_id: &str, resolved: bool, actor: &str) -> Result<String> {
        let at = now();
        let db = self.db.lock();
        let order_id: Option<String> = db
            .query_row(
                "SELECT order_id FROM order_notes WHERE id = ?1",
                params![note_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(Error::sql)?;
        let Some(order_id) = order_id else {
            return Err(Error::NotFound(note_id.into()));
        };
        db.execute(
            "UPDATE order_notes SET resolved_at = ?2, resolved_by = ?3 WHERE id = ?1",
            params![
                note_id,
                resolved.then_some(at),
                resolved.then_some(actor.to_owned())
            ],
        )
        .map_err(Error::sql)?;
        touch(&db, &order_id, at)?;
        if resolved {
            insert_activity(
                &db,
                &ActivityEntry::new(
                    &order_id,
                    "note_resolved",
                    actor,
                    serde_json::json!({ "note_id": note_id }),
                    at,
                ),
            )?;
        }
        Ok(order_id)
    }

    /// Number of unresolved blocker notes of an order.
    pub fn open_blockers(&self, order_id: &str) -> Result<i64> {
        let db = self.db.lock();
        db.query_row(
            "SELECT COUNT(*) FROM order_notes
             WHERE order_id = ?1 AND kind = 'blocker' AND resolved_at IS NULL",
            params![order_id],
            |row| row.get(0),
        )
        .map_err(Error::sql)
    }

    /// Removes a mistakenly created note; the deletion stays in the trail.
    pub fn delete_note(&self, note_id: &str, actor: &str) -> Result<()> {
        let at = now();
        let db = self.db.lock();
        let row: Option<(String, String)> = db
            .query_row(
                "SELECT order_id, body FROM order_notes WHERE id = ?1",
                params![note_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(Error::sql)?;
        let Some((order_id, body)) = row else {
            return Err(Error::NotFound(note_id.into()));
        };
        db.execute("DELETE FROM order_notes WHERE id = ?1", params![note_id])
            .map_err(Error::sql)?;
        touch(&db, &order_id, at)?;
        insert_activity(
            &db,
            &ActivityEntry::new(
                &order_id,
                "note_deleted",
                actor,
                serde_json::json!({ "note_id": note_id, "body_excerpt": body.chars().take(80).collect::<String>() }),
                at,
            ),
        )?;
        Ok(())
    }

    /// Blockers and pinned notes first, then newest first.
    pub fn list_notes(&self, order_id: &str) -> Result<Vec<OrderNote>> {
        let db = self.db.lock();
        let mut statement = db
            .prepare(
                "SELECT id, order_id, kind, body, author, page_id, segment_id, pinned,
                        created_at, resolved_at, resolved_by
                 FROM order_notes WHERE order_id = ?1
                 ORDER BY (kind = 'blocker' AND resolved_at IS NULL) DESC,
                          pinned DESC, created_at DESC",
            )
            .map_err(Error::sql)?;
        let notes = statement
            .query_map(params![order_id], map_note)
            .map_err(Error::sql)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::sql)?;
        Ok(notes)
    }

    // ---- imports --------------------------------------------------------

    /// Persists a parsed import attempt (preview state); nothing touches the project.
    pub fn create_import(
        &self,
        order_id: &str,
        filename: String,
        checksum: String,
        report: ImportReport,
        actor: &str,
    ) -> Result<TranslationImport> {
        let at = now();
        let total = report.rows.len() as i64;
        let errors = report.rows.iter().filter(|row| row.has_error()).count() as i64
            + report
                .file_messages
                .iter()
                .filter(|message| message.severity == MessageSeverity::Error)
                .count() as i64;
        let warnings = report
            .rows
            .iter()
            .flat_map(|row| row.messages.iter())
            .chain(report.file_messages.iter())
            .filter(|message| message.severity == MessageSeverity::Warning)
            .count() as i64;
        let state = if errors > 0 {
            ImportState::Failed
        } else if warnings > 0 {
            ImportState::NeedsAttention
        } else {
            ImportState::Ready
        };
        let import = TranslationImport {
            id: uuid::Uuid::now_v7().to_string(),
            order_id: order_id.to_owned(),
            filename,
            checksum,
            state,
            total,
            accepted: total - report.rows.iter().filter(|row| row.has_error()).count() as i64,
            warnings,
            errors,
            actor: actor.to_owned(),
            created_at: at,
            applied_at: None,
            report,
        };
        let db = self.db.lock();
        if !order_exists(&db, order_id)? {
            return Err(Error::NotFound(order_id.into()));
        }
        insert_import(&db, &import)?;
        touch(&db, order_id, at)?;
        insert_activity(
            &db,
            &ActivityEntry::new(
                order_id,
                "import_uploaded",
                actor,
                serde_json::json!({
                    "import_id": import.id,
                    "filename": import.filename,
                    "state": import.state,
                    "total": total,
                    "errors": import.errors,
                    "warnings": warnings,
                }),
                at,
            ),
        )?;
        Ok(import)
    }

    /// Latest import of the order with the same checksum, for dedup.
    pub fn find_import_by_checksum(
        &self,
        order_id: &str,
        checksum: &str,
    ) -> Result<Option<TranslationImport>> {
        let id: Option<String> = {
            let db = self.db.lock();
            db.query_row(
                "SELECT id FROM translation_imports
                 WHERE order_id = ?1 AND checksum = ?2
                 ORDER BY created_at DESC, rowid DESC LIMIT 1",
                params![order_id, checksum],
                |row| row.get(0),
            )
            .optional()
            .map_err(Error::sql)?
        };
        id.map(|id| self.import(&id))
            .transpose()
            .map(Option::flatten)
    }

    pub fn import(&self, id: &str) -> Result<Option<TranslationImport>> {
        let db = self.db.lock();
        get_import(&db, id)
    }

    pub fn list_imports(&self, order_id: &str) -> Result<Vec<TranslationImport>> {
        let db = self.db.lock();
        let mut statement = db
            .prepare(
                "SELECT id, order_id, filename, checksum, state, total, accepted, warnings,
                        errors, actor, created_at, applied_at, report
                 FROM translation_imports WHERE order_id = ?1 ORDER BY created_at DESC, rowid DESC",
            )
            .map_err(Error::sql)?;
        let rows = statement
            .query_map(params![order_id], map_import_columns)
            .map_err(Error::sql)?;
        let mut imports = Vec::new();
        for row in rows {
            imports.push(row.map_err(Error::sql)??);
        }
        Ok(imports)
    }

    /// Finalizes a project import and all related administrative changes in one
    /// SQLite transaction. The import remains pending if any bookkeeping fails.
    pub fn finalize_import(
        &self,
        id: &str,
        notes: Vec<NewNote>,
        actor: &str,
    ) -> Result<TranslationImport> {
        let at = now();
        let mut db = self.db.lock();
        let transaction = db.transaction().map_err(Error::sql)?;
        let mut import = get_import(&transaction, id)?.ok_or_else(|| Error::NotFound(id.into()))?;
        if !import.state.pending() {
            return Err(Error::Conflict(format!(
                "import is {} and cannot be applied",
                import.state.as_str()
            )));
        }
        if import.errors > 0 {
            return Err(Error::Conflict(
                "an import with blocking errors cannot be applied".into(),
            ));
        }
        let blockers: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM order_notes
                 WHERE order_id = ?1 AND kind = 'blocker' AND resolved_at IS NULL",
                params![import.order_id],
                |row| row.get(0),
            )
            .map_err(Error::sql)?;
        if blockers > 0 {
            return Err(Error::Conflict(
                "unresolved blocker notes prevent applying the import".into(),
            ));
        }

        import.state = ImportState::Applied;
        import.applied_at = Some(at);
        update_import(&transaction, &import)?;
        set_stage(
            &transaction,
            &import.order_id,
            StageName::Translation,
            StageState::Done,
            actor,
            at,
        )?;
        set_stage(
            &transaction,
            &import.order_id,
            StageName::Editing,
            StageState::InProgress,
            actor,
            at,
        )?;
        for note in notes {
            insert_note(&transaction, &import.order_id, note, at)?;
        }
        touch(&transaction, &import.order_id, at)?;
        insert_activity(
            &transaction,
            &ActivityEntry::new(
                &import.order_id,
                "import_applied",
                actor,
                serde_json::json!({ "import_id": id, "accepted": import.accepted }),
                at,
            ),
        )?;
        transaction.commit().map_err(Error::sql)?;
        Ok(import)
    }

    /// Records a failed apply attempt without changing the pending state.
    pub fn record_import_failure(&self, id: &str, actor: &str, error: &str) -> Result<()> {
        let at = now();
        let db = self.db.lock();
        let import = get_import(&db, id)?.ok_or_else(|| Error::NotFound(id.into()))?;
        insert_activity(
            &db,
            &ActivityEntry::new(
                &import.order_id,
                "import_failed",
                actor,
                serde_json::json!({ "import_id": id, "error": error }),
                at,
            ),
        )?;
        Ok(())
    }

    /// Deletes a pending import attempt, keeping a trail entry.
    pub fn discard_import(&self, id: &str, actor: &str) -> Result<()> {
        let at = now();
        let db = self.db.lock();
        let import = get_import(&db, id)?.ok_or_else(|| Error::NotFound(id.into()))?;
        if !import.state.pending() && import.state != ImportState::Failed {
            return Err(Error::Conflict(
                "only pending or failed imports can be discarded".into(),
            ));
        }
        db.execute("DELETE FROM translation_imports WHERE id = ?1", params![id])
            .map_err(Error::sql)?;
        touch(&db, &import.order_id, at)?;
        insert_activity(
            &db,
            &ActivityEntry::new(
                &import.order_id,
                "import_discarded",
                actor,
                serde_json::json!({ "import_id": id, "filename": import.filename }),
                at,
            ),
        )?;
        Ok(())
    }

    // ---- activity --------------------------------------------------------

    /// Appends an audit entry from outside the store (e.g. pipeline events).
    pub fn log_activity(
        &self,
        order_id: &str,
        kind: &str,
        actor: &str,
        payload: serde_json::Value,
    ) -> Result<()> {
        let at = now();
        let db = self.db.lock();
        if !order_exists(&db, order_id)? {
            return Err(Error::NotFound(order_id.into()));
        }
        touch(&db, order_id, at)?;
        insert_activity(&db, &ActivityEntry::new(order_id, kind, actor, payload, at))?;
        Ok(())
    }

    pub fn list_activity(&self, order_id: &str, limit: u32) -> Result<Vec<ActivityEntry>> {
        let db = self.db.lock();
        let mut statement = db
            .prepare(
                "SELECT id, order_id, kind, actor, payload, created_at
                 FROM order_activity WHERE order_id = ?1 ORDER BY created_at DESC, rowid DESC LIMIT ?2",
            )
            .map_err(Error::sql)?;
        let entries = statement
            .query_map(params![order_id, limit], |row| {
                let payload: String = row.get(4)?;
                Ok(ActivityEntry {
                    id: row.get(0)?,
                    order_id: row.get(1)?,
                    kind: row.get(2)?,
                    actor: row.get(3)?,
                    payload: serde_json::from_str(&payload).unwrap_or_default(),
                    created_at: row.get(5)?,
                })
            })
            .map_err(Error::sql)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::sql)?;
        Ok(entries)
    }

    // ---- internals -------------------------------------------------------

    fn scan(&self) -> Result<Scan> {
        let db = self.db.lock();
        let orders = {
            let mut statement = db
                .prepare(
                    "SELECT id, number, title, client_name, project_name, source_language,
                            target_language, lifecycle, priority, assignee, deadline_at, brief,
                            created_at, updated_at, pages_total, pages_prepared, pages_translated
                     FROM orders",
                )
                .map_err(Error::sql)?;
            statement
                .query_map([], map_order)
                .map_err(Error::sql)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Error::sql)?
        };
        let mut stages: HashMap<String, Vec<StageInfo>> = HashMap::new();
        {
            let mut statement = db
                .prepare(
                    "SELECT order_id, stage, state, started_at, completed_at FROM order_stages",
                )
                .map_err(Error::sql)?;
            let rows = statement
                .query_map([], |row| {
                    let stage_name: String = row.get(1)?;
                    let stage = StageName::parse(&stage_name).ok_or_else(|| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1,
                            rusqlite::types::Type::Text,
                            format!("unknown stage {stage_name}").into(),
                        )
                    })?;
                    Ok((
                        row.get::<_, String>(0)?,
                        StageInfo {
                            stage,
                            state: parse_stage_state(&row.get::<_, String>(2)?),
                            started_at: row.get(3)?,
                            completed_at: row.get(4)?,
                        },
                    ))
                })
                .map_err(Error::sql)?;
            for row in rows {
                let (order_id, info) = row.map_err(Error::sql)?;
                stages.entry(order_id).or_default().push(info);
            }
        }
        for infos in stages.values_mut() {
            infos.sort_by_key(|info| {
                StageName::ALL
                    .iter()
                    .position(|stage| *stage == info.stage)
                    .unwrap_or(0)
            });
        }
        let mut blockers: HashMap<String, i64> = HashMap::new();
        {
            let mut statement = db
                .prepare(
                    "SELECT order_id, COUNT(*) FROM order_notes
                     WHERE kind = 'blocker' AND resolved_at IS NULL GROUP BY order_id",
                )
                .map_err(Error::sql)?;
            let rows = statement
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(Error::sql)?;
            for row in rows {
                let (order_id, count) = row.map_err(Error::sql)?;
                blockers.insert(order_id, count);
            }
        }
        // Latest import per order.
        let mut last_imports: HashMap<String, ImportState> = HashMap::new();
        // Orders with at least one applied import.
        let mut applied: HashSet<String> = HashSet::new();
        {
            let mut statement = db
                .prepare(
                    "SELECT order_id, state FROM translation_imports ORDER BY created_at DESC, rowid DESC",
                )
                .map_err(Error::sql)?;
            let rows = statement
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(Error::sql)?;
            for row in rows {
                let (order_id, state) = row.map_err(Error::sql)?;
                let Some(state) = ImportState::parse(&state) else {
                    continue;
                };
                last_imports.entry(order_id.clone()).or_insert(state);
                if state == ImportState::Applied {
                    applied.insert(order_id);
                }
            }
        }
        // Pipeline failure after the last success, per order.
        let mut pipeline_failed: HashSet<String> = HashSet::new();
        {
            let mut statement = db
                .prepare(
                    "SELECT order_id, kind FROM order_activity
                     WHERE kind IN ('pipeline_failed', 'pipeline_succeeded')
                     ORDER BY created_at DESC, rowid DESC",
                )
                .map_err(Error::sql)?;
            let rows = statement
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(Error::sql)?;
            let mut seen: HashSet<String> = HashSet::new();
            for row in rows {
                let (order_id, kind) = row.map_err(Error::sql)?;
                if !seen.insert(order_id.clone()) {
                    continue;
                }
                if kind == "pipeline_failed" {
                    pipeline_failed.insert(order_id);
                }
            }
        }
        Ok(Scan {
            orders,
            stages,
            blockers,
            last_imports,
            applied,
            pipeline_failed,
        })
    }

    fn enrich(
        &self,
        scan: &Scan,
        order: &Order,
        clock: &Clock,
        available_projects: &HashSet<String>,
    ) -> OrderListItem {
        let stages = scan.stages.get(&order.id).map(Vec::as_slice).unwrap_or(&[]);
        let current = stages
            .iter()
            .find(|info| info.state != StageState::Done)
            .copied()
            .unwrap_or(StageInfo {
                stage: StageName::Delivery,
                state: StageState::Done,
                started_at: None,
                completed_at: None,
            });
        let open_blockers = scan.blockers.get(&order.id).copied().unwrap_or(0);
        let last_import_state = scan.last_imports.get(&order.id).copied();
        let import_unreviewed = scan.applied.contains(&order.id)
            && stages
                .iter()
                .find(|info| info.stage == StageName::Editing)
                .is_some_and(|info| info.state == StageState::NotStarted);
        let ready_to_deliver = StageName::BEFORE_DELIVERY.iter().all(|stage| {
            stages
                .iter()
                .find(|info| info.stage == *stage)
                .is_some_and(|info| info.state == StageState::Done)
        }) && stages
            .iter()
            .find(|info| info.stage == StageName::Delivery)
            .is_some_and(|info| info.state != StageState::Done);
        let mut attention = Vec::new();
        if order.lifecycle == Lifecycle::Active {
            match order.deadline_at {
                Some(deadline) if deadline < clock.now => attention.push(AttentionReason::Overdue),
                Some(deadline) if deadline < clock.soon => attention.push(AttentionReason::DueSoon),
                _ => {}
            }
            if open_blockers > 0 {
                attention.push(AttentionReason::Blocked);
            }
            if last_import_state == Some(ImportState::Failed) {
                attention.push(AttentionReason::ImportFailed);
            }
            if import_unreviewed {
                attention.push(AttentionReason::ImportUnreviewed);
            }
            if scan.pipeline_failed.contains(&order.id) {
                attention.push(AttentionReason::PipelineFailed);
            }
            if ready_to_deliver {
                attention.push(AttentionReason::ReadyToDeliver);
            }
            if order.updated_at < clock.stale_before {
                attention.push(AttentionReason::Stale);
            }
        }
        OrderListItem {
            order: order.clone(),
            current_stage: current.stage,
            current_stage_state: current.state,
            open_blockers,
            last_import_state,
            project_available: order
                .project_name
                .as_ref()
                .is_some_and(|name| available_projects.contains(name)),
            attention,
        }
    }
}

/// One pass over the store for list/summary enrichment.
struct Scan {
    orders: Vec<Order>,
    stages: HashMap<String, Vec<StageInfo>>,
    blockers: HashMap<String, i64>,
    last_imports: HashMap<String, ImportState>,
    applied: HashSet<String>,
    pipeline_failed: HashSet<String>,
}

fn matches_filter(item: &OrderListItem, filter: &OrderFilter) -> bool {
    if filter.view == OrderView::Attention && item.attention.is_empty() {
        return false;
    }
    if filter.view == OrderView::Mine {
        let matches = filter
            .actor
            .as_ref()
            .is_some_and(|actor| item.order.assignee.as_ref() == Some(actor));
        if !matches {
            return false;
        }
    }
    if let Some(lifecycle) = &filter.lifecycle
        && !lifecycle.contains(&item.order.lifecycle)
    {
        return false;
    }
    if let Some(stage) = filter.stage
        && item.current_stage != stage
    {
        return false;
    }
    if let Some(assignee) = &filter.assignee
        && item.order.assignee.as_deref() != Some(assignee.as_str())
    {
        return false;
    }
    if let Some(client) = &filter.client
        && !item.order.client_name.eq_ignore_ascii_case(client)
    {
        return false;
    }
    if let Some(has_blocker) = filter.has_blocker
        && (item.open_blockers > 0) != has_blocker
    {
        return false;
    }
    if let Some(state) = filter.import_state
        && item.last_import_state != Some(state)
    {
        return false;
    }
    if let Some(from) = filter.deadline_from
        && !item
            .order
            .deadline_at
            .is_some_and(|deadline| deadline >= from)
    {
        return false;
    }
    if let Some(to) = filter.deadline_to
        && !item.order.deadline_at.is_some_and(|deadline| deadline < to)
    {
        return false;
    }
    // OrderView::Today is expressed through deadline_from/deadline_to bounds.
    if let Some(search) = filter.search.as_deref().map(str::trim)
        && !search.is_empty()
    {
        let query = search.to_lowercase();
        let haystack = [
            item.order.number.as_str(),
            item.order.title.as_str(),
            item.order.client_name.as_str(),
            item.order.assignee.as_deref().unwrap_or_default(),
        ];
        if !haystack
            .iter()
            .any(|field| field.to_lowercase().contains(&query))
        {
            return false;
        }
    }
    true
}

// ---- row mapping ---------------------------------------------------------

fn parse_lifecycle(value: &str) -> Lifecycle {
    Lifecycle::ALL
        .into_iter()
        .find(|lifecycle| lifecycle.as_str() == value)
        .unwrap_or_default()
}

fn parse_priority(value: &str) -> Priority {
    Priority::ALL
        .into_iter()
        .find(|priority| priority.as_str() == value)
        .unwrap_or_default()
}

fn parse_stage_state(value: &str) -> StageState {
    StageState::ALL
        .into_iter()
        .find(|state| state.as_str() == value)
        .unwrap_or_default()
}

fn map_order(row: &rusqlite::Row<'_>) -> rusqlite::Result<Order> {
    Ok(Order {
        id: row.get(0)?,
        number: row.get(1)?,
        title: row.get(2)?,
        client_name: row.get(3)?,
        project_name: row.get(4)?,
        source_language: row.get(5)?,
        target_language: row.get(6)?,
        lifecycle: parse_lifecycle(&row.get::<_, String>(7)?),
        priority: parse_priority(&row.get::<_, String>(8)?),
        assignee: row.get(9)?,
        deadline_at: row.get(10)?,
        brief: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
        pages_total: row.get(14)?,
        pages_prepared: row.get(15)?,
        pages_translated: row.get(16)?,
    })
}

fn map_note(row: &rusqlite::Row<'_>) -> rusqlite::Result<OrderNote> {
    Ok(OrderNote {
        id: row.get(0)?,
        order_id: row.get(1)?,
        kind: NoteKind::parse(&row.get::<_, String>(2)?).unwrap_or_default(),
        body: row.get(3)?,
        author: row.get(4)?,
        page_id: row.get(5)?,
        segment_id: row.get(6)?,
        pinned: row.get::<_, i64>(7)? != 0,
        created_at: row.get(8)?,
        resolved_at: row.get(9)?,
        resolved_by: row.get(10)?,
    })
}

fn map_import_columns(row: &rusqlite::Row<'_>) -> rusqlite::Result<Result<TranslationImport>> {
    let report: String = row.get(12)?;
    let import = TranslationImport {
        id: row.get(0)?,
        order_id: row.get(1)?,
        filename: row.get(2)?,
        checksum: row.get(3)?,
        state: ImportState::parse(&row.get::<_, String>(4)?).unwrap_or(ImportState::Failed),
        total: row.get(5)?,
        accepted: row.get(6)?,
        warnings: row.get(7)?,
        errors: row.get(8)?,
        actor: row.get(9)?,
        created_at: row.get(10)?,
        applied_at: row.get(11)?,
        report: ImportReport::default(),
    };
    Ok(serde_json::from_str(&report)
        .map(|report| TranslationImport { report, ..import })
        .map_err(Error::json))
}

// ---- write helpers ---------------------------------------------------------

fn get_order(db: &Connection, id: &str) -> Result<Option<Order>> {
    db.query_row(
        "SELECT id, number, title, client_name, project_name, source_language, target_language,
                lifecycle, priority, assignee, deadline_at, brief, created_at, updated_at,
                pages_total, pages_prepared, pages_translated
         FROM orders WHERE id = ?1",
        params![id],
        map_order,
    )
    .optional()
    .map_err(Error::sql)
}

fn order_exists(db: &Connection, id: &str) -> Result<bool> {
    db.query_row(
        "SELECT 1 FROM orders WHERE id = ?1",
        params![id],
        |_| Ok(()),
    )
    .optional()
    .map(|row| row.is_some())
    .map_err(Error::sql)
}

fn get_import(db: &Connection, id: &str) -> Result<Option<TranslationImport>> {
    let row = db
        .query_row(
            "SELECT id, order_id, filename, checksum, state, total, accepted, warnings, errors,
                    actor, created_at, applied_at, report
             FROM translation_imports WHERE id = ?1",
            params![id],
            map_import_columns,
        )
        .optional()
        .map_err(Error::sql)?;
    row.transpose()
}
fn apply_order_patch(
    order: &mut Order,
    patch: OrderPatch,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut changes = serde_json::Map::new();
    macro_rules! field {
        ($name:ident) => {
            if let Some(value) = patch.$name {
                if order.$name != value {
                    changes.insert(
                        stringify!($name).into(),
                        serde_json::json!({ "from": order.$name, "to": value }),
                    );
                    order.$name = value;
                }
            }
        };
    }
    field!(title);
    field!(client_name);
    field!(source_language);
    field!(target_language);
    field!(priority);
    field!(brief);
    if let Some(number) = patch.number.filter(|number| !number.trim().is_empty())
        && order.number != number
    {
        changes.insert(
            "number".into(),
            serde_json::json!({ "from": order.number, "to": number }),
        );
        order.number = number;
    }
    if let Some(assignee) = patch.assignee {
        let assignee = assignee.filter(|name| !name.trim().is_empty());
        if order.assignee != assignee {
            changes.insert(
                "assignee".into(),
                serde_json::json!({ "from": order.assignee, "to": assignee }),
            );
            order.assignee = assignee;
        }
    }
    if let Some(deadline) = patch.deadline_at
        && order.deadline_at != deadline
    {
        changes.insert(
            "deadline_at".into(),
            serde_json::json!({ "from": order.deadline_at, "to": deadline }),
        );
        order.deadline_at = deadline;
    }
    if order.title.trim().is_empty() {
        return Err(Error::Input("order title must not be empty".into()));
    }
    Ok(changes)
}

fn insert_order(db: &Connection, order: &Order) -> Result<()> {
    db.execute(
        "INSERT INTO orders (id, number, title, client_name, project_name, source_language,
         target_language, lifecycle, priority, assignee, deadline_at, brief, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            order.id,
            order.number,
            order.title,
            order.client_name,
            order.project_name,
            order.source_language,
            order.target_language,
            order.lifecycle.as_str(),
            order.priority.as_str(),
            order.assignee,
            order.deadline_at,
            order.brief,
            order.created_at,
            order.updated_at,
        ],
    )
    .map_err(|error| match error {
        rusqlite::Error::SqliteFailure(inner, _)
            if inner.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Error::Conflict(format!("order number {:?} is already used", order.number))
        }
        other => Error::sql(other),
    })?;
    Ok(())
}

fn update_order_row(db: &Connection, order: &Order) -> Result<()> {
    db.execute(
        "UPDATE orders SET number = ?2, title = ?3, client_name = ?4, project_name = ?5,
         source_language = ?6, target_language = ?7, lifecycle = ?8, priority = ?9,
         assignee = ?10, deadline_at = ?11, brief = ?12, updated_at = ?13
         WHERE id = ?1",
        params![
            order.id,
            order.number,
            order.title,
            order.client_name,
            order.project_name,
            order.source_language,
            order.target_language,
            order.lifecycle.as_str(),
            order.priority.as_str(),
            order.assignee,
            order.deadline_at,
            order.brief,
            order.updated_at,
        ],
    )
    .map_err(|error| match error {
        rusqlite::Error::SqliteFailure(inner, _)
            if inner.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Error::Conflict(format!(
                "order number or project name collides with another order ({:?})",
                order.number
            ))
        }
        other => Error::sql(other),
    })?;
    Ok(())
}

fn set_stage(
    db: &Connection,
    id: &str,
    stage: StageName,
    state: StageState,
    actor: &str,
    at: Millis,
) -> Result<()> {
    let current: Option<(StageState, Option<Millis>)> = db
        .query_row(
            "SELECT state, started_at FROM order_stages WHERE order_id = ?1 AND stage = ?2",
            params![id, stage.as_str()],
            |row| Ok((parse_stage_state(&row.get::<_, String>(0)?), row.get(1)?)),
        )
        .optional()
        .map_err(Error::sql)?;
    let Some((current, started_at)) = current else {
        return Err(Error::NotFound(format!("order {id}")));
    };
    if current == state {
        return Ok(());
    }
    let started = match (started_at, state) {
        (None, StageState::NotStarted) => None,
        (None, _) => Some(at),
        (Some(at), _) => Some(at),
    };
    let completed = (state == StageState::Done).then_some(at);
    db.execute(
        "UPDATE order_stages SET state = ?3, started_at = ?4, completed_at = ?5
         WHERE order_id = ?1 AND stage = ?2",
        params![id, stage.as_str(), state.as_str(), started, completed],
    )
    .map_err(Error::sql)?;
    touch(db, id, at)?;
    let kind = match state {
        StageState::InProgress => "stage_started",
        StageState::Done => "stage_completed",
        StageState::Blocked => "stage_blocked",
        StageState::NotStarted => "stage_unblocked",
    };
    insert_activity(
        db,
        &ActivityEntry::new(
            id,
            kind,
            actor,
            serde_json::json!({ "stage": stage, "from": current, "to": state }),
            at,
        ),
    )
}

fn insert_note(db: &Connection, order_id: &str, input: NewNote, at: Millis) -> Result<OrderNote> {
    if input.body.trim().is_empty() {
        return Err(Error::Input("note body must not be empty".into()));
    }
    let note = OrderNote::new(order_id, input, at);
    db.execute(
        "INSERT INTO order_notes
         (id, order_id, kind, body, author, page_id, segment_id, pinned, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            note.id,
            note.order_id,
            note.kind.as_str(),
            note.body,
            note.author,
            note.page_id,
            note.segment_id,
            note.pinned,
            note.created_at,
        ],
    )
    .map_err(Error::sql)?;
    touch(db, order_id, at)?;
    insert_activity(
        db,
        &ActivityEntry::new(
            order_id,
            "note_added",
            &note.author,
            serde_json::json!({ "kind": note.kind, "note_id": note.id }),
            at,
        ),
    )?;
    Ok(note)
}

fn insert_import(db: &Connection, import: &TranslationImport) -> Result<()> {
    let report = serde_json::to_string(&import.report).map_err(Error::json)?;
    db.execute(
        "INSERT INTO translation_imports
         (id, order_id, filename, checksum, state, total, accepted, warnings, errors,
          actor, created_at, applied_at, report)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            import.id,
            import.order_id,
            import.filename,
            import.checksum,
            import.state.as_str(),
            import.total,
            import.accepted,
            import.warnings,
            import.errors,
            import.actor,
            import.created_at,
            import.applied_at,
            report,
        ],
    )
    .map_err(Error::sql)?;
    Ok(())
}

fn update_import(db: &Connection, import: &TranslationImport) -> Result<()> {
    db.execute(
        "UPDATE translation_imports SET state = ?2, applied_at = ?3 WHERE id = ?1",
        params![import.id, import.state.as_str(), import.applied_at],
    )
    .map_err(Error::sql)?;
    Ok(())
}

fn insert_activity(db: &Connection, entry: &ActivityEntry) -> Result<()> {
    let payload = serde_json::to_string(&entry.payload).map_err(Error::json)?;
    db.execute(
        "INSERT INTO order_activity (id, order_id, kind, actor, payload, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            entry.id,
            entry.order_id,
            entry.kind,
            entry.actor,
            payload,
            entry.created_at,
        ],
    )
    .map_err(Error::sql)?;
    Ok(())
}

fn touch(db: &Connection, order_id: &str, at: Millis) -> Result<()> {
    db.execute(
        "UPDATE orders SET updated_at = ?2 WHERE id = ?1",
        params![order_id, at],
    )
    .map_err(Error::sql)?;
    Ok(())
}
