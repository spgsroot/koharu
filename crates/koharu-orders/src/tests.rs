//! Store and exchange tests: one behavior per test, real behavior boundaries.

use std::collections::HashSet;

use crate::{exchange::SegmentRow, model::*, store::Store};

fn store() -> Store {
    Store::open_in_memory().expect("in-memory store")
}

fn clock(now: Millis) -> Clock {
    Clock::new(now, 0, Millis::MAX)
}

fn new_order(title: &str) -> NewOrder {
    NewOrder {
        title: title.into(),
        client_name: "Acme".into(),
        ..NewOrder::default()
    }
}

#[test]
fn create_order_generates_number_and_stages() {
    let store = store();
    let detail = store.create_order(new_order("First"), "alice").unwrap();
    assert!(detail.item.order.number.starts_with("ORD-"));
    assert_eq!(detail.stages.len(), 7);
    assert!(
        detail
            .stages
            .iter()
            .all(|stage| stage.state == StageState::NotStarted)
    );
    assert_eq!(detail.item.current_stage, StageName::Intake);

    let second = store.create_order(new_order("Second"), "alice").unwrap();
    assert_ne!(detail.item.order.number, second.item.order.number);

    let custom = store
        .create_order(
            NewOrder {
                number: Some("CLIENT-42".into()),
                ..new_order("Third")
            },
            "alice",
        )
        .unwrap();
    assert_eq!(custom.item.order.number, "CLIENT-42");
    assert!(
        store
            .create_order(
                NewOrder {
                    number: Some("CLIENT-42".into()),
                    ..new_order("Dup")
                },
                "alice",
            )
            .is_err(),
        "duplicate explicit number must be rejected"
    );
}
#[test]
fn linked_order_sync_is_atomic_and_preserves_admin_fields() {
    let store = store();
    let synced = store
        .sync_linked_order(
            "TG-ABC",
            "telegram-project",
            OrderPatch {
                title: Some("Telegram title".into()),
                client_name: Some("Client".into()),
                source_language: Some("ja-JP".into()),
                target_language: Some("ru-RU".into()),
                ..OrderPatch::default()
            },
            "telegram",
        )
        .unwrap();
    assert_eq!(synced.project_name.as_deref(), Some("telegram-project"));
    let stages = store.order_detail(&synced.id).unwrap().unwrap().stages;
    assert_eq!(stages[0].state, StageState::Done);
    assert_eq!(stages[1].state, StageState::InProgress);

    store
        .update_order(
            &synced.id,
            OrderPatch {
                priority: Some(Priority::Urgent),
                ..OrderPatch::default()
            },
            "alice",
        )
        .unwrap();
    let occupied = store.create_order(new_order("Other"), "alice").unwrap();
    store
        .attach_project(
            &occupied.item.order.id,
            Some("occupied-project".into()),
            "alice",
        )
        .unwrap();
    assert!(
        store
            .sync_linked_order(
                "TG-ABC",
                "occupied-project",
                OrderPatch {
                    title: Some("Must roll back".into()),
                    ..OrderPatch::default()
                },
                "telegram",
            )
            .is_err()
    );
    let after = store.get_order(&synced.id).unwrap().unwrap();
    assert_eq!(after.title, "Telegram title");
    assert_eq!(after.priority, Priority::Urgent);
    assert_eq!(after.project_name.as_deref(), Some("telegram-project"));
}

#[test]
fn current_stage_is_first_not_done() {
    let store = store();
    let detail = store.create_order(new_order("Flow"), "alice").unwrap();
    let id = detail.item.order.id.clone();
    store
        .set_stage(&id, StageName::Intake, StageState::Done, "alice")
        .unwrap();
    store
        .set_stage(&id, StageName::Preparation, StageState::InProgress, "alice")
        .unwrap();
    let detail = store.order_detail(&id).unwrap().unwrap();
    assert_eq!(detail.item.current_stage, StageName::Preparation);
    assert_eq!(detail.item.current_stage_state, StageState::InProgress);
    let intake = detail
        .stages
        .iter()
        .find(|s| s.stage == StageName::Intake)
        .unwrap();
    assert!(intake.completed_at.is_some());
}

#[test]
fn attention_rules_fire_and_clear() {
    let store = store();
    let now = 1_800_000_000_000; // fixed point in time
    let detail = store
        .create_order(
            NewOrder {
                deadline_at: Some(now - 1000),
                ..new_order("Late")
            },
            "alice",
        )
        .unwrap();
    let id = detail.item.order.id.clone();
    let items = store
        .list_orders(&OrderFilter::default(), &clock(now), &HashSet::new())
        .unwrap();
    assert_eq!(items.len(), 1);
    assert!(items[0].attention.contains(&AttentionReason::Overdue));

    // Overdue cleared by completing the order.
    store
        .set_lifecycle(&id, Lifecycle::Completed, "alice")
        .unwrap();
    let items = store
        .list_orders(&OrderFilter::default(), &clock(now), &HashSet::new())
        .unwrap();
    assert!(
        items.is_empty(),
        "completed orders leave the attention view"
    );

    // Blocker drives both the flag and the summary chip.
    let detail = store.create_order(new_order("Blocked"), "alice").unwrap();
    let blocked_id = detail.item.order.id.clone();
    store
        .add_note(
            &blocked_id,
            NewNote {
                kind: NoteKind::Blocker,
                body: "waiting for client".into(),
                author: "alice".into(),
                page_id: None,
                segment_id: None,
                pinned: false,
            },
        )
        .unwrap();
    let items = store
        .list_orders(&OrderFilter::default(), &clock(now), &HashSet::new())
        .unwrap();
    let blocked = items
        .iter()
        .find(|item| item.order.id == blocked_id)
        .unwrap();
    assert!(blocked.attention.contains(&AttentionReason::Blocked));
    assert_eq!(blocked.open_blockers, 1);
    assert_eq!(store.summary(&clock(now)).unwrap().blocked, 1);

    // Resolving removes the blocker attention.
    let notes = store.list_notes(&blocked_id).unwrap();
    store.set_note_resolved(&notes[0].id, true, "bob").unwrap();
    let items = store
        .list_orders(&OrderFilter::default(), &clock(now), &HashSet::new())
        .unwrap();
    let unblocked = items
        .iter()
        .find(|item| item.order.id == blocked_id)
        .unwrap();
    assert!(!unblocked.attention.contains(&AttentionReason::Blocked));
    assert_eq!(store.summary(&clock(now)).unwrap().blocked, 0);

    // Ready to deliver: all stages before delivery done.
    for stage in StageName::BEFORE_DELIVERY {
        store
            .set_stage(&blocked_id, stage, StageState::Done, "alice")
            .unwrap();
    }
    let items = store
        .list_orders(&OrderFilter::default(), &clock(now), &HashSet::new())
        .unwrap();
    let ready = items
        .iter()
        .find(|item| item.order.id == blocked_id)
        .unwrap();
    assert!(ready.attention.contains(&AttentionReason::ReadyToDeliver));
    assert_eq!(store.summary(&clock(now)).unwrap().ready, 1);
}

#[test]
fn project_link_is_exclusive_and_survives_detach() {
    let store = store();
    let first = store.create_order(new_order("A"), "alice").unwrap();
    let second = store.create_order(new_order("B"), "alice").unwrap();
    store
        .attach_project(&first.item.order.id, Some("book".into()), "alice")
        .unwrap();
    assert!(
        store
            .attach_project(&second.item.order.id, Some("book".into()), "alice")
            .is_err(),
        "a project links to at most one order"
    );
    let order = store
        .attach_project(&first.item.order.id, None, "alice")
        .unwrap();
    assert_eq!(order.project_name, None);
    assert!(
        store
            .attach_project(&second.item.order.id, Some("book".into()), "alice")
            .is_ok()
    );
    assert_eq!(
        store.order_for_project("book").unwrap().as_deref(),
        Some(second.item.order.id.as_str())
    );
}

#[test]
fn progress_is_stored_not_manual() {
    let store = store();
    let detail = store.create_order(new_order("P"), "alice").unwrap();
    store
        .set_progress(
            &detail.item.order.id,
            PageCounts {
                total: 10,
                prepared: 6,
                translated: 3,
            },
        )
        .unwrap();
    let order = store.get_order(&detail.item.order.id).unwrap().unwrap();
    assert_eq!(order.pages_total, 10);
    assert_eq!(order.pages_translated, 3);
}

fn segments() -> Vec<SegmentRow> {
    vec![
        SegmentRow {
            id: "seg-1".into(),
            page: "1".into(),
            source: "こんにちは".into(),
            translation: None,
        },
        SegmentRow {
            id: "seg-2".into(),
            page: "1".into(),
            source: "同じテキスト".into(),
            translation: Some("Same text".into()),
        },
        SegmentRow {
            id: "seg-3".into(),
            page: "2".into(),
            source: "同じテキスト".into(),
            translation: None,
        },
    ]
}

/// Segments as returned by a translator: every empty translation filled in.
fn translated(segments: &[SegmentRow]) -> Vec<SegmentRow> {
    segments
        .iter()
        .map(|segment| SegmentRow {
            translation: Some(
                segment
                    .translation
                    .clone()
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| format!("TR({})", segment.id)),
            ),
            ..segment.clone()
        })
        .collect()
}

#[test]
fn csv_template_roundtrip_validates() {
    let csv = crate::template_csv("ORD-2026-0001", &translated(&segments())).unwrap();
    let report = crate::parse_import("ORD-2026-0001", "t.csv", &csv, &segments());
    assert!(report.file_messages.is_empty());
    assert_eq!(report.rows.len(), 3);
    assert!(
        report.rows.iter().all(|row| !row.has_error()),
        "filled template must validate: {:?}",
        report
            .rows
            .iter()
            .map(|row| &row.messages)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.rows[0].change, RowChange::New);
    assert_eq!(report.rows[1].change, RowChange::Unchanged);
    assert_eq!(report.rows[0].previous_translation, None);
    assert_eq!(
        report.rows[1].previous_translation.as_deref(),
        Some("Same text")
    );
    // Identical source texts (seg-2 vs seg-3) map by segment_id, never by text.
    assert_eq!(report.rows[2].segment_id, "seg-3");
    assert_eq!(report.rows[2].change, RowChange::New);
}

#[test]
fn xlsx_template_roundtrip_validates() {
    let xlsx = crate::template_xlsx("ORD-2026-0001", &translated(&segments())).unwrap();
    let report = crate::parse_import("ORD-2026-0001", "t.xlsx", &xlsx, &segments());
    assert!(report.file_messages.is_empty());
    assert_eq!(report.rows.len(), 3);
    assert!(report.rows.iter().all(|row| !row.has_error()));
}

#[test]
fn import_validation_blocks_on_contract_violations() {
    let csv = |rows: &str| {
        format!("order_number,page,segment_id,source_text,translation\n{rows}").into_bytes()
    };
    // Wrong order number.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-9999,1,seg-1,こんにちは,Hello"),
        &segments(),
    );
    assert!(
        report.rows[0]
            .messages
            .iter()
            .any(|m| m.kind == "wrong_order")
    );
    // Unknown segment.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-0001,1,nope,こんにちは,Hello"),
        &segments(),
    );
    assert!(
        report.rows[0]
            .messages
            .iter()
            .any(|m| m.kind == "unknown_segment")
    );
    // Duplicate segment.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-0001,1,seg-1,こんにちは,Hello\nORD-2026-0001,1,seg-1,こんにちは,Hi"),
        &segments(),
    );
    assert!(
        report.rows[1]
            .messages
            .iter()
            .any(|m| m.kind == "duplicate_segment")
    );
    // Source drift.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-0001,1,seg-1,edited,Hello"),
        &segments(),
    );
    assert!(
        report.rows[0]
            .messages
            .iter()
            .any(|m| m.kind == "source_changed")
    );
    // Page labels are part of the row identity contract.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-0001,2,seg-1,こんにちは,Hello"),
        &segments(),
    );
    assert!(
        report.rows[0]
            .messages
            .iter()
            .any(|m| m.kind == "wrong_page")
    );
    // Empty translation.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-0001,1,seg-1,こんにちは,"),
        &segments(),
    );
    assert!(
        report.rows[0]
            .messages
            .iter()
            .any(|m| m.kind == "empty_translation")
    );
    // Changed translation detected.
    let report = crate::parse_import(
        "ORD-2026-0001",
        "t.csv",
        &csv("ORD-2026-0001,1,seg-2,同じテキスト,Same but better"),
        &segments(),
    );
    assert_eq!(report.rows[0].change, RowChange::Changed);
    // Every project segment must be present; partial files never apply.
    assert!(
        report
            .file_messages
            .iter()
            .any(|message| message.kind == "missing_segments")
    );
    assert!(!report.rows[0].has_error());
    // Windows-1251 fallback: the whole file is 1251-encoded.
    let target = vec![SegmentRow {
        id: "seg-r".into(),
        page: "1".into(),
        source: "Привет".into(),
        translation: None,
    }];
    let mut bytes =
        b"order_number,page,segment_id,source_text,translation\nORD-2026-0001,1,seg-r,".to_vec();
    bytes.extend([0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2]); // "Привет" in cp1251
    bytes.push(b',');
    bytes.extend([0xC7, 0xE4, 0xF0, 0xE0, 0xE2, 0xF1, 0xF2, 0xE2, 0xF3, 0xE9]); // "Здравствуй"
    let report = crate::parse_import("ORD-2026-0001", "t.csv", &bytes, &target);
    assert!(
        !report.rows[0].has_error(),
        "cp1251 file must decode: {:?}",
        report.rows[0].messages
    );
    assert_eq!(report.rows[0].translation, "Здравствуй");
}

#[test]
fn import_lifecycle_in_store() {
    let store = store();
    let detail = store.create_order(new_order("I"), "alice").unwrap();
    let id = detail.item.order.id.clone();
    let csv = crate::template_csv("ORD-x", &translated(&segments())).unwrap();
    let checksum = crate::checksum(&csv);
    let report = crate::parse_import("ORD-x", "t.csv", &csv, &segments());
    let import = store
        .create_import(&id, "t.csv".into(), checksum.clone(), report, "alice")
        .unwrap();
    assert_eq!(import.state, ImportState::Ready);
    assert_eq!(import.accepted, 3);
    // Same checksum finds the pending import again (no duplicate attempts).
    let found = store
        .find_import_by_checksum(&id, &checksum)
        .unwrap()
        .unwrap();
    assert_eq!(found.id, import.id);
    let blocker = store
        .add_note(
            &id,
            NewNote {
                kind: NoteKind::Blocker,
                body: "Clarify terminology".into(),
                author: "alice".into(),
                page_id: None,
                segment_id: None,
                pinned: true,
            },
        )
        .unwrap();
    assert!(
        store
            .finalize_import(&import.id, Vec::new(), "alice")
            .is_err()
    );
    store.set_note_resolved(&blocker.id, true, "alice").unwrap();
    // Finalization is one administrative transaction; re-apply must fail.
    let applied = store
        .finalize_import(&import.id, Vec::new(), "alice")
        .unwrap();
    assert_eq!(applied.state, ImportState::Applied);
    assert!(
        store
            .finalize_import(&import.id, Vec::new(), "alice")
            .is_err()
    );
    // Failed state when the report carries errors.
    let bad = crate::parse_import(
        "ORD-x",
        "t.csv",
        b"order_number,page,segment_id,source_text,translation\nORD-x,1,nope,x,y",
        &segments(),
    );
    let failed = store
        .create_import(&id, "bad.csv".into(), crate::checksum(b"bad"), bad, "alice")
        .unwrap();
    assert_eq!(failed.state, ImportState::Failed);
    // Activity trail covers everything.
    let activity = store.list_activity(&id, 50).unwrap();
    let kinds: Vec<&str> = activity.iter().map(|entry| entry.kind.as_str()).collect();
    assert!(kinds.contains(&"import_uploaded"));
    assert!(kinds.contains(&"import_applied"));
}
#[test]
fn malformed_persisted_import_report_is_not_treated_as_empty() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("orders.sqlite3");
    let store = Store::open(&path).unwrap();
    let order = store.create_order(new_order("I"), "alice").unwrap();
    let csv = crate::template_csv("ORD-x", &translated(&segments())).unwrap();
    let report = crate::parse_import("ORD-x", "t.csv", &csv, &segments());
    let import = store
        .create_import(
            &order.item.order.id,
            "t.csv".into(),
            crate::checksum(&csv),
            report,
            "alice",
        )
        .unwrap();
    drop(store);

    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute(
        "UPDATE translation_imports SET report = '{' WHERE id = ?1",
        rusqlite::params![import.id],
    )
    .unwrap();
    drop(db);

    let store = Store::open(&path).unwrap();
    assert!(matches!(
        store.import(&import.id),
        Err(crate::Error::Json(_))
    ));
    assert!(matches!(
        store.list_imports(&order.item.order.id),
        Err(crate::Error::Json(_))
    ));
}

#[test]
fn filters_select_expected_rows() {
    let store = store();
    let now = 1_800_000_000_000;
    let mine = store
        .create_order(
            NewOrder {
                assignee: Some("alice".into()),
                ..new_order("Manga Alpha")
            },
            "alice",
        )
        .unwrap();
    store.create_order(new_order("Comic Beta"), "bob").unwrap();
    let available = HashSet::new();

    let search = store
        .list_orders(
            &OrderFilter {
                view: OrderView::All,
                search: Some("alpha".into()),
                ..OrderFilter::default()
            },
            &clock(now),
            &available,
        )
        .unwrap();
    assert_eq!(search.len(), 1);
    assert_eq!(search[0].order.id, mine.item.order.id);

    let mine_items = store
        .list_orders(
            &OrderFilter {
                view: OrderView::Mine,
                actor: Some("alice".into()),
                ..OrderFilter::default()
            },
            &clock(now),
            &available,
        )
        .unwrap();
    assert_eq!(mine_items.len(), 1);

    let none = store
        .list_orders(
            &OrderFilter {
                view: OrderView::All,
                lifecycle: Some(vec![Lifecycle::Delivered]),
                ..OrderFilter::default()
            },
            &clock(now),
            &available,
        )
        .unwrap();
    assert!(none.is_empty());
}
