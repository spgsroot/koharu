//! Translation exchange format: template export and import validation.
//!
//! One spreadsheet row maps to exactly one text segment, keyed solely by
//! `segment_id` (repeated source lines are common, so text never keys a row).
//! A parsed file never mutates a project: parsing produces an [`ImportReport`]
//! and only an explicit apply step consumes it.

use std::{borrow::Cow, collections::HashMap, io::Cursor};

use calamine::{Data, Reader as _, Xlsx};

use crate::model::*;

/// Canonical column order of the exchange format.
pub const COLUMNS: [&str; 7] = [
    "order_number",
    "page",
    "segment_id",
    "source_text",
    "translation",
    "translator_note",
    "review_status",
];

const REQUIRED: [&str; 5] = [
    "order_number",
    "page",
    "segment_id",
    "source_text",
    "translation",
];

const REVIEW_STATUSES: [&str; 2] = ["translated", "needs_review"];

/// One translatable segment of a project, provided by the caller (the app
/// layer maps scene entities to this plain shape so the format stays
/// testable without the document model).
#[derive(Clone, Debug)]
pub struct SegmentRow {
    /// Layer entity id, serialized UUID — the stable `segment_id`.
    pub id: String,
    /// Human-facing page label.
    pub page: String,
    pub source: String,
    pub translation: Option<String>,
}

impl SegmentRow {
    /// Translation as it appears in the `translation` column.
    fn current(&self) -> &str {
        self.translation.as_deref().unwrap_or("")
    }
}

/// Renders the order's translation template as `.xlsx` (primary format).
pub fn template_xlsx(order_number: &str, segments: &[SegmentRow]) -> crate::Result<Vec<u8>> {
    let mut workbook = rust_xlsxwriter::Workbook::new();
    let worksheet = workbook
        .add_worksheet()
        .set_name("translations")
        .map_err(|error| crate::Error::Input(error.to_string()))?;
    let header = rust_xlsxwriter::Format::new().set_bold();
    for (column, name) in COLUMNS.iter().enumerate() {
        worksheet
            .write_string_with_format(0, column as u16, *name, &header)
            .map_err(|error| crate::Error::Input(error.to_string()))?;
    }
    for (index, segment) in segments.iter().enumerate() {
        let row = (index + 1) as u32;
        let cells = [
            order_number,
            &segment.page,
            &segment.id,
            &segment.source,
            segment.current(),
            "",
            "",
        ];
        for (column, value) in cells.iter().enumerate() {
            worksheet
                .write_string(row, column as u16, *value)
                .map_err(|error| crate::Error::Input(error.to_string()))?;
        }
    }
    workbook
        .save_to_buffer()
        .map_err(|error| crate::Error::Input(error.to_string()))
}

/// Renders the template as UTF-8 CSV with a BOM so spreadsheet tools
/// (Excel) keep non-ASCII text intact.
pub fn template_csv(order_number: &str, segments: &[SegmentRow]) -> crate::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = b"\xEF\xBB\xBF".to_vec();
    {
        let mut writer = csv::Writer::from_writer(&mut bytes);
        writer.write_record(COLUMNS).map_err(csv_error)?;
        for segment in segments {
            writer
                .write_record([
                    order_number,
                    &segment.page,
                    &segment.id,
                    &segment.source,
                    segment.current(),
                    "",
                    "",
                ])
                .map_err(csv_error)?;
        }
        writer
            .flush()
            .map_err(|error| crate::Error::io("csv flush", error))?;
    }
    Ok(bytes)
}

/// Renders an error report of a failed/blocked import as UTF-8 CSV with BOM.
pub fn report_csv(report: &ImportReport) -> crate::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = b"\xEF\xBB\xBF".to_vec();
    {
        let mut writer = csv::Writer::from_writer(&mut bytes);
        writer
            .write_record(["row", "page", "segment_id", "severity", "kind", "detail"])
            .map_err(csv_error)?;
        for message in &report.file_messages {
            writer
                .write_record([
                    "",
                    "",
                    "",
                    severity_name(message.severity),
                    &message.kind,
                    &message.detail,
                ])
                .map_err(csv_error)?;
        }
        for row in &report.rows {
            for message in &row.messages {
                writer
                    .write_record([
                        &row.row.to_string(),
                        &row.page,
                        &row.segment_id,
                        severity_name(message.severity),
                        &message.kind,
                        &message.detail,
                    ])
                    .map_err(csv_error)?;
            }
        }
        writer
            .flush()
            .map_err(|error| crate::Error::io("csv flush", error))?;
    }
    Ok(bytes)
}

fn severity_name(severity: MessageSeverity) -> &'static str {
    match severity {
        MessageSeverity::Warning => "warning",
        MessageSeverity::Error => "error",
    }
}

fn csv_error(error: csv::Error) -> crate::Error {
    crate::Error::Input(format!("csv: {error}"))
}

/// Parses and validates an uploaded translation file against the order's
/// current segments. The result lists every row with its problems; apply is
/// possible only when no error-severity message exists.
pub fn parse_import(
    order_number: &str,
    filename: &str,
    bytes: &[u8],
    segments: &[SegmentRow],
) -> ImportReport {
    let mut report = ImportReport::default();
    let rows = if filename.to_ascii_lowercase().ends_with(".xlsx") {
        parse_xlsx(bytes, &mut report)
    } else if filename.to_ascii_lowercase().ends_with(".csv") {
        parse_csv(bytes, &mut report)
    } else {
        report.file_messages.push(ImportMessage::error(
            "unsupported_format",
            format!("{filename}: only .xlsx and .csv files are supported"),
        ));
        None
    };
    let Some(rows) = rows else { return report };
    validate(order_number, rows, segments, &mut report);
    report
}

fn parse_xlsx(bytes: &[u8], report: &mut ImportReport) -> Option<Vec<Vec<String>>> {
    let mut workbook = match Xlsx::new(Cursor::new(bytes)) {
        Ok(workbook) => workbook,
        Err(error) => {
            report.file_messages.push(ImportMessage::error(
                "unreadable_file",
                format!("not a valid .xlsx workbook: {error}"),
            ));
            return None;
        }
    };
    let range = match workbook.worksheet_range_at(0) {
        Some(Ok(range)) => range,
        Some(Err(error)) => {
            report.file_messages.push(ImportMessage::error(
                "unreadable_file",
                format!("failed to read the first sheet: {error}"),
            ));
            return None;
        }
        None => {
            report.file_messages.push(ImportMessage::error(
                "empty_file",
                "the workbook has no sheets",
            ));
            return None;
        }
    };
    Some(
        range
            .rows()
            .map(|row| row.iter().map(cell_text).collect())
            .collect(),
    )
}

fn cell_text(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(text) => text.to_string(),
        other => other.to_string(),
    }
}

fn parse_csv(bytes: &[u8], report: &mut ImportReport) -> Option<Vec<Vec<String>>> {
    let (text, encoding_warning) = match std::str::from_utf8(bytes) {
        Ok(text) => {
            let stripped = text.strip_prefix('\u{FEFF}').unwrap_or(text);
            let warning = stripped
                .contains('\u{FFFD}')
                .then(|| "file contains replacement characters; check the encoding".to_string());
            (Cow::Borrowed(stripped), warning)
        }
        Err(_) => {
            let (decoded, _, _) = encoding_rs::WINDOWS_1251.decode(bytes);
            (
                Cow::Owned(decoded.into_owned()),
                Some("file is not UTF-8; interpreted as Windows-1251".to_string()),
            )
        }
    };
    if let Some(detail) = encoding_warning {
        report
            .file_messages
            .push(ImportMessage::warning("encoding", detail));
    }
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(false)
        .from_reader(text.as_bytes());
    let mut rows = Vec::new();
    for record in reader.records() {
        match record {
            Ok(record) => rows.push(record.iter().map(str::to_string).collect()),
            Err(error) => {
                report.file_messages.push(ImportMessage::error(
                    "unreadable_file",
                    format!("malformed CSV: {error}"),
                ));
                return None;
            }
        }
    }
    Some(rows)
}

fn validate(
    order_number: &str,
    rows: Vec<Vec<String>>,
    segments: &[SegmentRow],
    report: &mut ImportReport,
) {
    let Some(header) = rows.first() else {
        report.file_messages.push(ImportMessage::error(
            "empty_file",
            "the file has no header row",
        ));
        return;
    };
    let mut index: HashMap<String, usize> = HashMap::new();
    for (position, name) in header.iter().enumerate() {
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() {
            continue;
        }
        if !COLUMNS.contains(&name.as_str()) {
            report.file_messages.push(ImportMessage::warning(
                "unknown_column",
                format!("column {name:?} is not part of the template and is ignored"),
            ));
        }
        index.entry(name).or_insert(position);
    }
    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|name| !index.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        report.file_messages.push(ImportMessage::error(
            "missing_columns",
            format!("missing required columns: {}", missing.join(", ")),
        ));
        return;
    }
    let cell = |row: &[String], name: &str| -> String {
        index
            .get(name)
            .and_then(|position| row.get(*position))
            .map(|value| value.trim().to_string())
            .unwrap_or_default()
    };
    let segments_by_id: HashMap<&str, &SegmentRow> = segments
        .iter()
        .map(|segment| (segment.id.as_str(), segment))
        .collect();
    let mut seen: HashMap<String, u32> = HashMap::new();
    for (offset, row) in rows.iter().enumerate().skip(1) {
        let row_number = (offset + 1) as u32;
        if row.iter().all(|value| value.trim().is_empty()) {
            continue;
        }
        let row_order = cell(row, "order_number");
        let page = cell(row, "page");
        let segment_id = cell(row, "segment_id");
        let source_text = cell(row, "source_text");
        let translation = cell(row, "translation");
        let translator_note = cell(row, "translator_note");
        let review_status = cell(row, "review_status");
        let mut messages = Vec::new();
        if row_order != order_number {
            messages.push(ImportMessage::error(
                "wrong_order",
                format!("row targets order {row_order:?}, this order is {order_number:?}"),
            ));
        }
        let mut change = RowChange::Unchanged;
        let mut previous_translation = None;
        match (
            seen.get(&segment_id),
            segments_by_id.get(segment_id.as_str()),
        ) {
            (Some(first), _) => {
                messages.push(ImportMessage::error(
                    "duplicate_segment",
                    format!("segment_id already used in row {first}"),
                ));
            }
            (None, None) => {
                messages.push(ImportMessage::error(
                    "unknown_segment",
                    "segment_id does not exist in the linked project",
                ));
            }
            (None, Some(segment)) => {
                previous_translation = segment.translation.clone();
                if segment.page != page {
                    messages.push(ImportMessage::error(
                        "wrong_page",
                        format!("segment belongs to page {:?}, not {:?}", segment.page, page),
                    ));
                }
                if segment.source != source_text {
                    messages.push(ImportMessage::error(
                        "source_changed",
                        "source text differs from the project; re-export the template",
                    ));
                }
                if translation.is_empty() {
                    messages.push(ImportMessage::error(
                        "empty_translation",
                        "translation must not be empty",
                    ));
                }
                change = match segment.translation.as_deref() {
                    None => RowChange::New,
                    Some(current) if current == translation => RowChange::Unchanged,
                    Some(_) => RowChange::Changed,
                };
                if !review_status.is_empty() && !REVIEW_STATUSES.contains(&review_status.as_str()) {
                    messages.push(ImportMessage::warning(
                        "unknown_review_status",
                        format!("{review_status:?} is not translated|needs_review"),
                    ));
                }
            }
        }
        if !segment_id.is_empty() {
            seen.entry(segment_id.clone()).or_insert(row_number);
        }
        report.rows.push(ImportRow {
            row: row_number,
            page,
            segment_id,
            source_text,
            translation,
            previous_translation,
            translator_note: (!translator_note.is_empty()).then_some(translator_note),
            review_status: REVIEW_STATUSES
                .contains(&review_status.as_str())
                .then_some(review_status),
            change,
            messages,
        });
    }
    let missing_segments = segments
        .iter()
        .filter(|segment| !seen.contains_key(&segment.id))
        .count();
    if missing_segments > 0 {
        report.file_messages.push(ImportMessage::error(
            "missing_segments",
            format!(
                "translation file omits {missing_segments} project segment(s); re-export the template"
            ),
        ));
    }
    if report.rows.is_empty() {
        report.file_messages.push(ImportMessage::error(
            "empty_file",
            "the file has no data rows",
        ));
    }
}
