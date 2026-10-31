//! Administrative order management for Koharu: a standalone SQLite store and
//! the `.xlsx`/`.csv` translation exchange format.
//!
//! Orders are administrative entities; projects remain editor documents. The
//! store never lives inside a `.khrproj`, and parsing a translation file never
//! mutates a project (see `docs/workflow/orders.md`).

mod error;
mod exchange;
mod model;
mod store;

pub use error::{Error, Result};
pub use exchange::{COLUMNS, SegmentRow, parse_import, report_csv, template_csv, template_xlsx};
pub use model::*;
pub use store::Store;

/// blake3 hex digest used to deduplicate uploaded translation files.
pub fn checksum(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests;
