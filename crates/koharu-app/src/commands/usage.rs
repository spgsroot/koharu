use anyhow::Context as _;
use koharu_usage::{UsageJournal, UsagePage, UsageQuery};

use super::Error;

/// One page of the local inference usage journal, newest first, with totals
/// over everything the filters match.
///
/// The journal is local-only and stays out of `koharu_metrics`.
#[tauri::command]
#[specta::specta]
pub(crate) async fn get_usage_log(query: UsageQuery) -> std::result::Result<UsagePage, Error> {
    // Totals scan every matching row, so keep the query off the async workers.
    let page = tokio::task::spawn_blocking(move || UsageJournal::shared()?.query(&query))
        .await
        .context("the usage journal query was interrupted")??;
    Ok(page)
}
