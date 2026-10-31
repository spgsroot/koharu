use serde::{Deserialize, Serialize};
use specta::Type;

use crate::model::{UsageOperation, UsageRecord};

pub(crate) const DEFAULT_LIMIT: u32 = 100;
pub(crate) const MAX_LIMIT: u32 = 500;

/// Filters and paging for [`UsageJournal::query`](crate::UsageJournal::query).
/// Every filter is optional and filters combine with AND.
#[derive(Clone, Debug, Default, Deserialize, Type)]
#[serde(default)]
pub struct UsageQuery {
    pub provider: Option<String>,
    /// Matches the model that was requested.
    pub model: Option<String>,
    pub operation: Option<UsageOperation>,
    /// `true` keeps only succeeded requests, `false` only failed ones.
    pub succeeded: Option<bool>,
    /// Inclusive lower bound on `started_at`, in Unix milliseconds.
    #[specta(type = Option<f64>)]
    pub from: Option<i64>,
    /// Exclusive upper bound on `started_at`, in Unix milliseconds.
    #[specta(type = Option<f64>)]
    pub to: Option<i64>,
    /// The `next_cursor` of the previous page.
    #[specta(type = Option<f64>)]
    pub cursor: Option<i64>,
    /// Page size; defaults to 100 and is capped at 500.
    pub limit: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Type)]
pub struct UsageModel {
    pub provider: String,
    pub model: String,
}

/// One page of the journal, newest first, with totals for the whole filter.
#[derive(Clone, Debug, PartialEq, Serialize, Type)]
pub struct UsagePage {
    pub entries: Vec<UsageRecord>,
    /// Pass back as [`UsageQuery::cursor`] for the next older page.
    #[specta(type = Option<f64>)]
    pub next_cursor: Option<i64>,
    pub has_more: bool,
    /// Aggregates over every record matching the filters, ignoring the cursor
    /// and the page size.
    pub totals: UsageTotals,
    /// Every provider in the journal, regardless of the filters.
    pub providers: Vec<String>,
    /// Every requested model in the journal, regardless of the filters.
    pub models: Vec<UsageModel>,
}

/// Sum of one counter over the requests that reported it.
///
/// Requests that did not report the counter are neither zero nor part of
/// `sum`; they are `UsageTotals::requests - known`.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Type)]
pub struct TokenTotal {
    #[specta(type = f64)]
    pub sum: u64,
    #[specta(type = f64)]
    pub known: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Type)]
pub struct CostTotal {
    pub currency: String,
    pub amount: f64,
    /// Requests that reported a cost in this currency.
    #[specta(type = f64)]
    pub requests: u64,
}

/// Aggregates over the requests matching a query.
///
/// `reasoning` is part of `output`, and `cache_read`, `cache_write`,
/// `input_audio` and `input_image` are part of `input`: show them next to
/// those sums and never add them to `input`, `output` or `total`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Type)]
pub struct UsageTotals {
    #[specta(type = f64)]
    pub requests: u64,
    #[specta(type = f64)]
    pub succeeded: u64,
    #[specta(type = f64)]
    pub failed: u64,
    pub input: TokenTotal,
    pub output: TokenTotal,
    pub total: TokenTotal,
    pub reasoning: TokenTotal,
    pub cache_read: TokenTotal,
    pub cache_write: TokenTotal,
    pub input_audio: TokenTotal,
    pub input_image: TokenTotal,
    pub output_audio: TokenTotal,
    /// Requests that reported none of `input`, `output` and `total`.
    #[specta(type = f64)]
    pub requests_without_usage: u64,
    /// Requests whose provider reported no cost.
    #[specta(type = f64)]
    pub requests_without_cost: u64,
    /// One entry per currency; currencies are never mixed.
    pub costs: Vec<CostTotal>,
}
