//! A local journal of every inference request Koharu sends, with the token
//! usage and cost the provider reported for it.
//!
//! The journal is one SQLite file next to `config.toml` (`usage.sqlite3`),
//! shared by every part of the application that talks to a model: translation,
//! the agent and OCR. Writers build a [`UsageRecord`] per HTTP attempt and
//! hand it to [`record`]; the desktop reads it back through
//! [`UsageJournal::query`].
//!
//! # What a record may contain
//!
//! Numbers, opaque identifiers and short codes. Prompts, model output,
//! reasoning text, images, credentials and provider error messages are never
//! stored, and there is no export: the file never leaves the machine.
//!
//! # Unknown is not zero
//!
//! Anything the provider did not state is `None`. Token counts are never
//! estimated from text and cost is never derived from a price list; a request
//! has a cost only when the provider put one in its response. Totals report how
//! many requests contributed to each sum so that missing data stays visible.
//!
//! Subset counters do not add up: see [`TokenUsage`].

mod journal;
mod model;
mod query;
pub mod wire;

pub use journal::UsageJournal;
pub use model::{
    ProviderCost, RequestSettings, TokenUsage, UsageContext, UsageOperation, UsageOutcome,
    UsageRecord,
};
pub use query::{CostTotal, TokenTotal, UsageModel, UsagePage, UsageQuery, UsageTotals};

/// Appends `record` to the shared journal.
///
/// Accounting must never fail the paid request it describes, so a journal that
/// cannot be written is logged and the caller carries on.
pub fn record(record: &UsageRecord) {
    if let Err(error) = UsageJournal::shared().and_then(|journal| journal.append(record)) {
        tracing::error!(
            %error,
            provider = %record.provider,
            record = %record.id,
            "failed to write the usage journal"
        );
    }
}

/// Downgrades a request recorded as succeeded whose output a later stage
/// rejected; see [`UsageJournal::amend_outcome`]. Failures are logged.
pub fn amend_outcome(id: &str, outcome: UsageOutcome) {
    if let Err(error) =
        UsageJournal::shared().and_then(|journal| journal.amend_outcome(id, outcome))
    {
        tracing::error!(%error, record = id, "failed to amend the usage journal");
    }
}
