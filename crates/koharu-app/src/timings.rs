//! Append-only journal of stage, commit and job durations, kept beside
//! `config.toml` so performance work is measured on real runs. Records carry
//! identifiers and durations only, never page text, prompts or images.

use std::{
    fs::OpenOptions,
    io::Write as _,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result};
use serde::Serialize;

const FILE_NAME: &str = "stage-timings.jsonl";
/// Beyond this size the journal rotates to one `.1` file, bounding disk use.
const MAX_BYTES: u64 = 4 * 1024 * 1024;

static WRITE: Mutex<()> = Mutex::new(());

#[derive(Serialize)]
pub(crate) struct Timing {
    pub(crate) job: String,
    pub(crate) event: &'static str,
    pub(crate) page: Option<String>,
    pub(crate) stage: Option<String>,
    pub(crate) model: Option<String>,
    #[serde(serialize_with = "milliseconds")]
    pub(crate) elapsed: Duration,
}

#[derive(Serialize)]
struct Record<'a> {
    at_ms: u128,
    #[serde(flatten)]
    timing: &'a Timing,
}

pub(crate) fn record(timing: &Timing) {
    if let Err(error) = append(timing) {
        tracing::debug!(%error, "failed to record a stage timing");
    }
}

fn append(timing: &Timing) -> Result<()> {
    let path = koharu_config::path()?.with_file_name(FILE_NAME);
    let line = serde_json::to_string(&Record {
        at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        timing,
    })?;
    let _write = WRITE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if std::fs::metadata(&path).is_ok_and(|metadata| metadata.len() > MAX_BYTES) {
        std::fs::rename(&path, path.with_extension("jsonl.1"))
            .context("failed to rotate the stage timing journal")?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    writeln!(file, "{line}")?;
    Ok(())
}

fn milliseconds<S: serde::Serializer>(
    elapsed: &Duration,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(elapsed.as_secs_f64() * 1000.0)
}
