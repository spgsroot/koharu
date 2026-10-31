use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use rusqlite::{Connection, Row, params, params_from_iter, types::Value};

use crate::{
    CostTotal, TokenTotal, UsageModel, UsageOperation, UsageOutcome, UsagePage, UsageQuery,
    UsageRecord, UsageTotals,
    model::{ProviderCost, RequestSettings, TokenUsage, UsageContext},
    query::{DEFAULT_LIMIT, MAX_LIMIT},
};

const SCHEMA_VERSION: u32 = 1;
const FILE_NAME: &str = "usage.sqlite3";

/// Idempotent so that two processes opening a fresh file cannot collide.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS usage_requests (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    started_at INTEGER NOT NULL,
    operation TEXT NOT NULL,
    provider TEXT NOT NULL,
    requested_model TEXT,
    resolved_model TEXT,
    provider_request_id TEXT,
    provider_response_id TEXT,
    outcome TEXT NOT NULL,
    error_code TEXT,
    http_status INTEGER,
    latency_ms INTEGER,
    attempt INTEGER NOT NULL,
    transport_retry INTEGER NOT NULL,
    reasoning INTEGER,
    reasoning_effort TEXT,
    temperature REAL,
    top_p REAL,
    top_k INTEGER,
    max_tokens INTEGER,
    frequency_penalty REAL,
    presence_penalty REAL,
    project_id TEXT,
    page_id TEXT,
    job_id TEXT,
    source_language TEXT,
    target_language TEXT,
    segments INTEGER,
    images INTEGER,
    context_entries INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    reasoning_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    input_audio_tokens INTEGER,
    input_image_tokens INTEGER,
    output_audio_tokens INTEGER,
    cost_amount REAL,
    cost_currency TEXT,
    cost_source TEXT,
    CHECK ((cost_amount IS NULL) = (cost_currency IS NULL)
       AND (cost_amount IS NULL) = (cost_source IS NULL))
) STRICT;
CREATE INDEX IF NOT EXISTS usage_requests_started ON usage_requests(started_at);
CREATE INDEX IF NOT EXISTS usage_requests_model ON usage_requests(provider, requested_model);
";

/// Every stored column except `seq`, in the order [`UsageJournal::append`]
/// binds them.
const COLUMNS: &str = "id, started_at, operation, provider, requested_model, resolved_model, \
    provider_request_id, provider_response_id, outcome, error_code, http_status, latency_ms, \
    attempt, transport_retry, reasoning, reasoning_effort, temperature, top_p, top_k, \
    max_tokens, frequency_penalty, presence_penalty, project_id, page_id, job_id, \
    source_language, target_language, segments, images, context_entries, input_tokens, \
    output_tokens, total_tokens, reasoning_tokens, cache_read_tokens, cache_write_tokens, \
    input_audio_tokens, input_image_tokens, output_audio_tokens, cost_amount, cost_currency, \
    cost_source";

const INSERT: &str = "INSERT INTO usage_requests (id, started_at, operation, provider, \
    requested_model, resolved_model, provider_request_id, provider_response_id, outcome, \
    error_code, http_status, latency_ms, attempt, transport_retry, reasoning, reasoning_effort, \
    temperature, top_p, top_k, max_tokens, frequency_penalty, presence_penalty, project_id, \
    page_id, job_id, source_language, target_language, segments, images, context_entries, \
    input_tokens, output_tokens, total_tokens, reasoning_tokens, cache_read_tokens, \
    cache_write_tokens, input_audio_tokens, input_image_tokens, output_audio_tokens, \
    cost_amount, cost_currency, cost_source) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, \
    ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, \
    ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40, ?41, ?42)";

static CONFIGURED: OnceLock<PathBuf> = OnceLock::new();
static SHARED: OnceLock<UsageJournal> = OnceLock::new();
/// Serializes `configure` and the first `shared` so they cannot disagree.
static OPENING: Mutex<()> = Mutex::new(());

/// Durable, append-only SQLite journal of inference requests.
///
/// One file per machine, next to Koharu's `config.toml`. The journal holds
/// numbers, identifiers and short codes only; see [`UsageRecord`].
pub struct UsageJournal {
    db: Mutex<Connection>,
    path: Option<PathBuf>,
}

impl UsageJournal {
    /// Selects the journal file before the shared journal is first used, in
    /// the manner of `koharu_runtime::Store::configure`. Calling it again with
    /// the same path is fine; a different path is an error.
    pub fn configure(path: impl Into<PathBuf>) -> Result<()> {
        let requested = path.into();
        ensure!(
            requested.is_absolute(),
            "usage journal must be an absolute path: {}",
            requested.display()
        );
        let _opening = OPENING.lock();
        if let Some(open) = SHARED.get() {
            ensure!(
                open.path.as_deref() == Some(requested.as_path()),
                "usage journal is already open at {}",
                open.path
                    .as_deref()
                    .map_or_else(String::new, |path| path.display().to_string())
            );
            return Ok(());
        }
        match CONFIGURED.get() {
            Some(active) => ensure!(
                active == &requested,
                "usage journal is already configured as {}",
                active.display()
            ),
            None => {
                let _ = CONFIGURED.set(requested);
            }
        }
        Ok(())
    }

    /// The process-wide journal, opened on first use at the configured path or
    /// at `usage.sqlite3` next to `config.toml`.
    pub fn shared() -> Result<&'static Self> {
        if let Some(journal) = SHARED.get() {
            return Ok(journal);
        }
        let _opening = OPENING.lock();
        if let Some(journal) = SHARED.get() {
            return Ok(journal);
        }
        let path = match CONFIGURED.get() {
            Some(path) => path.clone(),
            None => default_path()?,
        };
        let journal = Self::open(&path)?;
        Ok(SHARED.get_or_init(|| journal))
    }

    /// Opens (creating when missing) the journal at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create the usage journal directory {}",
                    parent.display()
                )
            })?;
        }
        let connection = Connection::open(path)
            .with_context(|| format!("failed to open the usage journal {}", path.display()))?;
        Self::init(connection, Some(path.to_owned()))
    }

    /// A journal that lives only as long as the value, for tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?, None)
    }

    fn init(connection: Connection, path: Option<PathBuf>) -> Result<Self> {
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        // Each row is a billing fact; losing the latest ones to a power cut is
        // not acceptable, and a request takes far longer than an fsync.
        connection.pragma_update(None, "synchronous", "FULL")?;
        let version: u32 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
        ensure!(
            version <= SCHEMA_VERSION,
            "usage journal schema {version} is newer than the supported {SCHEMA_VERSION}"
        );
        if version < SCHEMA_VERSION {
            connection
                .execute_batch(&format!(
                    "BEGIN IMMEDIATE;{SCHEMA}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;"
                ))
                .context("failed to create the usage journal schema")?;
        }
        Ok(Self {
            db: Mutex::new(connection),
            path,
        })
    }

    /// Appends one request. Every call writes a new row; a repeated id is an
    /// error.
    pub fn append(&self, record: &UsageRecord) -> Result<()> {
        let record = record.sanitized();
        let cost = record.cost.as_ref();
        let db = self.db.lock();
        db.prepare_cached(INSERT)?
            .execute(params![
                record.id,
                record.started_at,
                record.operation.as_str(),
                record.provider,
                record.requested_model,
                record.resolved_model,
                record.provider_request_id,
                record.provider_response_id,
                record.outcome.as_str(),
                record.error_code,
                record.http_status,
                record.latency_ms,
                record.attempt,
                record.transport_retry,
                record.settings.reasoning,
                record.settings.reasoning_effort,
                record.settings.temperature,
                record.settings.top_p,
                record.settings.top_k,
                record.settings.max_tokens,
                record.settings.frequency_penalty,
                record.settings.presence_penalty,
                record.context.project_id,
                record.context.page_id,
                record.context.job_id,
                record.context.source_language,
                record.context.target_language,
                record.context.segments,
                record.context.images,
                record.context.context_entries,
                record.tokens.input,
                record.tokens.output,
                record.tokens.total,
                record.tokens.reasoning,
                record.tokens.cache_read,
                record.tokens.cache_write,
                record.tokens.input_audio,
                record.tokens.input_image,
                record.tokens.output_audio,
                cost.map(|cost| cost.amount),
                cost.map(|cost| cost.currency.as_str()),
                cost.map(|cost| cost.source.as_str()),
            ])
            .with_context(|| format!("failed to journal usage record {}", record.id))?;
        Ok(())
    }

    /// Downgrades a request that was recorded as `Succeeded` once a later
    /// stage (parsing, validation) rejects its output. Failure classifications
    /// are never overwritten. Returns whether the row changed.
    pub fn amend_outcome(&self, id: &str, outcome: UsageOutcome) -> Result<bool> {
        let changed = self
            .db
            .lock()
            .execute(
                "UPDATE usage_requests SET outcome = ?1 WHERE id = ?2 AND outcome = 'succeeded'",
                params![outcome.as_str(), id],
            )
            .with_context(|| format!("failed to amend usage record {id}"))?;
        Ok(changed == 1)
    }

    /// One page of the journal, newest first, plus totals for the whole filter.
    pub fn query(&self, query: &UsageQuery) -> Result<UsagePage> {
        let limit = query.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT) as usize;
        let filter = Filter::new(query);
        let db = self.db.lock();

        let totals = totals(&db, &filter)?;

        let mut values = filter.values.clone();
        let cursor = query.cursor.map_or("", |cursor| {
            values.push(Value::Integer(cursor));
            " AND seq < ?"
        });
        values.push(Value::Integer(i64::try_from(limit + 1).unwrap_or(i64::MAX)));
        let mut statement = db.prepare(&format!(
            "SELECT seq, {COLUMNS} FROM usage_requests WHERE {}{cursor} ORDER BY seq DESC LIMIT ?",
            filter.clause
        ))?;
        let mut rows = statement
            .query_map(params_from_iter(values), |row| {
                Ok((row.get::<_, i64>("seq")?, read_record(row)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let has_more = rows.len() > limit;
        rows.truncate(limit);
        let next_cursor = has_more.then(|| rows.last().map(|(seq, _)| *seq)).flatten();

        let providers = db
            .prepare("SELECT DISTINCT provider FROM usage_requests ORDER BY provider")?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        let models = db
            .prepare(
                "SELECT DISTINCT provider, requested_model FROM usage_requests \
                 WHERE requested_model IS NOT NULL ORDER BY provider, requested_model",
            )?
            .query_map([], |row| {
                Ok(UsageModel {
                    provider: row.get(0)?,
                    model: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(UsagePage {
            entries: rows.into_iter().map(|(_, record)| record).collect(),
            next_cursor,
            has_more,
            totals,
            providers,
            models,
        })
    }
}

fn default_path() -> Result<PathBuf> {
    Ok(koharu_config::path()?.with_file_name(FILE_NAME))
}

/// A WHERE clause with its positional values.
struct Filter {
    clause: String,
    values: Vec<Value>,
}

impl Filter {
    fn new(query: &UsageQuery) -> Self {
        let mut clauses = Vec::new();
        let mut values = Vec::new();
        if let Some(provider) = &query.provider {
            clauses.push("provider = ?");
            values.push(Value::Text(provider.clone()));
        }
        if let Some(model) = &query.model {
            clauses.push("requested_model = ?");
            values.push(Value::Text(model.clone()));
        }
        if let Some(operation) = query.operation {
            clauses.push("operation = ?");
            values.push(Value::Text(operation.as_str().to_owned()));
        }
        match query.succeeded {
            Some(true) => clauses.push("outcome = 'succeeded'"),
            Some(false) => clauses.push("outcome <> 'succeeded'"),
            None => {}
        }
        if let Some(from) = query.from {
            clauses.push("started_at >= ?");
            values.push(Value::Integer(from));
        }
        if let Some(to) = query.to {
            clauses.push("started_at < ?");
            values.push(Value::Integer(to));
        }
        Self {
            clause: if clauses.is_empty() {
                "1 = 1".to_owned()
            } else {
                clauses.join(" AND ")
            },
            values,
        }
    }
}

fn totals(db: &Connection, filter: &Filter) -> Result<UsageTotals> {
    let mut totals = db.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(outcome = 'succeeded'), 0), \
             COUNT(input_tokens), COALESCE(SUM(input_tokens), 0), \
             COUNT(output_tokens), COALESCE(SUM(output_tokens), 0), \
             COUNT(total_tokens), COALESCE(SUM(total_tokens), 0), \
             COUNT(reasoning_tokens), COALESCE(SUM(reasoning_tokens), 0), \
             COUNT(cache_read_tokens), COALESCE(SUM(cache_read_tokens), 0), \
             COUNT(cache_write_tokens), COALESCE(SUM(cache_write_tokens), 0), \
             COUNT(input_audio_tokens), COALESCE(SUM(input_audio_tokens), 0), \
             COUNT(input_image_tokens), COALESCE(SUM(input_image_tokens), 0), \
             COUNT(output_audio_tokens), COALESCE(SUM(output_audio_tokens), 0), \
             COALESCE(SUM(input_tokens IS NULL AND output_tokens IS NULL \
                          AND total_tokens IS NULL), 0), \
             COALESCE(SUM(cost_amount IS NULL), 0) \
             FROM usage_requests WHERE {}",
            filter.clause
        ),
        params_from_iter(filter.values.iter()),
        |row| {
            let count = |index: usize| row.get::<_, i64>(index).map(|value| value.max(0) as u64);
            let token = |index: usize| -> rusqlite::Result<TokenTotal> {
                Ok(TokenTotal {
                    known: count(index)?,
                    sum: count(index + 1)?,
                })
            };
            let requests = count(0)?;
            let succeeded = count(1)?;
            Ok(UsageTotals {
                requests,
                succeeded,
                failed: requests - succeeded,
                input: token(2)?,
                output: token(4)?,
                total: token(6)?,
                reasoning: token(8)?,
                cache_read: token(10)?,
                cache_write: token(12)?,
                input_audio: token(14)?,
                input_image: token(16)?,
                output_audio: token(18)?,
                requests_without_usage: count(20)?,
                requests_without_cost: count(21)?,
                costs: Vec::new(),
            })
        },
    )?;
    totals.costs = db
        .prepare(&format!(
            "SELECT cost_currency, SUM(cost_amount), COUNT(*) FROM usage_requests \
             WHERE {} AND cost_amount IS NOT NULL GROUP BY cost_currency ORDER BY cost_currency",
            filter.clause
        ))?
        .query_map(params_from_iter(filter.values.iter()), |row| {
            Ok(CostTotal {
                currency: row.get(0)?,
                amount: row.get(1)?,
                requests: row.get::<_, i64>(2)?.max(0) as u64,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(totals)
}

fn read_record(row: &Row<'_>) -> rusqlite::Result<UsageRecord> {
    let cost = match row.get::<_, Option<f64>>("cost_amount")? {
        Some(amount) => Some(ProviderCost {
            amount,
            currency: row.get("cost_currency")?,
            source: row.get("cost_source")?,
        }),
        None => None,
    };
    Ok(UsageRecord {
        id: row.get("id")?,
        started_at: row.get("started_at")?,
        operation: text_enum(row, "operation", UsageOperation::parse)?,
        provider: row.get("provider")?,
        requested_model: row.get("requested_model")?,
        resolved_model: row.get("resolved_model")?,
        provider_request_id: row.get("provider_request_id")?,
        provider_response_id: row.get("provider_response_id")?,
        outcome: text_enum(row, "outcome", UsageOutcome::parse)?,
        error_code: row.get("error_code")?,
        http_status: row.get("http_status")?,
        latency_ms: row.get("latency_ms")?,
        attempt: row.get("attempt")?,
        transport_retry: row.get("transport_retry")?,
        settings: RequestSettings {
            reasoning: row.get("reasoning")?,
            reasoning_effort: row.get("reasoning_effort")?,
            temperature: row.get("temperature")?,
            top_p: row.get("top_p")?,
            top_k: row.get("top_k")?,
            max_tokens: row.get("max_tokens")?,
            frequency_penalty: row.get("frequency_penalty")?,
            presence_penalty: row.get("presence_penalty")?,
        },
        context: UsageContext {
            project_id: row.get("project_id")?,
            page_id: row.get("page_id")?,
            job_id: row.get("job_id")?,
            source_language: row.get("source_language")?,
            target_language: row.get("target_language")?,
            segments: row.get("segments")?,
            images: row.get("images")?,
            context_entries: row.get("context_entries")?,
        },
        tokens: TokenUsage {
            input: row.get("input_tokens")?,
            output: row.get("output_tokens")?,
            total: row.get("total_tokens")?,
            reasoning: row.get("reasoning_tokens")?,
            cache_read: row.get("cache_read_tokens")?,
            cache_write: row.get("cache_write_tokens")?,
            input_audio: row.get("input_audio_tokens")?,
            input_image: row.get("input_image_tokens")?,
            output_audio: row.get("output_audio_tokens")?,
        },
        cost,
    })
}

fn text_enum<T>(row: &Row<'_>, column: &str, parse: fn(&str) -> Option<T>) -> rusqlite::Result<T> {
    let text: String = row.get(column)?;
    parse(&text).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            format!("unknown {column} {text:?} in the usage journal").into(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(provider: &str, started_at: i64) -> UsageRecord {
        let mut record = UsageRecord::new(UsageOperation::Translation, provider);
        record.started_at = started_at;
        record.outcome = UsageOutcome::Succeeded;
        record
    }

    fn full_record() -> UsageRecord {
        UsageRecord {
            id: "018f0000-0000-7000-8000-000000000001".to_owned(),
            started_at: 1_700_000_000_123,
            operation: UsageOperation::Translation,
            provider: "openrouter".to_owned(),
            requested_model: Some("vendor/model".to_owned()),
            resolved_model: Some("vendor/model-2025".to_owned()),
            provider_request_id: Some("req-1".to_owned()),
            provider_response_id: Some("gen-1".to_owned()),
            outcome: UsageOutcome::ValidationRejected,
            error_code: Some("rate_limit".to_owned()),
            http_status: Some(200),
            latency_ms: Some(1234),
            attempt: 2,
            transport_retry: 1,
            settings: RequestSettings {
                reasoning: Some(true),
                reasoning_effort: Some("high".to_owned()),
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(40),
                max_tokens: Some(4096),
                frequency_penalty: Some(0.25),
                presence_penalty: Some(-0.5),
            },
            context: UsageContext {
                project_id: Some("project".to_owned()),
                page_id: Some("page".to_owned()),
                job_id: Some("job".to_owned()),
                source_language: Some("ja".to_owned()),
                target_language: Some("ru".to_owned()),
                segments: Some(12),
                images: Some(1),
                context_entries: Some(3),
            },
            tokens: TokenUsage {
                input: Some(1000),
                output: Some(300),
                total: Some(1300),
                reasoning: Some(120),
                cache_read: Some(400),
                cache_write: Some(50),
                input_audio: Some(7),
                input_image: Some(256),
                output_audio: Some(9),
            },
            cost: Some(ProviderCost {
                amount: 0.000_123_45,
                currency: "credits".to_owned(),
                source: "usage.cost".to_owned(),
            }),
        }
    }

    #[test]
    fn every_field_round_trips_through_the_journal() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let expected = full_record();
        journal.append(&expected).unwrap();

        let page = journal.query(&UsageQuery::default()).unwrap();

        assert_eq!(page.entries, [expected]);
        assert!(!page.has_more);
        assert_eq!(page.next_cursor, None);
    }

    #[test]
    fn insert_statement_binds_every_column() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let db = journal.db.lock();
        let statement = db.prepare(INSERT).unwrap();
        assert_eq!(statement.parameter_count(), COLUMNS.split(',').count());
    }

    #[test]
    fn unreported_counters_stay_null_and_do_not_become_zero() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let mut known = record("openai", 1);
        known.tokens.input = Some(10);
        known.tokens.output = Some(5);
        known.tokens.total = Some(15);
        let unknown = record("openai", 2);
        journal.append(&known).unwrap();
        journal.append(&unknown).unwrap();

        let page = journal.query(&UsageQuery::default()).unwrap();

        assert_eq!(page.entries[0].tokens, TokenUsage::default());
        assert_eq!(page.entries[0].cost, None);
        assert_eq!(page.totals.requests, 2);
        assert_eq!(page.totals.input, TokenTotal { sum: 10, known: 1 });
        assert_eq!(page.totals.reasoning, TokenTotal { sum: 0, known: 0 });
        assert_eq!(page.totals.requests_without_usage, 1);
        assert_eq!(page.totals.requests_without_cost, 2);
        assert!(page.totals.costs.is_empty());
    }

    #[test]
    fn subset_counters_are_reported_beside_totals_not_added_to_them() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let mut request = record("openai", 1);
        request.tokens = TokenUsage {
            input: Some(100),
            output: Some(40),
            total: Some(140),
            reasoning: Some(25),
            cache_read: Some(60),
            ..TokenUsage::default()
        };
        journal.append(&request).unwrap();

        let totals = journal.query(&UsageQuery::default()).unwrap().totals;

        assert_eq!(totals.input.sum, 100);
        assert_eq!(totals.output.sum, 40);
        assert_eq!(totals.total.sum, 140);
        assert_eq!(totals.reasoning.sum, 25);
        assert_eq!(totals.cache_read.sum, 60);
    }

    #[test]
    fn costs_are_summed_per_currency_and_never_mixed() {
        let journal = UsageJournal::open_in_memory().unwrap();
        for (index, (amount, currency)) in [(0.5, "USD"), (0.25, "USD"), (3.0, "credits")]
            .into_iter()
            .enumerate()
        {
            let mut request = record("grok", index as i64);
            request.cost = Some(ProviderCost {
                amount,
                currency: currency.to_owned(),
                source: "usage.cost".to_owned(),
            });
            journal.append(&request).unwrap();
        }
        journal.append(&record("grok", 10)).unwrap();

        let totals = journal.query(&UsageQuery::default()).unwrap().totals;

        assert_eq!(
            totals.costs,
            [
                CostTotal {
                    currency: "USD".to_owned(),
                    amount: 0.75,
                    requests: 2
                },
                CostTotal {
                    currency: "credits".to_owned(),
                    amount: 3.0,
                    requests: 1
                },
            ]
        );
        assert_eq!(totals.requests_without_cost, 1);
    }

    #[test]
    fn pages_walk_newest_first_without_gaps_or_repeats() {
        let journal = UsageJournal::open_in_memory().unwrap();
        for index in 0..5 {
            journal.append(&record("openai", index)).unwrap();
        }

        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let page = journal
                .query(&UsageQuery {
                    limit: Some(2),
                    cursor,
                    ..UsageQuery::default()
                })
                .unwrap();
            assert_eq!(
                page.totals.requests, 5,
                "totals ignore the cursor and limit"
            );
            seen.extend(page.entries.iter().map(|entry| entry.started_at));
            if !page.has_more {
                assert_eq!(page.next_cursor, None);
                break;
            }
            cursor = page.next_cursor;
            assert!(cursor.is_some());
        }

        assert_eq!(seen, [4, 3, 2, 1, 0]);
    }

    #[test]
    fn filters_combine_and_facets_ignore_them() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let mut a = record("openai", 100);
        a.requested_model = Some("gpt".to_owned());
        let mut b = record("openai", 200);
        b.requested_model = Some("gpt".to_owned());
        b.outcome = UsageOutcome::QuotaExceeded;
        let mut c = record("claude", 300);
        c.requested_model = Some("sonnet".to_owned());
        c.operation = UsageOperation::Agent;
        for request in [&a, &b, &c] {
            journal.append(request).unwrap();
        }

        let by_provider = |query: UsageQuery| journal.query(&query).unwrap();

        let page = by_provider(UsageQuery {
            provider: Some("openai".to_owned()),
            ..UsageQuery::default()
        });
        assert_eq!(page.totals.requests, 2);
        assert_eq!(page.providers, ["claude", "openai"]);
        assert_eq!(page.models.len(), 2);

        assert_eq!(
            by_provider(UsageQuery {
                succeeded: Some(false),
                ..UsageQuery::default()
            })
            .entries[0]
                .started_at,
            200
        );
        assert_eq!(
            by_provider(UsageQuery {
                operation: Some(UsageOperation::Agent),
                model: Some("sonnet".to_owned()),
                ..UsageQuery::default()
            })
            .totals
            .requests,
            1
        );
        let window = by_provider(UsageQuery {
            from: Some(100),
            to: Some(300),
            ..UsageQuery::default()
        });
        assert_eq!(
            window.totals.requests, 2,
            "from is inclusive, to is exclusive"
        );
        assert_eq!(window.totals.succeeded, 1);
        assert_eq!(window.totals.failed, 1);
    }

    #[test]
    fn amending_only_downgrades_succeeded_requests() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let ok = record("openai", 1);
        let mut failed = record("openai", 2);
        failed.outcome = UsageOutcome::QuotaExceeded;
        journal.append(&ok).unwrap();
        journal.append(&failed).unwrap();

        assert!(
            journal
                .amend_outcome(&ok.id, UsageOutcome::ValidationRejected)
                .unwrap()
        );
        assert!(
            !journal
                .amend_outcome(&failed.id, UsageOutcome::InvalidOutput)
                .unwrap()
        );
        assert!(
            !journal
                .amend_outcome("missing", UsageOutcome::Cancelled)
                .unwrap()
        );

        let page = journal.query(&UsageQuery::default()).unwrap();
        assert_eq!(page.entries[0].outcome, UsageOutcome::QuotaExceeded);
        assert_eq!(page.entries[1].outcome, UsageOutcome::ValidationRejected);
    }

    #[test]
    fn repeated_ids_are_rejected() {
        let journal = UsageJournal::open_in_memory().unwrap();
        let request = record("openai", 1);
        journal.append(&request).unwrap();
        assert!(journal.append(&request).is_err());
    }

    #[test]
    fn records_survive_reopening_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested").join(FILE_NAME);
        let request = full_record();
        UsageJournal::open(&path).unwrap().append(&request).unwrap();

        let reopened = UsageJournal::open(&path).unwrap();

        assert_eq!(
            reopened.query(&UsageQuery::default()).unwrap().entries,
            [request]
        );
    }

    #[test]
    fn newer_schemas_are_refused() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(FILE_NAME);
        Connection::open(&path)
            .unwrap()
            .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .unwrap();

        let error = UsageJournal::open(&path).err().unwrap();

        assert!(error.to_string().contains("newer"));
    }

    #[test]
    fn configure_requires_an_absolute_path() {
        assert!(UsageJournal::configure("relative/usage.sqlite3").is_err());
    }

    #[test]
    fn page_size_is_bounded() {
        let journal = UsageJournal::open_in_memory().unwrap();
        for index in 0..3 {
            journal.append(&record("openai", index)).unwrap();
        }
        let zero = journal
            .query(&UsageQuery {
                limit: Some(0),
                ..UsageQuery::default()
            })
            .unwrap();
        assert_eq!(zero.entries.len(), 1);
        assert!(zero.has_more);
    }
}
