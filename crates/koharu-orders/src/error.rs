//! Store and exchange error type.

/// Errors of the orders store and exchange parsing.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("failed to open store {0}: {1}")]
    Open(String, #[source] rusqlite::Error),

    #[error("io: {0}: {1}")]
    Io(String, #[source] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("invalid input: {0}")]
    Input(String),

    #[error("migration: {0}")]
    Migration(String),
}

impl Error {
    pub(crate) fn sql(source: rusqlite::Error) -> Self {
        Self::Sqlite(source)
    }

    pub(crate) fn open(path: String, source: rusqlite::Error) -> Self {
        Self::Open(path, source)
    }

    pub(crate) fn io(context: &str, source: std::io::Error) -> Self {
        Self::Io(context.to_owned(), source)
    }

    pub(crate) fn json(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
