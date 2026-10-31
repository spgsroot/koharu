use std::{env, path::PathBuf, sync::Arc};

use anyhow::{Context as _, Result, bail};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use koharu_orders::OrderPatch;
use koharu_pipeline::{Operation, Scope, Stage, TranslationOptions};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Cef, Manager as _};

use crate::commands::{
    lifecycle,
    orders::Orders,
    output::{self, ExportFormat},
    processing::{Job, Processing},
    project::{CurrentProject, Project},
};

#[derive(Clone)]
struct AutomationState {
    handle: AppHandle<Cef>,
    token: Arc<str>,
    root: Arc<PathBuf>,
    pinned: Arc<parking_lot::Mutex<Option<PinnedProject>>>,
    create_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone)]
struct PinnedProject {
    name: String,
    session: String,
    /// Changes whenever any project is opened, including a reopening of the
    /// same file, so an old client can never act on a later opening.
    processing_session: u64,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(StatusCode::BAD_REQUEST, format!("{error:#}"))
    }
}

type ApiResult<T> = std::result::Result<Json<T>, ApiError>;

#[derive(Serialize)]
struct Health {
    status: &'static str,
    pipeline_stages: &'static [Stage],
}

#[derive(Deserialize)]
struct CreateProject {
    name: String,
    order: Option<DashboardOrder>,
}

#[derive(Deserialize)]
struct DashboardOrder {
    number: String,
    title: String,
    #[serde(default)]
    client_name: String,
    #[serde(default)]
    source_language: String,
    #[serde(default)]
    target_language: String,
    assignee: Option<String>,
    #[serde(default)]
    brief: String,
}

#[derive(Serialize)]
struct ProjectCreated {
    name: String,
    session: String,
}

#[derive(Deserialize)]
struct ImportPages {
    paths: Vec<PathBuf>,
}

#[derive(Serialize)]
struct Imported {
    pages: usize,
}

#[derive(Serialize)]
struct PipelineStarted {
    job_id: String,
}

#[derive(Deserialize)]
struct StartPipeline {
    source_language: koharu_translator::Language,
    target_language: koharu_translator::Language,
    content_profile: koharu_translator::ContentProfile,
    stages: Option<Vec<Stage>>,
}

#[derive(Deserialize)]
struct ExportPages {
    directory: PathBuf,
    #[serde(default)]
    pages: Option<Vec<usize>>,
}

#[derive(Serialize)]
struct Exported {
    paths: Vec<PathBuf>,
}

/// Hash-then-compare keeps the length secret and the content compare
/// constant-time (blake3 output is fixed-size).
fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    blake3::hash(left) == blake3::hash(right)
}

fn authorize(headers: &HeaderMap, state: &AutomationState) -> std::result::Result<(), ApiError> {
    let supplied = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let expected = format!("Bearer {}", state.token);
    if !constant_time_equal(supplied.as_bytes(), expected.as_bytes()) {
        return Err(ApiError(StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    Ok(())
}

fn contained_existing(state: &AutomationState, path: PathBuf) -> Result<PathBuf> {
    // Reject symlinked final components before canonicalize to close most of
    // the check-then-use swap window for paths inside the automation root.
    let metadata = std::fs::symlink_metadata(&path)
        .with_context(|| format!("failed to inspect automation path {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("automation path must not be a symlink");
    }
    let path = std::fs::canonicalize(&path)
        .with_context(|| format!("failed to resolve automation path {}", path.display()))?;
    if !path.starts_with(state.root.as_ref()) {
        bail!("automation path is outside KOHARU_AUTOMATION_ROOT");
    }
    Ok(path)
}

/// Rejects stale clients and GUI project switches before any project-bound
/// operation can read or mutate the active document.
async fn require_pinned_project(
    state: &AutomationState,
    headers: &HeaderMap,
) -> std::result::Result<PinnedProject, ApiError> {
    let supplied = headers
        .get("x-koharu-automation-session")
        .and_then(|value| value.to_str().ok());
    let pinned = state.pinned.lock().clone().ok_or_else(|| {
        ApiError(
            StatusCode::CONFLICT,
            "create an automation project before continuing".into(),
        )
    })?;
    if supplied != Some(pinned.session.as_str()) {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "the automation project session is stale; create the project again".into(),
        ));
    }
    let current = state.handle.state::<CurrentProject>();
    let project = current.project.lock().await;
    let name = project.as_ref().map(|project| project.name.as_str());
    if name != Some(pinned.name.as_str())
        || state.handle.state::<Processing>().session() != pinned.processing_session
    {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "the automation project is no longer open; create it again before continuing".into(),
        ));
    }
    Ok(pinned)
}
fn sync_dashboard_order(
    state: &AutomationState,
    project_name: &str,
    input: DashboardOrder,
) -> Result<()> {
    if input.number.trim().is_empty() {
        bail!("dashboard order number is empty");
    }
    if input.title.trim().is_empty() {
        bail!("dashboard order title is empty");
    }
    state.handle.state::<Orders>().store.sync_linked_order(
        &input.number,
        project_name,
        OrderPatch {
            title: Some(input.title),
            client_name: Some(input.client_name),
            source_language: Some(input.source_language),
            target_language: Some(input.target_language),
            assignee: Some(input.assignee),
            brief: Some(input.brief),
            ..OrderPatch::default()
        },
        "telegram",
    )?;
    Ok(())
}

async fn health(State(state): State<AutomationState>, headers: HeaderMap) -> ApiResult<Health> {
    authorize(&headers, &state)?;
    Ok(Json(Health {
        status: "ok",
        pipeline_stages: &Stage::ALL,
    }))
}

async fn create_project(
    State(state): State<AutomationState>,
    headers: HeaderMap,
    Json(request): Json<CreateProject>,
) -> ApiResult<ProjectCreated> {
    authorize(&headers, &state)?;
    let _create = state.create_lock.lock().await;
    let CreateProject { name, order } = request;
    if name.trim().is_empty() {
        return Err(ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            "project name is empty".into(),
        ));
    }
    // Replacing the current project force-stops in-flight pipeline jobs;
    // refuse so a new run never silently kills another run's work.
    if !state.handle.state::<Processing>().stops.lock().is_empty() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "a pipeline job is still running; stop it before creating a project".into(),
        ));
    }
    lifecycle::create_project(name.clone(), state.handle.clone())
        .await
        .map_err(|error| ApiError(StatusCode::BAD_REQUEST, error.to_string()))?;
    if let Some(order) = order
        && let Err(error) = sync_dashboard_order(&state, &name, order)
    {
        let cleanup = lifecycle::delete_project(name.clone(), state.handle.clone()).await;
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            match cleanup {
                Ok(()) => error.to_string(),
                Err(cleanup) => {
                    format!("{error:#}; failed to remove the unsynchronized project: {cleanup}")
                }
            },
        ));
    }
    let session = uuid::Uuid::new_v4().to_string();
    *state.pinned.lock() = Some(PinnedProject {
        name: name.clone(),
        session: session.clone(),
        processing_session: state.handle.state::<Processing>().session(),
    });
    Ok(Json(ProjectCreated { name, session }))
}

async fn import_pages(
    State(state): State<AutomationState>,
    headers: HeaderMap,
    Json(request): Json<ImportPages>,
) -> ApiResult<Imported> {
    authorize(&headers, &state)?;
    require_pinned_project(&state, &headers).await?;
    let paths = request
        .paths
        .into_iter()
        .map(|path| contained_existing(&state, path))
        .collect::<Result<Vec<_>>>()?;
    let pages = lifecycle::import_pages_from_paths(&state.handle, paths).await?;
    Ok(Json(Imported { pages }))
}

async fn start_pipeline(
    State(state): State<AutomationState>,
    headers: HeaderMap,
    Json(request): Json<StartPipeline>,
) -> ApiResult<PipelineStarted> {
    authorize(&headers, &state)?;
    let pinned = require_pinned_project(&state, &headers).await?;
    let operation = match request.stages {
        Some(stages) if stages.is_empty() => {
            return Err(ApiError(
                StatusCode::UNPROCESSABLE_ENTITY,
                "at least one pipeline stage must be selected".into(),
            ));
        }
        Some(stages) => Operation::Stages { stages },
        None => Operation::Full,
    };
    let job = crate::commands::processing::process_with_translation(
        state.handle.clone(),
        Scope::Project,
        operation,
        state.handle.state::<CurrentProject>(),
        state.handle.state::<Processing>(),
        state
            .handle
            .state::<crate::commands::processing::JobChannel>(),
        TranslationOptions {
            source_language: request.source_language,
            target_language: request.target_language,
            content_profile: request.content_profile,
        },
        pinned.processing_session,
    )
    .await
    .map_err(|error| ApiError(StatusCode::CONFLICT, error.to_string()))?;
    Ok(Json(PipelineStarted {
        job_id: job.to_string(),
    }))
}

fn find_job(state: &AutomationState, id: &str) -> Option<Job> {
    let processing = state.handle.state::<Processing>();
    processing
        .jobs
        .lock()
        .values()
        .chain(processing.finished.lock().values())
        .find(|job| job.id.to_string() == id)
        .cloned()
}

async fn pipeline_status(
    State(state): State<AutomationState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Job> {
    authorize(&headers, &state)?;
    find_job(&state, &id)
        .map(Json)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "pipeline job not found".into()))
}

async fn stop_pipeline(
    State(state): State<AutomationState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Job> {
    authorize(&headers, &state)?;
    let processing = state.handle.state::<Processing>();
    let job = processing
        .jobs
        .lock()
        .values()
        .find(|job| job.id.to_string() == id)
        .cloned()
        .ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                "running pipeline job not found".into(),
            )
        })?;
    processing.stop(job.id)?;
    Ok(Json(job))
}

async fn export_pages(
    State(state): State<AutomationState>,
    headers: HeaderMap,
    Json(request): Json<ExportPages>,
) -> ApiResult<Exported> {
    authorize(&headers, &state)?;
    require_pinned_project(&state, &headers).await?;
    let directory = contained_existing(&state, request.directory)?;
    if !directory.is_dir() {
        return Err(ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            "export path is not a directory".into(),
        ));
    }
    let pages = match request.pages {
        None => Vec::new(),
        Some(numbers) => {
            let current = state.handle.state::<CurrentProject>();
            let project = current.project.lock().await;
            let snapshot = project.as_ref().context("no project is open")?.snapshot();
            let summaries = Project::pages(&snapshot)?;
            let mut selected = Vec::with_capacity(numbers.len());
            for number in numbers {
                let page = number
                    .checked_sub(1)
                    .and_then(|index| summaries.get(index))
                    .with_context(|| format!("export page {number} is out of range"))?;
                if selected.contains(&page.id) {
                    return Err(ApiError(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("export page {number} was selected more than once"),
                    ));
                }
                selected.push(page.id);
            }
            selected
        }
    };
    let paths =
        output::export_pages_to_directory(&state.handle, pages, ExportFormat::Png, directory)
            .await?;
    Ok(Json(Exported { paths }))
}

async fn list_pages(
    State(state): State<AutomationState>,
    headers: HeaderMap,
) -> ApiResult<Vec<crate::commands::project::PageSummary>> {
    authorize(&headers, &state)?;
    require_pinned_project(&state, &headers).await?;
    let current = state.handle.state::<CurrentProject>();
    let project = current.project.lock().await;
    let snapshot = project.as_ref().context("no project is open")?.snapshot();
    Ok(Json(Project::pages(&snapshot)?))
}

pub(crate) async fn serve(handle: AppHandle<Cef>) -> Result<()> {
    let Ok(token) = env::var("KOHARU_AUTOMATION_TOKEN") else {
        return Ok(());
    };
    if token.is_empty() || token.contains(['\r', '\n']) {
        bail!("KOHARU_AUTOMATION_TOKEN must be non-empty and single-line");
    }
    let root = env::var("KOHARU_AUTOMATION_ROOT")
        .context("KOHARU_AUTOMATION_ROOT is required when automation is enabled")?;
    let root = std::fs::canonicalize(&root)
        .with_context(|| format!("failed to resolve KOHARU_AUTOMATION_ROOT {root}"))?;
    let port = env::var("KOHARU_AUTOMATION_PORT")
        .unwrap_or_else(|_| "48765".into())
        .parse::<u16>()
        .context("KOHARU_AUTOMATION_PORT must be a valid TCP port")?;
    let state = AutomationState {
        handle,
        token: token.into(),
        root: Arc::new(root),
        pinned: Arc::new(parking_lot::Mutex::new(None)),
        create_lock: Arc::new(tokio::sync::Mutex::new(())),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/projects", post(create_project))
        .route("/v1/projects/import", post(import_pages))
        .route("/v1/projects/pages", get(list_pages))
        .route("/v1/projects/export", post(export_pages))
        .route("/v1/pipeline", post(start_pipeline))
        .route("/v1/pipeline/{id}", get(pipeline_status))
        .route("/v1/pipeline/{id}/stop", post(stop_pipeline))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    tracing::info!(port, "Koharu automation API listening on loopback");
    axum::serve(listener, app).await?;
    Ok(())
}
