use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::Arc,
};

use anyhow::{Context as _, Result};
use koharu_pipeline::{Committed, Committer, Progress, RunStatus, StageOutput, StopToken};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Cef, Manager as _, State, ipc::Channel};
use uuid::Uuid;

use super::{
    ChannelExt as _, Error,
    canvas::CanvasChannel,
    lifecycle::ProjectChannel,
    project::{CurrentProject, Rebased},
};
use crate::history::HistoryName;
use koharu_desktop::Desktop;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct JobId(Uuid);

impl JobId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Job {
    pub id: JobId,
    pub kind: JobKind,
    pub state: JobState,
    #[specta(type = f64)]
    pub completed: usize,
    #[specta(type = f64)]
    pub total: usize,
    pub page: Option<koharu_scene::EntityId>,
    pub stage: Option<koharu_pipeline::Stage>,
    pub model: Option<String>,
    #[specta(type = Vec<f64>)]
    pub policy_rejected_pages: Vec<usize>,
    pub error: Option<String>,
}

/// A pipeline run occupies the single pipeline slot; hand-painted repairs
/// queue in submission order beside it and execute concurrently with it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Pipeline,
    Cleanup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Finished,
    Failed,
    Stopped,
}

/// Repairs beyond this many waiting strokes are refused instead of growing
/// workers, mask payloads and pending UI state without bound.
const MAX_QUEUED_CLEANUPS: usize = 32;

#[derive(Default)]
pub(crate) struct Processing {
    pub(crate) stops: Mutex<HashMap<JobId, StopToken>>,
    pub(crate) jobs: Mutex<HashMap<JobId, Job>>,
    pub(crate) finished: Mutex<FinishedJobs>,
    queue: Mutex<JobQueue>,
    changed: tokio::sync::Notify,
    pub(crate) submission: tokio::sync::Mutex<()>,
    workers: Mutex<tokio::task::JoinSet<()>>,
    /// Counts admissions; unchanged across an idle delay means no job started.
    activity: std::sync::atomic::AtomicU64,
}

#[derive(Default)]
struct JobQueue {
    session: u64,
    pipeline: Option<JobId>,
    cleanups: VecDeque<JobId>,
}

impl Processing {
    pub(crate) fn session(&self) -> u64 {
        self.queue.lock().session
    }

    pub(crate) fn enqueue(
        &self,
        job: &Job,
        stop: StopToken,
        expected_session: Option<u64>,
    ) -> Result<u64> {
        let mut queue = self.queue.lock();
        if expected_session.is_some_and(|session| session != queue.session) {
            anyhow::bail!("the project changed before the job was accepted");
        }
        match job.kind {
            JobKind::Pipeline => {
                if queue.pipeline.is_some() {
                    anyhow::bail!("another process is already running");
                }
                queue.pipeline = Some(job.id);
            }
            JobKind::Cleanup => {
                if queue.cleanups.len() >= MAX_QUEUED_CLEANUPS {
                    anyhow::bail!(
                        "{MAX_QUEUED_CLEANUPS} repairs are already waiting; let the queue drain first"
                    );
                }
                queue.cleanups.push_back(job.id);
            }
        }
        self.stops.lock().insert(job.id, stop);
        self.activity
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.jobs.lock().insert(job.id, job.clone());
        Ok(queue.session)
    }

    pub(crate) fn is_current(&self, session: u64) -> bool {
        self.queue.lock().session == session
    }

    /// Waits until the job may start: a pipeline run starts at once, repairs
    /// start in submission order. Returns `false` when the job was stopped or
    /// its project session ended while it waited.
    async fn wait_turn(&self, id: JobId, kind: JobKind, session: u64, stop: &StopToken) -> bool {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            // Registered before the queue is inspected, so a `finish` between
            // the inspection and the await still wakes this waiter.
            changed.as_mut().enable();
            {
                let queue = self.queue.lock();
                if queue.session != session || stop.stopped() || !self.jobs.lock().contains_key(&id)
                {
                    return false;
                }
                if kind == JobKind::Pipeline || queue.cleanups.front() == Some(&id) {
                    return true;
                }
            }
            changed.await;
        }
    }

    pub(crate) fn finish(
        &self,
        id: JobId,
        session: u64,
        stopped: bool,
        error: Option<String>,
    ) -> Option<Job> {
        let mut queue = self.queue.lock();
        if queue.session != session {
            return None;
        }
        if queue.pipeline == Some(id) {
            queue.pipeline = None;
        }
        queue.cleanups.retain(|queued| *queued != id);
        self.stops.lock().remove(&id);
        let job = self.jobs.lock().remove(&id).map(|mut job| {
            job.state = if stopped {
                JobState::Stopped
            } else if error.is_some() {
                JobState::Failed
            } else {
                JobState::Finished
            };
            job.error = error;
            job
        });
        if let Some(job) = &job {
            self.finished.lock().insert(id, job.clone());
        }
        drop(queue);
        self.changed.notify_waiters();
        job
    }

    pub(crate) fn stop(&self, id: JobId) -> Result<()> {
        self.stops
            .lock()
            .get(&id)
            .with_context(|| format!("job {id} is not queued or running"))?
            .stop();
        self.changed.notify_waiters();
        Ok(())
    }

    /// Called while replacing the open project; old workers cannot publish or
    /// commit into the next session, including a reopening of the same file.
    pub(crate) fn reset(&self) -> impl Iterator<Item = Job> {
        let mut queue = self.queue.lock();
        queue.session = queue.session.wrapping_add(1);
        queue.pipeline = None;
        queue.cleanups.clear();
        for (_, stop) in self.stops.lock().drain() {
            stop.stop();
        }
        let mut jobs = std::mem::take(&mut *self.jobs.lock());
        {
            let mut finished = self.finished.lock();
            for job in jobs.values_mut() {
                job.state = JobState::Stopped;
                finished.insert(job.id, job.clone());
            }
        }
        drop(queue);
        self.changed.notify_waiters();
        jobs.into_values()
    }
}

/// Completed jobs kept for status polling, evicted oldest-first at capacity.
/// (HashMap iteration order is arbitrary, so eviction tracks insertion order.)
#[derive(Default)]
pub(crate) struct FinishedJobs {
    map: HashMap<JobId, Job>,
    order: std::collections::VecDeque<JobId>,
}

impl FinishedJobs {
    const CAPACITY: usize = 128;

    pub(crate) fn values(&self) -> impl Iterator<Item = &Job> {
        self.map.values()
    }

    fn get(&self, id: &JobId) -> Option<&Job> {
        self.map.get(id)
    }

    pub(crate) fn insert(&mut self, id: JobId, job: Job) {
        if let Some(existing) = self.map.get_mut(&id) {
            *existing = job;
            return;
        }
        while self.map.len() >= Self::CAPACITY {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.map.remove(&oldest);
        }
        self.order.push_back(id);
        self.map.insert(id, job);
    }
}

#[derive(Default)]
pub(crate) struct JobChannel {
    pub(crate) channel: Mutex<Option<Channel<Job>>>,
}

#[tauri::command]
#[specta::specta]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn process(
    handle: AppHandle<Cef>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
) -> std::result::Result<JobId, Error> {
    process_inner(
        handle,
        scope,
        operation,
        project,
        processing,
        job_channel,
        None,
        None,
        None,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn process_with_translation(
    handle: AppHandle<Cef>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
    translation: koharu_pipeline::TranslationOptions,
    expected_session: u64,
) -> std::result::Result<JobId, Error> {
    process_inner(
        handle,
        scope,
        operation,
        project,
        processing,
        job_channel,
        Some(Arc::new(translation)),
        None,
        Some(expected_session),
        None,
    )
    .await
}

/// Starts a manual cleanup job over the pixels painted in `inpainting_mask`.
///
/// The stroke travels with the job it creates instead of waiting in shared
/// state, so a refused request leaves nothing for the next run to pick up.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn process_with_inpainting_mask(
    handle: AppHandle<Cef>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
    inpainting_mask: koharu_pipeline::InpaintingMask,
    expected_session: u64,
    pipeline: koharu_pipeline::PipelineSnapshot,
) -> std::result::Result<JobId, Error> {
    process_inner(
        handle,
        scope,
        operation,
        project,
        processing,
        job_channel,
        None,
        Some(inpainting_mask),
        Some(expected_session),
        Some(pipeline),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn process_inner(
    handle: AppHandle<Cef>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
    translation: Option<Arc<koharu_pipeline::TranslationOptions>>,
    inpainting_mask: Option<koharu_pipeline::InpaintingMask>,
    expected_session: Option<u64>,
    prepared_pipeline: Option<koharu_pipeline::PipelineSnapshot>,
) -> std::result::Result<JobId, Error> {
    let id = JobId::new();
    let stop = StopToken::default();
    let manual = inpainting_mask.as_ref();
    let job = Job {
        id,
        kind: if manual.is_some() {
            JobKind::Cleanup
        } else {
            JobKind::Pipeline
        },
        state: JobState::Queued,
        completed: 0,
        total: usize::from(manual.is_some()),
        page: manual.map(|mask| mask.page),
        stage: manual.map(|mask| mask.target.stage()),
        model: manual.map(|mask| {
            match mask.target {
                koharu_pipeline::CleanupTarget::Fast => "fast",
                koharu_pipeline::CleanupTarget::Inpainting => "inpainting",
                koharu_pipeline::CleanupTarget::Decensor => "decensor",
            }
            .to_owned()
        }),
        policy_rejected_pages: Vec::new(),
        error: None,
    };
    let (project_id, session, pipeline) = {
        let current = project.project.lock().await;
        let current = current.as_ref().context("no project is open")?;
        let snapshot = current.snapshot();
        if let Some(mask) = manual {
            snapshot.page(mask.page)?;
        }
        let pipeline = match prepared_pipeline {
            Some(pipeline) => pipeline,
            None => handle.state::<koharu_pipeline::Pipeline>().capture()?,
        };
        let session = processing.enqueue(&job, stop.clone(), expected_session)?;
        (snapshot.project_id(), session, pipeline)
    };
    let kind = job.kind;
    job_channel.channel.publish(job);
    let task_handle = handle.clone();
    let mut workers = processing.workers.lock();
    while let Some(result) = workers.try_join_next() {
        if let Err(error) = result {
            tracing::error!(%error, "processing worker stopped unexpectedly");
        }
    }
    workers.spawn(async move {
        let state = task_handle.state::<Processing>();
        // Whatever ends this task — completion, cancellation, a panic or an
        // abort — the job leaves the queue and its terminal state is published.
        let mut finalizer = Finalizer {
            handle: task_handle.clone(),
            id,
            session,
            armed: true,
        };
        if !state.wait_turn(id, kind, session, &stop).await {
            finalizer.finish(true, None);
            return;
        }
        // Deliberately captured after the previous queue item has committed.
        let snapshot = {
            let current = task_handle.state::<CurrentProject>();
            let current = current.project.lock().await;
            current
                .as_ref()
                .filter(|project| {
                    state.is_current(session) && project.snapshot().project_id() == project_id
                })
                .map(|project| project.snapshot())
        };
        let Some(snapshot) = snapshot else {
            finalizer.finish(true, None);
            return;
        };
        let started = state.jobs.lock().get_mut(&id).map(|job| {
            job.state = JobState::Running;
            job.clone()
        });
        if let Some(job) = started {
            task_handle.state::<JobChannel>().channel.publish(job);
        }
        let progress = Arc::new(Mutex::new((0_usize, 0_usize)));
        let page_numbers = Arc::new(Mutex::new(HashMap::<koharu_scene::EntityId, usize>::new()));
        let progress_page_numbers = Arc::clone(&page_numbers);
        let progress_handle = task_handle.clone();
        let mut request = koharu_pipeline::Request {
            operation,
            scope,
            stop: stop.clone(),
            progress: None,
            inpainting_mask,
            translation,
        };
        request.progress = Some(Arc::new(move |event| {
            let update = match event {
                Progress::Started { pages, stages } => {
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "pipeline_start",
                        page_count = pages.len(),
                        stage_count = stages.len(),
                    );
                    *progress_page_numbers.lock() = pages
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(index, page)| (page, index + 1))
                        .collect();
                    let mut progress = progress.lock();
                    *progress = (0, pages.len().saturating_mul(stages.len()));
                    Some((0, progress.1, None, None, None, None))
                }
                Progress::Loading { page, stage, model } => {
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "stage_loading",
                        stage = %stage,
                        model,
                    );
                    let progress = progress.lock();
                    Some((
                        progress.0,
                        progress.1,
                        Some(page),
                        Some(stage),
                        Some(model),
                        None,
                    ))
                }
                Progress::Finished {
                    page,
                    stage,
                    model,
                    elapsed,
                } => {
                    if stage != koharu_pipeline::Stage::Translation {
                        tracing::info!(
                            target: "koharu_metrics",
                            metric = "model_run",
                            stage = %stage,
                            model,
                            duration_ms = elapsed.as_secs_f64() * 1000.0,
                        );
                    }
                    crate::timings::record(&crate::timings::Timing {
                        job: id.to_string(),
                        event: "stage",
                        page: Some(page.to_string()),
                        stage: Some(stage.to_string()),
                        model: Some(model.clone()),
                        elapsed,
                    });
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((
                        progress.0,
                        progress.1,
                        Some(page),
                        Some(stage),
                        Some(model),
                        None,
                    ))
                }
                Progress::Skipped { page, stage } => {
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "stage_skip",
                        stage = %stage,
                    );
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), None, None))
                }
                Progress::PolicyRejected { page, stage, model } => {
                    tracing::warn!(
                        target: "koharu_metrics",
                        metric = "translation_policy_rejected",
                        %page,
                        %stage,
                        model,
                    );
                    let page_number = progress_page_numbers.lock().get(&page).copied();
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((
                        progress.0,
                        progress.1,
                        Some(page),
                        Some(stage),
                        Some(model),
                        page_number,
                    ))
                }
                Progress::Running { stage, model, .. } => {
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "stage_running",
                        stage = %stage,
                        model,
                    );
                    None
                }
            };
            if let Some((completed, total, page, stage, model, policy_rejected_page)) = update {
                let job = {
                    let processing = progress_handle.state::<Processing>();
                    let mut jobs = processing.jobs.lock();
                    jobs.get_mut(&id).map(|job| {
                        job.completed = completed;
                        job.total = total;
                        job.page = page;
                        job.stage = stage;
                        job.model = model;
                        if let Some(page) = policy_rejected_page
                            && !job.policy_rejected_pages.contains(&page)
                        {
                            job.policy_rejected_pages.push(page);
                        }
                        job.clone()
                    })
                };
                if let Some(job) = job {
                    progress_handle.state::<JobChannel>().channel.publish(job);
                }
            }
        }));

        struct PipelineCommitter {
            handle: AppHandle<Cef>,
            project_id: koharu_scene::ProjectId,
            session: u64,
            job: JobId,
        }

        #[async_trait::async_trait]
        impl Committer for PipelineCommitter {
            async fn commit(&mut self, output: StageOutput) -> Result<Committed> {
                let started = std::time::Instant::now();
                let (stage, output_page) = (output.stage, output.page);
                let (commit, page, info) = {
                    let projects = self.handle.state::<CurrentProject>();
                    let mut projects = projects.project.lock().await;
                    let project = projects.as_mut().context("no project is open")?;
                    anyhow::ensure!(
                        self.handle.state::<Processing>().is_current(self.session)
                            && project.snapshot().project_id() == self.project_id,
                        "the project changed before the pipeline result could be committed"
                    );
                    let commit = match project.commit_rebased(output.patch).await? {
                        Rebased::Applied(commit) => commit,
                        Rebased::Unchanged => return Ok(Committed::Unchanged(project.snapshot())),
                        Rebased::Superseded => {
                            return Ok(Committed::Superseded(project.snapshot()));
                        }
                    };
                    project.record_named(
                        HistoryName::PipelineStage,
                        Some(output.stage.to_string()),
                        None,
                        &commit,
                    );
                    let page = project.active_page();
                    (commit, page, project.info())
                };
                let snapshot = commit.snapshot.clone();
                let desktop = self.handle.state::<Desktop>();
                desktop.synchronize(&commit.snapshot, page, &commit).await?;
                let canvas = desktop.canvas_state();
                self.handle.state::<CanvasChannel>().channel.publish(canvas);
                self.handle
                    .state::<ProjectChannel>()
                    .channel
                    .publish(Some(info));
                // Includes waiting for the project lock and the canvas update.
                crate::timings::record(&crate::timings::Timing {
                    job: self.job.to_string(),
                    event: "commit",
                    page: Some(output_page.to_string()),
                    stage: Some(stage.to_string()),
                    model: None,
                    elapsed: started.elapsed(),
                });
                Ok(Committed::Applied(snapshot))
            }
        }

        let mut committer = PipelineCommitter {
            handle: task_handle.clone(),
            project_id,
            session,
            job: id,
        };
        let result = pipeline.execute(snapshot, request, &mut committer).await;
        let (stopped, error) = match result {
            Ok(report) => {
                crate::timings::record(&crate::timings::Timing {
                    job: id.to_string(),
                    event: "job",
                    page: None,
                    stage: None,
                    model: Some(format!("{kind:?}").to_lowercase()),
                    elapsed: report.elapsed,
                });
                (report.status == RunStatus::Stopped, None)
            }
            Err(error) => {
                let message = format!("{error:#}");
                tracing::error!(stage = ?error.stage, error = %message, "processing failed");
                (false, Some(message))
            }
        };
        tracing::info!(
            target: "koharu_metrics",
            metric = "pipeline_result",
            outcome = if stopped {
                "stopped"
            } else if error.is_some() {
                "failed"
            } else {
                "completed"
            },
        );
        let processing = task_handle.state::<Processing>();
        if let Some(job) = finalizer.complete(stopped, error) {
            let publish = job.clone();
            // Audit the pipeline outcome on the order backed by the open project.
            if let Some(orders) = task_handle.try_state::<crate::commands::orders::Orders>() {
                let current = task_handle.state::<CurrentProject>();
                let project = current.project.lock().await;
                if let Some(project) = project.as_ref()
                    && processing.is_current(session)
                    && project.snapshot().project_id() == project_id
                    && let Ok(Some(order_id)) = orders.store.order_for_project(&project.name)
                {
                    use crate::commands::orders::sync_progress;
                    match job.state {
                        JobState::Finished => {
                            let _ = orders.store.log_activity(
                                &order_id,
                                "pipeline_succeeded",
                                orders.actor(),
                                serde_json::json!({ "job_id": job.id.to_string() }),
                            );
                        }
                        JobState::Failed => {
                            let _ = orders.store.log_activity(
                                &order_id,
                                "pipeline_failed",
                                orders.actor(),
                                serde_json::json!({
                                    "job_id": job.id.to_string(),
                                    "error": job.error,
                                }),
                            );
                        }
                        _ => {}
                    }
                    sync_progress(&orders, project);
                }
            }
            if processing.is_current(session) {
                task_handle.state::<JobChannel>().channel.publish(publish);
            }
            if kind == JobKind::Cleanup {
                release_models_when_idle(&task_handle);
            }
        }
    });
    Ok(id)
}

/// Ends a worker's job exactly once. Dropping it while still armed — a panic
/// or an aborted task — reports the job as failed instead of leaving it at the
/// head of the queue, where it would block every later repair.
struct Finalizer {
    handle: AppHandle<Cef>,
    id: JobId,
    session: u64,
    armed: bool,
}

impl Finalizer {
    fn complete(&mut self, stopped: bool, error: Option<String>) -> Option<Job> {
        self.armed = false;
        self.handle
            .state::<Processing>()
            .finish(self.id, self.session, stopped, error)
    }

    fn finish(&mut self, stopped: bool, error: Option<String>) {
        if let Some(job) = self.complete(stopped, error) {
            self.handle.state::<JobChannel>().channel.publish(job);
        }
    }
}

impl Drop for Finalizer {
    fn drop(&mut self) {
        if self.armed {
            tracing::error!(job = %self.id, "processing worker ended without finishing its job");
            self.finish(false, Some("processing stopped unexpectedly".to_owned()));
        }
    }
}

/// Hand-painted repairs keep their generative model resident between strokes;
/// once no job has started for this long its weights are released.
const IDLE_MODEL_RELEASE: std::time::Duration = std::time::Duration::from_secs(300);

fn release_models_when_idle(handle: &AppHandle<Cef>) {
    let processing = handle.state::<Processing>();
    let activity = processing
        .activity
        .load(std::sync::atomic::Ordering::Relaxed);
    let handle = handle.clone();
    processing.workers.lock().spawn(async move {
        tokio::time::sleep(IDLE_MODEL_RELEASE).await;
        let processing = handle.state::<Processing>();
        if processing
            .activity
            .load(std::sync::atomic::Ordering::Relaxed)
            != activity
            || !processing.jobs.lock().is_empty()
        {
            return;
        }
        match handle.state::<koharu_pipeline::Pipeline>().release(&[
            koharu_pipeline::Stage::Inpainting,
            koharu_pipeline::Stage::Decensor,
        ]) {
            Ok(true) => {
                tracing::info!(target: "koharu_metrics", metric = "model_unload", reason = "idle");
            }
            Ok(false) => {}
            Err(error) => tracing::warn!(%error, "failed to release idle repair models"),
        }
    });
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "pipeline_stop",
    skip_all,
    fields(state = "requested")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn stop_job(
    job: JobId,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    processing.stop(job)?;
    Ok(())
}

/// Current states of the given jobs, for reconciling the editor after a lost
/// channel event. Jobs absent from the result are no longer known; the editor
/// treats them as stopped.
#[tauri::command]
#[specta::specta]
pub(crate) async fn get_jobs(
    jobs: Vec<JobId>,
    processing: State<'_, Processing>,
) -> std::result::Result<Vec<Job>, Error> {
    let active = processing.jobs.lock();
    let finished = processing.finished.lock();
    Ok(jobs
        .iter()
        .filter_map(|id| active.get(id).or_else(|| finished.get(id)).cloned())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submitted(processing: &Processing, kind: JobKind) -> Result<(Job, StopToken, u64)> {
        let job = Job {
            id: JobId::new(),
            kind,
            state: JobState::Queued,
            completed: 0,
            total: 1,
            page: None,
            stage: Some(koharu_pipeline::Stage::Inpainting),
            model: None,
            policy_rejected_pages: Vec::new(),
            error: None,
        };
        let stop = StopToken::default();
        let session = processing.enqueue(&job, stop.clone(), None)?;
        Ok((job, stop, session))
    }

    fn repair(processing: &Processing) -> (Job, StopToken, u64) {
        submitted(processing, JobKind::Cleanup).unwrap()
    }

    async fn turn(processing: &Processing, job: &Job, session: u64, stop: &StopToken) -> bool {
        processing.wait_turn(job.id, job.kind, session, stop).await
    }

    #[tokio::test]
    async fn repairs_run_in_submission_order_even_when_later_waiters_are_polled_first() {
        let processing = Processing::default();
        let (first, first_stop, session) = repair(&processing);
        let (second, second_stop, _) = repair(&processing);
        let (third, third_stop, _) = repair(&processing);
        let later = turn(&processing, &third, session, &third_stop);
        tokio::pin!(later);
        assert!(futures::poll!(later.as_mut()).is_pending());
        assert!(turn(&processing, &first, session, &first_stop).await);
        let second_waiter = turn(&processing, &second, session, &second_stop);
        tokio::pin!(second_waiter);
        assert!(futures::poll!(second_waiter.as_mut()).is_pending());
        let failed = processing
            .finish(first.id, session, false, Some("model load failed".into()))
            .unwrap();
        assert_eq!(failed.state, JobState::Failed);
        assert!(futures::poll!(later.as_mut()).is_pending());
        assert!(second_waiter.await);
        processing.finish(second.id, session, false, None).unwrap();
        assert!(later.await);
        processing.finish(third.id, session, false, None).unwrap();
        assert!(processing.jobs.lock().is_empty());
    }

    #[tokio::test]
    async fn cancelling_a_queued_repair_does_not_block_the_tail() {
        let processing = Processing::default();
        let (first, first_stop, session) = repair(&processing);
        let (cancelled, cancelled_stop, _) = repair(&processing);
        let (last, last_stop, _) = repair(&processing);
        assert!(turn(&processing, &first, session, &first_stop).await);
        let waiting = turn(&processing, &cancelled, session, &cancelled_stop);
        tokio::pin!(waiting);
        assert!(futures::poll!(waiting.as_mut()).is_pending());
        processing.stop(cancelled.id).unwrap();
        assert!(!waiting.await);
        let stopped = processing
            .finish(cancelled.id, session, true, None)
            .unwrap();
        assert_eq!(stopped.state, JobState::Stopped);
        processing.finish(first.id, session, false, None).unwrap();
        assert!(turn(&processing, &last, session, &last_stop).await);
        processing.finish(last.id, session, false, None).unwrap();
    }

    #[tokio::test]
    async fn replacing_a_project_invalidates_waiters_and_late_results_from_the_previous_session() {
        let processing = Processing::default();
        let (first, first_stop, session) = repair(&processing);
        let (second, second_stop, _) = repair(&processing);
        assert!(turn(&processing, &first, session, &first_stop).await);
        let waiting = turn(&processing, &second, session, &second_stop);
        tokio::pin!(waiting);
        assert!(futures::poll!(waiting.as_mut()).is_pending());
        let cancelled: Vec<_> = processing.reset().collect();
        assert_eq!(cancelled.len(), 2);
        assert!(cancelled.iter().all(|job| job.state == JobState::Stopped));
        for id in [first.id, second.id] {
            assert_eq!(
                processing
                    .finished
                    .lock()
                    .values()
                    .find(|job| job.id == id)
                    .unwrap()
                    .state,
                JobState::Stopped
            );
        }
        assert!(first_stop.stopped() && second_stop.stopped());
        assert!(!waiting.await);
        let (new, new_stop, next_session) = repair(&processing);
        assert!(processing.finish(first.id, session, false, None).is_none());
        assert!(processing.jobs.lock().contains_key(&new.id));
        assert!(
            processing
                .enqueue(&second, StopToken::default(), Some(session))
                .is_err()
        );
        assert!(turn(&processing, &new, next_session, &new_stop).await);
        processing
            .finish(new.id, next_session, false, None)
            .unwrap();
    }

    #[tokio::test]
    async fn a_repair_starts_beside_a_running_pipeline_but_a_second_pipeline_is_refused() {
        let processing = Processing::default();
        let (pipeline, pipeline_stop, session) = submitted(&processing, JobKind::Pipeline).unwrap();
        assert!(turn(&processing, &pipeline, session, &pipeline_stop).await);
        let (mask, mask_stop, _) = repair(&processing);
        assert!(turn(&processing, &mask, session, &mask_stop).await);
        assert!(submitted(&processing, JobKind::Pipeline).is_err());
        processing
            .finish(pipeline.id, session, false, None)
            .unwrap();
        assert!(submitted(&processing, JobKind::Pipeline).is_ok());
    }

    #[test]
    fn the_repair_queue_is_bounded_until_it_drains() {
        let processing = Processing::default();
        let queued: Vec<_> = (0..MAX_QUEUED_CLEANUPS)
            .map(|_| repair(&processing))
            .collect();
        assert!(submitted(&processing, JobKind::Cleanup).is_err());
        let (first, _, session) = &queued[0];
        processing.finish(first.id, *session, false, None).unwrap();
        assert!(submitted(&processing, JobKind::Cleanup).is_ok());
    }
}
