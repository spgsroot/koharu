use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use futures::{StreamExt as _, stream::FuturesUnordered};
use koharu_scene::{EntityId, Snapshot};

use crate::{
    Committed, Committer, ErrorKind, PipelineError, Progress, ProgressSink, Report, Request,
    RunStatus, Stage, StageOutput, StopToken, TranslationOptions,
    images::ImageCache,
    progress,
    resources::ResourceMonitor,
    scheduler::Scheduler,
    scope::NormalizedScope,
    stage_runner::{StageCompletion, StageJob, StageOutcome, StageRunner},
    stages::StageInput,
};

/// The stages that repair a hand-painted stroke in place of their saved masks.
/// A stroke is repaired by exactly one of them.
const PAINTED_CLEANUP: [Stage; 2] = [Stage::Inpainting, Stage::Decensor];

/// Whether `stage` is handed the stroke painted on its page: the cleanup stages
/// repair it, and censor detection stands aside so a stroke never replaces the
/// saved censor mask.
fn receives_painted_mask(stage: Stage) -> bool {
    stage == Stage::CensorDetection || PAINTED_CLEANUP.contains(&stage)
}

pub(crate) struct Execution<'a> {
    runner: Arc<StageRunner>,
    resources: Arc<ResourceMonitor>,
    committer: &'a mut dyn Committer,
    stop: StopToken,
    progress: Option<ProgressSink>,
    scope: NormalizedScope,
    scheduler: Scheduler,
    scene: Snapshot,
    images: BTreeMap<EntityId, Arc<ImageCache>>,
    busy_stages: BTreeSet<Stage>,
    completed: usize,
    failure: Option<PipelineError>,
    base: koharu_scene::Revision,
    started: Instant,
    inpainting_mask: Option<crate::InpaintingMask>,
    translation: Option<Arc<TranslationOptions>>,
}

impl<'a> Execution<'a> {
    pub(crate) fn new(
        runner: Arc<StageRunner>,
        resources: Arc<ResourceMonitor>,
        snapshot: Snapshot,
        request: Request,
        committer: &'a mut dyn Committer,
    ) -> std::result::Result<Self, PipelineError> {
        let started = Instant::now();
        let base = snapshot.revision();
        let stages = request
            .operation
            .stages()
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let scope = NormalizedScope::new(&snapshot, &request.scope, &stages)
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let pages = scope.pages().to_vec();
        let consumers = stages
            .iter()
            .filter(|stage| PAINTED_CLEANUP.contains(stage))
            .count();
        if let Some(mask) = request.inpainting_mask.as_ref()
            && (!pages.contains(&mask.page)
                || consumers != 1
                || !stages.contains(&mask.target.stage()))
        {
            return Err(PipelineError::new(
                ErrorKind::InvalidInput,
                None,
                anyhow::anyhow!(
                    "a painted cleanup mask needs its page in scope and exactly one cleanup stage"
                ),
            ));
        }
        progress::emit(
            request.progress.as_ref(),
            Progress::Started {
                pages: pages.clone(),
                stages: stages.clone(),
            },
        );

        Ok(Self {
            runner,
            resources,
            committer,
            stop: request.stop,
            progress: request.progress,
            scope,
            scheduler: Scheduler::new(&pages, &stages),
            scene: snapshot,
            images: BTreeMap::new(),
            busy_stages: BTreeSet::new(),
            completed: 0,
            failure: None,
            base,
            started,
            inpainting_mask: request.inpainting_mask,
            translation: request.translation,
        })
    }

    pub(crate) async fn run(mut self) -> std::result::Result<Report, PipelineError> {
        if self.stopped() {
            return Ok(self.report(RunStatus::Stopped));
        }

        self.resources.start();
        self.resources.wait_for_sample().await;

        let runner = self.runner.clone();
        let mut running = FuturesUnordered::new();
        loop {
            while let Some(job) = self.take_ready_job() {
                running.push(runner.run(job));
            }

            let Some(completion) = running.next().await else {
                break;
            };
            self.busy_stages.remove(&completion.stage);
            if self.stopped() || self.failure.is_some() {
                continue;
            }
            if let Err(error) = self.apply_completion(completion).await {
                self.failure = Some(error);
            }
        }

        self.finalize()
    }

    fn take_ready_job(&mut self) -> Option<StageJob> {
        if self.stopped() || self.failure.is_some() {
            return None;
        }
        let (page, stage) = self.scheduler.start_next(&self.busy_stages)?;
        self.busy_stages.insert(stage);
        let images = self
            .images
            .entry(page)
            .or_insert_with(|| Arc::new(ImageCache::default()))
            .clone();
        Some(StageJob::new(
            stage,
            StageInput::new(
                self.scene.clone(),
                page,
                self.scope.entities(),
                self.scope.region(page),
                images,
                self.inpainting_mask
                    .as_ref()
                    .filter(|mask| mask.page == page && receives_painted_mask(stage))
                    .cloned(),
                self.translation.clone(),
            )
            .stopped_by(self.stop.clone()),
            self.stop.clone(),
            self.progress.clone(),
            // A hand-painted repair is interactive; it must not wait behind
            // every queued stage of a concurrent pipeline run.
            self.inpainting_mask.is_some(),
        ))
    }

    async fn apply_completion(
        &mut self,
        completion: StageCompletion,
    ) -> std::result::Result<(), PipelineError> {
        let StageCompletion {
            page,
            stage,
            model,
            elapsed,
            outcome,
        } = completion;
        match outcome {
            Err(error) if stage == Stage::Translation && error.is_policy_rejected() => {
                self.mark_complete(page, stage);
                progress::emit(
                    self.progress.as_ref(),
                    Progress::PolicyRejected { page, stage, model },
                );
            }
            Err(error) => return Err(error),
            Ok(StageOutcome::Stopped) => {}
            Ok(StageOutcome::Skipped) => {
                self.mark_complete(page, stage);
                progress::emit(self.progress.as_ref(), Progress::Skipped { page, stage });
            }
            Ok(StageOutcome::Patch(patch)) => {
                if !self.commit_patch(page, stage, patch).await? {
                    return Ok(());
                }
                self.mark_complete(page, stage);
                progress::emit(
                    self.progress.as_ref(),
                    Progress::Finished {
                        page,
                        stage,
                        model,
                        elapsed,
                    },
                );
            }
        }
        Ok(())
    }

    async fn commit_patch(
        &mut self,
        page: EntityId,
        stage: Stage,
        patch: koharu_scene::Patch,
    ) -> std::result::Result<bool, PipelineError> {
        let base = patch.base_revision();
        let patch = match patch.rebase_on(&self.scene) {
            Ok(patch) => patch,
            // Another job committed over what this output observed after the
            // stage captured its input; recompute it on the latest scene.
            Err(
                koharu_scene::Error::PatchConflict(_)
                | koharu_scene::Error::EntityNotFound(_)
                | koharu_scene::Error::RelationNotFound(_),
            ) => {
                let latest = self.scene.clone();
                return self.supersede(page, stage, base, latest);
            }
            Err(error) => {
                return Err(PipelineError::new(
                    ErrorKind::InvalidOutput,
                    Some(stage),
                    anyhow::Error::from(error)
                        .context("failed to rebase stage output onto the latest scene"),
                ));
            }
        };
        let patch = patch
            .validate_on(&self.scene)
            .map(|()| patch.with_label(format!("Pipeline {stage} for page {page}")))
            .context("failed to validate stage output on the latest scene")
            .map_err(|error| PipelineError::new(ErrorKind::InvalidOutput, Some(stage), error))?;
        if self.stopped() {
            return Ok(false);
        }

        let committed = self
            .committer
            .commit(StageOutput { page, stage, patch })
            .await
            .with_context(|| format!("failed to commit {stage} output for page {page}"))
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        let (next, applied) = match committed {
            Committed::Applied(next) => (next, true),
            Committed::Unchanged(next) => (next, false),
            Committed::Superseded(next) => return self.supersede(page, stage, base, next),
        };
        validate_commit(&self.scene, &next, applied)
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        self.scene = next;
        Ok(true)
    }

    /// Re-runs a stage whose output conflicts with edits committed after it
    /// captured its input. Each re-run is paid for by another job's commit, so
    /// hand-painted repairs on the page being processed delay the stage
    /// instead of failing the run; a conflict without such progress is a
    /// genuine error.
    fn supersede(
        &mut self,
        page: EntityId,
        stage: Stage,
        base: koharu_scene::Revision,
        latest: Snapshot,
    ) -> std::result::Result<bool, PipelineError> {
        validate_commit(&self.scene, &latest, false)
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        if latest.revision() <= base {
            return Err(PipelineError::new(
                ErrorKind::Commit,
                Some(stage),
                anyhow::anyhow!(
                    "{stage} output for page {page} conflicts with the scene it was computed on"
                ),
            ));
        }
        self.scene = latest;
        tracing::info!(%stage, %page, "re-running stage superseded by a concurrent edit");
        self.scheduler.retry(page, stage);
        Ok(false)
    }

    fn mark_complete(&mut self, page: EntityId, stage: Stage) {
        if self.scheduler.complete_stage(page, stage) {
            self.images.remove(&page);
        }
        self.completed += 1;
    }

    fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    fn finalize(mut self) -> std::result::Result<Report, PipelineError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if !self.stopped() && self.completed != self.scheduler.total() {
            return Err(PipelineError::new(
                ErrorKind::InvalidOutput,
                None,
                anyhow::anyhow!(
                    "pipeline scheduler stopped after {} of {} work items",
                    self.completed,
                    self.scheduler.total()
                ),
            ));
        }
        let status = if self.stopped() {
            RunStatus::Stopped
        } else {
            RunStatus::Completed
        };
        Ok(self.report(status))
    }

    fn report(&self, status: RunStatus) -> Report {
        Report {
            status,
            base: self.base,
            final_revision: self.scene.revision(),
            completed: self.completed,
            total: self.scheduler.total(),
            elapsed: self.started.elapsed(),
        }
    }
}

fn validate_commit(previous: &Snapshot, next: &Snapshot, applied: bool) -> Result<()> {
    ensure!(
        previous.project_id() == next.project_id(),
        "committer returned a snapshot from another project"
    );
    if applied {
        ensure!(
            next.revision() > previous.revision(),
            "committer did not advance the scene revision"
        );
    } else {
        ensure!(
            next.revision() >= previous.revision(),
            "committer returned an older scene"
        );
    }
    Ok(())
}
