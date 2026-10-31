use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use crate::Stage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunStatus {
    Completed,
    Stopped,
}

#[derive(Debug)]
pub struct Report {
    pub status: RunStatus,
    pub base: koharu_scene::Revision,
    pub final_revision: koharu_scene::Revision,
    pub completed: usize,
    pub total: usize,
    pub elapsed: Duration,
}

#[derive(Debug)]
pub struct StageOutput {
    pub page: koharu_scene::EntityId,
    pub stage: Stage,
    pub patch: koharu_scene::Patch,
}

/// What a committer did with one stage's output. Every variant carries the
/// latest scene, which includes edits committed concurrently by other jobs.
#[derive(Debug)]
pub enum Committed {
    /// The output was applied.
    Applied(koharu_scene::Snapshot),
    /// Rebasing left nothing to change.
    Unchanged(koharu_scene::Snapshot),
    /// A concurrent edit conflicted with the output, which was discarded; the
    /// stage runs again on the returned scene.
    Superseded(koharu_scene::Snapshot),
}

#[async_trait]
pub trait Committer: Send {
    async fn commit(&mut self, output: StageOutput) -> Result<Committed>;
}
