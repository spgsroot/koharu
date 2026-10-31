use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use koharu_config::{Config, ConfigRevision};
use koharu_scene::Snapshot;

use crate::{
    Committer, ErrorKind, PipelineConfig, PipelineError, Report, Request, ResourceSnapshot,
    accelerator::Lanes, execution::Execution, resources::ResourceMonitor,
    stage_runner::StageRunner, stages::Models,
};

#[derive(Clone)]
pub struct Pipeline {
    state: Arc<PipelineState>,
}

struct PipelineState {
    config: Config<PipelineConfig>,
    translator: koharu_translator::Translator,
    device: koharu_ml::Device,
    models: Arc<Models>,
    current: Mutex<ConfiguredRunner>,
    resources: Arc<ResourceMonitor>,
    lanes: Arc<Lanes>,
}

struct ConfiguredRunner {
    revision: ConfigRevision,
    runner: Arc<StageRunner>,
}

/// Immutable model settings for a submitted job. Scene state is supplied when
/// the job starts, so queued masks see preceding edits without changing prompts.
/// Snapshots execute concurrently; the shared lanes serialize accelerator use.
#[derive(Clone)]
pub struct PipelineSnapshot {
    runner: Arc<StageRunner>,
    resources: Arc<ResourceMonitor>,
}

impl Pipeline {
    pub fn load(device: koharu_ml::Device) -> Result<Self> {
        Self::from_config(
            PipelineConfig::load()?,
            koharu_translator::ProvidersConfig::load()?,
            device,
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn from_config(
        config: Config<PipelineConfig>,
        providers: Config<koharu_translator::ProvidersConfig>,
        device: koharu_ml::Device,
    ) -> Result<Self> {
        let translator = koharu_translator::Translator::from_config(device.clone(), providers)?;
        let resources = ResourceMonitor::new(&device);
        let models = Arc::new(Models::default());
        let lanes = Arc::new(Lanes::new(&device, resources.clone()));
        let current = {
            let value = config.read()?;
            ConfiguredRunner {
                revision: value.revision(),
                runner: Arc::new(StageRunner::new(
                    &value,
                    translator.clone(),
                    &device,
                    lanes.clone(),
                    models.clone(),
                )?),
            }
        };
        Ok(Self {
            state: Arc::new(PipelineState {
                config,
                translator,
                device,
                models,
                current: Mutex::new(current),
                resources,
                lanes,
            }),
        })
    }

    pub fn subscribe_resources(&self) -> tokio::sync::watch::Receiver<ResourceSnapshot> {
        self.state.resources.start();
        self.state.resources.subscribe()
    }

    /// Releases the weights of `stages` in the current settings generation.
    /// A model that is processing right now keeps its weights.
    pub fn release(&self, stages: &[crate::Stage]) -> Result<bool> {
        let runner = self
            .state
            .current
            .lock()
            .map_err(|_| anyhow!("pipeline settings lock is poisoned"))?
            .runner
            .clone();
        // Every stage unloads; a short-circuiting `any` would keep the rest.
        let mut released = false;
        for stage in stages {
            released |= runner.unload(*stage);
        }
        Ok(released)
    }

    pub fn capture(&self) -> Result<PipelineSnapshot> {
        // Freeze the saved revision here, not in an asynchronous watcher that
        // may still hold the previous prompt when a stroke is accepted.
        let value = self.state.config.read()?;
        let mut current = self
            .state
            .current
            .lock()
            .map_err(|_| anyhow!("pipeline settings lock is poisoned"))?;
        if current.revision != value.revision() {
            *current = ConfiguredRunner {
                revision: value.revision(),
                runner: Arc::new(StageRunner::new(
                    &value,
                    self.state.translator.clone(),
                    &self.state.device,
                    self.state.lanes.clone(),
                    self.state.models.clone(),
                )?),
            };
        }
        Ok(PipelineSnapshot {
            runner: current.runner.clone(),
            resources: self.state.resources.clone(),
        })
    }

    #[tracing::instrument(skip_all)]
    pub async fn execute(
        &self,
        snapshot: Snapshot,
        request: Request,
        committer: &mut dyn Committer,
    ) -> std::result::Result<Report, PipelineError> {
        self.capture()
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?
            .execute(snapshot, request, committer)
            .await
    }
}

impl PipelineSnapshot {
    pub async fn execute(
        &self,
        snapshot: Snapshot,
        request: Request,
        committer: &mut dyn Committer,
    ) -> std::result::Result<Report, PipelineError> {
        Execution::new(
            self.runner.clone(),
            self.resources.clone(),
            snapshot,
            request,
            committer,
        )?
        .run()
        .await
    }
}
