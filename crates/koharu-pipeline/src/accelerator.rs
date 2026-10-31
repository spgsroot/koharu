use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{Stage, resources::ResourceMonitor, stages::Stages};

/// Execution resources shared by every concurrent job and every settings
/// generation of the pipeline: a pipeline run and hand-painted repairs execute
/// side by side, but never two accelerator workloads or two calls into one
/// stage at once.
pub(crate) struct Lanes {
    resources: Arc<ResourceMonitor>,
    accelerator: Option<Arc<PriorityLane>>,
    stages: BTreeMap<Stage, Arc<tokio::sync::Mutex<()>>>,
}

/// Held while a stage processes a page. Dropping it lets the next job use the
/// stage and the accelerator.
pub(crate) struct LanePermit {
    _accelerator: Option<AcceleratorPermit>,
    _stage: tokio::sync::OwnedMutexGuard<()>,
}

impl Lanes {
    pub(crate) fn new(device: &koharu_ml::Device, resources: Arc<ResourceMonitor>) -> Self {
        let accelerated = device.backend != koharu_ml::Backend::Cpu
            && device.device_type != koharu_ml::DeviceType::Cpu;
        Self {
            resources,
            // Heterogeneous CUDA model pairs took 2.5-4x longer together than
            // back-to-back on the target workload. Keep accelerator work
            // serialized while loaded weights remain resident between pages.
            accelerator: accelerated.then(|| Arc::new(PriorityLane::default())),
            stages: Stage::ALL
                .into_iter()
                .map(|stage| (stage, Arc::new(tokio::sync::Mutex::new(()))))
                .collect(),
        }
    }

    /// The stage lock is always taken before the accelerator, so a job never
    /// holds the accelerator while waiting for a stage another job is using.
    pub(crate) async fn acquire(
        &self,
        stage: Stage,
        accelerator: bool,
        urgent: bool,
    ) -> LanePermit {
        let stage = self.stages[&stage].clone().lock_owned().await;
        let accelerator = match (&self.accelerator, accelerator) {
            (Some(lane), true) => Some(lane.acquire(urgent).await),
            _ => None,
        };
        LanePermit {
            _accelerator: accelerator,
            _stage: stage,
        }
    }

    pub(crate) async fn recover(
        &self,
        stage: Stage,
        accelerator: bool,
        urgent: bool,
        stages: &Stages,
    ) -> LanePermit {
        let permit = self.acquire(stage, accelerator, urgent).await;
        if self.accelerator.is_none() || !accelerator {
            return permit;
        }
        if unload_other_models(stage, stages) {
            let mut changed = self.resources.subscribe();
            let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
        }
        permit
    }
}

/// A single accelerator slot in which interactive work goes first: a
/// hand-painted repair waits for at most the stage currently running, not for
/// every queued pipeline stage.
#[derive(Default)]
struct PriorityLane {
    state: Mutex<LaneState>,
    released: tokio::sync::Notify,
}

#[derive(Default)]
struct LaneState {
    busy: bool,
    urgent_waiting: usize,
}

struct AcceleratorPermit {
    lane: Arc<PriorityLane>,
}

/// Keeps the urgent-waiter count exact when an urgent acquisition is cancelled.
struct UrgentWaiter<'a> {
    lane: &'a PriorityLane,
    armed: bool,
}

impl PriorityLane {
    fn state(&self) -> std::sync::MutexGuard<'_, LaneState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    async fn acquire(self: &Arc<Self>, urgent: bool) -> AcceleratorPermit {
        if urgent {
            self.state().urgent_waiting += 1;
        }
        let mut waiter = UrgentWaiter {
            lane: self,
            armed: urgent,
        };
        loop {
            let released = self.released.notified();
            tokio::pin!(released);
            // Registered before the state is inspected, so a release between
            // the check and the await still wakes this waiter.
            released.as_mut().enable();
            {
                let mut state = self.state();
                if !state.busy && (urgent || state.urgent_waiting == 0) {
                    state.busy = true;
                    if urgent {
                        state.urgent_waiting -= 1;
                        waiter.armed = false;
                    }
                    return AcceleratorPermit { lane: self.clone() };
                }
            }
            released.await;
        }
    }
}

impl Drop for AcceleratorPermit {
    fn drop(&mut self) {
        self.lane.state().busy = false;
        self.lane.released.notify_waiters();
    }
}

impl Drop for UrgentWaiter<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.lane.state().urgent_waiting -= 1;
            self.lane.released.notify_waiters();
        }
    }
}

fn unload_other_models(requested: Stage, stages: &Stages) -> bool {
    let mut unloaded = false;
    for stage in Stage::ALL {
        if stage != requested && stages.unload(stage) {
            unloaded = true;
            tracing::info!(target: "koharu_metrics", metric = "model_unload", stage = %stage);
            tracing::debug!(%stage, "unloaded model while recovering from memory pressure");
        }
    }
    unloaded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn urgent_work_takes_the_lane_before_waiting_pipeline_work() {
        let lane = Arc::new(PriorityLane::default());
        let running = lane.acquire(false).await;
        let order = Arc::new(Mutex::new(Vec::new()));
        let normal = tokio::spawn({
            let (lane, order) = (lane.clone(), order.clone());
            async move {
                let _permit = lane.acquire(false).await;
                order.lock().unwrap().push("pipeline");
            }
        });
        tokio::task::yield_now().await;
        let urgent = tokio::spawn({
            let (lane, order) = (lane.clone(), order.clone());
            async move {
                let _permit = lane.acquire(true).await;
                order.lock().unwrap().push("manual");
            }
        });
        tokio::task::yield_now().await;
        drop(running);
        urgent.await.unwrap();
        normal.await.unwrap();
        assert_eq!(*order.lock().unwrap(), ["manual", "pipeline"]);
    }

    #[tokio::test]
    async fn a_cancelled_urgent_waiter_does_not_block_pipeline_work() {
        let lane = Arc::new(PriorityLane::default());
        let running = lane.acquire(false).await;
        let cancelled = tokio::spawn({
            let lane = lane.clone();
            async move {
                let _permit = lane.acquire(true).await;
            }
        });
        tokio::task::yield_now().await;
        cancelled.abort();
        let _ = cancelled.await;
        drop(running);
        tokio::time::timeout(Duration::from_secs(1), lane.acquire(false))
            .await
            .expect("pipeline work must not wait for a cancelled repair");
    }
}
