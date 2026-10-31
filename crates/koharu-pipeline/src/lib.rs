//! In-process, scene-native model orchestration for Koharu.

mod accelerator;
mod config;
mod error;
mod execution;
mod images;
mod model_cell;
mod pipeline;
mod progress;
mod report;
mod request;
mod resources;
mod scheduler;
mod scope;
mod stage;
mod stage_runner;
mod stages;

pub use config::{
    ArtworkInpaintingConfig, BubbleInpaintingKind, DecensorConfig, DecensorKind, DecensorModel,
    DetectionModel, InpaintingConfig, InpaintingKind, InpaintingModel, ManualInpaintingPreset,
    OcrModel, PipelineConfig, ProcessorConfig, SystemPrompts, TranslationConfig,
    builtin_manual_inpainting_presets,
};
pub use error::{ErrorKind, PipelineError};
pub use pipeline::{Pipeline, PipelineSnapshot};
pub use progress::{Progress, ProgressSink};
pub use report::{Committed, Committer, Report, RunStatus, StageOutput};
pub use request::{
    CleanupTarget, InpaintingMask, Operation, Request, StopToken, TranslationOptions,
};
pub use resources::{DeviceResources, ResourceSnapshot};
pub use scope::{Bounds, Scope};
pub use stage::Stage;
pub use stages::{
    CensorDetectionConfig, DiffusionCreamPyConfig, Flux2Klein9BConfig, Flux2KleinConfig,
    KoharuLayoutRFDetrSeg2XLConfig, RoremMixedConfig,
};

use images::ImageCache;
use model_cell::ModelCell;

#[cfg(test)]
mod tests;
