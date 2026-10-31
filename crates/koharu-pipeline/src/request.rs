use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{ProgressSink, Scope, Stage};
use koharu_scene::EntityId;

/// Per-run translation settings, such as a Telegram order's languages and
/// content profile. The profile selects the application's saved prompt.
#[derive(Clone, Debug)]
pub struct TranslationOptions {
    pub source_language: koharu_translator::Language,
    pub target_language: koharu_translator::Language,
    pub content_profile: koharu_translator::ContentProfile,
}

/// Which repair a hand-painted stroke runs over exactly its painted pixels.
///
/// Each target selects the model that repairs the stroke independently of the
/// others: its settings are never shared with another target, and none of them
/// reads the page's saved masks.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum CleanupTarget {
    /// Removes lettering with the fast model selected for speech bubbles.
    Fast,
    /// Repaints the stroke with FLUX.2 Klein 9B and the manual inpainting
    /// profile's own prompt and settings.
    Inpainting,
    /// Reconstructs censored artwork with the decensor model selected in
    /// settings.
    Decensor,
}

impl CleanupTarget {
    /// The pipeline stage that repairs the painted pixels. `Fast` and
    /// `Inpainting` are two routes of the inpainting stage.
    #[must_use]
    pub const fn stage(self) -> Stage {
        match self {
            Self::Fast | Self::Inpainting => Stage::Inpainting,
            Self::Decensor => Stage::Decensor,
        }
    }
}

/// A cleanup mask painted by hand for a single run.
///
/// It travels with the request and is never saved as a page asset. The one
/// cleanup stage the request selects repairs exactly its pixels instead of the
/// page's saved masks; `target` names that stage and the route inside it.
#[derive(Clone, Debug)]
pub struct InpaintingMask {
    pub page: EntityId,
    pub png: Arc<[u8]>,
    pub target: CleanupTarget,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    #[default]
    Full,
    Through {
        stage: Stage,
    },
    Only {
        stage: Stage,
    },
    Stages {
        stages: Vec<Stage>,
    },
}

impl Operation {
    pub(crate) fn stages(&self) -> Result<Vec<Stage>> {
        let stages = match self {
            Self::Full => Stage::ALL.to_vec(),
            Self::Through {
                stage: Stage::Detection,
            } => vec![Stage::Detection],
            Self::Through { stage: Stage::Ocr } => vec![Stage::Detection, Stage::Ocr],
            Self::Through {
                stage: Stage::Translation,
            } => {
                vec![Stage::Detection, Stage::Ocr, Stage::Translation]
            }
            Self::Through {
                stage: Stage::CensorDetection,
            } => vec![Stage::Detection, Stage::CensorDetection],
            Self::Through {
                stage: Stage::Inpainting,
            } => vec![Stage::Detection, Stage::Inpainting],
            Self::Through {
                stage: Stage::Decensor,
            } => vec![
                Stage::Detection,
                Stage::CensorDetection,
                Stage::Inpainting,
                Stage::Decensor,
            ],
            Self::Only { stage } => vec![*stage],
            Self::Stages { stages } => Stage::ALL
                .into_iter()
                .filter(|stage| stages.contains(stage))
                .collect(),
        };
        if stages.is_empty() {
            bail!("at least one pipeline stage must be selected");
        }
        Ok(stages)
    }
}

#[derive(Clone)]
pub struct Request {
    pub operation: Operation,
    pub scope: Scope,
    pub stop: StopToken,
    pub progress: Option<ProgressSink>,
    pub inpainting_mask: Option<InpaintingMask>,
    pub translation: Option<Arc<TranslationOptions>>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            operation: Operation::Full,
            scope: Scope::Project,
            stop: StopToken::default(),
            progress: None,
            inpainting_mask: None,
            translation: None,
        }
    }
}

#[derive(Clone, Default)]
pub struct StopToken(Arc<AtomicBool>);

impl StopToken {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn stopped(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
