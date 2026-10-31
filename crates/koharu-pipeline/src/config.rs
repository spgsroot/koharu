use anyhow::{Result, bail};
use koharu_translator::{
    ContentProfile, GenerationConfig, Language, TranslationValidationRule, TranslationValidator,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use specta::Type;

use crate::stages::{
    CensorDetectionConfig, DECENSOR_PROMPT, DiffusionCreamPyConfig, Flux2Klein9BConfig,
    Flux2KleinConfig, KoharuLayoutRFDetrSeg2XLConfig, RoremMixedConfig,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
#[specta(skip_attr = "serde")]
pub struct ManualInpaintingPreset {
    pub name: String,
    pub settings: Flux2Klein9BConfig,
}

impl Default for ManualInpaintingPreset {
    fn default() -> Self {
        Self {
            name: String::new(),
            settings: Flux2Klein9BConfig::manual_inpainting(),
        }
    }
}

pub fn builtin_manual_inpainting_presets() -> Vec<ManualInpaintingPreset> {
    [
        (
            "Text cleanup",
            "Remove all lettering and isolated glyphs from the masked area. Reconstruct the underlying background, line art and screentones. Match the surrounding drawing exactly and preserve bubble borders, panel borders and unrelated artwork.",
        ),
        (
            "Line art & screentone restore",
            "Restore the masked manga artwork with clean continuous ink lines and evenly patterned screentones. Match neighboring line weight, tone spacing, shading and contours. Remove unwanted marks without changing the composition.",
        ),
        (
            "Background fill",
            "Remove the masked objects and lettering, then fill the area with the surrounding background. Continue nearby textures, gradients, patterns and structural lines naturally. Do not introduce new objects or details.",
        ),
        (
            "Censor-bar reconstruction",
            DECENSOR_PROMPT,
        ),
    ]
    .into_iter()
    .map(|(name, prompt)| ManualInpaintingPreset {
        name: name.to_owned(),
        settings: Flux2Klein9BConfig {
            prompt: prompt.to_owned(),
            ..Flux2Klein9BConfig::manual_inpainting()
        },
    })
    .collect()
}

#[derive(Clone, Debug, PartialEq, Type)]
pub struct PipelineConfig {
    pub detection: DetectionModel,
    pub ocr: OcrModel,
    pub translation: TranslationConfig,
    /// Removes lettering. Speech bubbles and artwork select their models
    /// independently, and the stage owns its settings independently of the
    /// decensor stage.
    pub inpainting: InpaintingConfig,
    /// Reconstructs censored areas. Owns its model settings independently of
    /// the inpainting stage.
    pub decensor: DecensorConfig,
    /// Finds obscured areas and saves their mask.
    pub censor_detection: CensorDetectionConfig,
    /// Detection settings, kept independently of the active model.
    pub processor: ProcessorConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct PipelineFile {
    detection: ModelSelection,
    ocr: ModelSelection,
    translation: TranslationConfig,
    inpainting: InpaintingConfig,
    decensor: DecensorConfig,
    censor_detection: CensorDetectionConfig,
    #[serde(default)]
    processor: ProcessorConfig,
}

impl Default for PipelineFile {
    fn default() -> Self {
        Self {
            detection: ModelSelection {
                model: "koharu-layout-rfdetr-seg-2xl".to_owned(),
            },
            ocr: ModelSelection {
                model: "paddleocr-vl-1.6".to_owned(),
            },
            translation: TranslationConfig::default(),
            inpainting: InpaintingConfig::default(),
            decensor: DecensorConfig::default(),
            censor_detection: CensorDetectionConfig::default(),
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct ModelSelection {
    model: String,
}

impl Serialize for PipelineConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let detection = match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_) => "koharu-layout-rfdetr-seg-2xl",
        };
        let ocr = match &self.ocr {
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::HayaiOcr => "hayai-ocr",
        };
        let mut processor = self.processor.clone();
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = &self.detection;
        processor
            .koharu_layout_rfdetr_seg_2xl
            .get_or_insert_with(|| config.clone());
        PipelineFile {
            detection: ModelSelection {
                model: detection.to_owned(),
            },
            ocr: ModelSelection {
                model: ocr.to_owned(),
            },
            translation: self.translation.clone(),
            inpainting: self.inpainting.clone(),
            decensor: self.decensor.clone(),
            censor_detection: self.censor_detection.clone(),
            processor,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PipelineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let file = PipelineFile::deserialize(deserializer)?;
        let detection = match file.detection.model.as_str() {
            "koharu-layout-rfdetr-seg-2xl" => DetectionModel::KoharuLayoutRFDetrSeg2XL(
                file.processor
                    .koharu_layout_rfdetr_seg_2xl
                    .clone()
                    .unwrap_or_default(),
            ),
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported detection model {model}"
                )));
            }
        };
        let ocr = match file.ocr.model.as_str() {
            "paddleocr-vl-1.6" => OcrModel::PaddleOcrVl1_6,
            "manga-ocr" => OcrModel::MangaOcr,
            "baberu-ocr" => OcrModel::BaberuOcr,
            "hayai-ocr" => OcrModel::HayaiOcr,
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported OCR model {model}"
                )));
            }
        };
        Ok(Self {
            detection,
            ocr,
            translation: file.translation,
            inpainting: file.inpainting,
            decensor: file.decensor,
            censor_detection: file.censor_detection,
            processor: file.processor,
        })
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(
                KoharuLayoutRFDetrSeg2XLConfig::default(),
            ),
            ocr: OcrModel::PaddleOcrVl1_6,
            translation: TranslationConfig::default(),
            inpainting: InpaintingConfig::default(),
            decensor: DecensorConfig::default(),
            censor_detection: CensorDetectionConfig::default(),
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
#[specta(skip_attr = "serde")]
pub struct TranslationConfig {
    pub model: koharu_translator::ModelSelection,
    pub generation: GenerationConfig,
    #[specta(type = String)]
    pub source_language: Language,
    #[specta(type = String)]
    pub target_language: Language,
    /// Profile used by runs that do not name their own, such as editor runs.
    pub content_profile: ContentProfile,
    pub system_prompts: SystemPrompts,
    pub instructions: Option<String>,
    pub validators: Vec<TranslationValidationRule>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            model: koharu_translator::ModelSelection::default(),
            generation: GenerationConfig::default(),
            source_language: Language::Japanese,
            target_language: Language::English,
            content_profile: ContentProfile::default(),
            system_prompts: SystemPrompts::default(),
            instructions: None,
            validators: Vec::new(),
        }
    }
}

/// Saved replacements for the built-in translation prompts, one per content
/// profile. A missing or blank entry uses that profile's built-in prompt.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
#[specta(skip_attr = "serde")]
pub struct SystemPrompts {
    pub hentai: Option<String>,
    pub manga: Option<String>,
    pub comic: Option<String>,
}

impl SystemPrompts {
    #[must_use]
    pub fn get(&self, profile: ContentProfile) -> Option<&str> {
        match profile {
            ContentProfile::Hentai => &self.hentai,
            ContentProfile::Manga => &self.manga,
            ContentProfile::Comic => &self.comic,
        }
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
    }
}

impl PipelineConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("pipeline")
    }

    pub fn detection(&self) -> Result<DetectionModel> {
        match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config) => {
                Ok(DetectionModel::KoharuLayoutRFDetrSeg2XL(
                    self.processor
                        .koharu_layout_rfdetr_seg_2xl
                        .clone()
                        .unwrap_or_else(|| config.clone()),
                ))
            }
        }
    }

    pub fn decensor(&self) -> Result<DecensorModel> {
        Ok(self.decensor.selected())
    }

    pub fn validate(&self) -> Result<()> {
        let _ = self.detection()?;
        let _ = self.decensor()?;
        crate::stages::validate(self)?;
        self.censor_detection.validate()?;
        if !matches!(
            self.ocr,
            OcrModel::PaddleOcrVl1_6
                | OcrModel::MangaOcr
                | OcrModel::BaberuOcr
                | OcrModel::HayaiOcr
        ) {
            bail!("unsupported OCR model")
        }
        TranslationValidator::new(&self.translation.validators)?;
        Ok(())
    }
}

/// Detection settings kept independently of the active model. The generative
/// stages keep their own settings inside their stage configuration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ProcessorConfig {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    pub koharu_layout_rfdetr_seg_2xl: Option<KoharuLayoutRFDetrSeg2XLConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum DetectionModel {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum OcrModel {
    #[serde(rename = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[serde(rename = "manga-ocr")]
    MangaOcr,
    #[serde(rename = "baberu-ocr")]
    BaberuOcr,
    #[serde(rename = "hayai-ocr")]
    HayaiOcr,
}

/// Direct models that clean lettering inside speech bubbles.
///
/// Bubble interiors are mostly flat fills that these models rebuild quickly, so
/// no generative model is selectable here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
pub enum BubbleInpaintingKind {
    #[default]
    #[serde(rename = "lama")]
    LaMa,
    #[serde(rename = "aot-inpainting")]
    AotInpainting,
}

/// Models that can clean lettering over artwork.
///
/// Generative models repaint the covered drawing. LaMa and AOT only spread the
/// surrounding pixels, which is fast but smears detailed artwork.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
pub enum InpaintingKind {
    #[serde(rename = "lama")]
    LaMa,
    #[serde(rename = "aot-inpainting")]
    AotInpainting,
    #[serde(rename = "flux2-klein")]
    Flux2Klein,
    #[serde(rename = "flux2-klein-9b-uncensored")]
    Flux2Klein9B,
    #[serde(rename = "diffusioncreampy-animagine-xl-4")]
    DiffusionCreamPyAnimagineXl4,
    #[serde(rename = "diffusioncreampy-illustrious-xl-2")]
    DiffusionCreamPyIllustriousXl2,
    #[serde(rename = "rorem-mixed")]
    RoremMixed,
}

/// The inpainting stage. Lettering inside speech bubbles and lettering over
/// artwork are cleaned by independently selected models.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct InpaintingConfig {
    pub bubbles: BubbleInpaintingKind,
    pub artwork: ArtworkInpaintingConfig,
    /// The manual FLUX brush owns its prompt and sampling settings.
    pub manual: Flux2Klein9BConfig,
    pub manual_presets: Vec<ManualInpaintingPreset>,
}

impl Default for InpaintingConfig {
    fn default() -> Self {
        Self {
            bubbles: BubbleInpaintingKind::default(),
            artwork: ArtworkInpaintingConfig::default(),
            manual: Flux2Klein9BConfig::manual_inpainting(),
            manual_presets: Vec::new(),
        }
    }
}

/// The artwork route of the inpainting stage: its selected model and the
/// settings it keeps for every model.
///
/// Settings belong to the stage rather than to the model. The decensor stage
/// keeps its own copy, so selecting one model in both stages never shares a
/// prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ArtworkInpaintingConfig {
    pub model: InpaintingKind,
    #[serde(rename = "flux2-klein")]
    pub flux2_klein: Flux2KleinConfig,
    #[serde(rename = "flux2-klein-9b-uncensored")]
    pub flux2_klein_9b: Flux2Klein9BConfig,
    #[serde(rename = "diffusioncreampy-animagine-xl-4")]
    pub diffusioncreampy_animagine_xl_4: DiffusionCreamPyConfig,
    #[serde(rename = "diffusioncreampy-illustrious-xl-2")]
    pub diffusioncreampy_illustrious_xl_2: DiffusionCreamPyConfig,
    #[serde(rename = "rorem-mixed")]
    pub rorem_mixed: RoremMixedConfig,
}

impl Default for ArtworkInpaintingConfig {
    fn default() -> Self {
        Self {
            // 4B is Apache-2.0 and fits a 16 GB card with the other models;
            // docs/fork/cleanup-evidence.md compares it with 9B on lettering.
            model: InpaintingKind::Flux2Klein,
            flux2_klein: Flux2KleinConfig::text_cleanup(),
            flux2_klein_9b: Flux2Klein9BConfig::text_cleanup(),
            diffusioncreampy_animagine_xl_4: DiffusionCreamPyConfig::animagine_xl_4(),
            diffusioncreampy_illustrious_xl_2: DiffusionCreamPyConfig::illustrious_xl_2(),
            rorem_mixed: RoremMixedConfig::default(),
        }
    }
}

impl ArtworkInpaintingConfig {
    /// The selected model carrying this route's own settings for it.
    pub fn selected(&self) -> InpaintingModel {
        match self.model {
            InpaintingKind::LaMa => InpaintingModel::LaMa {},
            InpaintingKind::AotInpainting => InpaintingModel::AotInpainting {},
            InpaintingKind::Flux2Klein => InpaintingModel::Flux2Klein(self.flux2_klein.clone()),
            InpaintingKind::Flux2Klein9B => {
                InpaintingModel::Flux2Klein9B(self.flux2_klein_9b.clone())
            }
            InpaintingKind::DiffusionCreamPyAnimagineXl4 => {
                InpaintingModel::DiffusionCreamPyAnimagineXl4(
                    self.diffusioncreampy_animagine_xl_4.clone(),
                )
            }
            InpaintingKind::DiffusionCreamPyIllustriousXl2 => {
                InpaintingModel::DiffusionCreamPyIllustriousXl2(
                    self.diffusioncreampy_illustrious_xl_2.clone(),
                )
            }
            InpaintingKind::RoremMixed => InpaintingModel::RoremMixed(self.rorem_mixed.clone()),
        }
    }
}

/// A model together with the settings it runs with, as the inpainting stage
/// processor consumes it.
#[derive(Clone, Debug, PartialEq)]
pub enum InpaintingModel {
    LaMa {},
    AotInpainting {},
    Flux2Klein(Flux2KleinConfig),
    Flux2Klein9B(Flux2Klein9BConfig),
    DiffusionCreamPyAnimagineXl4(DiffusionCreamPyConfig),
    DiffusionCreamPyIllustriousXl2(DiffusionCreamPyConfig),
    RoremMixed(RoremMixedConfig),
}

/// Generative models that can reconstruct censored artwork.
///
/// Direct models such as LaMa and AOT only spread surrounding pixels, so they
/// are not selectable here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
pub enum DecensorKind {
    #[serde(rename = "flux2-klein-9b-uncensored")]
    Flux2Klein9B,
    #[serde(rename = "flux2-klein")]
    Flux2Klein,
    #[serde(rename = "diffusioncreampy-animagine-xl-4")]
    DiffusionCreamPyAnimagineXl4,
    #[serde(rename = "diffusioncreampy-illustrious-xl-2")]
    DiffusionCreamPyIllustriousXl2,
    #[serde(rename = "rorem-mixed")]
    RoremMixed,
}

/// The decensor stage: its selected model and the settings it keeps for every
/// model it can run.
///
/// Settings belong to the stage rather than to the model. The inpainting stage
/// keeps its own copy, so selecting one model in both stages never shares a
/// prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct DecensorConfig {
    pub model: DecensorKind,
    #[serde(rename = "flux2-klein")]
    pub flux2_klein: Flux2KleinConfig,
    #[serde(rename = "flux2-klein-9b-uncensored")]
    pub flux2_klein_9b: Flux2Klein9BConfig,
    #[serde(rename = "diffusioncreampy-animagine-xl-4")]
    pub diffusioncreampy_animagine_xl_4: DiffusionCreamPyConfig,
    #[serde(rename = "diffusioncreampy-illustrious-xl-2")]
    pub diffusioncreampy_illustrious_xl_2: DiffusionCreamPyConfig,
    #[serde(rename = "rorem-mixed")]
    pub rorem_mixed: RoremMixedConfig,
}

impl Default for DecensorConfig {
    fn default() -> Self {
        Self {
            model: DecensorKind::Flux2Klein9B,
            flux2_klein: Flux2KleinConfig::decensor(),
            flux2_klein_9b: Flux2Klein9BConfig::decensor(),
            diffusioncreampy_animagine_xl_4: DiffusionCreamPyConfig {
                positive_prompt: DECENSOR_PROMPT.to_owned(),
                ..DiffusionCreamPyConfig::animagine_xl_4()
            },
            diffusioncreampy_illustrious_xl_2: DiffusionCreamPyConfig {
                positive_prompt: DECENSOR_PROMPT.to_owned(),
                ..DiffusionCreamPyConfig::illustrious_xl_2()
            },
            rorem_mixed: RoremMixedConfig::decensor(),
        }
    }
}

impl DecensorConfig {
    /// The selected model carrying this stage's own settings for it.
    pub fn selected(&self) -> DecensorModel {
        match self.model {
            DecensorKind::Flux2Klein9B => DecensorModel::Flux2Klein9B(self.flux2_klein_9b.clone()),
            DecensorKind::Flux2Klein => DecensorModel::Flux2Klein(self.flux2_klein.clone()),
            DecensorKind::DiffusionCreamPyAnimagineXl4 => {
                DecensorModel::DiffusionCreamPyAnimagineXl4(
                    self.diffusioncreampy_animagine_xl_4.clone(),
                )
            }
            DecensorKind::DiffusionCreamPyIllustriousXl2 => {
                DecensorModel::DiffusionCreamPyIllustriousXl2(
                    self.diffusioncreampy_illustrious_xl_2.clone(),
                )
            }
            DecensorKind::RoremMixed => DecensorModel::RoremMixed(self.rorem_mixed.clone()),
        }
    }
}

/// The decensor stage's selected model together with its settings, as the
/// stage processor consumes it.
#[derive(Clone, Debug, PartialEq)]
pub enum DecensorModel {
    Flux2Klein9B(Flux2Klein9BConfig),
    Flux2Klein(Flux2KleinConfig),
    DiffusionCreamPyAnimagineXl4(DiffusionCreamPyConfig),
    DiffusionCreamPyIllustriousXl2(DiffusionCreamPyConfig),
    RoremMixed(RoremMixedConfig),
}

impl Default for DecensorModel {
    fn default() -> Self {
        Self::Flux2Klein9B(Flux2Klein9BConfig::decensor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Saved settings travel as TOML on disk and as JSON through the settings
    /// page, so both encodings must preserve every stage exactly.
    fn round_trips(config: &PipelineConfig) -> [PipelineConfig; 2] {
        [
            toml::from_str(&toml::to_string(config).unwrap()).unwrap(),
            serde_json::from_str(&serde_json::to_string(config).unwrap()).unwrap(),
        ]
    }

    #[test]
    fn defaults_select_one_processor_for_each_phase() {
        let config = PipelineConfig::default();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::PaddleOcrVl1_6));
        assert_eq!(config.inpainting.bubbles, BubbleInpaintingKind::LaMa);
        assert_eq!(config.decensor.model, DecensorKind::Flux2Klein9B);
        assert_eq!(
            config.censor_detection,
            CensorDetectionConfig {
                enabled: true,
                detect_mosaic: true,
                threshold: 0.5,
                mask_grow: 12,
                mask_blur: 4.0,
            }
        );
        assert_eq!(config.translation.source_language, Language::Japanese);
    }

    #[test]
    fn each_stage_starts_from_its_own_prompts() {
        let config = PipelineConfig::default();

        assert!(matches!(
            config.decensor().unwrap(),
            DecensorModel::Flux2Klein9B(profile) if profile.prompt == DECENSOR_PROMPT
        ));
        assert_eq!(config.decensor().unwrap(), DecensorModel::default());
        assert_eq!(config.decensor.flux2_klein.prompt, DECENSOR_PROMPT);
        assert_eq!(config.decensor.rorem_mixed.prompt, DECENSOR_PROMPT);
        assert_eq!(
            config
                .decensor
                .diffusioncreampy_animagine_xl_4
                .positive_prompt,
            DECENSOR_PROMPT
        );
        assert_eq!(
            config
                .decensor
                .diffusioncreampy_illustrious_xl_2
                .positive_prompt,
            DECENSOR_PROMPT
        );
        assert_eq!(
            config.inpainting.artwork.flux2_klein_9b,
            Flux2Klein9BConfig::text_cleanup()
        );
        assert_eq!(
            config.inpainting.artwork.flux2_klein,
            Flux2KleinConfig::text_cleanup()
        );
    }

    #[test]
    fn fills_missing_translation_fields_from_defaults() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [translation]
                target_language = "ru-RU"
            "#,
        )
        .unwrap();

        assert_eq!(config.translation.source_language, Language::Japanese);
        assert_eq!(config.translation.target_language, Language::Russian);
        assert!(config.translation.validators.is_empty());
    }

    #[test]
    fn rejects_invalid_translation_validator_configuration() {
        let mut config = PipelineConfig::default();
        config.translation.validators = vec![TranslationValidationRule {
            name: "Broken".to_owned(),
            pattern: "[".to_owned(),
        }];

        assert!(config.validate().is_err());
    }

    #[test]
    fn parses_stage_owned_model_settings() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [ocr]
                model = "baberu-ocr"

                [inpainting.artwork]
                model = "rorem-mixed"

                [inpainting.artwork.rorem-mixed]
                prompt = "Remove the lettering."
                negative_prompt = "letters, words"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::BaberuOcr));
        assert!(matches!(
            config.inpainting.artwork.selected(),
            InpaintingModel::RoremMixed(config)
                if config.prompt == "Remove the lettering."
                    && config.negative_prompt == "letters, words"
        ));
    }

    #[test]
    fn missing_slots_use_defaults() {
        let config = toml::from_str::<PipelineConfig>("").unwrap();

        assert_eq!(config, PipelineConfig::default());
    }

    #[test]
    fn rejects_removed_flux9b_profiles() {
        for model in ["flux2-klein-9b-lineart", "flux2-klein-base-9b-lineart"] {
            let document = format!("[inpainting.artwork]\nmodel = \"{model}\"\n");
            let error = toml::from_str::<PipelineConfig>(&document).unwrap_err();
            assert!(error.to_string().contains("unknown variant"));
        }
    }

    #[test]
    fn ignores_unknown_model_configuration_fields() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"
                legacy_threshold = 0.5

                [ocr]
                model = "paddleocr-vl-1.6"
                legacy_language = "ja"

                [inpainting]
                bubbles = "aot-inpainting"
                legacy_resolution = 1024

                [inpainting.artwork]
                model = "lama"
                legacy_resolution = 1024
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::PaddleOcrVl1_6));
        assert_eq!(
            config.inpainting.bubbles,
            BubbleInpaintingKind::AotInpainting
        );
        assert_eq!(
            config.inpainting.artwork.selected(),
            InpaintingModel::LaMa {}
        );
    }

    #[test]
    fn parses_detection_and_generative_inpainting_options() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting.artwork]
                model = "flux2-klein"

                [inpainting.artwork.flux2-klein]
                prompt = "Reconstruct the illustration without text."

                [processor."koharu-layout-rfdetr-seg-2xl"]
                text_threshold = 0.25
                bubble_threshold = 0.45
                panel_threshold = 0.55
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection().unwrap(),
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config)
                if config.text_threshold == Some(0.25)
                    && config.bubble_threshold == Some(0.45)
                    && config.panel_threshold == Some(0.55)
        ));
        assert!(matches!(
            config.inpainting.artwork.selected(),
            InpaintingModel::Flux2Klein(config)
                if config.prompt == "Reconstruct the illustration without text."
        ));
    }

    #[test]
    fn keeps_diffusioncreampy_profiles_independent() {
        let mut config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting.artwork]
                model = "diffusioncreampy-animagine-xl-4"

                [inpainting.artwork.diffusioncreampy-animagine-xl-4]
                endpoint = "http://127.0.0.1:8188"

                [inpainting.artwork.diffusioncreampy-illustrious-xl-2]
                endpoint = "http://127.0.0.1:8189"
                checkpoint = "custom-illustrious.safetensors"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.inpainting.artwork.selected(),
            InpaintingModel::DiffusionCreamPyAnimagineXl4(profile)
                if profile.endpoint == "http://127.0.0.1:8188"
        ));

        config.inpainting.artwork.model = InpaintingKind::DiffusionCreamPyIllustriousXl2;
        assert!(matches!(
            config.inpainting.artwork.selected(),
            InpaintingModel::DiffusionCreamPyIllustriousXl2(profile)
                if profile.endpoint == "http://127.0.0.1:8189"
                    && profile.checkpoint == "custom-illustrious.safetensors"
        ));
    }

    #[test]
    fn serializes_each_stage_with_its_own_model_tables() {
        let config = PipelineConfig {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(0.25),
                ..Default::default()
            }),
            inpainting: InpaintingConfig {
                bubbles: BubbleInpaintingKind::AotInpainting,
                artwork: ArtworkInpaintingConfig {
                    model: InpaintingKind::Flux2Klein,
                    flux2_klein: Flux2KleinConfig {
                        prompt: "Keep the line art.".to_owned(),
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            decensor: DecensorConfig {
                model: DecensorKind::RoremMixed,
                rorem_mixed: RoremMixedConfig {
                    prompt: "Erase the bars.".to_owned(),
                    negative_prompt: "color".to_owned(),
                },
                ..Default::default()
            },
            censor_detection: CensorDetectionConfig {
                mask_grow: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let document = toml::to_string(&config).unwrap();

        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();
        assert_eq!(
            restored.inpainting.bubbles,
            BubbleInpaintingKind::AotInpainting
        );
        assert!(matches!(
            restored.inpainting.artwork.selected(),
            InpaintingModel::Flux2Klein(config) if config.prompt == "Keep the line art."
        ));
        assert!(matches!(
            restored.decensor().unwrap(),
            DecensorModel::RoremMixed(config) if config.prompt == "Erase the bars."
        ));
        assert_eq!(restored.censor_detection.mask_grow, 3);
    }

    #[test]
    fn parses_an_independent_decensoring_selection() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [inpainting]
                bubbles = "aot-inpainting"

                [inpainting.artwork]
                model = "lama"

                [decensor]
                model = "diffusioncreampy-animagine-xl-4"

                [decensor.diffusioncreampy-animagine-xl-4]
                endpoint = "http://127.0.0.1:8188"
                checkpoint = "animagine-xl-4.0-opt.safetensors"
                controlnet = "controlnet-union-sdxl-1.0.safetensors"
                num_inference_steps = 28

                [censor_detection]
                enabled = false
                detect_mosaic = false
                threshold = 0.7
                mask_grow = 3
                mask_blur = 0.0
            "#,
        )
        .unwrap();

        assert_eq!(
            config.inpainting.bubbles,
            BubbleInpaintingKind::AotInpainting
        );
        assert_eq!(
            config.inpainting.artwork.selected(),
            InpaintingModel::LaMa {}
        );
        assert!(matches!(
            config.decensor().unwrap(),
            DecensorModel::DiffusionCreamPyAnimagineXl4(profile)
                if profile.endpoint == "http://127.0.0.1:8188"
                    && profile.num_inference_steps == 28
        ));
        assert_eq!(
            config.censor_detection,
            CensorDetectionConfig {
                enabled: false,
                detect_mosaic: false,
                threshold: 0.7,
                mask_grow: 3,
                mask_blur: 0.0,
            }
        );
    }

    #[test]
    fn selecting_a_direct_model_for_decensoring_is_rejected() {
        for model in ["lama", "aot-inpainting"] {
            let document = format!("[decensor]\nmodel = \"{model}\"\n");
            let error = toml::from_str::<PipelineConfig>(&document).unwrap_err();
            assert!(error.to_string().contains("unknown variant"));
        }
    }

    #[test]
    fn selecting_a_generative_model_for_bubbles_is_rejected() {
        for model in ["flux2-klein", "flux2-klein-9b-uncensored", "rorem-mixed"] {
            let document = format!("[inpainting]\nbubbles = \"{model}\"\n");
            let error = toml::from_str::<PipelineConfig>(&document).unwrap_err();
            assert!(error.to_string().contains("unknown variant"));
        }
    }

    #[test]
    fn rejects_invalid_censor_detection_settings() {
        let mut config = PipelineConfig::default();
        config.censor_detection.threshold = 1.5;
        assert!(config.validate().is_err());

        let mut config = PipelineConfig::default();
        config.censor_detection.mask_grow = 4096;
        assert!(config.validate().is_err());

        let mut config = PipelineConfig::default();
        config.censor_detection.mask_blur = f32::NAN;
        assert!(config.validate().is_err());
    }

    #[test]
    fn stages_keep_independent_prompts_for_every_generative_model() {
        let mut config = PipelineConfig::default();
        config.inpainting.bubbles = BubbleInpaintingKind::AotInpainting;
        config.inpainting.artwork.flux2_klein.prompt = "inpainting flux".to_owned();
        config.decensor.flux2_klein.prompt = "decensor flux".to_owned();
        config.inpainting.artwork.flux2_klein_9b.prompt = "inpainting flux 9b".to_owned();
        config.decensor.flux2_klein_9b.prompt = "decensor flux 9b".to_owned();
        for (animagine, illustrious, stage) in [
            (
                &mut config.inpainting.artwork.diffusioncreampy_animagine_xl_4,
                &mut config.inpainting.artwork.diffusioncreampy_illustrious_xl_2,
                "inpainting",
            ),
            (
                &mut config.decensor.diffusioncreampy_animagine_xl_4,
                &mut config.decensor.diffusioncreampy_illustrious_xl_2,
                "decensor",
            ),
        ] {
            animagine.positive_prompt = format!("{stage} animagine");
            animagine.negative_prompt = format!("{stage} animagine negative");
            illustrious.positive_prompt = format!("{stage} illustrious");
            illustrious.negative_prompt = format!("{stage} illustrious negative");
        }
        config.inpainting.artwork.rorem_mixed = RoremMixedConfig {
            prompt: "inpainting rorem".to_owned(),
            negative_prompt: "inpainting rorem negative".to_owned(),
        };
        config.decensor.rorem_mixed = RoremMixedConfig {
            prompt: "decensor rorem".to_owned(),
            negative_prompt: "decensor rorem negative".to_owned(),
        };

        for restored in round_trips(&config) {
            assert_eq!(restored.inpainting, config.inpainting);
            assert_eq!(restored.decensor, config.decensor);
        }
    }

    #[test]
    fn a_model_selected_by_both_stages_resolves_each_stages_own_prompt() {
        let mut config = PipelineConfig::default();
        config.inpainting.artwork.model = InpaintingKind::Flux2Klein9B;
        config.inpainting.artwork.flux2_klein_9b.prompt = "Erase the lettering.".to_owned();
        config.decensor.model = DecensorKind::Flux2Klein9B;

        for restored in round_trips(&config) {
            assert!(matches!(
                restored.inpainting.artwork.selected(),
                InpaintingModel::Flux2Klein9B(profile) if profile.prompt == "Erase the lettering."
            ));
            assert!(matches!(
                restored.decensor().unwrap(),
                DecensorModel::Flux2Klein9B(profile) if profile.prompt == DECENSOR_PROMPT
            ));
        }
    }

    #[test]
    fn switching_models_keeps_every_profile_of_the_stage() {
        let mut config = PipelineConfig::default();
        config.decensor.flux2_klein.prompt = "Repaint the bars.".to_owned();
        config.decensor.rorem_mixed.negative_prompt = "black bars".to_owned();

        config.decensor.model = DecensorKind::RoremMixed;
        let [mut config, _] = round_trips(&config);
        assert!(matches!(
            config.decensor().unwrap(),
            DecensorModel::RoremMixed(profile) if profile.negative_prompt == "black bars"
        ));

        config.decensor.model = DecensorKind::Flux2Klein;
        let [_, config] = round_trips(&config);
        assert!(matches!(
            config.decensor().unwrap(),
            DecensorModel::Flux2Klein(profile) if profile.prompt == "Repaint the bars."
        ));
        assert_eq!(config.decensor.rorem_mixed.negative_prompt, "black bars");
        assert_eq!(config.inpainting, InpaintingConfig::default());
    }

    #[test]
    fn missing_stage_tables_use_that_stages_defaults() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [inpainting.artwork]
                model = "flux2-klein"

                [decensor]
                model = "flux2-klein"
            "#,
        )
        .unwrap();

        assert_eq!(
            config.inpainting.artwork.selected(),
            InpaintingModel::Flux2Klein(Flux2KleinConfig::text_cleanup())
        );
        assert!(matches!(
            config.decensor().unwrap(),
            DecensorModel::Flux2Klein(profile) if profile.prompt == DECENSOR_PROMPT
        ));
    }
}
