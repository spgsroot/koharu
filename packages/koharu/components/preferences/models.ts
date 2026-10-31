import type {
  ArtworkInpaintingConfig,
  BubbleInpaintingKind,
  DecensorConfig,
  DecensorKind,
  DetectionModel,
  DiffusionCreamPyConfig,
  Flux2Klein9BConfig,
  Flux2KleinConfig,
  InpaintingKind,
  OcrModel,
  PipelineConfig,
  RoremMixedConfig,
} from '@koharu/bridge/protocol'

/**
 * A model together with the settings its stage keeps for it. The artwork
 * inpainting and decensor stages each store their own entry per model, so a
 * model selected by both is edited independently in each.
 */
export type PipelineModel =
  | DetectionModel
  | OcrModel
  | { model: 'lama' | 'aot-inpainting' }
  | ({ model: 'flux2-klein' } & Flux2KleinConfig)
  | ({ model: 'flux2-klein-9b-uncensored' } & Flux2Klein9BConfig)
  | ({
      model: 'diffusioncreampy-animagine-xl-4' | 'diffusioncreampy-illustrious-xl-2'
    } & DiffusionCreamPyConfig)
  | ({ model: 'rorem-mixed' } & RoremMixedConfig)

/**
 * Where a model is chosen. Automatic inpainting chooses two models: one for lettering
 * inside speech bubbles and one for lettering over artwork. `manual` is the profile of
 * the Inpainting brush, which always runs FLUX.2 Klein 9B with its own settings.
 */
export type ModelStage = 'detection' | 'ocr' | 'bubbles' | 'artwork' | 'manual' | 'decensor'
export type ModelName = PipelineModel['model']

/** The stages that keep their settings under each model's own name. */
type Profiles = ArtworkInpaintingConfig | DecensorConfig
type StoredModel = Exclude<keyof Profiles, 'model'>

/** The first model of each stage is the one the backend selects by default. */
export const modelOptions = {
  detection: ['koharu-layout-rfdetr-seg-2xl'],
  ocr: ['paddleocr-vl-1.6', 'manga-ocr', 'baberu-ocr', 'hayai-ocr'],
  bubbles: ['lama', 'aot-inpainting'],
  artwork: [
    'flux2-klein-9b-uncensored',
    'flux2-klein',
    'diffusioncreampy-animagine-xl-4',
    'diffusioncreampy-illustrious-xl-2',
    'rorem-mixed',
    'lama',
    'aot-inpainting',
  ],
  manual: ['flux2-klein-9b-uncensored'],
  decensor: [
    'flux2-klein-9b-uncensored',
    'flux2-klein',
    'diffusioncreampy-animagine-xl-4',
    'diffusioncreampy-illustrious-xl-2',
    'rorem-mixed',
  ],
} satisfies Record<ModelStage, ModelName[]> & {
  bubbles: BubbleInpaintingKind[]
  artwork: InpaintingKind[]
  decensor: DecensorKind[]
}

export const modelNames: Record<ModelName, string> = {
  'koharu-layout-rfdetr-seg-2xl': 'Koharu Layout RF-DETR Seg 2XL',
  'paddleocr-vl-1.6': 'PaddleOCR-VL 1.6',
  'manga-ocr': 'Manga OCR',
  'baberu-ocr': 'Baberu OCR',
  'hayai-ocr': 'Hayai OCR',
  lama: 'LaMa',
  'aot-inpainting': 'AOT Inpainting',
  'flux2-klein': 'FLUX.2 Klein',
  'flux2-klein-9b-uncensored': 'FLUX.2 Klein 9B Uncensored',
  'diffusioncreampy-animagine-xl-4': 'DiffusionCreamPy · Animagine XL 4.0 Opt',
  'diffusioncreampy-illustrious-xl-2': 'DiffusionCreamPy · Illustrious XL 2.0',
  'rorem-mixed': 'RORem Mixed',
}

/**
 * The stage's selected model with the settings it keeps for that model. The
 * backend always sends every entry, so defaults are never invented here.
 */
export function stageModel(config: PipelineConfig, stage: ModelStage): PipelineModel {
  switch (stage) {
    case 'detection': {
      const settings = config.processor?.['koharu-layout-rfdetr-seg-2xl']
      return settings ? { ...config.detection, ...settings } : config.detection
    }
    case 'ocr':
      return config.ocr
    case 'bubbles':
      return { model: config.inpainting.bubbles ?? modelOptions.bubbles[0] }
    case 'artwork':
    case 'decensor': {
      const settings = stage === 'artwork' ? config.inpainting.artwork : config.decensor
      const model = settings?.model ?? modelOptions[stage][0]
      return { model, ...settings?.[model as StoredModel] } as PipelineModel
    }
    case 'manual':
      return { model: 'flux2-klein-9b-uncensored', ...config.inpainting.manual }
  }
}

/** Switches a stage's model without touching any stored settings. */
export function selectModel(
  config: PipelineConfig,
  stage: ModelStage,
  model: ModelName,
): PipelineConfig {
  switch (stage) {
    case 'detection':
      return { ...config, detection: { ...config.detection, model } as DetectionModel }
    case 'ocr':
      return { ...config, ocr: { model } as OcrModel }
    case 'bubbles':
      return {
        ...config,
        inpainting: { ...config.inpainting, bubbles: model as BubbleInpaintingKind },
      }
    case 'artwork':
      return {
        ...config,
        inpainting: {
          ...config.inpainting,
          artwork: { ...config.inpainting.artwork, model: model as InpaintingKind },
        },
      }
    case 'decensor':
      return { ...config, decensor: { ...config.decensor, model: model as DecensorKind } }
    case 'manual':
      // The manual profile is always FLUX.2 Klein 9B, so there is no model to switch.
      return config
  }
}

/** Stores edited settings for a stage's selected model, leaving the other stage alone. */
export function editModel(
  config: PipelineConfig,
  stage: ModelStage,
  edited: PipelineModel,
): PipelineConfig {
  switch (stage) {
    case 'detection':
      return {
        ...config,
        detection: edited as DetectionModel,
        processor: {
          ...config.processor,
          'koharu-layout-rfdetr-seg-2xl':
            edited.model === 'koharu-layout-rfdetr-seg-2xl'
              ? {
                  text_threshold: edited.text_threshold ?? null,
                  bubble_threshold: edited.bubble_threshold ?? null,
                  panel_threshold: edited.panel_threshold ?? null,
                }
              : (config.processor?.['koharu-layout-rfdetr-seg-2xl'] ?? null),
        },
      }
    case 'ocr':
      return { ...config, ocr: edited as OcrModel }
    case 'bubbles':
      return selectModel(config, stage, edited.model)
    case 'artwork':
      return {
        ...config,
        inpainting: {
          ...config.inpainting,
          artwork: withSettings(config.inpainting.artwork, edited),
        },
      }
    case 'decensor':
      return { ...config, decensor: withSettings(config.decensor, edited) }
    case 'manual': {
      if (edited.model !== 'flux2-klein-9b-uncensored') return config
      const { model: _model, ...manual } = edited
      return { ...config, inpainting: { ...config.inpainting, manual } }
    }
  }
}

/** Selects the edited model and replaces only that model's entry in the stage. */
function withSettings<Config extends Profiles>(
  stage: Config | undefined,
  edited: PipelineModel,
): Config {
  const { model, ...settings } = edited
  const next: Record<string, unknown> = { ...stage, model, [model]: settings }
  return next as Config
}
