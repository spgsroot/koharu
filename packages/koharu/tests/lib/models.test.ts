import { describe, expect, it } from 'vitest'

import { editModel, selectModel, stageModel } from '@/components/preferences/models'
import type { PipelineConfig } from '@koharu/bridge/protocol'

/** Settings one stage stores for every generative model, as the backend always sends them. */
function entries(stage: string) {
  return {
    'flux2-klein': { prompt: `${stage} flux` },
    'flux2-klein-9b-uncensored': {
      prompt: `${stage} flux 9b`,
      num_inference_steps: 8,
      guidance_scale: 1,
    },
    'diffusioncreampy-animagine-xl-4': {
      checkpoint: 'animagine-xl-4.0-opt.safetensors',
      positive_prompt: `${stage} animagine`,
      negative_prompt: `${stage} animagine negative`,
    },
    'diffusioncreampy-illustrious-xl-2': {
      checkpoint: 'Illustrious-XL-v2.0.safetensors',
      positive_prompt: `${stage} illustrious`,
      negative_prompt: `${stage} illustrious negative`,
    },
    'rorem-mixed': { prompt: `${stage} rorem`, negative_prompt: `${stage} rorem negative` },
  }
}

const baseConfig: PipelineConfig = {
  detection: { model: 'koharu-layout-rfdetr-seg-2xl' },
  ocr: { model: 'paddleocr-vl-1.6' },
  translation: {
    model: {
      provider: 'local',
      model: 'gemma4-e2b-it',
      quantization: null,
      vision: true,
      reasoning: true,
    },
    generation: { vision: true, reasoning: false },
    source_language: 'ja-JP',
    target_language: 'en-US',
    content_profile: 'manga',
    system_prompts: { hentai: null, manga: null, comic: null },
    instructions: null,
    validators: [],
  },
  inpainting: {
    bubbles: 'lama',
    artwork: { model: 'flux2-klein-9b-uncensored', ...entries('artwork') },
    manual: {},
  },
  decensor: { model: 'flux2-klein-9b-uncensored', ...entries('decensor') },
  censor_detection: {},
  processor: {},
}

const bothOnFlux9b = selectModel(baseConfig, 'artwork', 'flux2-klein-9b-uncensored')

describe('pipeline model registry', () => {
  it('edits one stage without changing the other stage selecting the same model', () => {
    const decensorEdited = editModel(bothOnFlux9b, 'decensor', {
      model: 'flux2-klein-9b-uncensored',
      prompt: 'Rebuild the obscured anatomy.',
      num_inference_steps: 8,
      guidance_scale: 2,
    })

    expect(decensorEdited.decensor['flux2-klein-9b-uncensored']).toEqual({
      prompt: 'Rebuild the obscured anatomy.',
      num_inference_steps: 8,
      guidance_scale: 2,
    })
    expect(decensorEdited.inpainting).toEqual(bothOnFlux9b.inpainting)

    const inpaintingEdited = editModel(decensorEdited, 'artwork', {
      model: 'flux2-klein-9b-uncensored',
      prompt: 'Erase the lettering.',
      num_inference_steps: 8,
      guidance_scale: 1,
    })

    expect(inpaintingEdited.inpainting.artwork?.['flux2-klein-9b-uncensored']?.prompt).toBe(
      'Erase the lettering.',
    )
    expect(inpaintingEdited.decensor).toEqual(decensorEdited.decensor)
  })

  it('keeps every stored profile when a stage switches models', () => {
    const edited = editModel(baseConfig, 'decensor', {
      model: 'rorem-mixed',
      prompt: 'Erase the bars.',
      negative_prompt: 'black bars',
    })
    const onFlux = selectModel(edited, 'decensor', 'flux2-klein')

    expect(onFlux.decensor.model).toBe('flux2-klein')
    expect(stageModel(onFlux, 'decensor')).toMatchObject({
      model: 'flux2-klein',
      prompt: 'decensor flux',
    })
    expect(stageModel(selectModel(onFlux, 'decensor', 'rorem-mixed'), 'decensor')).toMatchObject({
      model: 'rorem-mixed',
      prompt: 'Erase the bars.',
      negative_prompt: 'black bars',
    })
    expect(onFlux.inpainting).toEqual(baseConfig.inpainting)
  })

  it('stores an edited ComfyUI profile under its own stage only', () => {
    const config = editModel(baseConfig, 'artwork', {
      model: 'diffusioncreampy-animagine-xl-4',
      ...entries('artwork')['diffusioncreampy-animagine-xl-4'],
      denoise: 0.65,
    })

    expect(config.inpainting.artwork).toMatchObject({
      model: 'diffusioncreampy-animagine-xl-4',
      'diffusioncreampy-animagine-xl-4': {
        checkpoint: 'animagine-xl-4.0-opt.safetensors',
        denoise: 0.65,
      },
    })
    expect(config.decensor).toBe(baseConfig.decensor)
    expect(config.inpainting.bubbles).toBe(baseConfig.inpainting.bubbles)
  })

  it('switches bubble models without changing artwork or decensor settings', () => {
    const selected = selectModel(baseConfig, 'bubbles', 'aot-inpainting')
    expect(stageModel(selected, 'bubbles')).toEqual({ model: 'aot-inpainting' })
    expect(selected.inpainting.artwork).toEqual(baseConfig.inpainting.artwork)
    expect(selected.decensor).toEqual(baseConfig.decensor)
  })
})
