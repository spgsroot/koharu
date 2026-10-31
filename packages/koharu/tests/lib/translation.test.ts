import { describe, expect, it } from 'vitest'

import { translationWithGeneration, translationWithModel } from '@/lib/translation'
import type { Model, TranslationConfig } from '@koharu/bridge/protocol'

function model(reasoning_efforts: Model['reasoning_efforts']): Model {
  return {
    provider: 'openai',
    model: 'gpt-5.5',
    name: 'GPT-5.5',
    quantizations: [],
    vision: true,
    reasoning: true,
    reasoning_efforts,
    reasoning_always_on: false,
  }
}

const translation: TranslationConfig = {
  model: {
    provider: 'local',
    model: 'gemma4-e2b-it',
    quantization: null,
    vision: true,
    reasoning: true,
  },
  generation: { reasoning: true, reasoning_effort: 'xhigh' },
  source_language: 'ja-JP',
  target_language: 'en-US',
  content_profile: 'manga',
  system_prompts: { hentai: null, manga: null, comic: null },
  instructions: null,
  validators: [],
}

describe('translation settings', () => {
  it('keeps a reasoning level the new model offers and clears one it does not', () => {
    const kept = translationWithModel(translation, model(['low', 'xhigh']))
    expect(kept.generation).toEqual({ reasoning: true, reasoning_effort: 'xhigh' })
    expect(kept.model.reasoning_efforts).toEqual(['low', 'xhigh'])

    const cleared = translationWithModel(translation, model(['low', 'high']))
    expect(cleared.generation).toEqual({ reasoning: true, reasoning_effort: null })
    expect(cleared.model.reasoning_efforts).toEqual(['low', 'high'])
  })

  it('does not rewrite generation settings that hold no level', () => {
    const plain = { ...translation, generation: { reasoning: false } }
    expect(translationWithModel(plain, model([])).generation).toBe(plain.generation)
  })

  it('stores the reasoning capability a level was chosen against', () => {
    const chosen = translationWithGeneration(
      translation,
      { ...translation.generation, reasoning_effort: 'high' },
      model(['low', 'high']),
    )
    expect(chosen.generation.reasoning_effort).toBe('high')
    expect(chosen.model).toMatchObject({ reasoning: true, reasoning_efforts: ['low', 'high'] })

    const edited = translationWithGeneration(
      translation,
      { ...translation.generation, temperature: 0.3 },
      model(['low']),
    )
    expect(edited.model).toBe(translation.model)
  })
})
