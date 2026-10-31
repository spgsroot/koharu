import type {
  GenerationConfig,
  LanguageChoice,
  Model,
  ModelSelection,
  Provider,
  ProviderPreference,
  TranslationConfig,
} from '@koharu/bridge/protocol'

export function providerName(entries: ProviderPreference[], provider: Provider): string {
  return entries.find((entry) => entry.config.provider === provider)?.name ?? provider
}

export function modelKey(model: Model | ModelSelection): string {
  return `${model.provider}:${model.model ?? ''}`
}

/**
 * Moves the translation settings to another model. A reasoning level is only
 * valid for a model that offers it, so a level the new model lacks is cleared
 * instead of being carried to a model that would reject it.
 */
export function translationWithModel(
  translation: TranslationConfig,
  model: Model,
): TranslationConfig {
  const effort = translation.generation.reasoning_effort
  return {
    ...translation,
    model: {
      provider: model.provider,
      model: model.model,
      quantization: model.quantizations[0]?.id ?? null,
      vision: model.vision,
      reasoning: model.reasoning,
      reasoning_efforts: model.reasoning_efforts,
      reasoning_always_on: model.reasoning_always_on,
    },
    generation:
      effort && !model.reasoning_efforts.includes(effort)
        ? { ...translation.generation, reasoning_effort: null }
        : translation.generation,
  }
}

/**
 * Applies edited generation settings. The levels a model accepts are stored
 * with its selection, so choosing a different level also stores the reasoning
 * capability of the model it was chosen for.
 */
export function translationWithGeneration(
  translation: TranslationConfig,
  generation: GenerationConfig,
  model: Model,
): TranslationConfig {
  const levelChanged =
    (generation.reasoning_effort ?? null) !== (translation.generation.reasoning_effort ?? null)
  return {
    ...translation,
    model: levelChanged
      ? {
          ...translation.model,
          reasoning: model.reasoning,
          reasoning_efforts: model.reasoning_efforts,
          reasoning_always_on: model.reasoning_always_on,
        }
      : translation.model,
    generation,
  }
}

export function orderedLanguageChoices(
  languages: readonly LanguageChoice[],
): Array<{ tag: string; name: string }> {
  return languages
    .map((language) => ({ tag: language.tag, name: language.name }))
    .sort((left, right) =>
      left.name.localeCompare(right.name, undefined, { numeric: true, sensitivity: 'base' }),
    )
}
