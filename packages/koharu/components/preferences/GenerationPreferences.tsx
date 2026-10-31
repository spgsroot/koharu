'use client'

import { useTranslation } from 'react-i18next'

import {
  NumberField,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import type { GenerationConfig, Model, ReasoningEffort } from '@koharu/bridge/protocol'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Switch } from '@koharu/ui/components/switch'

/** Stands for "no level chosen": the provider keeps its own default depth. */
const providerDefault = 'default'

export function GenerationPreferences({
  value,
  model,
  onChange,
}: {
  value: GenerationConfig
  /** Reasoning capabilities of the model these settings apply to. */
  model: Pick<Model, 'reasoning' | 'reasoning_efforts'>
  onChange: (value: GenerationConfig) => void
}) {
  const { t } = useTranslation()
  const update = (changes: Partial<GenerationConfig>) => onChange({ ...value, ...changes })
  const levels = model.reasoning_efforts
  const level = value.reasoning_effort ?? null
  const levelUnsupported = level !== null && !levels.includes(level)
  const levelLabel = (effort: ReasoningEffort) => t(`settings.generation.reasoningLevels.${effort}`)
  const levelOptions = levelUnsupported ? [...levels, level] : levels
  const reasoningNote = !model.reasoning
    ? t('settings.generation.reasoningUnsupported')
    : levels.length === 0
      ? t('settings.generation.reasoningOnOffOnly')
      : null
  return (
    <PreferenceSection
      title={t('settings.generation.title')}
      description={t('settings.generation.description')}
    >
      <PreferenceRow
        title={t('settings.generation.sampling')}
        description={t('settings.generation.samplingDescription')}
        align='start'
      >
        <div className='grid grid-cols-2 gap-2'>
          <NumberField
            label={t('settings.generation.temperature')}
            value={value.temperature ?? null}
            min={0}
            max={2}
            step={0.1}
            onChange={(temperature) => update({ temperature })}
          />
          <NumberField
            label={t('settings.generation.topK')}
            value={value.top_k ?? null}
            min={1}
            step={1}
            onChange={(top_k) => update({ top_k })}
          />
          <NumberField
            label={t('settings.generation.topP')}
            value={value.top_p ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(top_p) => update({ top_p })}
          />
          <NumberField
            label={t('settings.generation.minP')}
            value={value.min_p ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(min_p) => update({ min_p })}
          />
          <NumberField
            label={t('settings.generation.maxTokens')}
            value={value.max_tokens ?? null}
            min={1}
            step={1}
            onChange={(max_tokens) => update({ max_tokens })}
          />
          <NumberField
            label={t('settings.generation.repeatPenalty')}
            value={value.repeat_penalty ?? null}
            min={0}
            step={0.05}
            onChange={(repeat_penalty) => update({ repeat_penalty })}
          />
          <NumberField
            label={t('settings.generation.frequencyPenalty')}
            value={value.frequency_penalty ?? null}
            step={0.1}
            onChange={(frequency_penalty) => update({ frequency_penalty })}
          />
          <NumberField
            label={t('settings.generation.presencePenalty')}
            value={value.presence_penalty ?? null}
            step={0.1}
            onChange={(presence_penalty) => update({ presence_penalty })}
          />
        </div>
      </PreferenceRow>
      <PreferenceRow
        title={t('settings.generation.reasoning')}
        description={t('settings.generation.reasoningDescription')}
      >
        <div className='flex h-8 items-center justify-end'>
          <Switch
            aria-label={t('settings.generation.enableReasoning')}
            checked={value.reasoning ?? false}
            onCheckedChange={(reasoning) => update({ reasoning })}
          />
        </div>
        {reasoningNote && (
          <p className='mt-1 text-right text-[10px] leading-4 text-muted-foreground'>
            {reasoningNote}
          </p>
        )}
      </PreferenceRow>
      {model.reasoning && (levels.length > 0 || level !== null) && (
        <PreferenceRow
          title={t('settings.generation.reasoningLevel')}
          description={t('settings.generation.reasoningLevelDescription')}
        >
          <Select
            value={level ?? providerDefault}
            items={{
              [providerDefault]: t('model.default'),
              ...Object.fromEntries(levelOptions.map((effort) => [effort, levelLabel(effort)])),
            }}
            disabled={!(value.reasoning ?? false)}
            onValueChange={(next) =>
              update({
                reasoning_effort: next === providerDefault ? null : (next as ReasoningEffort),
              })
            }
          >
            <SelectTrigger
              aria-label={t('settings.generation.reasoningLevel')}
              className='h-8 w-full text-[11px]'
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={providerDefault}>{t('model.default')}</SelectItem>
              {levelOptions.map((effort) => (
                <SelectItem key={effort} value={effort}>
                  {levelLabel(effort)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {levelUnsupported && (
            <p className='mt-1 text-[10px] leading-4 text-destructive'>
              {t('settings.generation.reasoningLevelUnsupported')}
            </p>
          )}
        </PreferenceRow>
      )}
      <PreferenceRow
        title={t('settings.generation.vision')}
        description={t('settings.generation.visionDescription')}
      >
        <div className='flex h-8 items-center justify-end'>
          <Switch
            aria-label={t('settings.generation.vision')}
            checked={value.vision ?? false}
            onCheckedChange={(vision) => update({ vision })}
          />
        </div>
      </PreferenceRow>
    </PreferenceSection>
  )
}
