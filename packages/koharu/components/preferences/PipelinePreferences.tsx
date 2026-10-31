'use client'

import { Brush, Eraser, FileText, Search, Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import {
  editModel,
  modelNames,
  modelOptions,
  selectModel,
  stageModel,
  type ModelName,
  type ModelStage,
  type PipelineModel,
} from '@/components/preferences/models'
import {
  NumberField,
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
  TextField,
} from '@/components/preferences/PreferenceFields'
import type { PipelineConfig } from '@koharu/bridge/protocol'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Switch } from '@koharu/ui/components/switch'

const stages = [
  ['detection', Search],
  ['ocr', FileText],
  ['bubbles', Eraser],
  ['artwork', Brush],
  ['manual', Sparkles],
  ['decensor', Sparkles],
] as const satisfies ReadonlyArray<readonly [ModelStage, typeof Search]>

export function PipelinePreferences({
  value,
  onChange,
}: {
  value: PipelineConfig
  onChange: (value: PipelineConfig) => void
}) {
  const { t } = useTranslation()
  return (
    <PreferencePage
      title={t('settings.pipeline.title')}
      description={t('settings.pipeline.description')}
    >
      <PreferenceSection title={t('settings.pipeline.processing')}>
        {stages.map(([stage, Icon]) => {
          const model = stageModel(value, stage)
          const title = t(`settings.pipeline.stages.${stage}.title`)
          return (
            <PreferenceRow
              key={stage}
              title={title}
              description={t(`settings.pipeline.stages.${stage}.description`)}
              align='start'
            >
              <div className='grid gap-3'>
                <div className='flex items-center gap-2'>
                  <Icon className='size-3.5 shrink-0 text-muted-foreground' />
                  <Select
                    value={model.model}
                    items={Object.fromEntries(
                      modelOptions[stage].map((name) => [name, modelNames[name]]),
                    )}
                    onValueChange={(name) => {
                      if (name) {
                        onChange(selectModel(value, stage, name as ModelName))
                      }
                    }}
                  >
                    <SelectTrigger
                      aria-label={t('settings.pipeline.modelLabel', { stage: title })}
                      className='h-8 min-w-0 flex-1 text-[11px]'
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {modelOptions[stage].map((name) => (
                        <SelectItem key={name} value={name}>
                          {modelNames[name]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <ModelOptions
                  model={model}
                  onChange={(next) => onChange(editModel(value, stage, next))}
                />
              </div>
            </PreferenceRow>
          )
        })}
        <PreferenceRow
          title={t('settings.pipeline.censorDetection.title')}
          description={t('settings.pipeline.censorDetection.description')}
          align='start'
        >
          <CensorDetectionOptions
            config={value.censor_detection}
            onChange={(censor_detection) => onChange({ ...value, censor_detection })}
          />
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  )
}

function CensorDetectionOptions({
  config,
  onChange,
}: {
  config: PipelineConfig['censor_detection']
  onChange: (config: PipelineConfig['censor_detection']) => void
}) {
  const { t } = useTranslation()
  return (
    <div className='grid gap-2'>
      <div className='grid grid-cols-2 gap-2'>
        <label className='flex h-8 items-center justify-between gap-3 text-[10px] text-muted-foreground'>
          {t('settings.pipeline.options.automaticCensorMask')}
          <Switch
            checked={config.enabled ?? true}
            onCheckedChange={(enabled) => onChange({ ...config, enabled })}
          />
        </label>
        <label className='flex h-8 items-center justify-between gap-3 text-[10px] text-muted-foreground'>
          {t('settings.pipeline.options.detectMosaic')}
          <Switch
            checked={config.detect_mosaic ?? true}
            onCheckedChange={(detect_mosaic) => onChange({ ...config, detect_mosaic })}
          />
        </label>
      </div>
      <div className='grid grid-cols-3 gap-2'>
        <NumberField
          label={t('settings.pipeline.options.censorThreshold')}
          value={config.threshold ?? 0.5}
          min={0}
          max={1}
          step={0.05}
          onChange={(threshold) => onChange({ ...config, threshold: threshold ?? undefined })}
        />
        <NumberField
          label={t('settings.pipeline.options.censorMaskGrow')}
          value={config.mask_grow ?? 12}
          min={0}
          max={255}
          step={1}
          onChange={(mask_grow) => onChange({ ...config, mask_grow: mask_grow ?? undefined })}
        />
        <NumberField
          label={t('settings.pipeline.options.censorMaskBlur')}
          value={config.mask_blur ?? 4}
          min={0}
          max={64}
          step={0.5}
          onChange={(mask_blur) => onChange({ ...config, mask_blur: mask_blur ?? undefined })}
        />
      </div>
    </div>
  )
}

export function ModelOptions({
  model,
  onChange,
}: {
  model: PipelineModel
  onChange: (model: PipelineModel) => void
}) {
  const { t } = useTranslation()
  switch (model.model) {
    case 'koharu-layout-rfdetr-seg-2xl':
      return (
        <div className='grid grid-cols-3 gap-2'>
          <NumberField
            label={t('settings.pipeline.options.textThreshold')}
            value={model.text_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(text_threshold) => onChange({ ...model, text_threshold })}
          />
          <NumberField
            label={t('settings.pipeline.options.bubbleThreshold')}
            value={model.bubble_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(bubble_threshold) => onChange({ ...model, bubble_threshold })}
          />
          <NumberField
            label={t('settings.pipeline.options.panelThreshold')}
            value={model.panel_threshold ?? null}
            min={0}
            max={1}
            step={0.05}
            onChange={(panel_threshold) => onChange({ ...model, panel_threshold })}
          />
        </div>
      )
    case 'flux2-klein':
      return (
        <TextField
          label={t('settings.pipeline.options.prompt')}
          value={model.prompt ?? ''}
          onChange={(prompt) => onChange({ ...model, prompt })}
        />
      )
    case 'flux2-klein-9b-uncensored':
      return (
        <div className='grid gap-2'>
          <p className='text-[10px] leading-4 text-muted-foreground'>
            {t(`settings.pipeline.profiles.${model.model}`)}
          </p>
          <TextField
            label={t('settings.pipeline.options.prompt')}
            value={model.prompt ?? ''}
            onChange={(prompt) => onChange({ ...model, prompt })}
          />
          <div className='grid grid-cols-2 gap-2'>
            <NumberField
              label={t('settings.pipeline.options.inferenceSteps')}
              value={model.num_inference_steps ?? 8}
              min={1}
              max={100}
              step={1}
              onChange={(num_inference_steps) =>
                onChange({ ...model, num_inference_steps: num_inference_steps ?? undefined })
              }
            />
            <NumberField
              label={t('settings.pipeline.options.guidanceScale')}
              value={model.guidance_scale ?? 1}
              min={0.1}
              max={20}
              step={0.1}
              onChange={(guidance_scale) =>
                onChange({ ...model, guidance_scale: guidance_scale ?? undefined })
              }
            />
          </div>
          <div className='grid grid-cols-2 gap-2'>
            <NumberField
              label={t('settings.pipeline.options.seed')}
              value={model.seed ?? -1}
              min={-1}
              max={Number.MAX_SAFE_INTEGER}
              step={1}
              onChange={(seed) => onChange({ ...model, seed: seed ?? -1 })}
            />
            <NumberField
              label={t('settings.pipeline.options.denoise')}
              value={model.strength ?? 1}
              min={0.01}
              max={1}
              step={0.05}
              onChange={(strength) => onChange({ ...model, strength: strength ?? 1 })}
            />
          </div>
          <TextField
            label={t('settings.pipeline.options.loraPath')}
            value={model.lora_path ?? ''}
            onChange={(lora_path) => onChange({ ...model, lora_path: lora_path || null })}
          />
          <NumberField
            label={t('settings.pipeline.options.loraStrength')}
            value={model.lora_strength ?? 1}
            min={0}
            step={0.1}
            onChange={(lora_strength) =>
              onChange({ ...model, lora_strength: lora_strength ?? undefined })
            }
          />
        </div>
      )
    case 'diffusioncreampy-animagine-xl-4':
    case 'diffusioncreampy-illustrious-xl-2':
      return (
        <div className='grid gap-2'>
          <p className='text-[10px] leading-4 text-muted-foreground'>
            {t(`settings.pipeline.profiles.${model.model}`)}
          </p>
          <div className='grid grid-cols-2 gap-2'>
            <TextField
              label={t('settings.pipeline.options.endpoint')}
              value={model.endpoint ?? 'http://127.0.0.1:8188'}
              onChange={(endpoint) => onChange({ ...model, endpoint })}
            />
            <TextField
              label={t('settings.pipeline.options.checkpoint')}
              value={model.checkpoint ?? ''}
              onChange={(checkpoint) => onChange({ ...model, checkpoint })}
            />
          </div>
          <TextField
            label={t('settings.pipeline.options.controlnet')}
            value={model.controlnet ?? 'controlnet-union-sdxl-1.0.safetensors'}
            onChange={(controlnet) => onChange({ ...model, controlnet })}
          />
          <div className='grid grid-cols-2 gap-2'>
            <TextField
              label={t('settings.pipeline.options.positivePrompt')}
              value={model.positive_prompt ?? ''}
              onChange={(positive_prompt) => onChange({ ...model, positive_prompt })}
            />
            <TextField
              label={t('settings.pipeline.options.negativePrompt')}
              value={model.negative_prompt ?? ''}
              onChange={(negative_prompt) => onChange({ ...model, negative_prompt })}
            />
          </div>
          <div className='grid grid-cols-3 gap-2'>
            <NumberField
              label={t('settings.pipeline.options.seed')}
              value={model.seed ?? 424242}
              min={0}
              max={4294967295}
              step={1}
              onChange={(seed) => onChange({ ...model, seed: seed ?? undefined })}
            />
            <NumberField
              label={t('settings.pipeline.options.inferenceSteps')}
              value={model.num_inference_steps ?? 10}
              min={1}
              max={100}
              step={1}
              onChange={(num_inference_steps) =>
                onChange({ ...model, num_inference_steps: num_inference_steps ?? undefined })
              }
            />
            <NumberField
              label={t('settings.pipeline.options.guidanceScale')}
              value={model.guidance_scale ?? 7}
              min={0.1}
              max={20}
              step={0.1}
              onChange={(guidance_scale) =>
                onChange({ ...model, guidance_scale: guidance_scale ?? undefined })
              }
            />
            <NumberField
              label={t('settings.pipeline.options.denoise')}
              value={model.denoise ?? 0.5}
              min={0}
              max={1}
              step={0.05}
              onChange={(denoise) => onChange({ ...model, denoise: denoise ?? undefined })}
            />
            <NumberField
              label={t('settings.pipeline.options.maskPadding')}
              value={model.mask_padding ?? 128}
              min={0}
              max={512}
              step={8}
              onChange={(mask_padding) =>
                onChange({ ...model, mask_padding: mask_padding ?? undefined })
              }
            />
            <NumberField
              label={t('settings.pipeline.options.maskGrow')}
              value={model.mask_grow ?? 0}
              min={0}
              max={128}
              step={1}
              onChange={(mask_grow) => onChange({ ...model, mask_grow: mask_grow ?? undefined })}
            />
            <NumberField
              label={t('settings.pipeline.options.maskBlur')}
              value={model.mask_blur ?? 16}
              min={0}
              max={128}
              step={1}
              onChange={(mask_blur) => onChange({ ...model, mask_blur: mask_blur ?? undefined })}
            />
            <NumberField
              label={t('settings.pipeline.options.colorMatchStrength')}
              value={model.color_match_strength ?? 1}
              min={0}
              max={1}
              step={0.05}
              onChange={(color_match_strength) =>
                onChange({
                  ...model,
                  color_match_strength: color_match_strength ?? undefined,
                })
              }
            />
            <NumberField
              label={t('settings.pipeline.options.timeoutSeconds')}
              value={model.timeout_seconds ?? 900}
              min={30}
              max={3600}
              step={30}
              onChange={(timeout_seconds) =>
                onChange({ ...model, timeout_seconds: timeout_seconds ?? undefined })
              }
            />
          </div>
        </div>
      )
    case 'rorem-mixed':
      return (
        <div className='grid grid-cols-2 gap-2'>
          <TextField
            label={t('settings.pipeline.options.prompt')}
            value={model.prompt ?? ''}
            onChange={(prompt) => onChange({ ...model, prompt })}
          />
          <TextField
            label={t('settings.pipeline.options.negativePrompt')}
            value={model.negative_prompt ?? ''}
            onChange={(negative_prompt) => onChange({ ...model, negative_prompt })}
          />
        </div>
      )
    case 'paddleocr-vl-1.6':
    case 'manga-ocr':
    case 'baberu-ocr':
    case 'hayai-ocr':
    case 'lama':
    case 'aot-inpainting':
      return null
  }
}
