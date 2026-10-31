'use client'

import { Bot, History, SlidersHorizontal } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { AgentPanel } from '@/components/editor/AgentPanel'
import { HistoryPanel } from '@/components/editor/HistoryPanel'
import { Inspector } from '@/components/editor/Inspector'
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
import { ModelOptions } from '@/components/preferences/PipelinePreferences'
import { TextField } from '@/components/preferences/PreferenceFields'
import { savePreferences } from '@/lib/backend'
import { isCleanupTool, receivePreferences, useKoharuStore } from '@/lib/store'
import type {
  CleanupTarget,
  Flux2Klein9BConfig,
  ManualInpaintingPreset,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'

export function RightSidebar() {
  const { t } = useTranslation()
  const [panel, setPanel] = useState<'agent' | 'history' | 'properties'>('properties')
  const tool = useKoharuStore((state) => state.tool)

  return (
    <aside className='flex h-full min-h-0 flex-col bg-[var(--surface-panel)]'>
      <div className='flex h-10 shrink-0 items-center gap-1 border-b border-border/80 px-2.5'>
        <Button
          variant={panel === 'properties' ? 'secondary' : 'ghost'}
          size='sm'
          className='h-7 flex-1 text-[10px]'
          onClick={() => setPanel('properties')}
        >
          <SlidersHorizontal className='size-3' /> {t('agent.properties')}
        </Button>
        <Button
          variant={panel === 'history' ? 'secondary' : 'ghost'}
          size='sm'
          className='h-7 flex-1 text-[10px]'
          onClick={() => setPanel('history')}
        >
          <History className='size-3' /> {t('history.title')}
        </Button>
        <Button
          variant={panel === 'agent' ? 'secondary' : 'ghost'}
          size='sm'
          className='h-7 flex-1 text-[10px]'
          onClick={() => setPanel('agent')}
        >
          <Bot className='size-3' /> {t('agent.title')}
        </Button>
      </div>
      <div className={panel === 'agent' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <AgentPanel />
      </div>
      <div className={panel === 'history' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <HistoryPanel />
      </div>
      <div className={panel === 'properties' ? 'min-h-0 flex-1' : 'hidden'}>
        {isCleanupTool(tool) ? <RepairSettings tool={tool} /> : <Inspector />}
      </div>
    </aside>
  )
}

function RepairSettings({ tool }: { tool: CleanupTarget }) {
  const { t } = useTranslation()
  const preferences = useKoharuStore((state) => state.preferences)
  const stage: ModelStage =
    tool === 'fast' ? 'bubbles' : tool === 'inpainting' ? 'manual' : 'decensor'
  // Saving presets rewrites the preferences object; only a change to this
  // stage's stored settings may replace an unapplied draft.
  const stored = preferences ? JSON.stringify(stageModel(preferences.pipeline, stage)) : null
  const selected = useMemo(() => (stored ? (JSON.parse(stored) as PipelineModel) : null), [stored])
  const [draft, setDraft] = useState<PipelineModel | null>(selected)
  const [saving, setSaving] = useState(false)
  useEffect(() => setDraft(selected), [selected])

  const apply = () => {
    if (!preferences || !draft || saving) return
    setSaving(true)
    void savePreferences(
      editModel(preferences.pipeline, stage, draft),
      preferences.providers,
      preferences.typesetting,
    )
      .then(receivePreferences)
      .catch(() => undefined)
      .finally(() => setSaving(false))
  }

  const savePresets = (manual_presets: ManualInpaintingPreset[]) => {
    if (!preferences || saving) return
    setSaving(true)
    const pipeline = {
      ...preferences.pipeline,
      inpainting: { ...preferences.pipeline.inpainting, manual_presets },
    }
    void savePreferences(pipeline, preferences.providers, preferences.typesetting)
      .then(receivePreferences)
      .catch(() => undefined)
      .finally(() => setSaving(false))
  }

  if (!preferences || !draft) return null
  const title = t(`tools.${tool}`)
  return (
    <section className='h-full overflow-y-auto p-4' aria-label={title}>
      <h2 className='mb-2 text-sm font-medium'>{title}</h2>
      <p className='mb-4 text-xs leading-5 text-muted-foreground'>
        {t('tools.repairSettingsHint')}
      </p>
      <fieldset disabled={saving} className='grid gap-4'>
        <Select
          value={draft.model}
          items={Object.fromEntries(modelOptions[stage].map((name) => [name, modelNames[name]]))}
          onValueChange={(name) => {
            if (name) {
              setDraft(
                stageModel(selectModel(preferences.pipeline, stage, name as ModelName), stage),
              )
            }
          }}
        >
          <SelectTrigger aria-label={t('settings.pipeline.modelLabel', { stage: title })}>
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
        {stage === 'manual' && (
          <ManualPresets
            builtins={preferences.default_manual_inpainting_presets}
            presets={preferences.pipeline.inpainting.manual_presets ?? []}
            draft={draft}
            onApply={(settings) => setDraft({ model: 'flux2-klein-9b-uncensored', ...settings })}
            onSave={savePresets}
          />
        )}
        <ModelOptions model={draft} onChange={setDraft} />
        <Button type='button' disabled={draft === selected} onClick={apply}>
          {t('tools.applyRepairSettings')}
        </Button>
      </fieldset>
    </section>
  )
}

/**
 * Choosing a preset only fills the draft; "Apply settings" still decides when
 * later strokes use it, so masks already accepted keep their settings.
 */
function ManualPresets({
  builtins,
  presets,
  draft,
  onApply,
  onSave,
}: {
  builtins: ManualInpaintingPreset[]
  presets: ManualInpaintingPreset[]
  draft: PipelineModel
  onApply: (settings: Flux2Klein9BConfig) => void
  onSave: (presets: ManualInpaintingPreset[]) => void
}) {
  const { t } = useTranslation()
  const [selected, setSelected] = useState('')
  const [name, setName] = useState('')
  const trimmed = name.trim()
  const user = selected.startsWith('user:') ? Number(selected.slice('user:'.length)) : null
  const current = user === null ? undefined : presets[user]
  const taken = presets.some((preset, index) => index !== user && preset.name === trimmed)
  const settings = () => {
    const { model: _model, ...rest } = draft
    return rest as Flux2Klein9BConfig
  }

  return (
    <div className='grid gap-2'>
      <Select
        value={selected || null}
        items={Object.fromEntries([
          ...builtins.map((preset, index) => [`builtin:${index}`, preset.name]),
          ...presets.map((preset, index) => [`user:${index}`, preset.name]),
        ])}
        onValueChange={(value) => {
          if (!value) return
          const [source, index] = value.split(':')
          const preset = (source === 'user' ? presets : builtins)[Number(index)]
          if (!preset) return
          setSelected(value)
          setName(source === 'user' ? preset.name : '')
          onApply(preset.settings)
        }}
      >
        <SelectTrigger aria-label={t('repairPresets.label')}>
          <SelectValue placeholder={t('repairPresets.placeholder')} />
        </SelectTrigger>
        <SelectContent>
          {builtins.map((preset, index) => (
            <SelectItem key={`builtin:${index}`} value={`builtin:${index}`}>
              {preset.name}
            </SelectItem>
          ))}
          {presets.map((preset, index) => (
            <SelectItem key={`user:${index}`} value={`user:${index}`}>
              {preset.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <TextField label={t('repairPresets.name')} value={name} onChange={setName} />
      <div className='flex flex-wrap gap-2'>
        <Button
          type='button'
          size='sm'
          variant='secondary'
          disabled={!trimmed || (taken && current?.name !== trimmed)}
          onClick={() => {
            const saved = { name: trimmed, settings: settings() }
            const existing = presets.findIndex((preset) => preset.name === trimmed)
            onSave(
              existing === -1
                ? [...presets, saved]
                : presets.map((preset, index) => (index === existing ? saved : preset)),
            )
            setSelected(`user:${existing === -1 ? presets.length : existing}`)
          }}
        >
          {t('repairPresets.save')}
        </Button>
        {current && (
          <>
            <Button
              type='button'
              size='sm'
              variant='outline'
              disabled={!trimmed || taken || trimmed === current.name}
              onClick={() =>
                onSave(
                  presets.map((preset, index) =>
                    index === user ? { ...preset, name: trimmed } : preset,
                  ),
                )
              }
            >
              {t('repairPresets.rename')}
            </Button>
            <Button
              type='button'
              size='sm'
              variant='ghost'
              className='text-destructive hover:text-destructive'
              onClick={() => {
                setSelected('')
                setName('')
                onSave(presets.filter((_, index) => index !== user))
              }}
            >
              {t('repairPresets.delete')}
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
