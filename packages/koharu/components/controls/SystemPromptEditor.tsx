'use client'

import { useTranslation } from 'react-i18next'

import type { ContentProfile, DefaultSystemPrompts, SystemPrompts } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Textarea } from '@koharu/ui/components/textarea'
import { cn } from '@koharu/ui/lib/utils'

export const contentProfiles = [
  'hentai',
  'manga',
  'comic',
] as const satisfies readonly ContentProfile[]

/**
 * Selects the content profile of editor runs and edits that profile's saved
 * system prompt. Every profile keeps its own prompt; an empty prompt uses the
 * built-in template shown as the placeholder.
 */
export function SystemPromptEditor({
  profile,
  prompts,
  defaults,
  disabled = false,
  compact = false,
  onChange,
}: {
  profile: ContentProfile
  prompts: SystemPrompts
  defaults: DefaultSystemPrompts | null
  disabled?: boolean
  compact?: boolean
  onChange: (profile: ContentProfile, prompts: SystemPrompts) => void
}) {
  const { t } = useTranslation()
  const prompt = prompts[profile] ?? ''
  const edit = (next: string) => onChange(profile, { ...prompts, [profile]: next || null })

  return (
    <div className='grid gap-1.5'>
      <Select
        value={profile}
        items={Object.fromEntries(
          contentProfiles.map((item) => [item, t(`systemPrompt.profiles.${item}`)]),
        )}
        disabled={disabled}
        onValueChange={(next) => next && onChange(next as ContentProfile, prompts)}
      >
        <SelectTrigger
          aria-label={t('systemPrompt.profile')}
          className={cn('w-full text-[11px]', compact ? 'h-7' : 'h-8')}
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {contentProfiles.map((item) => (
            <SelectItem key={item} value={item}>
              {t(`systemPrompt.profiles.${item}`)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Textarea
        aria-label={t('systemPrompt.label')}
        value={prompt}
        disabled={disabled}
        placeholder={defaults?.[profile]}
        className={
          compact
            ? 'max-h-28 min-h-20 resize-none overflow-y-auto text-[11px] leading-4'
            : 'field-sizing-fixed min-h-32 resize-y overflow-y-auto text-[12px] leading-5'
        }
        onChange={(event) => edit(event.currentTarget.value)}
      />
      <div className='flex items-start justify-between gap-2'>
        <span className='text-[9px] leading-3 text-muted-foreground'>{t('systemPrompt.hint')}</span>
        {!prompt && defaults && (
          <Button
            type='button'
            variant='ghost'
            size='xs'
            disabled={disabled}
            className='shrink-0 text-[10px]'
            onClick={() => edit(defaults[profile])}
          >
            {t('systemPrompt.editBuiltIn')}
          </Button>
        )}
      </div>
    </div>
  )
}
