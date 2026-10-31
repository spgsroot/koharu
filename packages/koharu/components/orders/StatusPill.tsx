'use client'

import { Badge } from '@koharu/ui/components/badge'
import { cn } from '@koharu/ui/lib/utils'

export type PillTone = 'neutral' | 'ok' | 'warn' | 'bad'

const toneClass: Record<PillTone, string> = {
  neutral: '',
  ok: 'border-transparent bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
  warn: 'border-transparent bg-amber-500/10 text-amber-700 dark:text-amber-400',
  bad: '',
}

const toneVariant: Record<PillTone, 'outline' | 'secondary' | 'destructive'> = {
  neutral: 'outline',
  ok: 'secondary',
  warn: 'secondary',
  bad: 'destructive',
}

export function StatusPill({ tone, label }: { tone: PillTone; label: string }) {
  return (
    <Badge variant={toneVariant[tone]} className={cn('font-normal', toneClass[tone])}>
      {label}
    </Badge>
  )
}
