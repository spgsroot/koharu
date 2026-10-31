'use client'

import { Check } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { STAGE_ORDER, useAddOrderNote, useSetOrderStage } from '@/lib/orders'
import type { OrderDetail, OrderNote, StageInfo, StageState } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Textarea } from '@koharu/ui/components/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'
import { cn } from '@koharu/ui/lib/utils'

const segmentClass: Record<StageState, string> = {
  done: 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-400',
  in_progress: 'bg-muted ring-1 ring-foreground/20 ring-inset',
  blocked: 'bg-destructive/15 text-destructive',
  not_started: 'bg-muted text-muted-foreground',
}

export function StageBar({ order, notes }: { order: OrderDetail; notes: OrderNote[] }) {
  const { t } = useTranslation()
  const setStage = useSetOrderStage()
  const addNote = useAddOrderNote()
  const [blocking, setBlocking] = useState(false)
  const [blockBody, setBlockBody] = useState('')

  const stages: StageInfo[] = STAGE_ORDER.map(
    (name) =>
      order.stages.find((info) => info.stage === name) ?? {
        stage: name,
        state: 'not_started',
        started_at: null,
        completed_at: null,
      },
  )
  const currentIndex = stages.findIndex((info) => info.state !== 'done')
  const current = currentIndex >= 0 ? stages[currentIndex] : null
  const openBlockers = notes.filter((note) => note.kind === 'blocker' && !note.resolved_at)
  const locked = order.open_blockers > 0
  const busy = setStage.isPending || addNote.isPending

  const apply = (changes: [StageInfo['stage'], StageState][]) =>
    setStage.mutate({ id: order.id, changes })

  const blockCurrent = () => {
    const body = blockBody.trim()
    if (!body) return
    addNote.mutate(
      {
        orderId: order.id,
        blockStage: true,
        input: {
          kind: 'blocker',
          body,
          author: '',
          page_id: null,
          segment_id: null,
          pinned: false,
        },
      },
      {
        onSuccess: () => {
          setBlocking(false)
          setBlockBody('')
        },
      },
    )
  }

  return (
    <section aria-label={t('orders.stages.title')}>
      <ol className='grid grid-cols-7 gap-1'>
        {stages.map((info) => {
          const blockers =
            info.state === 'blocked' ? openBlockers.map((note) => note.body).filter(Boolean) : []
          const label = `${t(`orders.stages.${info.stage}`)}: ${t(`orders.stageStates.${info.state}`)}`
          const segment = (
            <div
              className={cn(
                'flex h-7 items-center justify-center gap-1 rounded-md text-[10px] select-none',
                segmentClass[info.state],
              )}
            >
              {info.state === 'done' ? <Check className='size-3' aria-hidden /> : null}
              <span className='truncate px-1'>{t(`orders.stages.${info.stage}`)}</span>
            </div>
          )
          return (
            <li
              key={info.stage}
              aria-current={info === current ? 'step' : undefined}
              aria-label={label}
            >
              {blockers.length > 0 ? (
                <Tooltip>
                  <TooltipTrigger render={<button type='button' className='w-full' />}>
                    {segment}
                  </TooltipTrigger>
                  <TooltipContent side='bottom' className='max-w-64'>
                    <ul className='grid gap-1'>
                      {blockers.map((body) => (
                        <li key={body} className='text-[10px] leading-4'>
                          {body}
                        </li>
                      ))}
                    </ul>
                  </TooltipContent>
                </Tooltip>
              ) : (
                segment
              )}
            </li>
          )
        })}
      </ol>

      {current ? (
        <div className='mt-3'>
          {locked ? (
            <p className='text-[10px] text-muted-foreground'>{t('orders.stages.lockedHint')}</p>
          ) : null}
          <div className='mt-1.5 flex flex-wrap items-center gap-1.5'>
            {current.state === 'blocked' ? (
              <Button
                type='button'
                size='sm'
                variant='outline'
                className='h-7 text-[11px]'
                disabled={busy || locked}
                onClick={() => apply([[current.stage, 'in_progress']])}
              >
                {t('orders.stages.unblock')}
              </Button>
            ) : (
              <>
                <Button
                  type='button'
                  size='sm'
                  variant='outline'
                  className='h-7 text-[11px]'
                  disabled={busy || locked || blocking}
                  onClick={() => setBlocking(true)}
                >
                  {t('orders.stages.block')}
                </Button>
                <Button
                  type='button'
                  size='sm'
                  variant='outline'
                  className='h-7 text-[11px]'
                  disabled={busy || locked}
                  onClick={() => apply([[current.stage, 'done']])}
                >
                  {t('orders.stages.complete')}
                </Button>
                {currentIndex < STAGE_ORDER.length - 1 ? (
                  <Button
                    type='button'
                    size='sm'
                    variant='outline'
                    className='h-7 text-[11px]'
                    disabled={busy || locked}
                    onClick={() =>
                      apply([
                        [current.stage, 'done'],
                        [STAGE_ORDER[currentIndex + 1], 'in_progress'],
                      ])
                    }
                  >
                    {t('orders.stages.next')}
                  </Button>
                ) : null}
                {currentIndex > STAGE_ORDER.indexOf('editing') ? (
                  <Button
                    type='button'
                    size='sm'
                    variant='ghost'
                    className='h-7 text-[11px] text-muted-foreground'
                    disabled={busy || locked}
                    onClick={() =>
                      apply([
                        ['editing', 'in_progress'],
                        ...STAGE_ORDER.slice(STAGE_ORDER.indexOf('editing') + 1).map(
                          (name) => [name, 'not_started'] as [typeof name, StageState],
                        ),
                      ])
                    }
                  >
                    {t('orders.stages.returnToEditing')}
                  </Button>
                ) : null}
              </>
            )}
          </div>

          {blocking ? (
            <div className='mt-2 rounded-lg border border-border/70 bg-muted/40 p-2'>
              <Textarea
                value={blockBody}
                onChange={(event) => setBlockBody(event.target.value)}
                placeholder={t('orders.stages.blockPlaceholder')}
                rows={2}
                autoFocus
                className='text-[11px]'
              />
              <div className='mt-1.5 flex justify-end gap-1.5'>
                <Button
                  type='button'
                  size='sm'
                  variant='ghost'
                  className='h-7 text-[11px]'
                  onClick={() => setBlocking(false)}
                  disabled={addNote.isPending}
                >
                  {t('common.cancel')}
                </Button>
                <Button
                  type='button'
                  size='sm'
                  className='h-7 text-[11px]'
                  onClick={blockCurrent}
                  disabled={!blockBody.trim() || addNote.isPending}
                >
                  {t('orders.stages.blockSubmit')}
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <p className='mt-3 text-[10px] text-muted-foreground'>{t('orders.stages.allDone')}</p>
      )}
    </section>
  )
}
