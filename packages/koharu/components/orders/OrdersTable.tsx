'use client'

import { Flag, RotateCcw } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { formatDay } from '@/components/orders/format'
import { StatusPill, type PillTone } from '@/components/orders/StatusPill'
import { useAddOrderNote, useOrdersUiStore } from '@/lib/orders'
import type { OrderListItem } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@koharu/ui/components/table'
import { Textarea } from '@koharu/ui/components/textarea'
import { cn } from '@koharu/ui/lib/utils'

const lifecycleTone: Record<OrderListItem['lifecycle'], PillTone> = {
  active: 'neutral',
  on_hold: 'warn',
  completed: 'ok',
  delivered: 'ok',
  cancelled: 'neutral',
}

function slaOf(item: OrderListItem): 'none' | 'ok' | 'risk' | 'breached' {
  if (item.deadline_at === null) return 'none'
  if (item.attention.includes('overdue')) return 'breached'
  if (item.attention.includes('due_soon')) return 'risk'
  return 'ok'
}

export function OrdersTable({ items, loading }: { items: OrderListItem[]; loading: boolean }) {
  const { t } = useTranslation()
  const setSelectedOrder = useOrdersUiStore((state) => state.setSelectedOrder)
  const addNote = useAddOrderNote()
  const [nearestFirst, setNearestFirst] = useState(false)
  const [blockNoteOrder, setBlockNoteOrder] = useState<OrderListItem | null>(null)
  const [blockNoteBody, setBlockNoteBody] = useState('')

  const rows = useMemo(() => {
    if (!nearestFirst) return items
    return [...items].sort((a, b) => {
      if (a.deadline_at === null) return 1
      if (b.deadline_at === null) return -1
      return a.deadline_at - b.deadline_at
    })
  }, [items, nearestFirst])

  const submitBlockNote = () => {
    const order = blockNoteOrder
    const body = blockNoteBody.trim()
    if (!order || !body) return
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
          setBlockNoteOrder(null)
          setBlockNoteBody('')
        },
      },
    )
  }

  return (
    <section className='flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border/80 bg-[var(--surface-panel)]'>
      <div className='min-h-0 flex-1 overflow-auto'>
        <Table>
          <TableHeader className='sticky top-0 bg-[var(--surface-panel)]'>
            <TableRow className='border-border/70 hover:bg-transparent'>
              <TableHead className='px-4 text-[11px]'>{t('orders.table.order')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.assignee')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.deadline')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.progressTitle')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.sla')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.stage')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.status')}</TableHead>
              <TableHead className='text-[11px]'>{t('orders.table.attention')}</TableHead>
              <TableHead className='pr-4 text-right text-[11px]'>
                {t('orders.table.actions')}
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow className='hover:bg-transparent'>
                <TableCell colSpan={9} className='px-4 py-10 text-center'>
                  <span className='text-[11px] text-muted-foreground'>{t('common.loading')}</span>
                </TableCell>
              </TableRow>
            ) : rows.length === 0 ? (
              <TableRow className='hover:bg-transparent'>
                <TableCell colSpan={9} className='px-4 py-10 text-center'>
                  <span className='text-[11px] text-muted-foreground'>
                    {t('orders.table.empty')}
                  </span>
                </TableCell>
              </TableRow>
            ) : (
              rows.map((item) => {
                const sla = slaOf(item)
                return (
                  <TableRow
                    key={item.id}
                    className='cursor-pointer border-border/60'
                    onClick={() => setSelectedOrder(item.id)}
                  >
                    <TableCell className='px-4 py-2.5 align-top'>
                      <div className='text-[12px] font-medium'>{item.number}</div>
                      <div className='mt-0.5 max-w-[220px] truncate text-[10px] text-muted-foreground'>
                        {item.client_name}
                        {item.title ? ` · ${item.title}` : ''}
                      </div>
                    </TableCell>
                    <TableCell className='py-2.5 align-top text-[11px]'>
                      {item.assignee ?? '—'}
                    </TableCell>
                    <TableCell
                      className={cn(
                        'py-2.5 align-top text-[11px] tabular-nums',
                        sla === 'breached' && 'text-destructive',
                      )}
                    >
                      {formatDay(item.deadline_at) ?? '—'}
                    </TableCell>
                    <TableCell className='py-2.5 align-top'>
                      {item.pages_total > 0 ? (
                        <div className='min-w-[110px]'>
                          <div className='text-[10px] text-muted-foreground tabular-nums'>
                            {t('orders.table.progress', {
                              translated: item.pages_translated,
                              total: item.pages_total,
                            })}
                          </div>
                          <div
                            className='mt-1 h-1 w-full overflow-hidden rounded-full bg-muted'
                            role='progressbar'
                            aria-valuenow={item.pages_translated}
                            aria-valuemax={item.pages_total}
                            aria-label={t('orders.table.progressLabel')}
                          >
                            <div
                              className='h-full rounded-full bg-primary/60'
                              style={{
                                width: `${Math.min(100, Math.round((item.pages_translated / item.pages_total) * 100))}%`,
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <span className='text-[11px] text-muted-foreground'>
                          {t('orders.table.noPages')}
                        </span>
                      )}
                    </TableCell>
                    <TableCell className='py-2.5 align-top'>
                      {sla === 'none' ? (
                        <span className='text-[11px] text-muted-foreground'>—</span>
                      ) : (
                        <StatusPill
                          tone={sla === 'breached' ? 'bad' : sla === 'risk' ? 'warn' : 'ok'}
                          label={t(`orders.sla.${sla}`)}
                        />
                      )}
                    </TableCell>
                    <TableCell className='py-2.5 align-top'>
                      <div className='text-[11px]'>{t(`orders.stages.${item.current_stage}`)}</div>
                      <div className='mt-0.5 text-[10px] text-muted-foreground'>
                        {t(`orders.stageStates.${item.current_stage_state}`)}
                      </div>
                    </TableCell>
                    <TableCell className='py-2.5 align-top'>
                      <StatusPill
                        tone={lifecycleTone[item.lifecycle]}
                        label={t(`orders.lifecycle.${item.lifecycle}`)}
                      />
                    </TableCell>
                    <TableCell className='max-w-[180px] py-2.5 align-top'>
                      {item.attention.length === 0 ? (
                        <span className='text-[11px] text-muted-foreground'>—</span>
                      ) : (
                        <div className='flex flex-wrap gap-1'>
                          {item.attention.map((reason) => (
                            <StatusPill
                              key={reason}
                              tone={
                                reason === 'overdue' ||
                                reason === 'blocked' ||
                                reason === 'import_failed' ||
                                reason === 'pipeline_failed'
                                  ? 'bad'
                                  : reason === 'ready_to_deliver'
                                    ? 'ok'
                                    : 'warn'
                              }
                              label={t(`orders.attention.${reason}`)}
                            />
                          ))}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className='py-2 pr-4 align-top'>
                      <div
                        className='flex items-center justify-end gap-1'
                        onClick={(event) => event.stopPropagation()}
                      >
                        <Button
                          type='button'
                          variant='ghost'
                          size='icon-sm'
                          className='size-7 text-muted-foreground hover:text-foreground'
                          aria-label={t('orders.table.addBlocker', { number: item.number })}
                          title={t('orders.table.addBlockerHint')}
                          onClick={() => setBlockNoteOrder(item)}
                        >
                          <Flag className='size-3.5' />
                        </Button>
                        <Button
                          type='button'
                          variant='ghost'
                          size='sm'
                          className='h-7 text-[11px]'
                          onClick={() => setSelectedOrder(item.id)}
                        >
                          {t('orders.table.open')}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })
            )}
          </TableBody>
        </Table>
      </div>

      <footer className='flex h-10 shrink-0 items-center justify-between border-t border-border/70 px-4'>
        <span className='text-[10px] text-muted-foreground tabular-nums'>
          {t('orders.table.total', { count: rows.length })}
        </span>
        <Button
          type='button'
          variant='ghost'
          size='sm'
          className={cn('h-7 gap-1.5 text-[10px]', nearestFirst && 'text-foreground')}
          aria-pressed={nearestFirst}
          onClick={() => setNearestFirst((value) => !value)}
        >
          <RotateCcw className={cn('size-3', !nearestFirst && 'text-muted-foreground')} />
          {nearestFirst ? t('orders.table.sortNearestOn') : t('orders.table.sortNearestOff')}
        </Button>
      </footer>

      <Dialog
        open={blockNoteOrder !== null}
        onOpenChange={(open) => {
          if (!open && !addNote.isPending) setBlockNoteOrder(null)
        }}
      >
        <DialogContent className='max-w-md gap-4 p-5'>
          <DialogHeader className='gap-1'>
            <DialogTitle className='text-[15px]'>{t('orders.blockerDialog.title')}</DialogTitle>
            <DialogDescription className='text-[11px]'>
              {t('orders.blockerDialog.description', { number: blockNoteOrder?.number })}
            </DialogDescription>
          </DialogHeader>
          <Textarea
            value={blockNoteBody}
            onChange={(event) => setBlockNoteBody(event.target.value)}
            placeholder={t('orders.blockerDialog.placeholder')}
            rows={4}
            autoFocus
            className='text-[12px]'
          />
          <DialogFooter>
            <Button
              type='button'
              variant='ghost'
              size='sm'
              onClick={() => setBlockNoteOrder(null)}
              disabled={addNote.isPending}
            >
              {t('common.cancel')}
            </Button>
            <Button
              type='button'
              size='sm'
              onClick={submitBlockNote}
              disabled={!blockNoteBody.trim() || addNote.isPending}
            >
              {t('orders.blockerDialog.submit')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}
