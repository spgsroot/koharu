'use client'

import { useTranslation } from 'react-i18next'

import { formatDateTime } from '@/components/orders/format'
import { StatusPill } from '@/components/orders/StatusPill'
import { useApplyImport, useDiscardImport, useTranslationImport } from '@/lib/orders'
import type { ImportMessage } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import { Separator } from '@koharu/ui/components/separator'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@koharu/ui/components/table'
import { cn } from '@koharu/ui/lib/utils'

const importStateTone = {
  ready: 'ok',
  needs_attention: 'warn',
  failed: 'bad',
  applied: 'neutral',
} as const

function messageLine(message: ImportMessage, key: string) {
  return (
    <li key={key} className='text-[11px] leading-4'>
      <span className='font-medium'>{message.kind}</span>
      <span className='text-muted-foreground'> — {message.detail}</span>
    </li>
  )
}

export function ImportPreviewDialog({
  importId,
  onClose,
}: {
  importId: string | null
  onClose: () => void
}) {
  const { t } = useTranslation()
  const detail = useTranslationImport(importId)
  const applyImport = useApplyImport()
  const discardImport = useDiscardImport()
  const busy = applyImport.isPending || discardImport.isPending

  const item = detail.data ?? null
  const report = item?.report ?? null
  const errors: { key: string; message: ImportMessage }[] = []
  const warnings: { key: string; message: ImportMessage }[] = []
  if (report) {
    report.file_messages.forEach((message, index) =>
      (message.severity === 'error' ? errors : warnings).push({ key: `f${index}`, message }),
    )
    report.rows.forEach((row) =>
      row.messages.forEach((message, index) =>
        (message.severity === 'error' ? errors : warnings).push({
          key: `r${row.row}m${index}`,
          message: {
            ...message,
            detail: `${t('orders.import.row', { row: row.row })}: ${message.detail}`,
          },
        }),
      ),
    )
  }
  const pending = item !== null && (item.state === 'ready' || item.state === 'needs_attention')
  const canApply = pending && item.errors === 0

  const close = () => {
    if (!busy) onClose()
  }

  return (
    <Dialog open={importId !== null} onOpenChange={(open) => (!open ? close() : undefined)}>
      <DialogContent className='flex max-h-[85vh] max-w-3xl flex-col gap-4 p-5'>
        <DialogHeader className='gap-1'>
          <DialogTitle className='text-[15px]'>{t('orders.import.title')}</DialogTitle>
          <DialogDescription className='text-[11px]'>
            {item ? (
              <>
                {item.filename} · {formatDateTime(item.created_at)} · {item.actor}
              </>
            ) : (
              t('common.loading')
            )}
          </DialogDescription>
        </DialogHeader>

        {item && report ? (
          <ScrollArea className='-mx-1 min-h-0 flex-1 px-1'>
            <div className='flex flex-wrap items-center gap-1.5'>
              <StatusPill
                tone={importStateTone[item.state]}
                label={t(`orders.importStates.${item.state}`)}
              />
              <StatusPill
                tone={item.errors > 0 ? 'bad' : 'neutral'}
                label={t('orders.import.errorsCount', { count: item.errors })}
              />
              <StatusPill
                tone={item.warnings > 0 ? 'warn' : 'neutral'}
                label={t('orders.import.warningsCount', { count: item.warnings })}
              />
              <StatusPill
                tone='neutral'
                label={t('orders.import.acceptedCount', {
                  accepted: item.accepted,
                  total: item.total,
                })}
              />
            </div>

            {errors.length > 0 ? (
              <section className='mt-4 rounded-lg bg-destructive/10 p-3' aria-live='polite'>
                <h3 className='text-[11px] font-medium text-destructive'>
                  {t('orders.import.errorsHeading', { count: errors.length })}
                </h3>
                <ul className='mt-1.5 grid gap-1'>
                  {errors.map(({ key, message }) => messageLine(message, key))}
                </ul>
              </section>
            ) : null}
            {warnings.length > 0 ? (
              <section className='mt-3 rounded-lg bg-amber-500/10 p-3'>
                <h3 className='text-[11px] font-medium text-amber-700 dark:text-amber-400'>
                  {t('orders.import.warningsHeading', { count: warnings.length })}
                </h3>
                <ul className='mt-1.5 grid gap-1'>
                  {warnings.map(({ key, message }) => messageLine(message, key))}
                </ul>
              </section>
            ) : null}

            <Separator className='my-4 bg-border/70' />

            <Table>
              <TableHeader>
                <TableRow className='border-border/70 hover:bg-transparent'>
                  <TableHead className='w-16 text-[11px]'>{t('orders.import.page')}</TableHead>
                  <TableHead className='w-28 text-[11px]'>{t('orders.import.segment')}</TableHead>
                  <TableHead className='text-[11px]'>{t('orders.import.source')}</TableHead>
                  <TableHead className='text-[11px]'>{t('orders.import.translation')}</TableHead>
                  <TableHead className='w-24 text-[11px]'>{t('orders.import.change')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {report.rows.map((row) => (
                  <TableRow key={row.row} className='border-border/60'>
                    <TableCell className='py-2 align-top text-[11px] tabular-nums'>
                      {row.page}
                    </TableCell>
                    <TableCell
                      className='max-w-28 truncate py-2 align-top font-mono text-[10px] text-muted-foreground'
                      title={row.segment_id}
                    >
                      {row.segment_id}
                    </TableCell>
                    <TableCell className='max-w-56 py-2 align-top'>
                      <p className='line-clamp-3 text-[11px] whitespace-pre-line text-muted-foreground'>
                        {row.source_text}
                      </p>
                    </TableCell>
                    <TableCell className='max-w-56 py-2 align-top'>
                      <p className='line-clamp-3 text-[11px] whitespace-pre-line'>
                        {row.translation}
                      </p>
                    </TableCell>
                    <TableCell className='py-2 align-top'>
                      <StatusPill
                        tone={
                          row.change === 'changed'
                            ? 'warn'
                            : row.change === 'new'
                              ? 'ok'
                              : 'neutral'
                        }
                        label={t(`orders.importChanges.${row.change}`)}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        ) : (
          <div className='grid min-h-32 place-items-center text-[11px] text-muted-foreground'>
            {t('common.loading')}
          </div>
        )}

        <DialogFooter className={cn('items-center')}>
          {pending && !canApply ? (
            <p className='mr-auto text-[10px] text-destructive'>{t('orders.import.applyHint')}</p>
          ) : null}
          {item && pending ? (
            <>
              <Button
                type='button'
                variant='ghost'
                size='sm'
                disabled={busy}
                onClick={() =>
                  discardImport.mutate(
                    { importId: item.id, orderId: item.order_id },
                    { onSuccess: onClose },
                  )
                }
              >
                {t('orders.import.discard')}
              </Button>
              <Button
                type='button'
                size='sm'
                disabled={busy || !canApply}
                onClick={() => applyImport.mutate(item.id, { onSuccess: onClose })}
              >
                {t('orders.import.apply')}
              </Button>
            </>
          ) : (
            <Button type='button' size='sm' variant='ghost' onClick={onClose}>
              {t('orders.import.close')}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
