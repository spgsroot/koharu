'use client'

import { Check, RotateCcw, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { deadlineToInput, formatDateTime, inputToDeadline } from '@/components/orders/format'
import { StatusPill } from '@/components/orders/StatusPill'
import { exportImportReport } from '@/lib/export'
import {
  useAddOrderNote,
  useApplyImport,
  useDeleteOrderNote,
  useDiscardImport,
  useOrderActivity,
  useOrderImports,
  useOrderNotes,
  useResolveOrderNote,
  useSetOrderLifecycle,
  useUpdateOrder,
} from '@/lib/orders'
import type {
  ImportMeta,
  Lifecycle,
  NoteKind,
  OrderDetail,
  OrderPatch,
  Priority,
} from '@koharu/bridge/protocol'
import { Badge } from '@koharu/ui/components/badge'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'
import { Label } from '@koharu/ui/components/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@koharu/ui/components/tabs'
import { Textarea } from '@koharu/ui/components/textarea'
import { toast } from '@koharu/ui/components/toast'
import { cn } from '@koharu/ui/lib/utils'

const lifecycles: Lifecycle[] = ['active', 'on_hold', 'completed', 'delivered', 'cancelled']
const priorities: Priority[] = ['normal', 'high', 'urgent']
const noteKinds: NoteKind[] = ['internal', 'client', 'blocker', 'qa']
const destructiveLifecycle: Lifecycle[] = ['completed', 'delivered', 'cancelled']

export function OrderTabs({
  order,
  onPreviewImport,
}: {
  order: OrderDetail
  onPreviewImport: (importId: string) => void
}) {
  const { t } = useTranslation()
  return (
    <Tabs defaultValue='main' className='mt-4 min-h-0 flex-1'>
      <TabsList className='h-8'>
        <TabsTrigger value='main' className='text-[11px]'>
          {t('orders.tabs.main')}
        </TabsTrigger>
        <TabsTrigger value='notes' className='text-[11px]'>
          {t('orders.tabs.notes')}
        </TabsTrigger>
        <TabsTrigger value='activity' className='text-[11px]'>
          {t('orders.tabs.activity')}
        </TabsTrigger>
        <TabsTrigger value='imports' className='text-[11px]'>
          {t('orders.tabs.imports')}
        </TabsTrigger>
      </TabsList>
      <TabsContent value='main' className='pt-3'>
        <MainTab order={order} />
      </TabsContent>
      <TabsContent value='notes' className='pt-3'>
        <NotesTab order={order} />
      </TabsContent>
      <TabsContent value='activity' className='pt-3'>
        <ActivityTab order={order} />
      </TabsContent>
      <TabsContent value='imports' className='pt-3'>
        <ImportsTab order={order} onPreview={onPreviewImport} />
      </TabsContent>
    </Tabs>
  )
}

function MainTab({ order }: { order: OrderDetail }) {
  const { t } = useTranslation()
  const updateOrder = useUpdateOrder()
  const setLifecycle = useSetOrderLifecycle()
  const [clientName, setClientName] = useState(order.client_name)
  const [assignee, setAssignee] = useState(order.assignee ?? '')
  const [deadline, setDeadline] = useState(deadlineToInput(order.deadline_at))
  const [priority, setPriority] = useState<Priority>(order.priority)
  const [brief, setBrief] = useState(order.brief)

  useEffect(() => {
    setClientName(order.client_name)
    setAssignee(order.assignee ?? '')
    setDeadline(deadlineToInput(order.deadline_at))
    setPriority(order.priority)
    setBrief(order.brief)
  }, [order])

  const deadlineAt = inputToDeadline(deadline)
  const dirty =
    clientName.trim() !== order.client_name ||
    (assignee.trim() || null) !== order.assignee ||
    deadlineAt !== order.deadline_at ||
    priority !== order.priority ||
    brief !== order.brief

  const save = () => {
    const patch: Partial<OrderPatch> = {}
    if (clientName.trim() !== order.client_name) patch.client_name = clientName.trim()
    if ((assignee.trim() || null) !== order.assignee) patch.assignee = assignee.trim() || null
    if (deadlineAt !== order.deadline_at) patch.deadline_at = deadlineAt
    if (priority !== order.priority) patch.priority = priority
    if (brief !== order.brief) patch.brief = brief
    if (Object.keys(patch).length === 0) return
    updateOrder.mutate({ id: order.id, patch: patch as OrderPatch })
  }

  return (
    <div className='grid gap-3'>
      <div className='grid grid-cols-2 gap-3'>
        <div className='grid gap-1.5'>
          <Label htmlFor='tab-client' className='text-[11px]'>
            {t('orders.form.clientName')}
          </Label>
          <Input
            id='tab-client'
            value={clientName}
            onChange={(event) => setClientName(event.target.value)}
            className='h-8 text-[12px]'
            autoComplete='off'
          />
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='tab-assignee' className='text-[11px]'>
            {t('orders.form.assignee')}
          </Label>
          <Input
            id='tab-assignee'
            value={assignee}
            onChange={(event) => setAssignee(event.target.value)}
            className='h-8 text-[12px]'
            autoComplete='off'
          />
        </div>
      </div>

      <div className='grid grid-cols-3 gap-3'>
        <div className='grid gap-1.5'>
          <Label htmlFor='tab-deadline' className='text-[11px]'>
            {t('orders.form.deadline')}
          </Label>
          <Input
            id='tab-deadline'
            type='date'
            value={deadline}
            onChange={(event) => setDeadline(event.target.value)}
            className='h-8 text-[12px]'
          />
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='tab-priority' className='text-[11px]'>
            {t('orders.form.priority')}
          </Label>
          <Select value={priority} onValueChange={(value) => setPriority(value as Priority)}>
            <SelectTrigger id='tab-priority' className='h-8 w-full text-[12px]'>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {priorities.map((value) => (
                <SelectItem key={value} value={value} className='text-[12px]'>
                  {t(`orders.priority.${value}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='tab-lifecycle' className='text-[11px]'>
            {t('orders.form.lifecycle')}
          </Label>
          <Select
            value={order.lifecycle}
            onValueChange={(value) => {
              const lifecycle = value as Lifecycle
              if (lifecycle === order.lifecycle) return
              if (
                destructiveLifecycle.includes(lifecycle) &&
                !window.confirm(
                  t('orders.lifecycle.confirm', { status: t(`orders.lifecycle.${lifecycle}`) }),
                )
              )
                return
              setLifecycle.mutate({ id: order.id, lifecycle })
            }}
          >
            <SelectTrigger id='tab-lifecycle' className='h-8 w-full text-[12px]'>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {lifecycles.map((value) => (
                <SelectItem key={value} value={value} className='text-[12px]'>
                  {t(`orders.lifecycle.${value}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className='grid gap-1.5'>
        <Label htmlFor='tab-brief' className='text-[11px]'>
          {t('orders.form.brief')}
        </Label>
        <Textarea
          id='tab-brief'
          value={brief}
          onChange={(event) => setBrief(event.target.value)}
          rows={5}
          className='text-[12px]'
        />
      </div>

      <div className='flex justify-end'>
        <Button
          type='button'
          size='sm'
          className='h-8 text-[11px]'
          onClick={save}
          disabled={!dirty || updateOrder.isPending}
        >
          {t('orders.form.save')}
        </Button>
      </div>
    </div>
  )
}

function NotesTab({ order }: { order: OrderDetail }) {
  const { t } = useTranslation()
  const notes = useOrderNotes(order.id)
  const addNote = useAddOrderNote()
  const resolveNote = useResolveOrderNote()
  const deleteNote = useDeleteOrderNote()
  const [kind, setKind] = useState<NoteKind>('internal')
  const [body, setBody] = useState('')

  const sorted = [...(notes.data ?? [])].sort((a, b) => {
    const aHot = a.kind === 'blocker' && !a.resolved_at ? 1 : 0
    const bHot = b.kind === 'blocker' && !b.resolved_at ? 1 : 0
    if (aHot !== bHot) return bHot - aHot
    return b.created_at - a.created_at
  })

  const submit = () => {
    const text = body.trim()
    if (!text) return
    addNote.mutate(
      {
        orderId: order.id,
        blockStage: kind === 'blocker',
        input: { kind, body: text, author: '', page_id: null, segment_id: null, pinned: false },
      },
      { onSuccess: () => setBody('') },
    )
  }

  return (
    <div className='grid gap-3'>
      <div className='rounded-lg border border-border/70 p-2.5'>
        <div className='flex items-center gap-2'>
          <Select value={kind} onValueChange={(value) => setKind(value as NoteKind)}>
            <SelectTrigger className='h-8 w-40 text-[12px]' aria-label={t('orders.notes.kind')}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {noteKinds.map((value) => (
                <SelectItem key={value} value={value} className='text-[12px]'>
                  {t(`orders.noteKinds.${value}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Textarea
          value={body}
          onChange={(event) => setBody(event.target.value)}
          placeholder={t('orders.notes.placeholder')}
          rows={3}
          className='mt-2 text-[12px]'
        />
        <div className='mt-2 flex justify-end'>
          <Button
            type='button'
            size='sm'
            className='h-8 text-[11px]'
            onClick={submit}
            disabled={!body.trim() || addNote.isPending}
          >
            {t('orders.notes.add')}
          </Button>
        </div>
      </div>

      {notes.isLoading ? (
        <p className='py-6 text-center text-[11px] text-muted-foreground'>{t('common.loading')}</p>
      ) : sorted.length === 0 ? (
        <p className='py-6 text-center text-[11px] text-muted-foreground'>
          {t('orders.notes.empty')}
        </p>
      ) : (
        <ul className='grid gap-2'>
          {sorted.map((note) => (
            <li
              key={note.id}
              className={cn(
                'rounded-lg border p-2.5',
                note.kind === 'blocker' && !note.resolved_at
                  ? 'border-destructive/30 bg-destructive/5'
                  : 'border-border/70',
              )}
            >
              <div className='flex items-center gap-2'>
                <Badge
                  variant={note.kind === 'blocker' ? 'destructive' : 'outline'}
                  className='text-[10px] font-normal'
                >
                  {t(`orders.noteKinds.${note.kind}`)}
                </Badge>
                <span className='text-[10px] text-muted-foreground'>{note.author}</span>
                <span className='text-[10px] text-muted-foreground tabular-nums'>
                  {formatDateTime(note.created_at)}
                </span>
                {note.resolved_at ? (
                  <span className='text-[10px] text-emerald-700 dark:text-emerald-400'>
                    {t('orders.notes.resolved', { by: note.resolved_by ?? '—' })}
                  </span>
                ) : null}
              </div>
              <p className='mt-1.5 text-[12px] leading-5 whitespace-pre-line'>{note.body}</p>
              {note.kind === 'blocker' ? (
                <div className='mt-1.5 flex gap-1'>
                  <Button
                    type='button'
                    variant='ghost'
                    size='sm'
                    className='h-7 gap-1 text-[10px] text-muted-foreground'
                    disabled={resolveNote.isPending}
                    onClick={() =>
                      resolveNote.mutate({ noteId: note.id, resolved: !note.resolved_at })
                    }
                  >
                    {note.resolved_at ? (
                      <RotateCcw className='size-3' />
                    ) : (
                      <Check className='size-3' />
                    )}
                    {note.resolved_at ? t('orders.notes.reopen') : t('orders.notes.resolve')}
                  </Button>
                  <Button
                    type='button'
                    variant='ghost'
                    size='sm'
                    className='h-7 gap-1 text-[10px] text-muted-foreground hover:text-destructive'
                    disabled={deleteNote.isPending}
                    onClick={() => {
                      if (window.confirm(t('orders.notes.deleteConfirm')))
                        deleteNote.mutate({ noteId: note.id, orderId: order.id })
                    }}
                  >
                    <Trash2 className='size-3' />
                    {t('orders.notes.delete')}
                  </Button>
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ActivityTab({ order }: { order: OrderDetail }) {
  const { t } = useTranslation()
  const activity = useOrderActivity(order.id)

  return activity.isLoading ? (
    <p className='py-6 text-center text-[11px] text-muted-foreground'>{t('common.loading')}</p>
  ) : !activity.data || activity.data.length === 0 ? (
    <p className='py-6 text-center text-[11px] text-muted-foreground'>
      {t('orders.activity.empty')}
    </p>
  ) : (
    <ul className='grid gap-2'>
      {activity.data.map((entry) => {
        const key = `orders.activity.${entry.kind}`
        const label = t(key, { defaultValue: entry.kind })
        return (
          <li key={entry.id} className='rounded-lg border border-border/70 p-2.5'>
            <div className='flex items-baseline justify-between gap-2'>
              <span className='text-[11px] font-medium'>{label}</span>
              <span className='shrink-0 text-[10px] text-muted-foreground tabular-nums'>
                {entry.actor} · {formatDateTime(entry.created_at)}
              </span>
            </div>
            {Object.keys(entry.payload).length > 0 ? (
              <dl className='mt-1 grid gap-0.5'>
                {Object.entries(entry.payload).map(([name, value]) => (
                  <div key={name} className='flex gap-1.5 text-[10px]'>
                    <dt className='text-muted-foreground'>{name}:</dt>
                    <dd className='min-w-0 truncate'>{value}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </li>
        )
      })}
    </ul>
  )
}

const importStateTone = {
  ready: 'ok',
  needs_attention: 'warn',
  failed: 'bad',
  applied: 'neutral',
} as const

function ImportRow({
  item,
  onPreview,
}: {
  item: ImportMeta
  onPreview: (importId: string) => void
}) {
  const { t } = useTranslation()
  const applyImport = useApplyImport()
  const discardImport = useDiscardImport()
  const pending = item.state === 'ready' || item.state === 'needs_attention'
  const busy = applyImport.isPending || discardImport.isPending
  return (
    <div
      role='button'
      tabIndex={0}
      className='grid w-full cursor-pointer gap-1 rounded-lg border border-border/70 p-2.5 text-left hover:bg-foreground/[0.03]'
      onClick={() => onPreview(item.id)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' && event.target === event.currentTarget) onPreview(item.id)
      }}
    >
      <span className='flex items-center justify-between gap-2'>
        <span className='truncate text-[11px] font-medium'>{item.filename}</span>
        <StatusPill
          tone={importStateTone[item.state]}
          label={t(`orders.importStates.${item.state}`)}
        />
      </span>
      <span className='text-[10px] text-muted-foreground tabular-nums'>
        {formatDateTime(item.created_at)} · {item.actor} ·{' '}
        {t('orders.imports.counts', {
          accepted: item.accepted,
          errors: item.errors,
          warnings: item.warnings,
        })}
      </span>
      <span className='flex items-center gap-1' onClick={(event) => event.stopPropagation()}>
        {pending ? (
          <>
            <Button
              type='button'
              variant='ghost'
              size='sm'
              className='h-7 text-[10px]'
              disabled={busy || item.errors > 0}
              onClick={() => applyImport.mutate(item.id)}
            >
              {t('orders.import.apply')}
            </Button>
            <Button
              type='button'
              variant='ghost'
              size='sm'
              className='h-7 text-[10px] text-muted-foreground'
              disabled={busy}
              onClick={() => discardImport.mutate({ importId: item.id, orderId: item.order_id })}
            >
              {t('orders.import.discard')}
            </Button>
          </>
        ) : null}
        <ReportButton importId={item.id} />
      </span>
    </div>
  )
}

function ReportButton({ importId }: { importId: string }) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  return (
    <Button
      type='button'
      variant='ghost'
      size='sm'
      className='h-7 text-[10px] text-muted-foreground'
      disabled={busy}
      onClick={() => {
        setBusy(true)
        void exportImportReport(importId)
          .then((path) => {
            if (path)
              toast.add({
                type: 'success',
                title: t('orders.imports.reportSaved'),
                description: path,
              })
          })
          .catch(() => undefined)
          .finally(() => setBusy(false))
      }}
    >
      {t('orders.imports.report')}
    </Button>
  )
}

function ImportsTab({
  order,
  onPreview,
}: {
  order: OrderDetail
  onPreview: (importId: string) => void
}) {
  const { t } = useTranslation()
  const imports = useOrderImports(order.id)

  return imports.isLoading ? (
    <p className='py-6 text-center text-[11px] text-muted-foreground'>{t('common.loading')}</p>
  ) : !imports.data || imports.data.length === 0 ? (
    <p className='py-6 text-center text-[11px] text-muted-foreground'>
      {t('orders.imports.empty')}
    </p>
  ) : (
    <ul className='grid gap-2'>
      {imports.data.map((item) => (
        <li key={item.id}>
          <ImportRow item={item} onPreview={onPreview} />
        </li>
      ))}
    </ul>
  )
}
