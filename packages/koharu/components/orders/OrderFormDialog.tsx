'use client'

import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { deadlineToInput, inputToDeadline } from '@/components/orders/format'
import { useCreateOrder, useUpdateOrder } from '@/lib/orders'
import type { NewOrder, OrderListItem, OrderPatch, Priority } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Input } from '@koharu/ui/components/input'
import { Label } from '@koharu/ui/components/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Textarea } from '@koharu/ui/components/textarea'

const priorities: Priority[] = ['normal', 'high', 'urgent']

interface FormState {
  number: string
  title: string
  clientName: string
  sourceLanguage: string
  targetLanguage: string
  assignee: string
  deadline: string
  priority: Priority
  brief: string
}

const emptyForm: FormState = {
  number: '',
  title: '',
  clientName: '',
  sourceLanguage: '',
  targetLanguage: '',
  assignee: '',
  deadline: '',
  priority: 'normal',
  brief: '',
}

function formFromOrder(order: OrderListItem): FormState {
  return {
    number: order.number,
    title: order.title,
    clientName: order.client_name,
    sourceLanguage: order.source_language,
    targetLanguage: order.target_language,
    assignee: order.assignee ?? '',
    deadline: deadlineToInput(order.deadline_at),
    priority: order.priority,
    brief: order.brief,
  }
}

export function OrderFormDialog({
  order,
  onClose,
}: {
  /** Present → edit mode; null → hidden; 'create' → create mode. */
  order: OrderListItem | 'create' | null
  onClose: () => void
}) {
  const { t } = useTranslation()
  const createOrder = useCreateOrder()
  const updateOrder = useUpdateOrder()
  const editing = order !== null && order !== 'create' ? order : null
  const [form, setForm] = useState<FormState>(emptyForm)

  useEffect(() => {
    if (order === 'create') setForm(emptyForm)
    else if (editing) setForm(formFromOrder(editing))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [order])

  const busy = createOrder.isPending || updateOrder.isPending
  const set = (patch: Partial<FormState>) => setForm((value) => ({ ...value, ...patch }))

  const submit = () => {
    const title = form.title.trim()
    const clientName = form.clientName.trim()
    if (!title || !clientName) return
    if (editing) {
      const patch: Partial<OrderPatch> = {}
      if (form.number.trim() !== editing.number) patch.number = form.number.trim()
      if (title !== editing.title) patch.title = title
      if (clientName !== editing.client_name) patch.client_name = clientName
      if (form.sourceLanguage.trim() !== editing.source_language)
        patch.source_language = form.sourceLanguage.trim()
      if (form.targetLanguage.trim() !== editing.target_language)
        patch.target_language = form.targetLanguage.trim()
      if (form.priority !== editing.priority) patch.priority = form.priority
      if ((form.assignee.trim() || null) !== editing.assignee)
        patch.assignee = form.assignee.trim() || null
      const deadline = inputToDeadline(form.deadline)
      if (deadline !== editing.deadline_at) patch.deadline_at = deadline
      if (form.brief !== editing.brief) patch.brief = form.brief
      if (Object.keys(patch).length === 0) {
        onClose()
        return
      }
      updateOrder.mutate({ id: editing.id, patch: patch as OrderPatch }, { onSuccess: onClose })
    } else {
      const input: NewOrder = {
        number: form.number.trim() || null,
        title,
        assignee: form.assignee.trim() || null,
        deadline_at: inputToDeadline(form.deadline),
      }
      if (clientName) input.client_name = clientName
      if (form.sourceLanguage.trim()) input.source_language = form.sourceLanguage.trim()
      if (form.targetLanguage.trim()) input.target_language = form.targetLanguage.trim()
      if (form.priority !== 'normal') input.priority = form.priority
      if (form.brief.trim()) input.brief = form.brief
      createOrder.mutate(input, { onSuccess: onClose })
    }
  }

  const fields = (
    <div className='grid gap-3'>
      <div className='grid grid-cols-2 gap-3'>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-number' className='text-[11px]'>
            {t('orders.form.number')}
          </Label>
          <Input
            id='order-number'
            value={form.number}
            onChange={(event) => set({ number: event.target.value })}
            placeholder={t('orders.form.numberPlaceholder')}
            autoComplete='off'
            className='h-8 text-[12px]'
          />
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-priority' className='text-[11px]'>
            {t('orders.form.priority')}
          </Label>
          <Select
            value={form.priority}
            onValueChange={(value) => set({ priority: value as Priority })}
          >
            <SelectTrigger id='order-priority' className='h-8 w-full text-[12px]'>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {priorities.map((priority) => (
                <SelectItem key={priority} value={priority} className='text-[12px]'>
                  {t(`orders.priority.${priority}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className='grid gap-1.5'>
        <Label htmlFor='order-title' className='text-[11px]'>
          {t('orders.form.title')}
        </Label>
        <Input
          id='order-title'
          value={form.title}
          onChange={(event) => set({ title: event.target.value })}
          autoComplete='off'
          className='h-8 text-[12px]'
        />
      </div>

      <div className='grid grid-cols-2 gap-3'>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-client' className='text-[11px]'>
            {t('orders.form.clientName')}
          </Label>
          <Input
            id='order-client'
            value={form.clientName}
            onChange={(event) => set({ clientName: event.target.value })}
            autoComplete='off'
            className='h-8 text-[12px]'
          />
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-assignee' className='text-[11px]'>
            {t('orders.form.assignee')}
          </Label>
          <Input
            id='order-assignee'
            value={form.assignee}
            onChange={(event) => set({ assignee: event.target.value })}
            autoComplete='off'
            className='h-8 text-[12px]'
          />
        </div>
      </div>

      <div className='grid grid-cols-3 gap-3'>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-source' className='text-[11px]'>
            {t('orders.form.sourceLanguage')}
          </Label>
          <Input
            id='order-source'
            value={form.sourceLanguage}
            onChange={(event) => set({ sourceLanguage: event.target.value })}
            autoComplete='off'
            className='h-8 text-[12px]'
          />
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-target' className='text-[11px]'>
            {t('orders.form.targetLanguage')}
          </Label>
          <Input
            id='order-target'
            value={form.targetLanguage}
            onChange={(event) => set({ targetLanguage: event.target.value })}
            autoComplete='off'
            className='h-8 text-[12px]'
          />
        </div>
        <div className='grid gap-1.5'>
          <Label htmlFor='order-deadline' className='text-[11px]'>
            {t('orders.form.deadline')}
          </Label>
          <Input
            id='order-deadline'
            type='date'
            value={form.deadline}
            onChange={(event) => set({ deadline: event.target.value })}
            className='h-8 text-[12px]'
          />
        </div>
      </div>

      <div className='grid gap-1.5'>
        <Label htmlFor='order-brief' className='text-[11px]'>
          {t('orders.form.brief')}
        </Label>
        <Textarea
          id='order-brief'
          value={form.brief}
          onChange={(event) => set({ brief: event.target.value })}
          rows={4}
          className='text-[12px]'
        />
      </div>
    </div>
  )

  return (
    <Dialog
      open={order !== null}
      onOpenChange={(open) => {
        if (!open && !busy) onClose()
      }}
    >
      <DialogContent className='max-w-lg gap-4 p-5'>
        <DialogHeader className='gap-1'>
          <DialogTitle className='text-[15px]'>
            {editing
              ? t('orders.form.editTitle', { number: editing.number })
              : t('orders.form.createTitle')}
          </DialogTitle>
          <DialogDescription className='text-[11px]'>
            {t('orders.form.description')}
          </DialogDescription>
        </DialogHeader>
        {fields}
        <DialogFooter>
          <Button type='button' variant='ghost' size='sm' onClick={onClose} disabled={busy}>
            {t('common.cancel')}
          </Button>
          <Button
            type='button'
            size='sm'
            onClick={submit}
            disabled={busy || !form.title.trim() || !form.clientName.trim()}
          >
            {editing ? t('orders.form.save') : t('orders.form.create')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
