'use client'

import { useQuery } from '@tanstack/react-query'
import { FileInput, FileSpreadsheet, FolderOpen, FolderPlus, Pencil } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ImportPreviewDialog } from '@/components/orders/ImportPreviewDialog'
import { OrderFormDialog } from '@/components/orders/OrderFormDialog'
import { OrderTabs } from '@/components/orders/OrderTabs'
import { StageBar } from '@/components/orders/StageBar'
import { call } from '@/lib/backend'
import { exportTranslationTemplate, importProject } from '@/lib/export'
import {
  useAttachOrderProject,
  useBeginTranslationImport,
  useOrder,
  useOrderNotes,
  useOrdersUiStore,
  useUpdateOrder,
} from '@/lib/orders'
import { pageKey, pagesKey, projectKey, refresh } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type OrderPatch, type TemplateFormat } from '@koharu/bridge/protocol'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@koharu/ui/components/alert-dialog'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@koharu/ui/components/sheet'
import { toast } from '@koharu/ui/components/toast'
import { cn } from '@koharu/ui/lib/utils'

export function OrderDetailSheet() {
  const { t } = useTranslation()
  const selectedOrderId = useOrdersUiStore((state) => state.selectedOrderId)
  const setSelectedOrder = useOrdersUiStore((state) => state.setSelectedOrder)
  const setView = useKoharuStore((state) => state.setView)
  const detail = useOrder(selectedOrderId)
  const notes = useOrderNotes(selectedOrderId)
  const updateOrder = useUpdateOrder()
  const beginImport = useBeginTranslationImport()
  const [titleDraft, setTitleDraft] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [attaching, setAttaching] = useState(false)
  const [duplicateOpen, setDuplicateOpen] = useState(false)
  const [previewImportId, setPreviewImportId] = useState<string | null>(null)

  const order = detail.data ?? null

  const close = () => {
    setSelectedOrder(null)
    setTitleDraft(null)
  }

  const saveTitle = () => {
    if (!order || titleDraft === null) return
    const title = titleDraft.trim()
    setTitleDraft(null)
    if (title && title !== order.title)
      updateOrder.mutate({ id: order.id, patch: { title } as OrderPatch })
  }

  const exportTemplate = (format: TemplateFormat) => {
    if (!order) return
    void exportTranslationTemplate({ orderId: order.id, format })
      .then((path) => {
        if (path)
          toast.add({
            type: 'success',
            title: t('orders.actions.exportSaved'),
            description: path,
          })
      })
      .catch(() => undefined)
  }

  const startImport = () => {
    const id = order?.id
    if (!id) return
    beginImport.mutate(id, {
      onSuccess: (result) => {
        if (!result) return
        if (result.duplicate) setDuplicateOpen(true)
        else setPreviewImportId(result.import.id)
      },
    })
  }

  const openEditor = () => {
    if (!order?.project_name) return
    void call(commands.openProject, order.project_name)
      .then(() => refresh(projectKey, pagesKey, pageKey))
      .then(() => setView('start'))
      .catch(() => undefined)
  }

  return (
    <>
      <Sheet
        open={selectedOrderId !== null}
        onOpenChange={(open) => {
          if (!open) close()
        }}
      >
        <SheetContent
          side='right'
          className='w-full gap-0 p-0 sm:max-w-xl'
          aria-describedby={undefined}
        >
          <SheetHeader className='border-b border-border/70 px-5 pt-5 pb-4'>
            {order ? (
              <>
                <div className='flex items-center gap-2 pr-8'>
                  <SheetTitle className='shrink-0 text-[15px]'>{order.number}</SheetTitle>
                  <button
                    type='button'
                    className='group -ml-1 flex min-w-0 flex-1 items-center gap-1.5 rounded-md px-1.5 py-0.5 text-left hover:bg-foreground/[0.05]'
                    onClick={() => setTitleDraft(order.title)}
                    aria-label={t('orders.sheet.editTitle')}
                  >
                    <span className='truncate text-[12px] text-muted-foreground group-hover:text-foreground'>
                      {order.title}
                    </span>
                    <Pencil className='size-3 shrink-0 opacity-0 group-hover:opacity-100' />
                  </button>
                  <Button
                    type='button'
                    variant='ghost'
                    size='sm'
                    className='h-7 shrink-0 text-[11px] text-muted-foreground'
                    onClick={() => setEditing(true)}
                  >
                    {t('orders.sheet.edit')}
                  </Button>
                </div>
                <SheetDescription className='text-[11px]'>{order.client_name}</SheetDescription>
              </>
            ) : (
              <>
                <SheetTitle className='text-[15px]'>{t('orders.sheet.title')}</SheetTitle>
                <SheetDescription>{t('common.loading')}</SheetDescription>
              </>
            )}
          </SheetHeader>

          <ScrollArea className='min-h-0 flex-1'>
            <div className='px-5 py-4'>
              {detail.isLoading || !order ? (
                <p className='py-10 text-center text-[11px] text-muted-foreground'>
                  {t('common.loading')}
                </p>
              ) : (
                <>
                  <StageBar order={order} notes={notes.data ?? []} />

                  <div className='mt-4 flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground'>
                    <span className='rounded-md bg-muted px-2 py-1 tabular-nums'>
                      {t('orders.sheet.pagesChip', {
                        translated: order.pages_translated,
                        total: order.pages_total,
                      })}
                    </span>
                    <span
                      className={cn(
                        'rounded-md px-2 py-1 tabular-nums',
                        order.open_blockers > 0 ? 'bg-destructive/10 text-destructive' : 'bg-muted',
                      )}
                    >
                      {t('orders.sheet.blockersChip', { count: order.open_blockers })}
                    </span>
                    {order.last_import_state ? (
                      <span className='rounded-md bg-muted px-2 py-1'>
                        {t(`orders.importStates.${order.last_import_state}`)}
                      </span>
                    ) : null}
                  </div>

                  <div className='mt-4 flex flex-wrap items-center gap-1.5 border-y border-border/70 py-3'>
                    {order.project_available && order.project_name ? (
                      <Button
                        type='button'
                        size='sm'
                        variant='outline'
                        className='h-7 gap-1.5 text-[11px]'
                        onClick={openEditor}
                      >
                        <FolderOpen className='size-3.5' />
                        {t('orders.actions.openEditor')}
                      </Button>
                    ) : null}
                    {!order.project_name ? (
                      <Button
                        type='button'
                        size='sm'
                        variant='outline'
                        className='h-7 gap-1.5 text-[11px]'
                        onClick={() => setAttaching(true)}
                      >
                        <FolderPlus className='size-3.5' />
                        {t('orders.actions.attachProject')}
                      </Button>
                    ) : null}
                    <Button
                      type='button'
                      size='sm'
                      variant='ghost'
                      className='h-7 gap-1.5 text-[11px] text-muted-foreground'
                      onClick={() => exportTemplate('xlsx')}
                    >
                      <FileSpreadsheet className='size-3.5' />
                      {t('orders.actions.exportXlsx')}
                    </Button>
                    <Button
                      type='button'
                      size='sm'
                      variant='ghost'
                      className='h-7 gap-1.5 text-[11px] text-muted-foreground'
                      onClick={() => exportTemplate('csv')}
                    >
                      {t('orders.actions.exportCsv')}
                    </Button>
                    <Button
                      type='button'
                      size='sm'
                      variant='ghost'
                      className='h-7 gap-1.5 text-[11px] text-muted-foreground'
                      disabled={beginImport.isPending}
                      onClick={startImport}
                    >
                      <FileInput className='size-3.5' />
                      {t('orders.actions.import')}
                    </Button>
                  </div>

                  <OrderTabs order={order} onPreviewImport={setPreviewImportId} />
                </>
              )}
            </div>
          </ScrollArea>

          {order && titleDraft !== null ? (
            <div className='absolute inset-0 z-10 grid place-items-center bg-black/10'>
              <div className='w-80 rounded-2xl border border-border/80 bg-popover p-4 shadow-lg'>
                <label htmlFor='order-title-inline' className='text-[11px] font-medium'>
                  {t('orders.form.title')}
                </label>
                <Input
                  id='order-title-inline'
                  value={titleDraft}
                  onChange={(event) => setTitleDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') saveTitle()
                    if (event.key === 'Escape') setTitleDraft(null)
                  }}
                  autoFocus
                  className='mt-1.5 h-8 text-[12px]'
                />
                <div className='mt-2 flex justify-end gap-1.5'>
                  <Button
                    type='button'
                    variant='ghost'
                    size='sm'
                    className='h-7 text-[11px]'
                    onClick={() => setTitleDraft(null)}
                  >
                    {t('common.cancel')}
                  </Button>
                  <Button
                    type='button'
                    size='sm'
                    className='h-7 text-[11px]'
                    disabled={!titleDraft.trim()}
                    onClick={saveTitle}
                  >
                    {t('orders.form.save')}
                  </Button>
                </div>
              </div>
            </div>
          ) : null}
        </SheetContent>
      </Sheet>

      <OrderFormDialog order={editing && order ? order : null} onClose={() => setEditing(false)} />
      <ImportPreviewDialog importId={previewImportId} onClose={() => setPreviewImportId(null)} />
      <AttachProjectDialog
        orderId={order?.id ?? null}
        open={attaching}
        onClose={() => setAttaching(false)}
      />

      <AlertDialog open={duplicateOpen} onOpenChange={setDuplicateOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('orders.duplicate.title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('orders.duplicate.description')}</AlertDialogDescription>
          </AlertDialogHeader>
          <div className='flex justify-end'>
            <AlertDialogAction onClick={() => setDuplicateOpen(false)}>
              {t('orders.duplicate.ok')}
            </AlertDialogAction>
          </div>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

function AttachProjectDialog({
  orderId,
  open,
  onClose,
}: {
  orderId: string | null
  open: boolean
  onClose: () => void
}) {
  const { t } = useTranslation()
  const attach = useAttachOrderProject()
  const [newName, setNewName] = useState('')
  const projects = useQuery({
    queryKey: ['projects'],
    queryFn: () => call(commands.listProjects),
    enabled: open,
  })
  const [busy, setBusy] = useState(false)

  const attachTo = (name: string) => {
    if (!orderId || busy) return
    setBusy(true)
    void importProject(name)
      .then((projectName) => attach.mutateAsync({ id: orderId, projectName }))
      .then(onClose, () => undefined)
      .finally(() => setBusy(false))
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next && !busy) onClose()
      }}
    >
      <DialogContent className='max-w-md gap-4 p-5'>
        <DialogHeader className='gap-1'>
          <DialogTitle className='text-[15px]'>{t('orders.attach.title')}</DialogTitle>
          <DialogDescription className='text-[11px]'>
            {t('orders.attach.description')}
          </DialogDescription>
        </DialogHeader>

        {projects.data && projects.data.length > 0 ? (
          <ScrollArea className='max-h-56'>
            <ul className='grid gap-0.5'>
              {projects.data.map((project) => (
                <li key={project.name}>
                  <button
                    type='button'
                    className='flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[12px] hover:bg-foreground/[0.045] disabled:opacity-50'
                    disabled={busy}
                    onClick={() => attachTo(project.name)}
                  >
                    <span className='grid size-7 shrink-0 place-items-center rounded-md bg-accent text-accent-foreground'>
                      <FolderOpen className='size-3.5' />
                    </span>
                    <span className='min-w-0 truncate'>{project.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          </ScrollArea>
        ) : (
          <p className='text-[11px] text-muted-foreground'>
            {projects.isLoading ? t('common.loading') : t('orders.attach.noProjects')}
          </p>
        )}

        <div className='flex items-center gap-2 border-t border-border/70 pt-3'>
          <Input
            value={newName}
            onChange={(event) => setNewName(event.target.value)}
            placeholder={t('orders.attach.newPlaceholder')}
            autoComplete='off'
            className='h-8 text-[12px]'
            onKeyDown={(event) => {
              if (event.key === 'Enter' && newName.trim()) attachTo(newName.trim())
            }}
          />
          <Button
            type='button'
            size='sm'
            className='h-8 shrink-0 text-[11px]'
            disabled={!newName.trim() || busy}
            onClick={() => attachTo(newName.trim())}
          >
            {t('orders.attach.createAndAttach')}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
