'use client'

import { CircleAlert, Download, FileText, Square, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { isActiveJob } from '@/lib/jobs'
import { useApplyImport, useCurrentOrder, useDiscardImport, useOrderImports } from '@/lib/orders'
import { usePages } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import {
  commands,
  type Download as DownloadState,
  type ImportMeta,
  type Job,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'

export function ActivityCenter() {
  const { t } = useTranslation()
  const jobs = useKoharuStore((state) => state.jobs)
  const downloads = useKoharuStore((state) => state.downloads)
  const currentOrder = useCurrentOrder()
  const orderImports = useOrderImports(currentOrder?.id ?? null)
  const visibleJobs = Object.values(jobs).filter(
    (job) => isActiveJob(job) || job.state === 'failed',
  )
  const runningDownloads = Object.values(downloads).filter(
    (download) => download.state === 'running',
  )
  const failedDownloads = Object.values(downloads).filter((download) => download.state === 'failed')
  const visibleDownloads = [...runningDownloads, ...failedDownloads]
  const pendingImports = (orderImports.data ?? []).filter((item) => item.state !== 'applied')
  if (visibleJobs.length === 0 && visibleDownloads.length === 0 && pendingImports.length === 0)
    return null

  return (
    <aside className='absolute right-3 bottom-9 z-30 flex w-72 max-w-[calc(100%-1.5rem)] flex-col rounded-2xl border border-border/50 bg-popover shadow-md'>
      <div className='border-b px-3 py-2'>
        <span className='text-[10px] font-semibold tracking-[0.1em] uppercase'>
          {t('activity.title')}
        </span>
      </div>
      {runningDownloads.length > 1 ? (
        <DownloadGroup downloads={runningDownloads} />
      ) : (
        runningDownloads.map((download) => <DownloadItem key={download.id} download={download} />)
      )}
      {failedDownloads.map((download) => (
        <DownloadItem key={download.id} download={download} />
      ))}
      {visibleJobs.map((job) => (
        <JobItem key={job.id} job={job} />
      ))}
      {pendingImports.map((item) => (
        <ImportItem key={item.id} item={item} />
      ))}
    </aside>
  )
}

function ImportItem({ item }: { item: ImportMeta }) {
  const { t } = useTranslation()
  const applyImport = useApplyImport()
  const discardImport = useDiscardImport()
  const busy = applyImport.isPending || discardImport.isPending
  return (
    <div className='border-b p-3 last:border-b-0'>
      <div className='grid grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-x-2.5'>
        <FileText className='size-3.5 justify-self-center text-muted-foreground' />
        <div className='min-w-0'>
          <span className='block truncate text-[12px] font-medium'>{item.filename}</span>
          <p className='mt-0.5 truncate text-[10px] text-muted-foreground tabular-nums'>
            {t('activity.importIssues', { errors: item.errors, warnings: item.warnings })}
          </p>
        </div>
        <div className='flex items-center gap-1'>
          <Button
            size='xs'
            variant='secondary'
            disabled={item.errors > 0 || busy}
            onClick={() => applyImport.mutate(item.id)}
          >
            {t('activity.importApply')}
          </Button>
          <Button
            size='icon-xs'
            variant='ghost'
            aria-label={t('activity.importDiscard')}
            disabled={busy}
            className='text-destructive hover:text-destructive'
            onClick={() => discardImport.mutate({ importId: item.id, orderId: item.order_id })}
          >
            <X />
          </Button>
        </div>
      </div>
    </div>
  )
}

function DownloadGroup({ downloads }: { downloads: DownloadState[] }) {
  const { t } = useTranslation()
  const hasKnownTotal = downloads.every((download) => download.total > 0)
  const percent = hasKnownTotal
    ? progress(
        downloads.reduce((sum, download) => sum + download.completed, 0),
        downloads.reduce((sum, download) => sum + download.total, 0),
      )
    : null

  return (
    <div className='border-b p-3 last:border-b-0'>
      <div className='grid grid-cols-[1rem_minmax(0,1fr)_2.25rem_1.5rem] items-start gap-x-2.5'>
        <Download className='mt-0.5 size-3.5 justify-self-center text-primary' />
        <span className='truncate text-[11px]'>
          {t('activity.downloadingFiles', { count: downloads.length })}
        </span>
        <span className='text-right text-[10px] tabular-nums'>
          {percent !== null ? `${percent}%` : null}
        </span>
        <span aria-hidden='true' />
        <div className='col-start-2 col-end-4'>
          <Progress value={percent} />
        </div>
      </div>
    </div>
  )
}

function JobItem({ job }: { job: Job }) {
  const { t } = useTranslation()
  const dismiss = useKoharuStore((state) => state.dismissJob)
  // The pipeline reports which page it is on; the rail has already loaded the
  // labels, so this resolves from cache rather than fetching.
  const pages = usePages(job.page !== null).data
  // Numbered by position in the rail, so the row says where in the run this
  // is as well as which file: a scan filename alone rarely tells you.
  const pageIndex = job.page ? (pages?.findIndex((page) => page.id === job.page) ?? -1) : -1
  const page = pageIndex >= 0 ? pages?.[pageIndex] : undefined
  const pageLabel = page
    ? t('activity.pageLabel', { number: pageIndex + 1, label: page.label })
    : undefined
  if (job.state === 'failed') {
    return (
      <Failure
        message={job.error || t('activity.processingFailed')}
        onDismiss={() => dismiss(job.id)}
      />
    )
  }
  const percent = progress(job.completed, job.total)
  return (
    <div className='border-b p-3 last:border-b-0'>
      <div className='grid grid-cols-[1rem_minmax(0,1fr)_2.25rem_1.5rem] items-start gap-x-2.5'>
        <span className='mt-1.5 size-1.5 justify-self-center rounded-full bg-primary' />
        <div className='min-w-0'>
          <span className='block truncate text-[12px] font-medium capitalize'>
            {job.state === 'queued'
              ? t('activity.queued')
              : job.stage
                ? t(`phase.${job.stage}`, { defaultValue: job.stage })
                : t('activity.processing')}
          </span>
        </div>
        <span className='pt-0.5 text-right text-[10px] tabular-nums'>
          {percent !== null ? `${percent}%` : null}
        </span>
        <Button
          size='icon-xs'
          variant='ghost'
          className='-mt-1'
          aria-label={t('activity.stop')}
          onClick={() => void call(commands.stopJob, job.id).catch(() => undefined)}
        >
          <Square className='size-2.5 fill-current' />
        </Button>
        {pageLabel ? (
          <p className='col-start-2 col-end-4 mt-0.5 truncate text-[10px] text-muted-foreground'>
            {pageLabel}
          </p>
        ) : null}
        {job.model ? (
          <p className='col-start-2 col-end-4 truncate text-[10px] text-muted-foreground'>
            {job.model}
          </p>
        ) : null}
        <div className='col-start-2 col-end-4'>
          <Progress value={percent} />
        </div>
      </div>
    </div>
  )
}

function DownloadItem({ download }: { download: DownloadState }) {
  const { t } = useTranslation()
  const dismiss = useKoharuStore((state) => state.dismissDownload)
  if (download.state === 'failed') {
    return (
      <Failure
        message={download.error || t('activity.downloadFailed')}
        onDismiss={() => dismiss(download.id)}
      />
    )
  }
  const percent = progress(download.completed, download.total)
  return (
    <div className='border-b p-3 last:border-b-0'>
      <div className='grid grid-cols-[1rem_minmax(0,1fr)_2.25rem_1.5rem] items-start gap-x-2.5'>
        <Download className='mt-0.5 size-3.5 justify-self-center text-primary' />
        <span className='truncate text-[11px]'>{download.name || t('activity.modelDownload')}</span>
        <span className='text-right text-[10px] tabular-nums'>
          {percent !== null ? `${percent}%` : null}
        </span>
        <span aria-hidden='true' />
        <div className='col-start-2 col-end-4'>
          <Progress value={percent} />
        </div>
      </div>
    </div>
  )
}

function Failure({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  const { t } = useTranslation()
  return (
    <div className='grid grid-cols-[1rem_minmax(0,1fr)_2.25rem_1.5rem] items-start gap-x-2.5 border-b p-3 text-[11px] last:border-b-0'>
      <CircleAlert className='mt-0.5 size-3.5 justify-self-center text-destructive' />
      <span className='col-start-2 col-end-4 min-w-0 text-destructive'>{message}</span>
      <Button
        size='icon-xs'
        variant='ghost'
        className='-mt-1'
        aria-label={t('activity.dismiss')}
        onClick={onDismiss}
      >
        <X />
      </Button>
    </div>
  )
}

function Progress({ value }: { value: number | null }) {
  return (
    <div
      role='progressbar'
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value ?? undefined}
      className='mt-2 h-1 overflow-hidden rounded-full bg-muted'
    >
      <div
        className={`h-full rounded-full bg-primary ${value === null ? 'w-1/2' : ''}`}
        style={value === null ? undefined : { width: `${value}%` }}
      />
    </div>
  )
}

function progress(completed: number, total: number): number | null {
  return total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : null
}
