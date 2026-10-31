'use client'

import { ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { useTranslation } from 'react-i18next'

import { PreferencePage } from '@/components/preferences/PreferenceFields'
import { call } from '@/lib/backend'
import {
  commands,
  type TokenTotal,
  type UsagePage,
  type UsageQuery,
  type UsageRecord,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Input } from '@koharu/ui/components/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@koharu/ui/components/table'

const initialFilters = { provider: '', model: '', operation: '', from: '', to: '' }

function periodBoundary(day: string, exclusiveEnd = false): number | null {
  if (!day) return null
  const date = new Date(`${day}T00:00:00`)
  if (exclusiveEnd) date.setDate(date.getDate() + 1)
  return date.getTime()
}

export function UsagePreferences() {
  const { t, i18n } = useTranslation()
  const [draft, setDraft] = useState(initialFilters)
  const [filters, setFilters] = useState(initialFilters)
  const [cursors, setCursors] = useState<Array<number | null>>([null])
  const [pageIndex, setPageIndex] = useState(0)
  const [revision, setRevision] = useState(0)
  const [result, setResult] = useState<UsagePage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<UsageRecord | null>(null)
  const locale = i18n.resolvedLanguage ?? i18n.language
  const numbers = useMemo(() => new Intl.NumberFormat(locale), [locale])
  const amounts = useMemo(
    () => new Intl.NumberFormat(locale, { maximumSignificantDigits: 15 }),
    [locale],
  )
  const timestamps = useMemo(
    () => new Intl.DateTimeFormat(locale, { dateStyle: 'short', timeStyle: 'medium' }),
    [locale],
  )
  const query = useMemo<UsageQuery>(
    () => ({
      provider: filters.provider || null,
      model: filters.model || null,
      operation: (filters.operation || null) as UsageQuery['operation'],
      succeeded: null,
      from: periodBoundary(filters.from),
      to: periodBoundary(filters.to, true),
      cursor: cursors[pageIndex],
      limit: 50,
    }),
    [filters, cursors, pageIndex],
  )

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    void call(commands.getUsageLog, query)
      .then((page) => {
        if (active) setResult(page)
      })
      .catch(() => {
        if (active) {
          setResult(null)
          setError(t('usage.failed'))
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [query, revision, t])

  const resetPage = () => {
    setCursors([null])
    setPageIndex(0)
    setSelected(null)
  }
  const apply = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (draft.from && draft.to && draft.from > draft.to) {
      setError(t('usage.invalidDates'))
      return
    }
    setFilters({ ...draft, model: draft.model.trim() })
    resetPage()
  }
  const count = (value: number | null | undefined) => (value == null ? '—' : numbers.format(value))
  const aggregate = (value: TokenTotal | undefined) =>
    value && value.known > 0 ? count(value.sum) : '—'
  const date = (value: number) => {
    const parsed = new Date(value)
    return Number.isFinite(parsed.getTime()) ? timestamps.format(parsed) : String(value)
  }
  const cost = (value: { amount: number; currency: string } | null) =>
    value ? `${amounts.format(value.amount)} ${value.currency}` : '—'
  const totals = result?.totals
  const summary: Array<[string, string, number | null]> = [
    [t('usage.requests'), count(totals?.requests), null],
    [t('usage.input'), aggregate(totals?.input), totals?.input.known ?? null],
    [t('usage.output'), aggregate(totals?.output), totals?.output.known ?? null],
    [t('usage.total'), aggregate(totals?.total), totals?.total.known ?? null],
    [
      t('usage.fields.reasoning_tokens'),
      aggregate(totals?.reasoning),
      totals?.reasoning.known ?? null,
    ],
    [
      t('usage.fields.cache_read_tokens'),
      aggregate(totals?.cache_read),
      totals?.cache_read.known ?? null,
    ],
    [
      t('usage.fields.cache_write_tokens'),
      aggregate(totals?.cache_write),
      totals?.cache_write.known ?? null,
    ],
    [t('usage.cost'), totals?.costs.length ? totals.costs.map(cost).join(' · ') : '—', null],
  ]

  return (
    <PreferencePage title={t('usage.title')} description={t('usage.description')}>
      <div className='grid gap-4'>
        <p className='text-[11px] leading-5 text-muted-foreground'>{t('usage.privacy')}</p>
        <form onSubmit={apply} className='grid grid-cols-2 gap-3'>
          <label className='grid gap-1 text-[11px] text-muted-foreground'>
            {t('usage.provider')}
            <select
              className='h-9 min-w-0 rounded-md border border-input bg-background px-2 text-[12px] text-foreground'
              value={draft.provider}
              onChange={(event) => setDraft({ ...draft, provider: event.currentTarget.value })}
            >
              <option value=''>—</option>
              {result?.providers.map((provider) => (
                <option key={provider} value={provider}>
                  {provider}
                </option>
              ))}
            </select>
          </label>
          <label className='grid gap-1 text-[11px] text-muted-foreground'>
            {t('usage.model')}
            <Input
              value={draft.model}
              list='usage-models'
              onChange={(event) => setDraft({ ...draft, model: event.currentTarget.value })}
            />
            <datalist id='usage-models'>
              {result?.models
                .filter((item) => !draft.provider || item.provider === draft.provider)
                .map((item) => (
                  <option
                    key={`${item.provider}/${item.model}`}
                    value={item.model}
                    label={item.provider}
                  />
                ))}
            </datalist>
          </label>
          <label className='grid gap-1 text-[11px] text-muted-foreground'>
            {t('usage.operation')}
            <select
              className='h-9 min-w-0 rounded-md border border-input bg-background px-2 text-[12px] text-foreground'
              value={draft.operation}
              onChange={(event) => setDraft({ ...draft, operation: event.currentTarget.value })}
            >
              <option value=''>—</option>
              {['translation', 'agent', 'ocr'].map((operation) => (
                <option key={operation} value={operation}>
                  {operation}
                </option>
              ))}
            </select>
          </label>
          <div className='grid grid-cols-2 gap-2'>
            <label className='grid gap-1 text-[11px] text-muted-foreground'>
              {t('usage.from')}
              <Input
                type='date'
                value={draft.from}
                max={draft.to || undefined}
                onChange={(event) => setDraft({ ...draft, from: event.currentTarget.value })}
              />
            </label>
            <label className='grid gap-1 text-[11px] text-muted-foreground'>
              {t('usage.to')}
              <Input
                type='date'
                value={draft.to}
                min={draft.from || undefined}
                onChange={(event) => setDraft({ ...draft, to: event.currentTarget.value })}
              />
            </label>
          </div>
          <div className='col-span-2 flex flex-wrap gap-2'>
            <Button type='submit' variant='secondary' disabled={loading}>
              {t('usage.apply')}
            </Button>
            <Button
              type='button'
              variant='outline'
              disabled={loading}
              onClick={() => {
                resetPage()
                setRevision((current) => current + 1)
              }}
            >
              <RefreshCw className={loading ? 'animate-spin' : undefined} /> {t('usage.refresh')}
            </Button>
          </div>
        </form>
        {error && (
          <p role='alert' className='text-[12px] text-destructive'>
            {error}
          </p>
        )}
        <dl className='grid grid-cols-2 gap-2 sm:grid-cols-3'>
          {summary.map(([label, value, known]) => (
            <div
              key={label}
              className='min-w-0 rounded-xl border border-border/80 bg-[var(--surface-panel)] p-3'
            >
              <dt className='text-[10px] text-muted-foreground'>{label}</dt>
              <dd className='mt-1 text-[16px] font-semibold break-words tabular-nums'>{value}</dd>
              {known !== null && totals && known < totals.requests && (
                <p className='mt-1 text-[10px] text-muted-foreground'>
                  {t('usage.coverage', { known, total: totals.requests })}
                </p>
              )}
            </div>
          ))}
        </dl>
        {totals && (
          <div className='grid gap-1 text-[10px] text-muted-foreground'>
            <p>{t('usage.missingUsage', { count: totals.requests_without_usage })}</p>
            <p>{t('usage.missingCost', { count: totals.requests_without_cost })}</p>
          </div>
        )}
        <p className='text-[11px] leading-5 text-muted-foreground'>{t('usage.unknownNotice')}</p>
        <div className='overflow-x-auto rounded-xl border border-border/80' aria-busy={loading}>
          <Table className='min-w-[690px] text-[11px]'>
            <TableHeader>
              <TableRow>
                {[
                  t('usage.time'),
                  t('usage.model'),
                  t('usage.operation'),
                  t('usage.input'),
                  t('usage.output'),
                  t('usage.total'),
                  t('usage.cost'),
                  '',
                ].map((label, index) => (
                  <TableHead key={index} className='text-[10px]'>
                    {label}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {!result?.entries.length && (
                <TableRow>
                  <TableCell colSpan={8} className='py-8 text-center text-muted-foreground'>
                    {loading ? t('common.loading') : t('usage.empty')}
                  </TableCell>
                </TableRow>
              )}
              {result?.entries.map((entry) => (
                <TableRow
                  key={entry.id}
                  className={entry.outcome === 'succeeded' ? '' : 'bg-destructive/5'}
                >
                  <TableCell className='whitespace-nowrap'>
                    <span>{date(entry.started_at)}</span>
                    <div className='mt-1 text-[10px] text-muted-foreground'>{entry.outcome}</div>
                  </TableCell>
                  <TableCell className='max-w-[170px] break-words'>
                    {entry.resolved_model ?? entry.requested_model ?? '—'}
                    <div className='mt-1 text-[10px] text-muted-foreground'>{entry.provider}</div>
                  </TableCell>
                  <TableCell>{entry.operation}</TableCell>
                  <TableCell className='tabular-nums'>{count(entry.tokens.input)}</TableCell>
                  <TableCell className='tabular-nums'>{count(entry.tokens.output)}</TableCell>
                  <TableCell className='tabular-nums'>{count(entry.tokens.total)}</TableCell>
                  <TableCell className='whitespace-nowrap tabular-nums'>
                    {cost(entry.cost)}
                  </TableCell>
                  <TableCell>
                    <Button
                      type='button'
                      size='icon-sm'
                      variant='ghost'
                      aria-label={t('usage.detailsFor', { id: entry.id })}
                      onClick={() => setSelected(entry)}
                    >
                      <ChevronRight />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className='flex items-center justify-between gap-2'>
          <Button
            type='button'
            variant='outline'
            disabled={loading || pageIndex === 0}
            onClick={() => setPageIndex((current) => current - 1)}
          >
            <ChevronLeft /> {t('usage.previous')}
          </Button>
          <span className='text-[11px] text-muted-foreground'>
            {t('usage.page', { number: pageIndex + 1 })}
          </span>
          <Button
            type='button'
            variant='outline'
            disabled={loading || !result?.has_more || result.next_cursor === null}
            onClick={() => {
              if (!result || result.next_cursor === null) return
              setCursors((current) => [...current.slice(0, pageIndex + 1), result.next_cursor])
              setPageIndex((current) => current + 1)
            }}
          >
            {t('usage.next')} <ChevronRight />
          </Button>
        </div>
      </div>
      <Dialog
        open={selected !== null}
        onOpenChange={(open) => {
          if (!open) setSelected(null)
        }}
      >
        <DialogContent className='max-h-[85vh] overflow-y-auto sm:max-w-2xl'>
          <DialogHeader>
            <DialogTitle>{t('usage.details')}</DialogTitle>
            <DialogDescription>
              {selected
                ? `${date(selected.started_at)} · ${selected.provider} · ${selected.resolved_model ?? selected.requested_model ?? '—'}`
                : ''}
            </DialogDescription>
          </DialogHeader>
          {selected && (
            <div className='grid gap-5'>
              <DetailSection
                title={t('usage.identity')}
                values={[
                  [t('usage.fields.request_id'), selected.provider_request_id],
                  [t('usage.fields.response_id'), selected.provider_response_id],
                  [t('usage.fields.requested_model'), selected.requested_model],
                  [t('usage.fields.resolved_model'), selected.resolved_model],
                  [t('usage.status'), selected.outcome],
                  [t('usage.fields.http_status'), count(selected.http_status)],
                  [t('usage.fields.attempt'), count(selected.attempt + 1)],
                  [`HTTP · ${t('usage.fields.attempt')}`, count(selected.transport_retry + 1)],
                  [t('usage.fields.error_kind'), selected.error_code],
                  [t('usage.latency'), `${count(selected.latency_ms)} ms`],
                  [t('usage.fields.project_id'), selected.context.project_id],
                  [t('usage.fields.page_id'), selected.context.page_id],
                  [t('usage.fields.job_id'), selected.context.job_id],
                  [t('usage.fields.source_language'), selected.context.source_language],
                  [t('usage.fields.target_language'), selected.context.target_language],
                  [t('usage.fields.segment_count'), count(selected.context.segments)],
                  [t('usage.fields.image_count'), count(selected.context.images)],
                  [t('usage.fields.context_count'), count(selected.context.context_entries)],
                  [t('usage.fields.reasoning_effort'), selected.settings.reasoning_effort],
                ]}
              />
              <DetailSection
                title={t('usage.tokens')}
                values={[
                  [t('usage.input'), count(selected.tokens.input)],
                  [t('usage.output'), count(selected.tokens.output)],
                  [t('usage.total'), count(selected.tokens.total)],
                  [t('usage.fields.reasoning_tokens'), count(selected.tokens.reasoning)],
                  [t('usage.fields.cache_read_tokens'), count(selected.tokens.cache_read)],
                  [t('usage.fields.cache_write_tokens'), count(selected.tokens.cache_write)],
                  [t('usage.fields.image_input_tokens'), count(selected.tokens.input_image)],
                  [t('usage.fields.audio_input_tokens'), count(selected.tokens.input_audio)],
                  [t('usage.fields.audio_output_tokens'), count(selected.tokens.output_audio)],
                  [t('usage.cost'), cost(selected.cost)],
                  [t('usage.fields.cost_source'), selected.cost?.source ?? null],
                ]}
              />
              <section>
                <h3 className='mb-2 text-[12px] font-semibold'>{t('usage.parameters')}</h3>
                <pre className='overflow-x-auto rounded-lg bg-muted/50 p-3 text-[11px] leading-5 break-words whitespace-pre-wrap select-text'>
                  {JSON.stringify(selected.settings, null, 2)}
                </pre>
              </section>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </PreferencePage>
  )
}

function DetailSection({
  title,
  values,
}: {
  title: string
  values: Array<[string, string | null]>
}) {
  return (
    <section>
      <h3 className='mb-2 text-[12px] font-semibold'>{title}</h3>
      <dl className='grid gap-2 text-[11px]'>
        {values.map(([label, value]) => (
          <div key={label} className='grid grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)] gap-3'>
            <dt className='text-muted-foreground'>{label}</dt>
            <dd className='min-w-0 font-mono break-words select-text'>{value ?? '—'}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
