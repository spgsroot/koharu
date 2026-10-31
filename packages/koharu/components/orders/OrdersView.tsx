'use client'

import { ListFilter, Plus, Search } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { OrderDetailSheet } from '@/components/orders/OrderDetailSheet'
import { OrderFormDialog } from '@/components/orders/OrderFormDialog'
import { OrdersTable } from '@/components/orders/OrdersTable'
import { STAGE_ORDER, useOrders, useOrdersSummary, useOrdersUiStore } from '@/lib/orders'
import type { Lifecycle, OrderView, StageName } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Switch } from '@koharu/ui/components/switch'
import { cn } from '@koharu/ui/lib/utils'

const views: OrderView[] = ['attention', 'mine', 'today', 'all']
const lifecycles: Lifecycle[] = ['active', 'on_hold', 'completed', 'delivered', 'cancelled']

const summaryCards = [
  { key: 'overdue', view: 'attention' },
  { key: 'due_today', view: 'today' },
  { key: 'blocked', view: 'all' },
  { key: 'ready', view: 'attention' },
] as const

export default function OrdersView() {
  const { t } = useTranslation()
  const filter = useOrdersUiStore((state) => state.filter)
  const setFilter = useOrdersUiStore((state) => state.setFilter)
  const resetFilter = useOrdersUiStore((state) => state.resetFilter)
  const orders = useOrders()
  const summary = useOrdersSummary()
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [creating, setCreating] = useState(false)

  const activeFilterCount = [
    filter.lifecycle,
    filter.stage,
    filter.assignee,
    filter.client,
    filter.has_blocker,
    filter.import_state,
    filter.deadline_from,
    filter.deadline_to,
  ].filter((value) => value !== null).length

  return (
    <main className='flex min-h-0 flex-1 flex-col overflow-hidden bg-[var(--surface-canvas)]'>
      <ScrollArea className='min-h-0 flex-1'>
        <section className='mx-auto flex w-full max-w-[1240px] flex-col px-6 py-8 sm:px-10'>
          <header className='flex flex-wrap items-center justify-between gap-3'>
            <h1 className='text-[22px] font-semibold tracking-[-0.03em]'>
              {t('orders.workspace.orders')}
            </h1>
            <div className='flex items-center gap-2'>
              <div className='relative'>
                <Search className='absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground' />
                <Input
                  value={filter.search ?? ''}
                  onChange={(event) => setFilter({ search: event.target.value || null })}
                  placeholder={t('orders.header.searchPlaceholder')}
                  aria-label={t('orders.header.searchPlaceholder')}
                  className='h-8 w-56 bg-background pl-8 text-[12px]'
                  autoComplete='off'
                />
              </div>
              <Button
                type='button'
                variant='outline'
                size='sm'
                className='h-8 gap-1.5 text-[11px]'
                aria-pressed={filtersOpen}
                onClick={() => setFiltersOpen((value) => !value)}
              >
                <ListFilter className='size-3.5' />
                {t('orders.header.filters')}
                {activeFilterCount > 0 ? (
                  <span className='rounded-full bg-muted px-1.5 py-px text-[10px] tabular-nums'>
                    {activeFilterCount}
                  </span>
                ) : null}
              </Button>
              <Button
                type='button'
                size='sm'
                className='h-8 gap-1.5 text-[11px]'
                onClick={() => setCreating(true)}
              >
                <Plus className='size-3.5' />
                {t('orders.header.createOrder')}
              </Button>
            </div>
          </header>

          <div className='mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4'>
            {summaryCards.map(({ key, view }) => (
              <button
                key={key}
                type='button'
                className='rounded-xl border border-border/80 bg-[var(--surface-panel)] px-4 py-3 text-left transition-colors hover:border-border'
                onClick={() =>
                  key === 'blocked'
                    ? setFilter({ view, has_blocker: true })
                    : setFilter({ view, has_blocker: null })
                }
              >
                <div className='text-[20px] font-semibold tabular-nums'>
                  {summary.data ? summary.data[key] : '—'}
                </div>
                <div className='mt-0.5 text-[11px] text-muted-foreground'>
                  {t(`orders.summary.${key}`)}
                </div>
              </button>
            ))}
          </div>

          <nav
            className='mt-6 flex items-center gap-1 border-b border-border/70'
            aria-label={t('orders.views.label')}
          >
            {views.map((view) => (
              <button
                key={view}
                type='button'
                className={cn(
                  '-mb-px border-b-2 border-transparent px-3 pt-1 pb-2 text-[12px] text-muted-foreground transition-colors',
                  filter.view === view && 'border-foreground text-foreground',
                )}
                aria-current={filter.view === view ? 'page' : undefined}
                onClick={() => setFilter({ view })}
              >
                {t(`orders.views.${view}`)}
              </button>
            ))}
          </nav>

          {filtersOpen ? (
            <div className='mt-3 grid grid-cols-2 gap-3 rounded-xl border border-border/80 bg-[var(--surface-panel)] p-4 sm:grid-cols-4'>
              <div className='grid gap-1.5'>
                <span className='text-[10px] font-medium text-muted-foreground'>
                  {t('orders.filters.lifecycle')}
                </span>
                <div className='flex flex-wrap gap-1'>
                  {lifecycles.map((lifecycle) => {
                    const active = filter.lifecycle?.includes(lifecycle) ?? false
                    return (
                      <button
                        key={lifecycle}
                        type='button'
                        className={cn(
                          'rounded-full border px-2 py-0.5 text-[10px]',
                          active
                            ? 'border-foreground/30 bg-muted text-foreground'
                            : 'border-border/80 text-muted-foreground hover:border-border',
                        )}
                        aria-pressed={active}
                        onClick={() => {
                          const current = filter.lifecycle ?? []
                          setFilter({
                            lifecycle: active
                              ? current.filter((value) => value !== lifecycle).length > 0
                                ? current.filter((value) => value !== lifecycle)
                                : null
                              : [...current, lifecycle],
                          })
                        }}
                      >
                        {t(`orders.lifecycle.${lifecycle}`)}
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className='grid gap-1.5'>
                <span className='text-[10px] font-medium text-muted-foreground'>
                  {t('orders.filters.stage')}
                </span>
                <Select
                  value={filter.stage ?? ''}
                  onValueChange={(value) =>
                    setFilter({ stage: value === '' ? null : (value as StageName) })
                  }
                >
                  <SelectTrigger className='h-8 w-full text-[12px]'>
                    <SelectValue placeholder={t('orders.filters.anyStage')} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value='' className='text-[12px]'>
                      {t('orders.filters.anyStage')}
                    </SelectItem>
                    {STAGE_ORDER.map((stage) => (
                      <SelectItem key={stage} value={stage} className='text-[12px]'>
                        {t(`orders.stages.${stage}`)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className='grid gap-1.5'>
                <span className='text-[10px] font-medium text-muted-foreground'>
                  {t('orders.filters.assignee')}
                </span>
                <Input
                  value={filter.assignee ?? ''}
                  onChange={(event) => setFilter({ assignee: event.target.value || null })}
                  className='h-8 bg-background text-[12px]'
                  autoComplete='off'
                />
              </div>

              <div className='grid gap-1.5'>
                <span className='text-[10px] font-medium text-muted-foreground'>
                  {t('orders.filters.client')}
                </span>
                <Input
                  value={filter.client ?? ''}
                  onChange={(event) => setFilter({ client: event.target.value || null })}
                  className='h-8 bg-background text-[12px]'
                  autoComplete='off'
                />
              </div>

              <label className='flex items-center gap-2 text-[11px]'>
                <Switch
                  checked={filter.view === 'attention'}
                  onCheckedChange={(checked) => setFilter({ view: checked ? 'attention' : 'all' })}
                />
                {t('orders.filters.needsAttention')}
              </label>

              <div className='flex items-end'>
                <Button
                  type='button'
                  variant='ghost'
                  size='sm'
                  className='h-8 text-[11px] text-muted-foreground'
                  onClick={resetFilter}
                  disabled={activeFilterCount === 0 && filter.view === 'all' && !filter.search}
                >
                  {t('common.reset')}
                </Button>
              </div>
            </div>
          ) : null}

          <div className='mt-4 flex min-h-[320px] flex-col'>
            <OrdersTable items={orders.data ?? []} loading={orders.isLoading} />
          </div>
        </section>
      </ScrollArea>

      <OrderFormDialog order={creating ? 'create' : null} onClose={() => setCreating(false)} />
      <OrderDetailSheet />
    </main>
  )
}
