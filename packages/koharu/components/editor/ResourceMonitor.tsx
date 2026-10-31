'use client'

import { Cpu, Gauge, Gpu, MemoryStick } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { useKoharuStore } from '@/lib/store'
import {
  Popover,
  PopoverContent,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from '@koharu/ui/components/popover'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'

/** Shown instead of a number whenever native telemetry could not measure the value. */
const UNKNOWN = '—'

/** Native telemetry for the Koharu process and the accelerator used by models. */
export function ResourceMonitor() {
  const { t } = useTranslation()
  const resources = useKoharuStore((state) => state.resources)
  if (!resources) return null

  // Only the accelerator that runs models is reported. A CPU-only session has no
  // device entry, so no GPU is shown instead of substituting another adapter.
  const device = resources.devices.find((candidate) => candidate.selected) ?? null
  const unavailable = t('resources.unavailable')
  const cpu = percent(resources.process_cpu)
  const ram = percentOf(resources.process_memory, resources.system_memory)
  const ramDetail = t('resources.usedOfTotal', {
    used: bytes(resources.process_memory),
    total: bytes(resources.system_memory),
  })
  // Unmeasured utilization stays null so an idle GPU (0%) remains distinguishable.
  const gpu = device?.utilization == null ? null : percent(device.utilization)
  const vram =
    device?.memory_used != null && device.memory_budget != null
      ? {
          percent: percentOf(device.memory_used, device.memory_budget),
          detail: t('resources.usedOfTotal', {
            used: bytes(device.memory_used),
            total: bytes(device.memory_budget),
          }),
        }
      : null

  return (
    <div className='flex min-w-0 flex-1 justify-end'>
      <Popover>
        <Tooltip>
          <PopoverTrigger
            render={
              <TooltipTrigger
                render={
                  <button
                    type='button'
                    aria-label={
                      device
                        ? t('resources.summaryGpu', { ram, cpu, gpu: gpu ?? unavailable })
                        : t('resources.summary', { ram, cpu })
                    }
                    className='flex min-w-0 items-center gap-1 text-[10px] text-muted-foreground tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring/25'
                  />
                }
              >
                <Metric icon={MemoryStick} value={ram} />
                <Metric icon={Cpu} value={cpu} />
                {device && <Metric icon={Gpu} value={gpu ?? UNKNOWN} />}
              </TooltipTrigger>
            }
          />
          <TooltipContent side='top'>
            <dl className='grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 tabular-nums'>
              <Entry label={t('resources.ram')} value={`${ram} · ${ramDetail}`} />
              <Entry label={t('resources.cpu')} value={cpu} />
              {device && (
                <>
                  <Entry
                    label={t('resources.gpu')}
                    value={`${gpu ?? unavailable} · ${device.name}`}
                  />
                  <Entry
                    label={t('resources.vram')}
                    value={vram ? `${vram.percent} · ${vram.detail}` : unavailable}
                  />
                </>
              )}
            </dl>
          </TooltipContent>
        </Tooltip>
        <PopoverContent
          align='end'
          side='top'
          className='w-72 gap-0 overflow-hidden rounded-xl border-0 bg-[var(--surface-floating)] p-0 shadow-[var(--shadow-panel)] ring-1 ring-foreground/[0.06]'
        >
          <PopoverHeader className='border-b border-border/50 px-3 py-2'>
            <PopoverTitle className='flex items-center gap-2 text-[11px]'>
              <Gauge className='size-3.5 text-primary' />
              {t('resources.title')}
            </PopoverTitle>
          </PopoverHeader>
          <div className='divide-y divide-border/50 px-1.5 py-1 text-[11px]'>
            <ResourceRow
              icon={Cpu}
              label={t('resources.cpu')}
              value={cpu}
              detail={t('resources.processUtilization')}
            />
            <ResourceRow
              icon={MemoryStick}
              label={t('resources.ram')}
              value={ram}
              detail={ramDetail}
            />
            {device && (
              <>
                <ResourceRow
                  icon={Gpu}
                  label={device.name || t('resources.gpu')}
                  value={gpu ?? UNKNOWN}
                  detail={gpu ? t('resources.gpuUtilization') : unavailable}
                />
                <ResourceRow
                  icon={MemoryStick}
                  label={t('resources.vram')}
                  value={vram?.percent ?? UNKNOWN}
                  detail={vram?.detail ?? unavailable}
                />
              </>
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}

/**
 * One compact usage chip. Chips keep a stable width when the sidebar is wide and
 * shrink toward their content, never below it, when the sidebar is narrow.
 */
function Metric({ icon: Icon, value }: { icon: typeof Cpu; value: string }) {
  return (
    <span className='flex h-7 w-14 items-center justify-between rounded-lg bg-primary/[0.07] px-1 hover:bg-primary/[0.1] hover:text-foreground'>
      <Icon className='size-3.5 shrink-0' />
      {value}
    </span>
  )
}

function Entry({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className='opacity-70'>{label}</dt>
      <dd className='min-w-0 wrap-break-word'>{value}</dd>
    </>
  )
}

function ResourceRow({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Cpu
  label: string
  value: string
  detail: string
}) {
  return (
    <div className='grid grid-cols-[16px_minmax(0,1fr)_auto] items-center gap-x-2.5 px-2 py-2'>
      <Icon className='size-3.5 shrink-0 text-muted-foreground' />
      <Tooltip>
        <TooltipTrigger render={<span className='min-w-0' />}>
          <span className='block truncate text-foreground'>{label}</span>
          <span className='block truncate text-[9px] text-muted-foreground'>{detail}</span>
        </TooltipTrigger>
        <TooltipContent side='left'>
          {label}: {detail}
        </TooltipContent>
      </Tooltip>
      <span className='text-right font-medium text-foreground tabular-nums'>{value}</span>
    </div>
  )
}

function percent(value: number): string {
  return `${Math.min(Math.max(0, value), 100).toFixed(0)}%`
}

function percentOf(used: number, total: number): string {
  if (total <= 0) return UNKNOWN
  return percent(Math.min((used / total) * 100, 100))
}

function bytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${Math.round(value / 1024)} KB`
}
