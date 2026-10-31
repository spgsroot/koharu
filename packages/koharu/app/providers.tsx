'use client'

import { QueryClientProvider } from '@tanstack/react-query'
import { Channel } from '@tauri-apps/api/core'
import { useEffect, useRef, type ReactNode } from 'react'
import { I18nextProvider } from 'react-i18next'

import { StartupView } from '@/components/app/StartupView'
import ClientOnly from '@/components/ClientOnly'
import { refreshTranslationModels } from '@/lib/backend'
import i18n from '@/lib/i18n'
import { isActiveJob } from '@/lib/jobs'
import { pageKey, pagesKey, projectKey, queryClient, refresh } from '@/lib/queries'
import {
  receiveCanvas,
  receiveDownload,
  receiveStartupState,
  receiveJob,
  receiveResources,
  useKoharuStore,
} from '@/lib/store'
import {
  commands,
  type CanvasState,
  type Download,
  type Job,
  type ModelResources,
  type ProjectInfo,
} from '@koharu/bridge/protocol'
import { Toaster } from '@koharu/ui/components/toast'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

export function Providers({ children }: { children: ReactNode }) {
  const runtime = useRef({ active: false, bound: false })

  useEffect(() => {
    const lifecycle = runtime.current
    lifecycle.active = true
    if (!lifecycle.bound) {
      lifecycle.bound = true
      const completed = new Map<string, number>()
      const channel = <T,>(receive: (value: T) => void) =>
        new Channel<T>((value) => {
          if (lifecycle.active) receive(value)
        })

      void refreshTranslationModels().catch(() => undefined)

      void commands
        .subscribe(
          channel<CanvasState>(receiveCanvas),
          channel<Job>((job) => {
            const previous = completed.get(job.id) ?? 0
            if (isActiveJob(job)) completed.set(job.id, job.completed)
            else completed.delete(job.id)
            receiveJob(job)
            if (job.completed > previous || job.state !== 'running') {
              void refresh(projectKey, pagesKey, pageKey).catch(() => undefined)
            }
          }),
          channel<Download>(receiveDownload),
          channel<ModelResources>(receiveResources),
          channel<ProjectInfo | null>((project) => {
            const previous = queryClient.getQueryData<ProjectInfo | null>(projectKey)
            queryClient.setQueryData(projectKey, project)
            if (previous?.session !== project?.session) {
              const store = useKoharuStore.getState()
              store.selectPages(project?.active_page ? [project.active_page] : [])
              store.selectLayers([])
              store.clearPendingMasks()
            }
            if (project) {
              void refresh(pagesKey, pageKey).catch(() => undefined)
            } else {
              queryClient.setQueryData(pagesKey, [])
              queryClient.setQueryData(pageKey, null)
            }
          }),
        )
        .then((state) => {
          if (lifecycle.active) receiveStartupState(state)
        })
        .catch(() => undefined)

      // A dropped channel event would otherwise leave a job "running" and its
      // pending mask on the canvas forever; poll only while work is active.
      window.setInterval(() => {
        if (!lifecycle.active) return
        const active = Object.values(useKoharuStore.getState().jobs).filter(isActiveJob)
        if (active.length === 0) return
        void commands
          .getJobs(active.map((job) => job.id))
          .then((known) => {
            const current = new Map(known.map((job) => [job.id, job]))
            let ended = false
            for (const job of active) {
              // Unknown jobs belonged to an earlier project session.
              const next = current.get(job.id) ?? { ...job, state: 'stopped' as const }
              if (next.state === job.state && next.completed === job.completed) continue
              if (!isActiveJob(next)) {
                completed.delete(job.id)
                ended = true
              }
              receiveJob(next)
            }
            if (ended) void refresh(projectKey, pagesKey, pageKey).catch(() => undefined)
          })
          .catch(() => undefined)
      }, 3000)
    }

    return () => {
      lifecycle.active = false
    }
  }, [])

  useEffect(() => {
    const setLanguage = (language: string) => {
      document.documentElement.lang = language
    }
    setLanguage(i18n.language)
    i18n.on('languageChanged', setLanguage)
    return () => i18n.off('languageChanged', setLanguage)
  }, [])

  useEffect(() => {
    // Prevent the host webview from applying browser zoom; keep Ctrl+wheel for app handlers.
    const preventViewportScaling = (event: WheelEvent) => {
      if (event.ctrlKey) event.preventDefault()
    }

    window.addEventListener('wheel', preventViewportScaling, { capture: true, passive: false })
    return () => window.removeEventListener('wheel', preventViewportScaling, { capture: true })
  }, [])

  return (
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>
        <TooltipProvider delay={0}>
          <ClientOnly>
            <StartupBoundary>{children}</StartupBoundary>
            <Toaster />
          </ClientOnly>
        </TooltipProvider>
      </I18nextProvider>
    </QueryClientProvider>
  )
}

function StartupBoundary({ children }: { children: ReactNode }) {
  const initialized = useKoharuStore((state) => state.initialized)
  return initialized ? children : <StartupView />
}

export default Providers
