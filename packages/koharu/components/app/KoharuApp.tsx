'use client'

import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'

import { TitleBar } from '@/components/app/TitleBar'
import { Editor } from '@/components/editor/Editor'
import OrdersView from '@/components/orders/OrdersView'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import { StartView } from '@/components/start/StartView'
import { useProject } from '@/lib/queries'
import { receiveProjectInfo, useKoharuStore } from '@/lib/store'
import { cn } from '@koharu/ui/lib/utils'

export function KoharuApp() {
  const { t } = useTranslation()
  const project = useProject().data
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const view = useKoharuStore((state) => state.view)
  const ordersEnabled = useKoharuStore((state) => state.preferences?.ui.orders ?? false)
  const projectLoaded = project !== undefined
  const activePage = project?.active_page
  const editorOpen = view === 'start' && project !== undefined && project !== null
  const projectOpenRef = useRef(false)

  useEffect(() => {
    if (!projectLoaded || selectedPages.length > 0) return
    selectPages(activePage ? [activePage] : [])
    selectLayers([])
  }, [activePage, projectLoaded, selectLayers, selectPages, selectedPages.length])

  useEffect(() => {
    if (project === undefined) return
    if (project === null) {
      projectOpenRef.current = false
      return
    }
    if (!projectOpenRef.current) {
      projectOpenRef.current = true
      receiveProjectInfo(project)
    }
  }, [project])

  return (
    <div
      className={cn(
        'relative flex h-screen w-screen flex-col overflow-hidden text-foreground',
        editorOpen ? 'bg-transparent' : 'bg-[var(--surface-titlebar)]',
      )}
    >
      <TitleBar />
      {view === 'orders' && ordersEnabled ? (
        <OrdersView />
      ) : view === 'settings' ? (
        <SettingsPage />
      ) : project === undefined ? (
        <main className='grid min-h-0 flex-1 place-items-center bg-[var(--surface-canvas)]'>
          <div className='flex items-center gap-3 text-[12px] text-muted-foreground'>
            <span className='size-2 rounded-full bg-primary' />
            {t('startup.opening')}
          </div>
        </main>
      ) : project === null ? (
        <StartView />
      ) : (
        <Editor />
      )}
    </div>
  )
}
