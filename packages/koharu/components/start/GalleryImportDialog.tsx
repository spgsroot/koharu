'use client'

import { Channel } from '@tauri-apps/api/core'
import { CheckCircle2, CircleAlert, Images, LoaderCircle } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, projectKey, refresh } from '@/lib/queries'
import { commands } from '@koharu/bridge/protocol'
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
import { Textarea } from '@koharu/ui/components/textarea'
import { cn } from '@koharu/ui/lib/utils'

interface GalleryImportProgress {
  status: 'inspecting' | 'downloading' | 'importing'
  gallery: number
  galleries: number
  pages: number
  total_pages: number | null
}

interface GalleryImportResult {
  project: string
  pages: number
  manifest: string
}

export function GalleryImportDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const { t } = useTranslation()
  const [urlText, setUrlText] = useState('')
  const [name, setName] = useState('')
  const [cookies, setCookies] = useState('')
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<GalleryImportProgress | null>(null)
  const [result, setResult] = useState<GalleryImportResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const urls = urlText
    .split(/\r?\n/)
    .map((url) => url.trim())
    .filter(Boolean)
  const tooManyUrls = urls.length > 20
  const percentage =
    progress?.total_pages && progress.total_pages > 0
      ? Math.min(100, Math.round((progress.pages / progress.total_pages) * 100))
      : null
  const progressLabel = progress
    ? t(
        progress.status === 'importing'
          ? 'galleryImport.importingPages'
          : `galleryImport.${progress.status}`,
      )
    : t('galleryImport.importing')

  const changeOpen = (nextOpen: boolean) => {
    if (!nextOpen && running) return
    if (!nextOpen) {
      setUrlText('')
      setName('')
      setCookies('')
      setProgress(null)
      setResult(null)
      setError(null)
    }
    onOpenChange(nextOpen)
  }

  const importGalleries = async (event: FormEvent) => {
    event.preventDefault()
    if (running || result) return

    if (urls.length === 0 || urls.length > 20) {
      setError(t('galleryImport.error'))
      return
    }

    setRunning(true)
    setProgress(null)
    setError(null)
    const channel = new Channel<GalleryImportProgress>()
    channel.onmessage = setProgress

    try {
      const imported = await call(
        commands.importGalleries,
        {
          urls,
          name: name.trim() || null,
          exhentai_cookies: cookies.trim() || null,
        },
        channel,
      )
      setResult(imported)
      await refresh(projectKey, pagesKey, pageKey)
    } catch (cause: unknown) {
      setError(
        cause instanceof Error
          ? cause.message
          : typeof cause === 'string'
            ? cause
            : t('galleryImport.error'),
      )
    } finally {
      setCookies('')
      setRunning(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogContent
        className='flex max-h-[88vh] max-w-xl flex-col gap-4 overflow-y-auto p-5'
        showCloseButton={!running}
        aria-busy={running}
      >
        <DialogHeader className='shrink-0 gap-1.5 pr-6'>
          <div className='mb-1 grid size-9 place-items-center rounded-xl bg-primary/10 text-primary'>
            <Images className='size-4' aria-hidden='true' />
          </div>
          <DialogTitle className='text-[15px]'>{t('galleryImport.title')}</DialogTitle>
          <DialogDescription className='max-w-[52ch] text-[11px] leading-5'>
            {t('galleryImport.description')}
          </DialogDescription>
        </DialogHeader>

        <form id='gallery-import-form' className='grid gap-4' onSubmit={importGalleries}>
          {result ? (
            <section
              className='rounded-xl border border-emerald-500/20 bg-emerald-500/[0.07] p-4'
              role='status'
            >
              <div className='flex items-center gap-2 text-emerald-700 dark:text-emerald-400'>
                <CheckCircle2 className='size-4 shrink-0' aria-hidden='true' />
                <h3 className='text-[12px] font-medium'>{t('galleryImport.success')}</h3>
              </div>
              <p className='mt-3 text-[13px] font-medium break-words'>{result.project}</p>
              <p className='mt-1 text-[11px] text-muted-foreground tabular-nums'>
                {t('galleryImport.pages')}: {result.pages}
              </p>
              <p className='mt-3 rounded-lg bg-background/60 p-2.5 font-mono text-[10px] leading-4 break-all text-muted-foreground select-text'>
                {result.manifest}
              </p>
            </section>
          ) : (
            <>
              <div className='grid gap-1.5'>
                <div className='flex items-center justify-between gap-3'>
                  <Label htmlFor='gallery-import-urls' className='text-[11px]'>
                    {t('galleryImport.urls')}
                  </Label>
                  <span
                    className={cn(
                      'text-[10px] text-muted-foreground tabular-nums',
                      tooManyUrls && 'font-medium text-destructive',
                    )}
                  >
                    {urls.length}/20
                  </span>
                </div>
                <Textarea
                  id='gallery-import-urls'
                  value={urlText}
                  onChange={(event) => setUrlText(event.target.value)}
                  placeholder={t('galleryImport.urlsPlaceholder')}
                  disabled={running}
                  aria-invalid={tooManyUrls || (error !== null && urls.length === 0)}
                  aria-describedby='gallery-import-urls-count'
                  autoComplete='off'
                  autoCapitalize='none'
                  spellCheck={false}
                  className='max-h-48 min-h-32 resize-y font-mono text-[11px] leading-5'
                />
                <span id='gallery-import-urls-count' className='sr-only'>
                  {urls.length}/20
                </span>
              </div>

              <div className='grid gap-1.5'>
                <Label htmlFor='gallery-import-name' className='text-[11px]'>
                  {t('galleryImport.name')}{' '}
                  <span className='font-normal text-muted-foreground'>
                    {t('galleryImport.nameOptional')}
                  </span>
                </Label>
                <Input
                  id='gallery-import-name'
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  disabled={running}
                  autoComplete='off'
                  className='h-9 text-[11px]'
                />
              </div>

              <div className='grid gap-1.5'>
                <Label htmlFor='gallery-import-cookies' className='text-[11px]'>
                  {t('galleryImport.cookies')}{' '}
                  <span className='font-normal text-muted-foreground'>
                    {t('galleryImport.cookiesOptional')}
                  </span>
                </Label>
                <Input
                  id='gallery-import-cookies'
                  type='password'
                  value={cookies}
                  onChange={(event) => setCookies(event.target.value)}
                  disabled={running}
                  aria-describedby='gallery-import-cookies-hint'
                  autoComplete='off'
                  autoCapitalize='none'
                  spellCheck={false}
                  className='h-9 font-mono text-[11px]'
                />
                <p
                  id='gallery-import-cookies-hint'
                  className='text-[10px] leading-4 text-muted-foreground'
                >
                  {t('galleryImport.cookiesHint')}
                </p>
              </div>
            </>
          )}
        </form>

        {running && (
          <section className='grid gap-2.5 rounded-xl bg-muted/45 p-3.5' aria-live='polite'>
            <div className='flex items-center gap-2'>
              <LoaderCircle className='size-3.5 shrink-0 animate-spin' aria-hidden='true' />
              <span className='text-[11px] font-medium'>{progressLabel}</span>
            </div>
            <div
              className='h-1.5 overflow-hidden rounded-full bg-primary/10'
              role='progressbar'
              aria-label={t('galleryImport.progress')}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percentage ?? undefined}
            >
              <div
                className={cn(
                  'h-full rounded-full bg-primary transition-[width] duration-300',
                  percentage === null && 'w-1/3 animate-pulse',
                )}
                style={percentage === null ? undefined : { width: `${percentage}%` }}
              />
            </div>
            {progress && (
              <div className='flex flex-wrap gap-x-5 gap-y-1 text-[10px] text-muted-foreground tabular-nums'>
                <span>
                  {t('galleryImport.galleries')}: {progress.gallery}/{progress.galleries}
                </span>
                <span>
                  {t('galleryImport.pages')}: {progress.pages}
                  {progress.total_pages === null ? '' : `/${progress.total_pages}`}
                </span>
              </div>
            )}
          </section>
        )}

        {error && (
          <div
            className='flex items-start gap-2 rounded-xl bg-destructive/10 p-3 text-destructive'
            role='alert'
          >
            <CircleAlert className='mt-0.5 size-3.5 shrink-0' aria-hidden='true' />
            <div className='min-w-0'>
              <p className='text-[11px] font-medium'>{t('galleryImport.error')}</p>
              {error !== t('galleryImport.error') && (
                <p className='mt-1 text-[10px] leading-4 break-words'>{error}</p>
              )}
            </div>
          </div>
        )}

        <DialogFooter className='-mx-5 -mb-5 shrink-0 px-5 py-4'>
          {!running && (
            <Button
              type='button'
              size='sm'
              variant={result ? 'default' : 'ghost'}
              onClick={() => changeOpen(false)}
            >
              {result || error ? t('galleryImport.close') : t('galleryImport.cancel')}
            </Button>
          )}
          {!result && (
            <Button
              type='submit'
              form='gallery-import-form'
              size='sm'
              disabled={running}
              className='min-w-24 gap-1.5'
            >
              {running && <LoaderCircle className='size-3.5 animate-spin' aria-hidden='true' />}
              {running ? t('galleryImport.importing') : t('galleryImport.import')}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
