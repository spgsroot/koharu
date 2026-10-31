import { QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GalleryImportDialog } from '@/components/start/GalleryImportDialog'
import { queryClient } from '@/lib/queries'
import { commands } from '@koharu/bridge/protocol'

vi.mock('@tauri-apps/api/core', () => ({
  Channel: class<T> {
    onmessage: (message: T) => void = () => undefined
  },
}))

function renderDialog() {
  return render(
    <QueryClientProvider client={queryClient}>
      <GalleryImportDialog open onOpenChange={() => undefined} />
    </QueryClientProvider>,
  )
}

describe('GalleryImportDialog', () => {
  afterEach(() => vi.restoreAllMocks())

  it('imports the links in the order given, without blank lines', async () => {
    const importGalleries = vi.spyOn(commands, 'importGalleries').mockResolvedValue({
      project: 'Volume [1]',
      pages: 34,
      manifest: 'C:/Koharu/gallery-downloads/1/manifest.json',
    })
    renderDialog()
    fireEvent.change(screen.getByRole('textbox', { name: /Gallery links/ }), {
      target: {
        value: ' https://nhentai.net/g/2/ \n\nhttps://e-hentai.org/g/1/abcdef/\n',
      },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    await waitFor(() => expect(importGalleries).toHaveBeenCalledOnce())
    expect(importGalleries.mock.calls[0][0]).toEqual({
      urls: ['https://nhentai.net/g/2/', 'https://e-hentai.org/g/1/abcdef/'],
      name: null,
      exhentai_cookies: null,
    })
    expect(await screen.findByText('Project created')).toBeInTheDocument()
    expect(screen.getByText('Volume [1]')).toBeInTheDocument()
  })

  it('refuses more than twenty links before calling the backend', async () => {
    const importGalleries = vi.spyOn(commands, 'importGalleries')
    renderDialog()
    const links = Array.from({ length: 21 }, (_, index) => `https://nhentai.net/g/${index + 1}/`)
    fireEvent.change(screen.getByRole('textbox', { name: /Gallery links/ }), {
      target: { value: links.join('\n') },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    expect(await screen.findByText(/import failed/i)).toBeInTheDocument()
    expect(importGalleries).not.toHaveBeenCalled()
  })

  it('reports a backend failure and keeps the links for another try', async () => {
    vi.spyOn(commands, 'importGalleries').mockRejectedValue('exhentai requires cookies')
    renderDialog()
    const field = screen.getByRole('textbox', { name: /Gallery links/ })
    fireEvent.change(field, { target: { value: 'https://exhentai.org/g/1/abcdef/' } })
    fireEvent.click(screen.getByRole('button', { name: 'Import' }))

    expect(await screen.findByText('exhentai requires cookies')).toBeInTheDocument()
    expect(field).toHaveValue('https://exhentai.org/g/1/abcdef/')
  })
})
