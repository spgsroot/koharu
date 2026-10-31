'use client'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, projectKey, refresh } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type EntityId, type Page, type Point } from '@koharu/bridge/protocol'

/** Copies pasted onto their source page are shifted so they do not hide the originals. */
const PASTE_OFFSET = 16

/**
 * The text the selected text layers show. It is mirrored to the system
 * clipboard, so a later paste of exactly this text means "these layers",
 * while any other text pastes as a new text layer.
 */
export function selectionText(page: Page | null | undefined, layers: readonly EntityId[]): string {
  if (!page) return ''
  return page.layers
    .flatMap((layer) =>
      layer.type === 'text' && layers.includes(layer.id)
        ? [layer.content.translation?.text ?? layer.content.source?.text ?? '']
        : [],
    )
    .filter(Boolean)
    .join('\n')
}

/** Copies the selected layers natively; cutting removes them in a second history step. */
export async function copySelection(cut: boolean, text: string): Promise<void> {
  const layers = [...useKoharuStore.getState().selectedLayers]
  if (layers.length === 0) return
  const copied = await call(commands.copyLayers, layers)
  if (copied.count === 0) return
  useKoharuStore.setState({ layerClipboard: { ...copied, text } })
  if (!cut) return
  await call(commands.deleteLayers, layers)
  useKoharuStore.getState().selectLayers([])
  await refresh(projectKey, pagesKey, pageKey)
}

export async function pasteLayers(page: EntityId): Promise<void> {
  const copied = useKoharuStore.getState().layerClipboard
  if (!copied) return
  const offset = copied.page === page ? PASTE_OFFSET : 0
  const pasted = await call(commands.pasteLayers, page, { x: offset, y: offset })
  useKoharuStore.getState().selectLayers(pasted)
  await refresh(projectKey, pagesKey, pageKey)
}

/** Text copied outside Koharu becomes the translation of a new point-text layer. */
export async function pasteText(text: string, point: Point): Promise<void> {
  const created = await call(commands.addPointText, point)
  await call(commands.setTranslation, created.layer, text)
  useKoharuStore.getState().selectLayers([created.layer])
  await refresh(projectKey, pagesKey, pageKey)
}
