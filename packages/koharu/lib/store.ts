'use client'

import { create } from 'zustand'

import type {
  CanvasState,
  CleanupTarget,
  CopiedLayers,
  Download,
  EntityId,
  Job,
  JobId,
  Model,
  Frame,
  Point,
  Preferences,
  ProjectInfo,
  Stage,
  StartupState,
  ModelResources,
} from '@koharu/bridge/protocol'
import { toast } from '@koharu/ui/components/toast'

import { isActiveJob } from './jobs'
/** Every tool in toolbar order. A cleanup tool is named after the `CleanupTarget` it commits. */
export const canvasTools = [
  'select',
  'text',
  'draw',
  'eraser',
  'color_picker',
  'fast',
  'inpainting',
  'decensor',
  'pan',
] as const
export type CanvasTool = (typeof canvasTools)[number]
export const MIN_BRUSH_DIAMETER = 1
export const MAX_BRUSH_DIAMETER = 128

const cleanupTargets: Record<CleanupTarget, true> = { fast: true, inpainting: true, decensor: true }

export function isCleanupTool(tool: CanvasTool): tool is CleanupTarget {
  return tool in cleanupTargets
}

export function isBrushTool(tool: CanvasTool): boolean {
  return tool === 'draw' || tool === 'eraser' || isCleanupTool(tool)
}

export interface CanvasBrush {
  diameter: number
  color: string
}

/**
 * A cleanup stroke the native queue has accepted or is about to accept. The queue is
 * canonical; this only keeps the brushed area visible until its job settles.
 */
export interface PendingMask {
  /** Local identity: the job id only exists once the queue accepts the stroke. */
  id: string
  job: JobId | null
  page: EntityId
  target: CleanupTarget
  diameter: number
  points: Point[]
}

export type InspectorSection = 'copy' | 'type' | 'layers'
export type AppView = 'orders' | 'start' | 'settings'
export type ShortcutAction = CanvasTool | 'fit'
export type Shortcuts = Record<ShortcutAction, string>
export type PipelineScope = 'page' | 'selected-pages' | 'project'
export const pipelineStages: readonly Stage[] = [
  'detection',
  'ocr',
  'translation',
  'censor_detection',
  'inpainting',
  'decensor',
]

interface KoharuStore {
  initialized: boolean
  preferences: Preferences | null
  translationModels: Model[]
  resources: ModelResources | null
  jobs: Record<string, Job>
  downloads: Record<string, Download>
  camera: { zoom: number; translation: [number, number]; fitted: boolean }
  canvasPage: EntityId | null
  canvasRevision: number | null
  canvasGeneration: number
  canvasSize: [number, number]
  fitCanvasRequest: number
  layerFrames: Record<EntityId, Frame>
  selectedLayers: EntityId[]
  selectedPages: EntityId[]
  tool: CanvasTool
  brush: CanvasBrush
  pendingMasks: PendingMask[]
  inspector: InspectorSection
  processingScope: PipelineScope
  processingStages: Stage[]
  settingsOpen: boolean
  view: AppView
  shortcuts: Shortcuts
  /** Layers copied natively, with the text mirrored to the system clipboard. */
  layerClipboard: (CopiedLayers & { text: string }) | null
  selectPages: (pages: EntityId[]) => void
  showInspector: (section: InspectorSection) => void
  setProcessingScope: (scope: PipelineScope) => void
  setProcessingStages: (stages: Stage[]) => void
  setSettingsOpen: (open: boolean) => void
  setView: (view: AppView) => void
  selectLayers: (layers: EntityId[]) => void
  setTool: (tool: CanvasTool) => void
  setBrush: (brush: CanvasBrush) => void
  addPendingMask: (mask: Omit<PendingMask, 'id' | 'job'>) => string
  bindPendingMask: (id: string, job: JobId) => void
  removePendingMasks: (ids: readonly string[]) => void
  clearPendingMasks: () => void
  setShortcut: (action: ShortcutAction, key: string) => void
  requestCanvasFit: () => void
  dismissJob: (id: string) => void
  dismissDownload: (id: number) => void
}

export const defaultShortcuts: Shortcuts = {
  select: 'v',
  text: 't',
  draw: 'b',
  eraser: 'e',
  color_picker: 'i',
  fast: 'j',
  inpainting: 'p',
  decensor: 'd',
  pan: 'h',
  fit: '0',
}

let maskSequence = 0

export const useKoharuStore = create<KoharuStore>()((set) => ({
  initialized: false,
  preferences: null,
  translationModels: [],
  resources: null,
  jobs: {},
  downloads: {},
  camera: { zoom: 1, translation: [0, 0], fitted: true },
  canvasPage: null,
  canvasRevision: null,
  canvasGeneration: 0,
  canvasSize: [0, 0],
  fitCanvasRequest: 0,
  layerFrames: {},
  selectedLayers: [],
  selectedPages: [],
  tool: 'select',
  brush: { diameter: 48, color: '#FFFFFF' },
  pendingMasks: [],
  inspector: 'copy',
  processingScope: 'page',
  processingStages: [...pipelineStages],
  settingsOpen: false,
  view: 'orders' as AppView,
  shortcuts: defaultShortcuts,
  layerClipboard: null,
  selectPages: (selectedPages) => set({ selectedPages: [...new Set(selectedPages)] }),
  showInspector: (inspector) => set({ inspector }),
  setProcessingScope: (processingScope) => set({ processingScope }),
  setProcessingStages: (processingStages) => set({ processingStages }),
  setSettingsOpen: (settingsOpen) =>
    set(settingsOpen ? { settingsOpen, view: 'settings' } : { settingsOpen, view: 'start' }),
  setView: (view) => set({ view, settingsOpen: view === 'settings' }),
  selectLayers: (selectedLayers) => set({ selectedLayers: [...new Set(selectedLayers)] }),
  setTool: (tool) => set({ tool }),
  setBrush: (brush) => set({ brush }),
  addPendingMask: (mask) => {
    const id = `mask-${++maskSequence}`
    set((state) => ({ pendingMasks: [...state.pendingMasks, { ...mask, id, job: null }] }))
    return id
  },
  bindPendingMask: (id, job) =>
    set((state) => ({
      pendingMasks: state.pendingMasks.flatMap((mask) => {
        if (mask.id !== id) return [mask]
        const received = state.jobs[job]
        return received && !isActiveJob(received) ? [] : [{ ...mask, job }]
      }),
    })),
  removePendingMasks: (ids) =>
    set((state) => {
      const pendingMasks = state.pendingMasks.filter((mask) => !ids.includes(mask.id))
      return pendingMasks.length === state.pendingMasks.length ? state : { pendingMasks }
    }),
  clearPendingMasks: () =>
    set((state) => (state.pendingMasks.length === 0 ? state : { pendingMasks: [] })),
  setShortcut: (action, key) =>
    set((state) => ({
      shortcuts: { ...state.shortcuts, [action]: key.toLowerCase().slice(0, 1) },
    })),
  requestCanvasFit: () => set((state) => ({ fitCanvasRequest: state.fitCanvasRequest + 1 })),
  dismissJob: (id) =>
    set((state) => {
      const jobs = { ...state.jobs }
      delete jobs[id]
      return { jobs }
    }),
  dismissDownload: (id) =>
    set((state) => {
      const downloads = { ...state.downloads }
      delete downloads[String(id)]
      return { downloads }
    }),
}))

export const useView = (): AppView => useKoharuStore((state) => state.view)

export function receiveProjectInfo(info: ProjectInfo | null): void {
  if (info) useKoharuStore.setState({ view: 'start', settingsOpen: false })
}

export function receiveStartupState(state: StartupState): void {
  useKoharuStore.setState((current) => ({
    initialized: true,
    preferences: state.preferences,
    // The Orders workspace is the default only while `[ui] orders` enables it.
    view: current.view === 'orders' && !state.preferences.ui.orders ? 'start' : current.view,
    // Channels are live before this snapshot is read, so anything they
    // already delivered is newer than the snapshot.
    jobs: { ...byId(state.jobs), ...current.jobs },
    ...(current.canvasGeneration > state.canvas.generation ? {} : canvasSnapshot(state.canvas)),
  }))
}

export function receiveCanvas(canvas: CanvasState): void {
  useKoharuStore.setState(canvasSnapshot(canvas))
}

/** Finished and stopped jobs kept so a late admission reply still finds its terminal state. */
const RETAINED_TERMINAL_JOBS = 32

export function receiveJob(job: Job): void {
  useKoharuStore.setState((state) => {
    const jobs = { ...state.jobs, [job.id]: job }
    const terminal = Object.values(jobs).filter(
      (candidate) => !isActiveJob(candidate) && candidate.state !== 'failed',
    )
    for (const evicted of terminal.slice(0, -RETAINED_TERMINAL_JOBS)) delete jobs[evicted.id]
    return {
      jobs,
      pendingMasks: isActiveJob(job)
        ? state.pendingMasks
        : state.pendingMasks.filter((mask) => mask.job !== job.id),
    }
  })
}

export function receiveDownload(download: Download): void {
  useKoharuStore.setState((state) => {
    const downloads = { ...state.downloads }
    if (download.state === 'finished') delete downloads[String(download.id)]
    else downloads[String(download.id)] = download
    return { downloads }
  })
}

export function receivePreferences(preferences: Preferences): void {
  useKoharuStore.setState({ preferences })
}

export function receiveTranslationModels(translationModels: Model[]): void {
  useKoharuStore.setState({ translationModels })
}

export function receiveResources(resources: ModelResources): void {
  useKoharuStore.setState({ resources })
}

export function receiveError(message: string): void {
  toast.add({ type: 'error', title: 'Could not complete that action', description: message })
}

function canvasSnapshot(canvas: CanvasState) {
  return {
    canvasPage: canvas.page,
    canvasRevision: canvas.revision,
    canvasGeneration: canvas.generation,
    canvasSize: canvas.size,
    layerFrames: Object.fromEntries(
      canvas.element_frames.map(({ element, frame }) => [element, frame]),
    ),
  }
}

function byId<T extends { id: string | number }>(items: T[]): Record<string, T> {
  return Object.fromEntries(items.map((item) => [String(item.id), item]))
}
