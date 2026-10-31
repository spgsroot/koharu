'use client'

import { useMutation, useQuery } from '@tanstack/react-query'
import { create } from 'zustand'

import { call } from '@/lib/backend'
import { queryClient, useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import {
  commands,
  type DayBounds,
  type Lifecycle,
  type NewNote,
  type NewOrder,
  type OrderDetail,
  type OrderFilter,
  type OrderListItem,
  type OrderPatch,
  type StageName,
  type StageState,
  type TemplateFormat,
} from '@koharu/bridge/protocol'

/** Fixed production route; mirrors the StageName pipeline order from the backend. */
export const STAGE_ORDER: readonly StageName[] = [
  'intake',
  'preparation',
  'translation',
  'editing',
  'typesetting',
  'qa',
  'delivery',
]

export const ordersKey = ['orders'] as const
export const ordersSummaryKey = ['ordersSummary'] as const
export const ordersActorKey = ['ordersActor'] as const
export const orderKey = (id: string) => ['order', id] as const
export const orderNotesKey = (id: string) => ['orderNotes', id] as const
export const orderActivityKey = (id: string) => ['orderActivity', id] as const
export const orderImportsKey = (id: string) => ['orderImports', id] as const
const translationImportKey = (id: string) => ['translationImport', id] as const

export const defaultOrdersFilter: OrderFilter = {
  view: 'attention',
  search: null,
  lifecycle: null,
  stage: null,
  assignee: null,
  client: null,
  has_blocker: null,
  import_state: null,
  deadline_from: null,
  deadline_to: null,
  actor: null,
}

interface OrdersUiState {
  selectedOrderId: string | null
  setSelectedOrder: (id: string | null) => void
  filter: OrderFilter
  setFilter: (patch: Partial<OrderFilter>) => void
  resetFilter: () => void
}

export const useOrdersUiStore = create<OrdersUiState>()((set) => ({
  selectedOrderId: null,
  setSelectedOrder: (id) => set({ selectedOrderId: id }),
  filter: defaultOrdersFilter,
  setFilter: (patch) => set((state) => ({ filter: { ...state.filter, ...patch } })),
  resetFilter: () => set({ filter: defaultOrdersFilter }),
}))

/** Local-day bounds as epoch millis; the UI owns the time zone. */
export function todayBounds(): DayBounds {
  const now = new Date()
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  return { start: start.getTime(), end: start.getTime() + 86_400_000 }
}

async function resolveActor(filter: OrderFilter): Promise<OrderFilter> {
  if (filter.view !== 'mine' || filter.actor) return filter
  const actor = await call(commands.ordersActor)
  return { ...filter, actor }
}

export function useOrdersActor() {
  return useQuery({
    queryKey: ordersActorKey,
    queryFn: () => call(commands.ordersActor),
  })
}

export function useOrders() {
  const filter = useOrdersUiStore((state) => state.filter)
  return useOrdersByFilter(filter)
}

export function useOrdersByFilter(filter: OrderFilter, enabled = true) {
  return useQuery({
    queryKey: [...ordersKey, filter],
    queryFn: async () => call(commands.listOrders, await resolveActor(filter), todayBounds()),
    enabled,
  })
}

export function useOrdersSummary() {
  return useQuery({
    queryKey: ordersSummaryKey,
    queryFn: () => call(commands.ordersSummary, todayBounds()),
  })
}

export function useOrder(id: string | null) {
  return useQuery({
    queryKey: orderKey(id ?? ''),
    queryFn: () => call(commands.getOrder, id as string),
    enabled: id !== null,
  })
}

export function useOrderNotes(orderId: string | null) {
  return useQuery({
    queryKey: orderNotesKey(orderId ?? ''),
    queryFn: () => call(commands.listOrderNotes, orderId as string),
    enabled: orderId !== null,
  })
}

export function useOrderActivity(orderId: string | null) {
  return useQuery({
    queryKey: orderActivityKey(orderId ?? ''),
    queryFn: () => call(commands.listOrderActivity, orderId as string),
    enabled: orderId !== null,
  })
}

export function useOrderImports(orderId: string | null) {
  return useQuery({
    queryKey: orderImportsKey(orderId ?? ''),
    queryFn: () => call(commands.listOrderImports, orderId as string),
    enabled: orderId !== null,
  })
}

export function useTranslationImport(id: string | null) {
  return useQuery({
    queryKey: translationImportKey(id ?? ''),
    queryFn: () => call(commands.getTranslationImport, id as string),
    enabled: id !== null,
  })
}

/**
 * The order whose project is currently open in the editor, if any. Always
 * `null` while the Orders workspace is disabled in `[ui] orders`.
 */
export function useCurrentOrder(): OrderListItem | null | undefined {
  const enabled = useKoharuStore((state) => state.preferences?.ui.orders ?? false)
  const project = useProject()
  const orders = useOrdersByFilter({ ...defaultOrdersFilter, view: 'all' }, enabled)
  if (!enabled) return null
  if (project.data === undefined || orders.data === undefined) return undefined
  if (project.data === null) return null
  const projectName = project.data.name
  return orders.data.find((order) => order.project_name === projectName) ?? null
}

function invalidateOrder(orderId: string | null): Promise<unknown> {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: ordersKey }),
    queryClient.invalidateQueries({ queryKey: ordersSummaryKey }),
    ...(orderId
      ? [
          queryClient.invalidateQueries({ queryKey: orderKey(orderId) }),
          queryClient.invalidateQueries({ queryKey: orderNotesKey(orderId) }),
          queryClient.invalidateQueries({ queryKey: orderActivityKey(orderId) }),
          queryClient.invalidateQueries({ queryKey: orderImportsKey(orderId) }),
        ]
      : []),
  ])
}

export function useCreateOrder() {
  const setSelectedOrder = useOrdersUiStore((state) => state.setSelectedOrder)
  return useMutation({
    mutationFn: (input: NewOrder) => call(commands.createOrder, input),
    onSuccess: async (order) => {
      setSelectedOrder(order.id)
      await invalidateOrder(order.id)
    },
  })
}

export function useUpdateOrder() {
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: OrderPatch }) =>
      call(commands.updateOrder, id, patch),
    onSuccess: (order) => invalidateOrder(order.id),
  })
}

export function useSetOrderLifecycle() {
  return useMutation({
    mutationFn: ({ id, lifecycle }: { id: string; lifecycle: Lifecycle }) =>
      call(commands.setOrderLifecycle, id, lifecycle),
    onSuccess: (order) => invalidateOrder(order.id),
  })
}

export function useAttachOrderProject() {
  return useMutation({
    mutationFn: ({ id, projectName }: { id: string; projectName: string | null }) =>
      call(commands.attachOrderProject, id, projectName),
    onSuccess: (order) => invalidateOrder(order.id),
  })
}

export function useSetOrderStage() {
  return useMutation({
    mutationFn: async ({ id, changes }: { id: string; changes: [StageName, StageState][] }) => {
      let order: OrderDetail | null = null
      for (const [stage, state] of changes) {
        order = await call(commands.setOrderStage, id, stage, state)
      }
      return order
    },
    onSuccess: (order) => (order ? invalidateOrder(order.id) : undefined),
  })
}

export function useAddOrderNote() {
  return useMutation({
    mutationFn: ({
      orderId,
      input,
      blockStage,
    }: {
      orderId: string
      input: NewNote
      blockStage: boolean
    }) => call(commands.addOrderNote, orderId, input, blockStage),
    onSuccess: (order) => invalidateOrder(order.id),
  })
}

export function useResolveOrderNote() {
  return useMutation({
    mutationFn: ({ noteId, resolved }: { noteId: string; resolved: boolean }) =>
      call(commands.resolveOrderNote, noteId, resolved),
    onSuccess: (order) => invalidateOrder(order.id),
  })
}

/** `orderId` travels only for invalidation: the command takes just the note id. */
export function useDeleteOrderNote() {
  return useMutation({
    mutationFn: ({ noteId }: { noteId: string; orderId: string }) =>
      call(commands.deleteOrderNote, noteId),
    onSuccess: (_result, { orderId }) => invalidateOrder(orderId),
  })
}

export function useExportTranslationTemplate() {
  return useMutation({
    mutationFn: ({ orderId, format }: { orderId: string; format: TemplateFormat }) =>
      call(commands.exportTranslationTemplate, orderId, format),
  })
}

export function useBeginTranslationImport() {
  return useMutation({
    mutationFn: (orderId: string) => call(commands.beginTranslationImport, orderId),
    onSuccess: (result) => (result ? invalidateOrder(result.import.order_id) : undefined),
  })
}

export function useApplyImport() {
  return useMutation({
    mutationFn: (importId: string) => call(commands.applyTranslationImport, importId),
    onSuccess: (order) => invalidateOrder(order.id),
  })
}

export function useDiscardImport() {
  return useMutation({
    mutationFn: ({ importId }: { importId: string; orderId: string }) =>
      call(commands.discardTranslationImport, importId),
    onSuccess: (_result, { orderId }) => invalidateOrder(orderId),
  })
}

export function useExportImportReport() {
  return useMutation({
    mutationFn: (importId: string) => call(commands.exportImportReport, importId),
  })
}
