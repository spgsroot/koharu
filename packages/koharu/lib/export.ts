'use client'

import { call } from '@/lib/backend'
import { commands, type TemplateFormat } from '@koharu/bridge/protocol'

/**
 *  Ensures a project with `name` exists and returns its name — the attach
 *  flow attaches by name, never by id.
 */
export async function importProject(name: string): Promise<string> {
  const existing = await call(commands.listProjects)
  if (!existing.some((project) => project.name === name)) {
    await call(commands.createProject, name)
  }
  return name
}

/** Returns the saved path, or null when the save dialog was cancelled. */
export function exportTranslationTemplate(input: {
  orderId: string
  format: TemplateFormat
}): Promise<string | null> {
  return call(commands.exportTranslationTemplate, input.orderId, input.format)
}

/** Returns the saved report path, or null when the save dialog was cancelled. */
export function exportImportReport(importId: string): Promise<string | null> {
  return call(commands.exportImportReport, importId)
}
