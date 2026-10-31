'use client'

import { commands, type Job } from '@koharu/bridge/protocol'

/** A job the native queue still owes work for: waiting for its turn or executing. */
export function isActiveJob(job: Job): boolean {
  return job.state === 'queued' || job.state === 'running'
}

/** An automatic pipeline run; hand-painted repairs run beside it and do not block Run. */
export function isActivePipeline(job: Job): boolean {
  return job.kind === 'pipeline' && isActiveJob(job)
}

/**
 * Cancels queued jobs before the running one. Stopping the running job first would let the
 * queue start a waiting job that is about to be cancelled anyway.
 */
export async function stopJobs(jobs: readonly Job[]): Promise<void> {
  const ordered = [
    ...jobs.filter((job) => job.state === 'queued'),
    ...jobs.filter((job) => job.state === 'running'),
  ]
  for (const job of ordered) {
    await commands.stopJob(job.id).catch(() => undefined)
  }
}
