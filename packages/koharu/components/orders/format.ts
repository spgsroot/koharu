const dayFormat = new Intl.DateTimeFormat(undefined, {
  timeZone: 'UTC',
  day: 'numeric',
  month: 'short',
  year: 'numeric',
})

const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
})

/**
 *  Deadlines are whole calendar days stored at UTC midnight, so reading and
 *  the `<input type="date">` value go through the UTC getters on both sides.
 */
export function deadlineToInput(millis: number | null): string {
  if (millis === null) return ''
  const date = new Date(millis)
  const month = `${date.getUTCMonth() + 1}`.padStart(2, '0')
  const day = `${date.getUTCDate()}`.padStart(2, '0')
  return `${date.getUTCFullYear()}-${month}-${day}`
}

export function inputToDeadline(value: string): number | null {
  if (!value) return null
  const [year, month, day] = value.split('-').map(Number)
  return Date.UTC(year, month - 1, day)
}

export function formatDay(millis: number | null): string | null {
  return millis === null ? null : dayFormat.format(new Date(millis))
}

export function formatDateTime(millis: number): string {
  return dateTimeFormat.format(new Date(millis))
}
