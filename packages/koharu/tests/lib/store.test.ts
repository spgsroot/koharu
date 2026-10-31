import { describe, expect, it } from 'vitest'

import { defaultShortcuts, receiveDownload, useKoharuStore } from '@/lib/store'

describe('editor state', () => {
  it('retains failed downloads until they are dismissed', () => {
    useKoharuStore.setState({ downloads: {} })
    receiveDownload({
      state: 'failed',
      id: 9,
      name: 'weights.bin',
      completed: 0,
      total: 0,
      error: 'network unavailable',
    })
    expect(useKoharuStore.getState().downloads[9]).toMatchObject({
      state: 'failed',
      name: 'weights.bin',
    })

    useKoharuStore.getState().dismissDownload(9)
    expect(useKoharuStore.getState().downloads[9]).toBeUndefined()
  })

  it('defaults the paint brush to pure white', () => {
    expect(useKoharuStore.getState().brush.color).toBe('#FFFFFF')
  })

  it('gives every shortcut action its own default key', () => {
    const keys = Object.values(defaultShortcuts)
    expect(new Set(keys).size).toBe(keys.length)
  })
})
