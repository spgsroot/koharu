import { QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AgentPanel } from '@/components/editor/AgentPanel'
import { queryClient } from '@/lib/queries'
import { commands, type AgentStatus } from '@koharu/bridge/protocol'

vi.mock('@tauri-apps/api/core', () => ({
  Channel: class<T> {
    onmessage: (message: T) => void = () => undefined
  },
}))

const signedOut = (provider: 'codex' | 'xai' | 'kimi'): AgentStatus => ({
  account: null,
  models: [],
  config: { provider, model: null, reasoning: 'medium' },
  running: null,
})

function renderPanel() {
  return render(
    <QueryClientProvider client={queryClient}>
      <AgentPanel />
    </QueryClientProvider>,
  )
}

describe('AgentPanel providers', () => {
  afterEach(() => vi.restoreAllMocks())

  it('offers Kimi sign-in with its own copy and the shared device-code command', async () => {
    vi.spyOn(commands, 'getAgentStatus').mockResolvedValue(signedOut('kimi'))
    const login = vi.spyOn(commands, 'loginAgent').mockResolvedValue({
      ...signedOut('kimi'),
      account: { id: 'user', email: 'reader@example.com', plan: null },
    })
    renderPanel()

    expect(await screen.findByRole('heading', { name: 'Connect Kimi' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Sign in with Kimi' }))

    await waitFor(() => expect(login).toHaveBeenCalledOnce())
    expect(await screen.findByText('reader@example.com')).toBeInTheDocument()
  })

  it('does not label a Kimi account as Codex when the provider omits an email', async () => {
    vi.spyOn(commands, 'getAgentStatus').mockResolvedValue({
      ...signedOut('kimi'),
      account: { id: 'user', email: null, plan: null },
      models: [{ id: 'kimi-for-coding', name: 'Kimi for Coding', reasoning: [] }],
    })
    renderPanel()

    expect(await screen.findByText('Kimi Code')).toBeInTheDocument()
    expect(screen.getByText('Kimi Code account')).toBeInTheDocument()
    expect(screen.queryByText('Codex')).not.toBeInTheDocument()
  })
})
