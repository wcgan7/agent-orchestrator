import { describe, it, expect, beforeEach, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  url: string
  listeners: Record<string, Array<(event?: unknown) => void>> = {}

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
    setTimeout(() => this.dispatch('open'), 0)
  }

  addEventListener(event: string, cb: (event?: unknown) => void) {
    this.listeners[event] = this.listeners[event] || []
    this.listeners[event].push(cb)
  }

  send() {}

  close() {}

  dispatch(event: string) {
    for (const cb of this.listeners[event] || []) {
      cb({})
    }
  }
}

describe('App default route', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    window.location.hash = ''
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket

    global.fetch = vi.fn().mockImplementation((url) => {
      const u = String(url)
      if (u.includes('/api/v3/tasks/board')) {
        return Promise.resolve({ ok: true, json: async () => ({ columns: { backlog: [], ready: [], in_progress: [], in_review: [], blocked: [], done: [] } }) })
      }
      if (u.includes('/api/v3/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/v3/review-queue')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/v3/agents')) {
        return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
      }
      if (u.includes('/api/v3/projects')) {
        return Promise.resolve({ ok: true, json: async () => ({ projects: [] }) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }) as unknown as typeof fetch
  })

  it('lands on Board by default', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /board/i })).toBeInTheDocument()
    })
  })

  it('supports the Create Work modal tabs', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /^Create Work$/i }).length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getAllByRole('button', { name: /^Create Work$/i })[0])

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /Create Task/i }).length).toBeGreaterThan(0)
      expect(screen.getByRole('button', { name: /Import PRD/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Quick Action/i })).toBeInTheDocument()
    })
  })

  it('refreshes surfaces when websocket events arrive', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(mockedFetch).toHaveBeenCalled()
    })
    const baselineCalls = mockedFetch.mock.calls.length

    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    MockWebSocket.instances[0].dispatch('message')

    await waitFor(() => {
      expect(mockedFetch.mock.calls.length).toBeGreaterThan(baselineCalls)
    })
  })
})
