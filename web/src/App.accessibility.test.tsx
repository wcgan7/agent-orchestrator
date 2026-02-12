import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

class MockWebSocket {
  listeners: Record<string, Array<(event?: unknown) => void>> = {}

  constructor() {
    setTimeout(() => this.emit('open'), 0)
  }

  addEventListener(event: string, cb: (event?: unknown) => void) {
    this.listeners[event] = this.listeners[event] || []
    this.listeners[event].push(cb)
  }

  send() {}

  close() {}

  emit(event: string) {
    for (const cb of this.listeners[event] || []) cb({})
  }
}

function installFetchMock() {
  const projects = [{ id: 'p1', path: '/tmp/repo', source: 'pinned', is_git: true }]
  global.fetch = vi.fn().mockImplementation((url, init) => {
    const u = String(url)
    if (u.includes('/api/v3/tasks/board')) {
      return Promise.resolve({ ok: true, json: async () => ({ columns: { backlog: [], ready: [], in_progress: [], in_review: [], blocked: [], done: [] } }) })
    }
    if (u.includes('/api/v3/orchestrator/status')) {
      return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 1, in_progress: 0, draining: false, run_branch: null }) })
    }
    if (u.includes('/api/v3/review-queue')) {
      return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
    }
    if (u.includes('/api/v3/agents') && (init?.method || 'GET') === 'GET') {
      return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
    }
    if (u.includes('/api/v3/projects') && (init?.method || 'GET') === 'GET') {
      return Promise.resolve({ ok: true, json: async () => ({ projects }) })
    }
    if (u.includes('/api/v3/projects/pinned') && (init?.method || 'GET') === 'POST') {
      projects.push({ id: 'p2', path: '/abs/path', source: 'pinned', is_git: false })
      return Promise.resolve({ ok: true, json: async () => ({ project: { id: 'p2', path: '/abs/path', source: 'pinned', is_git: false } }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  }) as unknown as typeof fetch
}

describe('App navigation and settings flows', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    window.location.hash = ''
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket
    installFetchMock()
  })

  it('keeps core routes navigable via nav buttons', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /board/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Execution/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /execution/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Agents/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /agents/i })).toBeInTheDocument()
    })
  })

  it('pins manual project paths from Settings', async () => {
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Settings/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /settings/i })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Pin project by absolute path/i), {
      target: { value: '/abs/path' },
    })
    fireEvent.click(screen.getByRole('button', { name: /Pin project/i }))

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v3/projects/pinned'),
        expect.objectContaining({ method: 'POST' }),
      )
    })

    await waitFor(() => {
      expect(screen.getAllByRole('option', { name: /abs\/path/i }).length).toBeGreaterThan(0)
    })
  })
})
