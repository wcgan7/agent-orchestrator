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
      if (u.includes('/api/v3/tasks') && !u.includes('/api/v3/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
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

  it('submits task_type and advanced create fields from Create Task form', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /^Create Work$/i }).length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getAllByRole('button', { name: /^Create Work$/i })[0])

    const titleInput = screen.getByLabelText(/Title/i)
    fireEvent.change(titleInput, { target: { value: 'Review payment flow' } })
    fireEvent.change(screen.getByLabelText(/Task Type/i), { target: { value: 'bug' } })
    fireEvent.change(screen.getByLabelText(/Parent task ID/i), { target: { value: 'task-parent-01' } })
    fireEvent.change(screen.getByLabelText(/Pipeline template steps/i), { target: { value: 'plan, implement, verify' } })
    fireEvent.change(screen.getByLabelText(/Metadata JSON object/i), { target: { value: '{"area":"payments"}' } })
    fireEvent.submit(titleInput.closest('form') as HTMLFormElement)

    await waitFor(() => {
      const taskCreateCall = mockedFetch.mock.calls.find(([url, init]) => {
        return String(url).includes('/api/v3/tasks') && (init as RequestInit | undefined)?.method === 'POST'
      })
      expect(taskCreateCall).toBeTruthy()
    })

    const taskCreateCall = mockedFetch.mock.calls.find(([url, init]) => {
      return String(url).includes('/api/v3/tasks') && (init as RequestInit | undefined)?.method === 'POST'
    })
    expect(taskCreateCall).toBeTruthy()

    const body = JSON.parse(String((taskCreateCall?.[1] as RequestInit).body))
    expect(body.task_type).toBe('bug')
    expect(body.parent_id).toBe('task-parent-01')
    expect(body.pipeline_template).toEqual(['plan', 'implement', 'verify'])
    expect(body.metadata).toEqual({ area: 'payments' })
  })

  it('task explorer supports only blocked filter via GET /tasks', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText(/Task Explorer/i)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByLabelText(/Only blocked/i))

    await waitFor(() => {
      const blockedTasksCall = mockedFetch.mock.calls.find(([url, init]) => {
        return String(url).includes('/api/v3/tasks') &&
          !String(url).includes('/api/v3/tasks/') &&
          String(url).includes('status=blocked') &&
          ((init as RequestInit | undefined)?.method || 'GET') === 'GET'
      })
      expect(blockedTasksCall).toBeTruthy()
    })
  })

  it('task explorer paginates results', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    mockedFetch.mockImplementation((url) => {
      const u = String(url)
      if (u.includes('/api/v3/tasks/board')) {
        return Promise.resolve({ ok: true, json: async () => ({ columns: { backlog: [], ready: [], in_progress: [], in_review: [], blocked: [], done: [] } }) })
      }
      if (u.includes('/api/v3/tasks') && !u.includes('/api/v3/tasks/')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            tasks: Array.from({ length: 7 }).map((_, index) => ({
              id: `task-${index + 1}`,
              title: `Task ${index + 1}`,
              priority: 'P2',
              status: 'ready',
            })),
          }),
        })
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
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText(/Page 1 of 2/i)).toBeInTheDocument()
    })
    expect(screen.queryByText('Task 7')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /^Next$/i }))

    await waitFor(() => {
      expect(screen.getByText(/Page 2 of 2/i)).toBeInTheDocument()
      expect(screen.getByText('Task 7')).toBeInTheDocument()
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
