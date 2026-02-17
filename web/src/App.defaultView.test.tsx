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

  dispatch(event: string, payload: unknown = {}) {
    for (const cb of this.listeners[event] || []) {
      cb(payload)
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
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({ ok: true, json: async () => ({ columns: { backlog: [], queued: [], in_progress: [], in_review: [], blocked: [], done: [] } }) })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/review-queue')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/agents')) {
        return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
      }
      if (u.includes('/api/projects')) {
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

  it('requests metrics and worker compatibility endpoints during reload', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/metrics'))).toBe(true)
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/workers/health'))).toBe(true)
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/workers/routing'))).toBe(true)
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
        return String(url).includes('/api/tasks') && (init as RequestInit | undefined)?.method === 'POST'
      })
      expect(taskCreateCall).toBeTruthy()
    })

    const taskCreateCall = mockedFetch.mock.calls.find(([url, init]) => {
      return String(url).includes('/api/tasks') && (init as RequestInit | undefined)?.method === 'POST'
    })
    expect(taskCreateCall).toBeTruthy()

    const body = JSON.parse(String((taskCreateCall?.[1] as RequestInit).body))
    expect(body.task_type).toBe('bug')
    expect(body.parent_id).toBe('task-parent-01')
    expect(body.pipeline_template).toEqual(['plan', 'implement', 'verify'])
    expect(body.metadata).toEqual({ area: 'payments' })
  })

  it('opens task detail modal when clicking a kanban card', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    mockedFetch.mockImplementation((url) => {
      const u = String(url)
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            columns: {
              backlog: [{ id: 'task-1', title: 'Task 1', priority: 'P2', status: 'backlog', task_type: 'feature' }],
              queued: [],
              in_progress: [],
              in_review: [],
              blocked: [],
              done: [],
            },
          }),
        })
      }
      if (u.includes('/api/tasks/task-1')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            task: { id: 'task-1', title: 'Task 1', priority: 'P2', status: 'backlog', task_type: 'feature', blocked_by: [], blocks: [] },
          }),
        })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/review-queue')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/agents')) {
        return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
      }
      if (u.includes('/api/projects')) {
        return Promise.resolve({ ok: true, json: async () => ({ projects: [] }) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })

    render(<App />)

    // Click a task card to open the detail modal
    await waitFor(() => {
      expect(screen.getByText('Task 1')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Task 1'))
    await waitFor(() => {
      expect(screen.getByRole('dialog', { name: /Task detail/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /^Queue$/i })).toBeInTheDocument()
    })
  })

  it('board summary strip shows queue depth and worker count', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    mockedFetch.mockImplementation((url) => {
      const u = String(url)
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            columns: {
              backlog: [],
              queued: [{ id: 'task-1', title: 'Task 1', priority: 'P2', status: 'queued', task_type: 'feature' }],
              in_progress: [],
              in_review: [],
              blocked: [],
              done: [],
            },
          }),
        })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 3, in_progress: 1, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/review-queue')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/agents')) {
        return Promise.resolve({ ok: true, json: async () => ({ agents: [{ id: 'agent-1', role: 'general', status: 'running' }] }) })
      }
      if (u.includes('/api/projects')) {
        return Promise.resolve({ ok: true, json: async () => ({ projects: [] }) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText(/Queue: 3/i)).toBeInTheDocument()
      expect(screen.getByText(/In progress: 1/i)).toBeInTheDocument()
      expect(screen.getByText(/Workers: 1/i)).toBeInTheDocument()
    })
  })

  it('loads activity timeline for selected task via modal', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    mockedFetch.mockImplementation((url) => {
      const u = String(url)
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            columns: {
              backlog: [{ id: 'task-1', title: 'Task 1', priority: 'P2', status: 'queued', task_type: 'feature' }],
              queued: [],
              in_progress: [],
              in_review: [],
              blocked: [],
              done: [],
            },
          }),
        })
      }
      if (u.includes('/api/tasks/task-1')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            task: { id: 'task-1', title: 'Task 1', priority: 'P2', status: 'queued', task_type: 'feature', blocked_by: [], blocks: [] },
          }),
        })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/collaboration/timeline/task-1')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            events: [
              {
                id: 'evt-1',
                type: 'task.created',
                timestamp: '2026-02-13T00:00:00Z',
                actor: 'system',
                actor_type: 'system',
                summary: 'Task created',
                details: null,
              },
            ],
          }),
        })
      }
      if (u.includes('/api/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/review-queue')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/agents')) {
        return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
      }
      if (u.includes('/api/projects')) {
        return Promise.resolve({ ok: true, json: async () => ({ projects: [] }) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })

    render(<App />)

    // Click a task card to open the detail modal, then switch to Activity tab
    await waitFor(() => {
      expect(screen.getByText('Task 1')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Task 1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Activity$/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Activity$/i }))

    // Verify timeline endpoint was called
    await waitFor(() => {
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/collaboration/timeline/task-1'))).toBe(true)
    })

    // Verify timeline event is displayed
    await waitFor(() => {
      expect(screen.getByText('Task created')).toBeInTheDocument()
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
    MockWebSocket.instances[0].dispatch('message', { data: JSON.stringify({ channel: 'tasks', type: 'task.updated' }) })

    await waitFor(() => {
      expect(mockedFetch.mock.calls.length).toBeGreaterThan(baselineCalls)
    })
  })

  it('ignores websocket system frames without triggering reload', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(mockedFetch).toHaveBeenCalled()
    })
    const baselineCalls = mockedFetch.mock.calls.length

    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    MockWebSocket.instances[0].dispatch('message', { data: JSON.stringify({ channel: 'system', type: 'subscribed' }) })

    await new Promise((resolve) => setTimeout(resolve, 180))
    expect(mockedFetch.mock.calls.length).toBe(baselineCalls)
  })

  it('coalesces burst websocket task events into a single tasks refresh', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    const boardCallCount = () =>
      mockedFetch.mock.calls.filter(([url]) => String(url).includes('/api/tasks/board')).length

    await waitFor(() => {
      expect(boardCallCount()).toBeGreaterThan(0)
    })
    const baselineBoardCalls = boardCallCount()

    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    MockWebSocket.instances[0].dispatch('message', { data: JSON.stringify({ channel: 'tasks', type: 'task.updated' }) })
    MockWebSocket.instances[0].dispatch('message', { data: JSON.stringify({ channel: 'tasks', type: 'task.updated' }) })
    MockWebSocket.instances[0].dispatch('message', { data: JSON.stringify({ channel: 'tasks', type: 'task.updated' }) })

    await waitFor(() => {
      expect(boardCallCount()).toBeGreaterThan(baselineBoardCalls)
    })
    await new Promise((resolve) => setTimeout(resolve, 260))
    expect(boardCallCount()).toBe(baselineBoardCalls + 1)
  })

  it('ignores websocket events from other projects', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    localStorage.setItem('agent-orchestrator-project', '/tmp/repo-alpha')
    render(<App />)

    const boardCallCount = () =>
      mockedFetch.mock.calls.filter(([url]) => String(url).includes('/api/tasks/board')).length

    await waitFor(() => {
      expect(boardCallCount()).toBeGreaterThan(0)
    })
    const baselineBoardCalls = boardCallCount()

    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    MockWebSocket.instances[0].dispatch('message', {
      data: JSON.stringify({ channel: 'tasks', type: 'task.updated', project_id: 'repo-beta' }),
    })

    await new Promise((resolve) => setTimeout(resolve, 220))
    expect(boardCallCount()).toBe(baselineBoardCalls)
  })

  it('does not re-fetch root metadata on websocket task refreshes', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    const rootCallCount = () =>
      mockedFetch.mock.calls.filter(([url]) => String(url).startsWith('/?') || String(url) === '/').length
    const boardCallCount = () =>
      mockedFetch.mock.calls.filter(([url]) => String(url).includes('/api/tasks/board')).length

    await waitFor(() => {
      expect(rootCallCount()).toBeGreaterThan(0)
    })
    const baselineRootCalls = rootCallCount()
    const baselineBoardCalls = boardCallCount()

    expect(MockWebSocket.instances.length).toBeGreaterThan(0)
    MockWebSocket.instances[0].dispatch('message', { data: JSON.stringify({ channel: 'tasks', type: 'task.updated' }) })

    await waitFor(() => {
      expect(boardCallCount()).toBeGreaterThan(baselineBoardCalls)
    })
    await new Promise((resolve) => setTimeout(resolve, 220))
    expect(rootCallCount()).toBe(baselineRootCalls)
  })
})
