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
        return Promise.resolve({ ok: true, json: async () => ({ columns: { backlog: [], ready: [], in_progress: [], in_review: [], blocked: [], done: [] } }) })
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

  it('requests metrics and agent type compatibility endpoints during reload', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/metrics'))).toBe(true)
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/agents/types'))).toBe(true)
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

  it('task explorer supports only blocked filter via GET /tasks', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText(/Task Explorer/i)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByLabelText(/Only blocked/i))

    await waitFor(() => {
      const blockedTasksCall = mockedFetch.mock.calls.find(([url, init]) => {
        return String(url).includes('/api/tasks') &&
          !String(url).includes('/api/tasks/') &&
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
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({ ok: true, json: async () => ({ columns: { backlog: [], ready: [], in_progress: [], in_review: [], blocked: [], done: [] } }) })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
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

  it('loads collaboration endpoints for selected task and submits feedback/comments', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    mockedFetch.mockImplementation((url, init) => {
      const u = String(url)
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            columns: {
              backlog: [{ id: 'task-1', title: 'Task 1', priority: 'P2', status: 'ready', task_type: 'feature' }],
              ready: [],
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
            task: { id: 'task-1', title: 'Task 1', priority: 'P2', status: 'ready', task_type: 'feature', blocked_by: [], blocks: [] },
          }),
        })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      }
      if (u.includes('/api/collaboration/timeline/task-1')) {
        return Promise.resolve({ ok: true, json: async () => ({ events: [] }) })
      }
      if (u.includes('/api/collaboration/feedback/task-1')) {
        return Promise.resolve({ ok: true, json: async () => ({ feedback: [] }) })
      }
      if (u.includes('/api/collaboration/comments/task-1')) {
        return Promise.resolve({ ok: true, json: async () => ({ comments: [] }) })
      }
      if (u.includes('/api/collaboration/feedback') && (init as RequestInit | undefined)?.method === 'POST') {
        return Promise.resolve({ ok: true, json: async () => ({ feedback: { id: 'fb-1' } }) })
      }
      if (
        u.includes('/api/collaboration/comments') &&
        !u.includes('/resolve') &&
        (init as RequestInit | undefined)?.method === 'POST'
      ) {
        return Promise.resolve({ ok: true, json: async () => ({ comment: { id: 'cm-1' } }) })
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

    await waitFor(() => {
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/collaboration/timeline/task-1'))).toBe(true)
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/collaboration/feedback/task-1'))).toBe(true)
      expect(mockedFetch.mock.calls.some(([url]) => String(url).includes('/api/collaboration/comments/task-1'))).toBe(true)
    })

    fireEvent.change(screen.getByLabelText(/Summary/i), { target: { value: 'Please tighten validation.' } })
    fireEvent.click(screen.getByRole('button', { name: /Add feedback/i }))

    await waitFor(() => {
      const feedbackCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/collaboration/feedback') &&
        (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(feedbackCall).toBeTruthy()
    })

    fireEvent.change(screen.getByLabelText(/File path/i), { target: { value: 'src/App.tsx' } })
    fireEvent.change(screen.getByLabelText(/Comment/i), { target: { value: 'Looks good, but add tests.' } })
    fireEvent.click(screen.getByRole('button', { name: /Add comment/i }))

    await waitFor(() => {
      const commentCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/collaboration/comments') &&
        !String(url).includes('/resolve') &&
        (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(commentCall).toBeTruthy()
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
