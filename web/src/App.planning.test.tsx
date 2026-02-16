import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  listeners: Record<string, Array<(event?: unknown) => void>> = {}
  url: string

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

describe('Planning panel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    window.location.hash = ''
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket

    const task = {
      id: 'task-1',
      title: 'Task 1',
      description: 'Plan this',
      priority: 'P2',
      status: 'queued',
      task_type: 'feature',
      blocked_by: [],
      blocks: [],
    }

    const planDoc = {
      task_id: 'task-1',
      latest_revision_id: 'pr-2',
      committed_revision_id: 'pr-1',
      revisions: [
        {
          id: 'pr-1',
          task_id: 'task-1',
          created_at: '2026-02-14T00:00:00Z',
          source: 'worker_plan',
          parent_revision_id: null,
          step: 'plan',
          content: 'Initial plan',
          content_hash: 'x',
          status: 'committed',
        },
        {
          id: 'pr-2',
          task_id: 'task-1',
          created_at: '2026-02-14T00:01:00Z',
          source: 'human_edit',
          parent_revision_id: 'pr-1',
          step: 'plan_refine',
          feedback_note: 'manual',
          content: 'Updated plan',
          content_hash: 'y',
          status: 'draft',
        },
      ],
      active_refine_job: null,
      plans: [],
      latest: null,
    }

    global.fetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return Promise.resolve({ ok: true, json: async () => ({ project_id: 'repo-alpha' }) })
      if (u.includes('/api/collaboration/modes')) return Promise.resolve({ ok: true, json: async () => ({ modes: [] }) })
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ columns: { backlog: [task], queued: [], in_progress: [], in_review: [], blocked: [], done: [] } }),
        })
      }
      if (u.includes('/api/tasks/task-1/plan') && method === 'GET') {
        return Promise.resolve({ ok: true, json: async () => planDoc })
      }
      if (u.includes('/api/tasks/task-1/plan/refine') && method === 'POST') {
        return Promise.resolve({ ok: true, json: async () => ({ job: { id: 'prj-1', task_id: 'task-1', base_revision_id: 'pr-2', status: 'queued', feedback: 'tighten scope', created_at: '2026-02-14T00:02:00Z' } }) })
      }
      if (u.includes('/api/tasks/task-1/plan/revisions') && method === 'POST') {
        return Promise.resolve({ ok: true, json: async () => ({ revision: { ...planDoc.revisions[1], id: 'pr-3' } }) })
      }
      if (u.includes('/api/tasks/task-1/plan/commit') && method === 'POST') {
        return Promise.resolve({ ok: true, json: async () => ({ committed_revision_id: 'pr-2' }) })
      }
      if (u.includes('/api/tasks/task-1/generate-tasks') && method === 'POST') {
        return Promise.resolve({ ok: true, json: async () => ({ created_task_ids: ['task-2'] }) })
      }
      if (u.includes('/api/tasks/task-1') && method === 'GET') {
        return Promise.resolve({ ok: true, json: async () => ({ task }) })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [task] }) })
      }
      if (u.includes('/api/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/review-queue')) return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      if (u.includes('/api/agents')) return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
      if (u.includes('/api/projects')) return Promise.resolve({ ok: true, json: async () => ({ projects: [] }) })
      if (u.includes('/api/workers/health')) return Promise.resolve({ ok: true, json: async () => ({ providers: [] }) })
      if (u.includes('/api/workers/routing')) return Promise.resolve({ ok: true, json: async () => ({ default: 'codex', rows: [] }) })
      if (u.includes('/api/quick-actions')) return Promise.resolve({ ok: true, json: async () => ({ quick_actions: [] }) })
      if (u.includes('/api/tasks/execution-order')) return Promise.resolve({ ok: true, json: async () => ({ batches: [] }) })
      if (u.includes('/api/phases')) return Promise.resolve({ ok: true, json: async () => ([]) })
      if (u.includes('/api/metrics')) return Promise.resolve({ ok: true, json: async () => ({}) })
      if (u.includes('/api/collaboration/timeline/task-1')) return Promise.resolve({ ok: true, json: async () => ({ events: [] }) })
      if (u.includes('/api/collaboration/feedback/task-1')) return Promise.resolve({ ok: true, json: async () => ({ feedback: [] }) })
      if (u.includes('/api/collaboration/comments/task-1')) return Promise.resolve({ ok: true, json: async () => ({ comments: [] }) })
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }) as unknown as typeof fetch
  })

  it('shows planning controls and sends source-aware generate payload', async () => {
    const mockedFetch = global.fetch as unknown as ReturnType<typeof vi.fn>
    render(<App />)

    // Navigate to the standalone Planning route via the nav strip
    await waitFor(() => {
      expect(screen.getByRole('navigation', { name: /Main navigation/i })).toBeInTheDocument()
    })
    const navPlanning = screen.getByRole('navigation', { name: /Main navigation/i })
      .querySelector('button')
    // Find the Planning nav pill from the nav strip
    const navButtons = screen.getByRole('navigation', { name: /Main navigation/i }).querySelectorAll('button')
    const planningNavButton = Array.from(navButtons).find((btn) => btn.textContent?.trim() === 'Planning')
    expect(planningNavButton).toBeTruthy()
    fireEvent.click(planningNavButton!)

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /^Planning$/i })).toBeInTheDocument()
    })

    // Select the task from the planning sidebar list
    fireEvent.click(screen.getByRole('button', { name: /Task 1/i }))

    await waitFor(() => {
      expect(screen.getByLabelText(/Request changes from worker/i)).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Request changes from worker/i), { target: { value: 'tighten scope' } })
    fireEvent.click(screen.getByRole('button', { name: /Refine/i }))

    await waitFor(() => {
      const refineCall = mockedFetch.mock.calls.find(([url, init]) => String(url).includes('/api/tasks/task-1/plan/refine') && String((init as RequestInit).method).toUpperCase() === 'POST')
      expect(refineCall).toBeTruthy()
      const body = JSON.parse(String((refineCall?.[1] as RequestInit).body))
      expect(body.feedback).toBe('tighten scope')
    })

    fireEvent.change(screen.getByLabelText(/Generate tasks from/i), { target: { value: 'revision' } })
    fireEvent.change(screen.getByLabelText(/Generate from revision/i), { target: { value: 'pr-2' } })
    fireEvent.click(screen.getByRole('button', { name: /Generate Tasks/i }))

    await waitFor(() => {
      const generateCall = mockedFetch.mock.calls.find(([url, init]) => String(url).includes('/api/tasks/task-1/generate-tasks') && String((init as RequestInit).method).toUpperCase() === 'POST')
      expect(generateCall).toBeTruthy()
      const body = JSON.parse(String((generateCall?.[1] as RequestInit).body))
      expect(body.source).toBe('revision')
      expect(body.revision_id).toBe('pr-2')
      expect(body.infer_deps).toBe(true)
    })
  })
})

describe('Planning worker output', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    window.location.hash = ''
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket

    const task = {
      id: 'task-1',
      title: 'Task 1',
      description: 'Plan this',
      priority: 'P2',
      status: 'in_progress',
      task_type: 'feature',
      blocked_by: [],
      blocks: [],
    }

    const planDoc = {
      task_id: 'task-1',
      latest_revision_id: 'pr-2',
      committed_revision_id: 'pr-1',
      revisions: [
        {
          id: 'pr-1',
          task_id: 'task-1',
          created_at: '2026-02-14T00:00:00Z',
          source: 'worker_plan',
          parent_revision_id: null,
          step: 'plan',
          content: 'Initial plan',
          content_hash: 'x',
          status: 'committed',
        },
        {
          id: 'pr-2',
          task_id: 'task-1',
          created_at: '2026-02-14T00:01:00Z',
          source: 'worker_refine',
          parent_revision_id: 'pr-1',
          step: 'plan_refine',
          feedback_note: 'manual',
          content: 'Updated plan',
          content_hash: 'y',
          status: 'draft',
        },
      ],
      active_refine_job: {
        id: 'prj-1',
        task_id: 'task-1',
        base_revision_id: 'pr-2',
        status: 'running',
        created_at: '2026-02-14T00:02:00Z',
        started_at: '2026-02-14T00:02:10Z',
        feedback: 'tighten scope',
      },
      plans: [],
      latest: null,
    }

    const introText = "I'll research the browser background behavior constraints, especially on iPad/iOS, to refine the plan with accurate technical details."
    const introEvent = JSON.stringify({
      type: 'stream_event',
      event: {
        type: 'content_block_start',
        content_block: {
          type: 'text',
          text: introText,
        },
      },
    })
    const toolEvent = JSON.stringify({
      type: 'stream_event',
      event: {
        type: 'content_block_start',
        content_block: {
          type: 'tool_use',
          name: 'Task',
        },
      },
    })
    const deltaEvent = JSON.stringify({
      type: 'stream_event',
      event: {
        type: 'content_block_delta',
        delta: {
          type: 'text_delta',
          text: '\nKey constraints identified.',
        },
      },
    })
    const refinePollStdout = [
      introEvent,
      `-7151-4e37-b4a8-e888b870c7a0\"}\n${introEvent}\n${toolEvent}\n${deltaEvent}`,
    ]
    let refinePollCount = 0

    global.fetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return Promise.resolve({ ok: true, json: async () => ({ project_id: 'repo-alpha' }) })
      if (u.includes('/api/collaboration/modes')) return Promise.resolve({ ok: true, json: async () => ({ modes: [] }) })
      if (u.includes('/api/tasks/board')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ columns: { backlog: [task], queued: [], in_progress: [], in_review: [], blocked: [], done: [] } }),
        })
      }
      if (u.includes('/api/tasks/task-1/plan') && method === 'GET') {
        return Promise.resolve({ ok: true, json: async () => planDoc })
      }
      if (u.includes('/api/tasks/task-1/logs?max_chars=50000')) {
        const idx = Math.min(refinePollCount, refinePollStdout.length - 1)
        refinePollCount += 1
        return Promise.resolve({
          ok: true,
          json: async () => ({
            mode: 'active',
            step: 'plan_refine',
            stdout: refinePollStdout[idx],
            stderr: '',
            started_at: '2026-02-14T00:02:10Z',
          }),
        })
      }
      if (u.includes('/api/tasks/task-1/logs')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            mode: 'active',
            step: 'plan_refine',
            stdout: '',
            stderr: '',
            started_at: '2026-02-14T00:02:10Z',
          }),
        })
      }
      if (u.includes('/api/tasks/task-1') && method === 'GET') {
        return Promise.resolve({ ok: true, json: async () => ({ task }) })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) {
        return Promise.resolve({ ok: true, json: async () => ({ tasks: [task] }) })
      }
      if (u.includes('/api/orchestrator/status')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'running', queue_depth: 0, in_progress: 1, draining: false, run_branch: null }) })
      }
      if (u.includes('/api/review-queue')) return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
      if (u.includes('/api/agents')) return Promise.resolve({ ok: true, json: async () => ({ agents: [] }) })
      if (u.includes('/api/projects')) return Promise.resolve({ ok: true, json: async () => ({ projects: [] }) })
      if (u.includes('/api/workers/health')) return Promise.resolve({ ok: true, json: async () => ({ providers: [] }) })
      if (u.includes('/api/workers/routing')) return Promise.resolve({ ok: true, json: async () => ({ default: 'codex', rows: [] }) })
      if (u.includes('/api/quick-actions')) return Promise.resolve({ ok: true, json: async () => ({ quick_actions: [] }) })
      if (u.includes('/api/tasks/execution-order')) return Promise.resolve({ ok: true, json: async () => ({ batches: [] }) })
      if (u.includes('/api/phases')) return Promise.resolve({ ok: true, json: async () => ([]) })
      if (u.includes('/api/metrics')) return Promise.resolve({ ok: true, json: async () => ({}) })
      if (u.includes('/api/collaboration/timeline/task-1')) return Promise.resolve({ ok: true, json: async () => ({ events: [] }) })
      if (u.includes('/api/collaboration/feedback/task-1')) return Promise.resolve({ ok: true, json: async () => ({ feedback: [] }) })
      if (u.includes('/api/collaboration/comments/task-1')) return Promise.resolve({ ok: true, json: async () => ({ comments: [] }) })
      return Promise.resolve({ ok: true, json: async () => ({}) })
    }) as unknown as typeof fetch
  })

  it('keeps earlier worker output while refine logs update', async () => {
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('navigation', { name: /Main navigation/i })).toBeInTheDocument()
    })
    const navButtons = screen.getByRole('navigation', { name: /Main navigation/i }).querySelectorAll('button')
    const planningNavButton = Array.from(navButtons).find((btn) => btn.textContent?.trim() === 'Planning')
    expect(planningNavButton).toBeTruthy()
    fireEvent.click(planningNavButton!)
    fireEvent.click(screen.getByRole('button', { name: /Task 1/i }))

    await waitFor(() => {
      const panel = document.querySelector('pre.planning-worker-output')
      expect(panel?.textContent || '').toContain('I\'ll research the browser background behavior constraints')
    })

    await new Promise((resolve) => window.setTimeout(resolve, 2_100))

    await waitFor(() => {
      const panel = document.querySelector('pre.planning-worker-output')
      const text = panel?.textContent || ''
      expect(text).toContain('I\'ll research the browser background behavior constraints')
      expect(text).toContain('Key constraints identified.')
      expect(text).toContain('Tools used: Task')
      expect(text).not.toContain('-7151-4e37-b4a8-e888b870c7a0')
    })

    fireEvent.click(screen.getByRole('button', { name: /Manual Revision/i }))
    await waitFor(() => {
      const textarea = screen.getByPlaceholderText(/Paste or edit full plan text/i) as HTMLTextAreaElement
      expect(textarea.value).toContain('Initial plan')
    })
  }, 15_000)
})
