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

function installFetchMock(options?: {
  promptOverrides?: Record<string, string>
  promptInjections?: Record<string, string>
  clearConflictOnce?: boolean
}) {
  const jsonResponse = (payload: unknown) =>
    Promise.resolve({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => payload,
    })
  const jsonErrorResponse = (status: number, statusText: string, payload: unknown) =>
    Promise.resolve({
      ok: false,
      status,
      statusText,
      json: async () => payload,
    })

  let task = {
    id: 'task-1',
    title: 'Task 1',
    description: 'Ship task controls',
    priority: 'P2',
    status: 'queued',
    task_type: 'feature',
    labels: ['ui'],
    blocked_by: ['task-0'],
    blocks: ['task-2'],
    pending_gate: 'human_review',
    human_blocking_issues: [
      {
        summary: 'Need production API token',
        details: 'Grant read-only credentials for staging',
        action: 'Provide token',
      },
    ],
        hitl_mode: 'autopilot',
  }
  let terminalTask = {
    id: 'task-d1',
    title: 'Terminal task',
    description: 'Ready to delete',
    priority: 'P2',
    status: 'done',
    task_type: 'feature',
    labels: [],
    blocked_by: [],
    blocks: [],
  }
  let boardCleared = false
  let clearConflictEmitted = false
  let terminalDeleted = false

  const taskR1 = {
    id: 'task-r1',
    title: 'Review me',
    description: 'Review this task',
    priority: 'P2',
    status: 'in_review',
    task_type: 'feature',
    labels: [],
    blocked_by: [],
    blocks: [],
  }

  const settingsPayload = {
    orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
    agent_routing: {
      default_role: 'general',
      task_type_roles: {},
      role_provider_overrides: {},
    },
    defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 } },
    workers: {
      default: 'codex',
      default_model: '',
      routing: {},
      providers: {
        codex: { type: 'codex', command: 'codex exec' },
        claude: { type: 'claude', command: 'claude -p', model: 'sonnet', reasoning_effort: 'medium' },
      },
    },
    project: {
      commands: {},
      prompt_overrides: { ...(options?.promptOverrides || {}) },
      prompt_injections: { ...(options?.promptInjections || {}) },
      prompt_defaults: {
        implement: 'Implement the task completely and safely.',
        verify: 'Validate the implementation thoroughly.',
      },
    },
  }

  const mockedFetch = vi.fn().mockImplementation((url, init) => {
    const u = String(url)
    const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()

    if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
    if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })

    if (u.includes('/api/tasks/task-1/run') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/tasks/task-1/retry') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/tasks/task-1/cancel') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/tasks/task-1/transition') && method === 'POST') {
      const transitionBody = JSON.parse(String((init as RequestInit).body))
      if (transitionBody.status) {
        task = { ...task, status: transitionBody.status }
      }
      return jsonResponse({ task })
    }
    if (u.includes('/api/tasks/task-1/dependencies/task-0') && method === 'DELETE') return jsonResponse({ task })
    if (u.includes('/api/tasks/task-1/dependencies') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/tasks/analyze-dependencies') && method === 'POST') {
      return jsonResponse({ edges: [{ from: 'task-0', to: 'task-1' }] })
    }
    if (u.includes('/api/tasks/task-1/reset-dep-analysis') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/tasks/task-1/approve-gate') && method === 'POST') {
      task = {
        ...task,
        pending_gate: undefined,
        human_blocking_issues: [],
      }
      return jsonResponse({
        task,
        message: 'Approved Human Review. Task will resume shortly.',
        approved_at: '2026-02-22T00:00:00Z',
      })
    }
    if (u.includes('/api/tasks/task-d1') && method === 'DELETE') {
      terminalDeleted = true
      return jsonResponse({ deleted: true, task_id: 'task-d1' })
    }
    if (u.includes('/api/tasks/clear') && method === 'POST') {
      if (options?.clearConflictOnce && !clearConflictEmitted && !u.includes('force=true')) {
        clearConflictEmitted = true
        return jsonErrorResponse(409, 'Conflict', {
          detail: {
            detail: 'Cannot clear tasks while execution is still active. Retry with force=true.',
            code: 'active_execution',
            data: {
              active_execution: {
                count: 1,
                task_ids: ['task-1'],
                task_reasons: { 'task-1': ['active_future'] },
              },
            },
          },
        })
      }
      boardCleared = true
      return jsonResponse({
        cleared: true,
        archived_to: '/tmp/repo-alpha/.agent_orchestrator_archive/state_20260221T120000Z',
        message: 'Cleared all tasks. Archived previous runtime state to /tmp/repo-alpha/.agent_orchestrator_archive/state_20260221T120000Z.',
      })
    }
    if (u.includes('/api/tasks/task-1') && method === 'PATCH') return jsonResponse({ task })
    if (u.includes('/api/tasks') && !u.includes('/api/tasks/') && method === 'POST') return jsonResponse({ task })

    if (u.includes('/api/orchestrator/control') && method === 'POST') {
      return jsonResponse({ status: 'running', queue_depth: 1, in_progress: 0, draining: false, run_branch: null })
    }
    if (u.includes('/api/review/task-r1/approve') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/review/task-r1/request-changes') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/workers/health') && method === 'GET') {
      return jsonResponse({
        providers: [
          { name: 'codex', type: 'codex', configured: true, healthy: true, status: 'connected', detail: 'ok', checked_at: '2026-02-14T00:00:00Z', command: 'codex exec' },
          { name: 'claude', type: 'claude', configured: true, healthy: true, status: 'connected', detail: 'ok', checked_at: '2026-02-14T00:00:00Z', command: 'claude -p', model: 'sonnet' },
          { name: 'ollama', type: 'ollama', configured: false, healthy: false, status: 'not_configured', detail: 'Provider is not configured.', checked_at: '2026-02-14T00:00:00Z' },
        ],
      })
    }
    if (u.includes('/api/workers/routing') && method === 'GET') {
      return jsonResponse({
        default: 'codex',
        rows: [
          { step: 'plan', provider: 'claude', source: 'explicit', configured: true },
          { step: 'implement', provider: 'codex', source: 'default', configured: true },
          { step: 'review', provider: 'claude', source: 'explicit', configured: true },
        ],
      })
    }

    if (u.includes('/api/settings') && method === 'GET') return jsonResponse(settingsPayload)
    if (u.includes('/api/settings') && method === 'PATCH') return jsonResponse(settingsPayload)
    if (u.includes('/api/projects/pinned/pinned-1') && method === 'DELETE') return jsonResponse({ removed: true })
    if (u.includes('/api/projects/pinned') && method === 'GET') {
      return jsonResponse({ items: [{ id: 'pinned-1', path: '/tmp/repo-alpha', source: 'pinned', is_git: true }] })
    }
    if (u.includes('/api/projects/pinned') && method === 'POST') {
      return jsonResponse({ project: { id: 'pinned-2', path: '/tmp/repo-beta', source: 'pinned', is_git: true } })
    }

    if (u.includes('/api/terminal/session/') && u.includes('/logs') && method === 'GET') {
      return jsonResponse({ output: '', offset: 0, status: 'running', finished_at: null })
    }
    if (u.includes('/api/terminal/session/') && method === 'POST') {
      return jsonResponse({ session: { id: 'term-1', status: 'running', shell: '/bin/zsh', cwd: '/tmp/repo-alpha' } })
    }
    if (u.includes('/api/terminal/session') && method === 'POST') {
      return jsonResponse({ session: { id: 'term-1', status: 'running', shell: '/bin/zsh', cwd: '/tmp/repo-alpha' } })
    }
    if (u.includes('/api/terminal/session') && method === 'GET') {
      return jsonResponse({ session: null })
    }

    if (u.includes('/api/import/prd/preview') && method === 'POST') {
      return jsonResponse({
        job_id: 'job-1',
        preview: {
          nodes: [{ id: 'task-a', title: 'Task A', priority: 'P2' }],
          edges: [],
        },
      })
    }
    if (u.includes('/api/import/prd/commit') && method === 'POST') {
      return jsonResponse({ created_task_ids: ['task-a'] })
    }
    if (u.includes('/api/import/job-1') && method === 'GET') {
      return jsonResponse({
        job: {
          id: 'job-1',
          status: 'preview_ready',
          title: 'Import PRD',
          created_task_ids: ['task-a'],
          tasks: [{ title: 'Task A', priority: 'P2' }],
        },
      })
    }

    if (u.includes('/api/tasks/board')) {
      return jsonResponse({
        columns: {
          backlog: boardCleared ? [] : [task],
          queued: [],
          in_progress: [],
          in_review: boardCleared ? [] : [taskR1],
          blocked: [],
          done: boardCleared || terminalDeleted ? [] : [terminalTask],
          cancelled: [],
        },
      })
    }
    if (u.includes('/api/tasks/execution-order')) return jsonResponse({ batches: [['task-1']] })
    if (u.includes('/api/tasks/task-1') && method === 'GET') return jsonResponse({ task })
    if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
    if (u.includes('/api/orchestrator/status')) {
      return jsonResponse({ status: 'running', queue_depth: 1, in_progress: 0, draining: false, run_branch: null })
    }
    if (u.includes('/api/review-queue')) {
      return jsonResponse({
        tasks: [{ id: 'task-r1', title: 'Review me', priority: 'P2', status: 'in_review', task_type: 'feature' }],
      })
    }
    if (u.includes('/api/agents/types')) {
      return jsonResponse({ types: [{ role: 'general', display_name: 'General', task_type_affinity: [], allowed_steps: [] }] })
    }
    if (u.includes('/api/agents') && method === 'GET') {
      return jsonResponse({ agents: [{ id: 'agent-1', role: 'general', status: 'running' }] })
    }
    if (u.includes('/api/projects') && method === 'GET') {
      return jsonResponse({ projects: [{ id: 'repo-alpha', path: '/tmp/repo-alpha', source: 'workspace', is_git: true }] })
    }
    if (u.includes('/api/phases')) return jsonResponse([])
    if (u.includes('/api/collaboration/presence')) return jsonResponse({ users: [] })
    if (u.includes('/api/metrics')) {
      return jsonResponse({ api_calls: 1, wall_time_seconds: 1, phases_completed: 0, phases_total: 0, tokens_used: 10, estimated_cost_usd: 0.01 })
    }
    if (u.includes('/api/collaboration/timeline/task-1')) {
      return jsonResponse({
        events: [
          {
            id: 'evt-1',
            type: 'task.gate_waiting',
            timestamp: '2026-02-13T00:00:00Z',
            actor: 'system',
            actor_type: 'system',
            summary: 'task.gate_waiting',
            details: 'Need human intervention',
            human_blocking_issues: [{ summary: 'Need production API token' }],
          },
        ],
      })
    }
    if (u.includes('/api/collaboration/feedback/task-1')) return jsonResponse({ feedback: [] })
    if (u.includes('/api/collaboration/comments/task-1')) return jsonResponse({ comments: [] })

    if (u.includes('/api/tasks/task-r1') && method === 'GET') return jsonResponse({ task: taskR1 })
    if (u.includes('/api/tasks/task-d1') && method === 'GET') return jsonResponse({ task: terminalTask })
    if (u.includes('/api/collaboration/timeline/task-r1')) return jsonResponse({ events: [] })
    if (u.includes('/api/collaboration/feedback/task-r1')) return jsonResponse({ feedback: [] })
    if (u.includes('/api/collaboration/comments/task-r1')) return jsonResponse({ comments: [] })

    return jsonResponse({})
  })

  global.fetch = mockedFetch as unknown as typeof fetch
  return mockedFetch
}

describe('App action coverage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    window.location.hash = ''
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    MockWebSocket.instances = []
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket
  })

  it('executes task detail controls from the board route', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    // Click a task card to open the detail modal
    await waitFor(() => {
      expect(screen.getByText('Task 1')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Task 1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Move to Backlog/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Approve gate/i })).toBeInTheDocument()
      expect(screen.getByText('Need production API token')).toBeInTheDocument()
    })

    // Use the "Move to Backlog" status action button (task is queued, so this transitions to backlog)
    fireEvent.click(screen.getByRole('button', { name: /Move to Backlog/i }))
    await waitFor(() => {
      const transitionCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/tasks/task-1/transition') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(transitionCall).toBeTruthy()
      const body = JSON.parse(String((transitionCall?.[1] as RequestInit).body))
      expect(body.status).toBe('backlog')
    })

    fireEvent.click(screen.getByRole('button', { name: /Dependencies/i }))
    fireEvent.change(screen.getByLabelText(/Add dependency task ID/i), { target: { value: 'task-99' } })
    fireEvent.click(screen.getByRole('button', { name: /Add dependency/i }))
    await waitFor(() => {
      const addDepCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/tasks/task-1/dependencies') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(addDepCall).toBeTruthy()
      const body = JSON.parse(String((addDepCall?.[1] as RequestInit).body))
      expect(body.depends_on).toBe('task-99')
    })

    fireEvent.click(screen.getByRole('button', { name: /Remove/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/task-1/dependencies/task-0') && (init as RequestInit | undefined)?.method === 'DELETE'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Analyze dependencies/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/analyze-dependencies') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Reset inferred deps/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/task-1/reset-dep-analysis') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /^Overview$/i }))
    fireEvent.click(screen.getByRole('button', { name: /Approve gate/i }))
    await waitFor(() => {
      const gateCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/tasks/task-1/approve-gate') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(gateCall).toBeTruthy()
      const body = JSON.parse(String((gateCall?.[1] as RequestInit).body))
      expect(body.gate).toBe('human_review')
    })

    // Switch to Activity tab (formerly Collaboration)
    fireEvent.click(screen.getByRole('button', { name: /^Activity$/i }))
    // Activity timeline loads via a separate async request; give it its own waitFor window.
    await waitFor(() => {
      expect(screen.getByText(/Required human input/i)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /^Configuration$/i }))
    fireEvent.change(screen.getByLabelText(/Labels/i), { target: { value: 'ui,frontend' } })
    fireEvent.click(screen.getByRole('button', { name: /^Save$/i }))
    await waitFor(() => {
      const editCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/tasks/task-1') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(editCall).toBeTruthy()
      const body = JSON.parse(String((editCall?.[1] as RequestInit).body))
      expect(body.labels).toEqual(['ui', 'frontend'])
    })
  }, 30000)

  it('deletes terminal tasks and clears board with archive notice', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Terminal task')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Terminal task'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Delete$/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Delete$/i }))

    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/task-d1') && (init as RequestInit | undefined)?.method === 'DELETE'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /^Clear All Tasks$/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/clear') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })
    await waitFor(() => {
      expect(screen.getByText(/Archived previous runtime state to/i)).toBeInTheDocument()
    })
  })

  it('offers force clear when clear is blocked by active execution', async () => {
    const mockedFetch = installFetchMock({ clearConflictOnce: true })
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Clear All Tasks$/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /^Clear All Tasks$/i }))

    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/clear') &&
          !String(url).includes('force=true') &&
          (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })

    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/tasks/clear?') &&
          String(url).includes('force=true') &&
          String(url).includes('timeout_seconds=30') &&
          (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })
  })

  it('executes execution, review, and worker dashboard actions', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Execution/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Execution/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Execution/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Pause Queue$/i }))

    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) => {
          if (!String(url).includes('/api/orchestrator/control')) return false
          if ((init as RequestInit | undefined)?.method !== 'POST') return false
          const body = JSON.parse(String((init as RequestInit).body))
          return body.action === 'pause'
        })
      ).toBe(true)
    })

    // Navigate to Board and open an in_review task to test review actions
    fireEvent.click(screen.getByRole('button', { name: /Board/i }))
    await waitFor(() => {
      expect(screen.getByText('Review me')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Review me'))
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Guidance for changes/i)).toBeInTheDocument()
    })
    fireEvent.change(screen.getByPlaceholderText(/Guidance for changes/i), { target: { value: 'Looks solid.' } })
    fireEvent.click(screen.getByRole('button', { name: /^Request Changes$/i }))

    await waitFor(() => {
      const reviewCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/review/task-r1/request-changes') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(reviewCall).toBeTruthy()
      const body = JSON.parse(String((reviewCall?.[1] as RequestInit).body))
      expect(body.guidance).toBe('Looks solid.')
    })

    fireEvent.click(screen.getByRole('button', { name: /Workers/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /^Workers$/i })).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/workers/health') && (init as RequestInit | undefined)?.method === undefined
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Recheck providers/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.filter(([url]) => String(url).includes('/api/workers/health')).length
      ).toBeGreaterThan(1)
    })
  })

  it('saves settings payload and unpins projects', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Settings/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Settings/i })).toBeInTheDocument()
      expect(screen.getByLabelText(/Orchestrator concurrency/i)).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Orchestrator concurrency/i), { target: { value: '4' } })
    fireEvent.click(screen.getByLabelText(/Auto dependency analysis/i))
    fireEvent.change(screen.getByLabelText(/Max review attempts/i), { target: { value: '5' } })
    fireEvent.change(screen.getByLabelText(/Default role/i), { target: { value: 'reviewer' } })
    fireEvent.change(screen.getByLabelText(/Task type role map/i), { target: { value: '{"bug":"debugger"}' } })
    fireEvent.change(screen.getByLabelText(/Role provider overrides/i), { target: { value: '{"reviewer":"codex"}' } })
    fireEvent.change(screen.getByLabelText(/Default worker provider/i), { target: { value: 'claude' } })
    fireEvent.change(screen.getByLabelText(/Configure provider/i), { target: { value: 'codex' } })
    fireEvent.change(screen.getByLabelText(/Codex command/i), { target: { value: 'codex exec' } })
    fireEvent.change(screen.getByLabelText(/Codex model/i), { target: { value: 'gpt-5-codex' } })
    fireEvent.change(screen.getByLabelText(/Codex effort/i), { target: { value: 'high' } })
    fireEvent.change(screen.getByLabelText(/Configure provider/i), { target: { value: 'ollama' } })
    fireEvent.change(screen.getByLabelText(/Ollama endpoint/i), { target: { value: 'http://localhost:11434' } })
    fireEvent.change(screen.getByLabelText(/Ollama model/i), { target: { value: 'llama3.1:8b' } })
    fireEvent.change(screen.getByLabelText(/Configure provider/i), { target: { value: 'claude' } })
    fireEvent.change(screen.getByLabelText(/Claude command/i), { target: { value: 'claude -p' } })
    fireEvent.change(screen.getByLabelText(/Claude model/i), { target: { value: 'sonnet' } })
    fireEvent.change(screen.getByLabelText(/Claude effort/i), { target: { value: 'high' } })
    fireEvent.change(
      screen.getByLabelText(/Project commands by language/i),
      { target: { value: '{"python":{"test":"pytest -n auto","lint":"ruff check ."}}' } }
    )
    fireEvent.change(screen.getByLabelText(/Pipeline step/i), { target: { value: 'implement' } })
    fireEvent.change(screen.getByLabelText(/Override prompt \(optional\)/i), { target: { value: 'Custom implement prompt' } })
    fireEvent.change(screen.getByLabelText(/Quality gate critical/i), { target: { value: '1' } })
    fireEvent.change(screen.getByLabelText(/Quality gate high/i), { target: { value: '2' } })
    fireEvent.change(screen.getByLabelText(/Quality gate medium/i), { target: { value: '3' } })
    fireEvent.change(screen.getByLabelText(/Quality gate low/i), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: /Save settings/i }))

    await waitFor(() => {
      const settingsCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/settings') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(settingsCall).toBeTruthy()
      const body = JSON.parse(String((settingsCall?.[1] as RequestInit).body))
      expect(body.orchestrator.concurrency).toBe(4)
      expect(body.orchestrator.auto_deps).toBe(false)
      expect(body.orchestrator.max_review_attempts).toBe(5)
      expect(body.orchestrator.step_timeout_seconds).toBe(600)
      expect(body.defaults.quality_gate).toEqual({ critical: 1, high: 2, medium: 3, low: 4 })
      expect(body.agent_routing.task_type_roles).toEqual({ bug: 'debugger' })
      expect(body.workers.default).toBe('claude')
      expect(body.workers.default_model).toBe('')
      expect(body.workers.routing).toEqual({ plan: 'claude', review: 'claude' })
      expect(body.workers.providers.codex).toEqual({
        type: 'codex',
        command: 'codex exec',
        model: 'gpt-5-codex',
        reasoning_effort: 'high',
      })
      expect(body.workers.providers.ollama).toEqual({
        type: 'ollama',
        endpoint: 'http://localhost:11434',
        model: 'llama3.1:8b',
      })
      expect(body.workers.providers.claude).toEqual({
        type: 'claude',
        command: 'claude -p',
        model: 'sonnet',
        reasoning_effort: 'high',
      })
      expect(body.project.commands.python.test).toBe('pytest -n auto')
      expect(body.project.commands.python.lint).toBe('ruff check .')
      expect(body.project.prompt_overrides.implement).toBe('Custom implement prompt')
      expect(body.project.prompt_injections).toEqual({})
    })

    fireEvent.click(screen.getByRole('button', { name: /Unpin/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/projects/pinned/pinned-1') && (init as RequestInit | undefined)?.method === 'DELETE'
        )
      ).toBe(true)
    })
  })

  it('sends empty override value when clearing an existing step prompt override', async () => {
    const mockedFetch = installFetchMock({ promptOverrides: { implement: 'Existing implement override' } })
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Settings/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Settings/i })).toBeInTheDocument()
      expect(screen.getByLabelText(/Pipeline step/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/Override prompt \(optional\)/i)).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Pipeline step/i), { target: { value: 'implement' } })
    fireEvent.click(screen.getByRole('button', { name: /Clear override/i }))
    fireEvent.click(screen.getByRole('button', { name: /Save settings/i }))

    await waitFor(() => {
      const settingsCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/settings') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(settingsCall).toBeTruthy()
      const body = JSON.parse(String((settingsCall?.[1] as RequestInit).body))
      expect(body.project.prompt_overrides.implement).toBe('')
    })
  })

  it('sends empty override value when removing existing override text in the editor', async () => {
    const mockedFetch = installFetchMock({ promptOverrides: { implement: 'Existing implement override' } })
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Settings/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Settings/i })).toBeInTheDocument()
      expect(screen.getByLabelText(/Pipeline step/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/Override prompt \(optional\)/i)).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Pipeline step/i), { target: { value: 'implement' } })
    fireEvent.change(screen.getByLabelText(/Override prompt \(optional\)/i), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /Save settings/i }))

    await waitFor(() => {
      const settingsCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/settings') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(settingsCall).toBeTruthy()
      const body = JSON.parse(String((settingsCall?.[1] as RequestInit).body))
      expect(body.project.prompt_overrides.implement).toBe('')
    })
  })

  it('migrates existing step prompt injection into override and clears legacy injection', async () => {
    const mockedFetch = installFetchMock({ promptInjections: { implement: 'Existing implement injection' } })
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Settings/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Settings/i })).toBeInTheDocument()
      expect(screen.getByLabelText(/Pipeline step/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/Override prompt \(optional\)/i)).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Pipeline step/i), { target: { value: 'implement' } })
    fireEvent.click(screen.getByRole('button', { name: /Save settings/i }))

    await waitFor(() => {
      const settingsCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/settings') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(settingsCall).toBeTruthy()
      const body = JSON.parse(String((settingsCall?.[1] as RequestInit).body))
      expect(body.project.prompt_overrides.implement).toBe('Existing implement injection')
      expect(body.project.prompt_injections.implement).toBe('')
    })
  })

  it('submits task with worker model override', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /^Create Work$/i }).length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getAllByRole('button', { name: /^Create Work$/i })[0])
    fireEvent.change(screen.getByLabelText(/^Title$/i), { target: { value: 'Implement checkout' } })
    fireEvent.change(screen.getByLabelText(/Task Type/i), { target: { value: 'feature' } })
    fireEvent.change(screen.getByLabelText(/Worker model override/i), { target: { value: 'gpt-5-codex' } })
    fireEvent.click(screen.getByRole('button', { name: /Create & Queue/i }))

    await waitFor(() => {
      const taskCreateCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/tasks') &&
        !String(url).includes('/api/tasks/') &&
        (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(taskCreateCall).toBeTruthy()
      const body = JSON.parse(String((taskCreateCall?.[1] as RequestInit).body))
      expect(body.title).toBe('Implement checkout')
      expect(body.worker_model).toBe('gpt-5-codex')
    })
  })

  it('runs terminal and import modal workflows', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /^Create Work$/i }).length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getByRole('button', { name: /Toggle terminal/i }))

    await waitFor(() => {
      const startCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/terminal/session') &&
        (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(startCall).toBeTruthy()
    })

    fireEvent.click(screen.getAllByRole('button', { name: /^Create Work$/i })[0])
    fireEvent.click(screen.getByRole('button', { name: /Import PRD/i }))
    fireEvent.change(screen.getByLabelText(/PRD text/i), { target: { value: '- Task A' } })
    fireEvent.click(screen.getByRole('button', { name: /^Preview$/i }))

    await waitFor(() => {
      const previewCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/import/prd/preview') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(previewCall).toBeTruthy()
      const body = JSON.parse(String((previewCall?.[1] as RequestInit).body))
      expect(body.content).toBe('- Task A')
    })

    fireEvent.click(screen.getByRole('button', { name: /Commit to board/i }))
    await waitFor(() => {
      const commitCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/import/prd/commit') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(commitCall).toBeTruthy()
      const body = JSON.parse(String((commitCall?.[1] as RequestInit).body))
      expect(body.job_id).toBe('job-1')
    })
  })

  it('treats already-approved gate as silent no-op', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      const task = {
        id: 'task-1',
        title: 'Task 1',
        description: 'Ship task controls',
        priority: 'P2',
        status: 'queued',
        task_type: 'feature',
        labels: ['ui'],
        blocked_by: [],
        blocks: [],
        pending_gate: 'human_review',
                hitl_mode: 'autopilot',
      }
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({ columns: { backlog: [], queued: [task], in_progress: [], in_review: [], blocked: [], done: [] } })
      }
      if (u.includes('/api/tasks/task-1') && method === 'GET') return jsonResponse({ task })
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
      if (u.includes('/api/tasks/task-1/approve-gate') && method === 'POST') {
        return Promise.resolve({
          ok: false,
          status: 400,
          statusText: 'Bad Request',
          json: async () => ({ detail: 'No pending gate on this task' }),
        })
      }
      if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 1, in_progress: 0, draining: false, run_branch: null })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/settings')) return jsonResponse({
        orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
        agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
        defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 }, dependency_policy: 'prudent' },
        workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
        project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
      })
      if (u.includes('/api/tasks/task-1/plan')) return jsonResponse({ task_id: 'task-1', revisions: [] })
      if (u.includes('/api/tasks/task-1/plan/jobs')) return jsonResponse({ jobs: [] })
      if (u.includes('/api/tasks/task-1/workdoc')) return jsonResponse({ task_id: 'task-1', content: '', exists: false })
      if (u.includes('/api/collaboration/timeline/task-1')) return jsonResponse({ events: [] })
      if (u.includes('/api/collaboration/feedback/task-1')) return jsonResponse({ feedback: [] })
      if (u.includes('/api/collaboration/comments/task-1')) return jsonResponse({ comments: [] })
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Task 1')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Task 1'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Approve gate/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /Approve gate/i }))
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /Approve gate/i })).not.toBeInTheDocument()
      expect(screen.queryByText('Gate already approved.')).not.toBeInTheDocument()
    })
  })

  it('restores gate CTA when approve request fails', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      const task = {
        id: 'task-2',
        title: 'Task 2',
        description: 'Finalize commit',
        priority: 'P2',
        status: 'in_review',
        task_type: 'feature',
        labels: ['ui'],
        blocked_by: [],
        blocks: [],
        pending_gate: null,
                hitl_mode: 'review_only',
      }
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({ columns: { backlog: [], queued: [], in_progress: [], in_review: [task], blocked: [], done: [] } })
      }
      if (u.includes('/api/tasks/task-2') && method === 'GET') return jsonResponse({ task })
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
      if (u.includes('/api/review/task-2/approve') && method === 'POST') {
        return Promise.resolve({
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
          json: async () => ({ detail: 'Approval service timeout' }),
        })
      }
      if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 0, in_progress: 1, draining: false, run_branch: null })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/settings')) return jsonResponse({
        orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
        agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
        defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 }, dependency_policy: 'prudent' },
        workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
        project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
      })
      if (u.includes('/api/tasks/task-2/plan')) return jsonResponse({ task_id: 'task-2', revisions: [] })
      if (u.includes('/api/tasks/task-2/plan/jobs')) return jsonResponse({ jobs: [] })
      if (u.includes('/api/tasks/task-2/workdoc')) return jsonResponse({ task_id: 'task-2', content: '', exists: false })
      if (u.includes('/api/collaboration/timeline/task-2')) return jsonResponse({ events: [] })
      if (u.includes('/api/collaboration/feedback/task-2')) return jsonResponse({ feedback: [] })
      if (u.includes('/api/collaboration/comments/task-2')) return jsonResponse({ comments: [] })
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Task 2')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Task 2'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Approve$/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /^Approve$/i }))
    await waitFor(() => {
      expect(screen.getAllByText(/500 Internal Server Error/i).length).toBeGreaterThan(0)
    })
    expect(screen.getByRole('button', { name: /^Approve$/i })).toBeInTheDocument()
  })

  it('refresh reattaches scheduler when status reports stale attachment', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    let repaired = false
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({
          columns: { backlog: [], queued: [], in_progress: [], in_review: [], blocked: [], done: [], cancelled: [] },
        })
      }
      if (u.includes('/api/tasks/execution-order')) return jsonResponse({ batches: [], completed: [] })
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [] })
      if (u.includes('/api/orchestrator/status')) {
        return jsonResponse({
          status: 'running',
          queue_depth: 1,
          in_progress: 0,
          draining: false,
          run_branch: null,
          scheduler_attached: repaired,
          scheduler_stale: !repaired,
          tick_lag_seconds: repaired ? 0 : 120,
        })
      }
      if (u.includes('/api/orchestrator/control') && method === 'POST') {
        const body = JSON.parse(String((init as RequestInit).body || '{}'))
        if (body.action === 'reset') repaired = true
        return jsonResponse({
          status: 'running',
          queue_depth: 1,
          in_progress: 0,
          draining: false,
          run_branch: null,
          scheduler_attached: true,
          scheduler_stale: false,
          tick_lag_seconds: 0,
        })
      }
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/review-queue')) return jsonResponse({ tasks: [] })
      if (u.includes('/api/phases')) return jsonResponse([])
      if (u.includes('/api/collaboration/presence')) return jsonResponse({ users: [] })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/settings')) {
        return jsonResponse({
          orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
          agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
          defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 }, dependency_policy: 'prudent' },
          workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
          project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
        })
      }
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Refresh$/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Refresh$/i }))

    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) => {
          if (!String(url).includes('/api/orchestrator/control')) return false
          if ((init as RequestInit | undefined)?.method !== 'POST') return false
          const body = JSON.parse(String((init as RequestInit).body || '{}'))
          return body.action === 'reset'
        }),
      ).toBe(true)
    })
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) => {
          if (!String(url).includes('/api/orchestrator/control')) return false
          if ((init as RequestInit | undefined)?.method !== 'POST') return false
          const body = JSON.parse(String((init as RequestInit).body || '{}'))
          return body.action === 'reconcile'
        }),
      ).toBe(true)
    })
    await waitFor(() => {
      expect(screen.getByText('Scheduler reattached and reconciled.')).toBeInTheDocument()
    })
  })

  it('shows context-missing message in changes tab when task-scoped diff is unavailable', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    const task = {
      id: 'task-ctx',
      title: 'Contextless review task',
      description: 'Needs human review but has no scoped context',
      priority: 'P2',
      status: 'blocked',
      task_type: 'feature',
      labels: [],
      blocked_by: [],
      blocks: [],
      hitl_mode: 'review_only',
    }
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({
          columns: { backlog: [], queued: [], in_progress: [], in_review: [], blocked: [task], done: [], cancelled: [] },
        })
      }
      if (u.includes('/api/tasks/execution-order')) return jsonResponse({ batches: [], completed: [] })
      if (u.includes('/api/tasks/task-ctx/changes')) {
        return jsonResponse({ mode: 'none', reason: 'task_context_missing', commit: null, files: [], diff: '', stat: '' })
      }
      if (u.includes('/api/tasks/task-ctx') && method === 'GET') return jsonResponse({ task })
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
      if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/review-queue')) return jsonResponse({ tasks: [] })
      if (u.includes('/api/phases')) return jsonResponse([])
      if (u.includes('/api/collaboration/presence')) return jsonResponse({ users: [] })
      if (u.includes('/api/collaboration/timeline/task-ctx')) return jsonResponse({ events: [] })
      if (u.includes('/api/collaboration/feedback/task-ctx')) return jsonResponse({ feedback: [] })
      if (u.includes('/api/collaboration/comments/task-ctx')) return jsonResponse({ comments: [] })
      if (u.includes('/api/tasks/task-ctx/plan')) return jsonResponse({ task_id: 'task-ctx', revisions: [] })
      if (u.includes('/api/tasks/task-ctx/plan/jobs')) return jsonResponse({ jobs: [] })
      if (u.includes('/api/tasks/task-ctx/workdoc')) return jsonResponse({ task_id: 'task-ctx', content: '', exists: false })
      if (u.includes('/api/settings')) {
        return jsonResponse({
          orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
          agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
          defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 }, dependency_policy: 'prudent' },
          workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
          project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
        })
      }
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Contextless review task')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Contextless review task'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Changes$/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Changes$/i }))

    await waitFor(() => {
      expect(screen.getByText('No task-scoped changes available yet. Request changes to rerun implementation.')).toBeInTheDocument()
    })
    expect(screen.queryByText('Working tree changes')).not.toBeInTheDocument()
  })

  it('shows provenance and warning banner for low-confidence preserved-branch diff', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    const task = {
      id: 'task-diffwarn',
      title: 'Low-confidence diff task',
      description: 'Uses preserved branch fallback',
      priority: 'P2',
      status: 'blocked',
      task_type: 'feature',
      labels: [],
      blocked_by: [],
      blocks: [],
      hitl_mode: 'review_only',
    }
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({
          columns: { backlog: [], queued: [], in_progress: [], in_review: [], blocked: [task], done: [], cancelled: [] },
        })
      }
      if (u.includes('/api/tasks/execution-order')) return jsonResponse({ batches: [], completed: [] })
      if (u.includes('/api/tasks/task-diffwarn/changes')) {
        return jsonResponse({
          mode: 'preserved_branch',
          commit: null,
          branch: 'task-task-diffwarn',
          base_branch: 'main',
          base_ref: 'main',
          base_sha: 'aaaaaaaaaaaa1111111111111111111111111111',
          head_ref: 'task-task-diffwarn',
          head_sha: 'bbbbbbbbbbbb2222222222222222222222222222',
          base_source: 'heuristic',
          confidence: 'low',
          warnings: ['heuristic_base_inferred', 'large_file_count'],
          files: [{ path: 'tracked.txt', changes: '1 +' }],
          diff: 'diff --git a/tracked.txt b/tracked.txt\n--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1,2 @@\n base\n+change\n',
          stat: ' tracked.txt | 1 +\n 1 file changed, 1 insertion(+)\n',
        })
      }
      if (u.includes('/api/tasks/task-diffwarn') && method === 'GET') return jsonResponse({ task })
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
      if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 0, in_progress: 0, draining: false, run_branch: null })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/review-queue')) return jsonResponse({ tasks: [] })
      if (u.includes('/api/phases')) return jsonResponse([])
      if (u.includes('/api/collaboration/presence')) return jsonResponse({ users: [] })
      if (u.includes('/api/collaboration/timeline/task-diffwarn')) return jsonResponse({ events: [] })
      if (u.includes('/api/collaboration/feedback/task-diffwarn')) return jsonResponse({ feedback: [] })
      if (u.includes('/api/collaboration/comments/task-diffwarn')) return jsonResponse({ comments: [] })
      if (u.includes('/api/tasks/task-diffwarn/plan')) return jsonResponse({ task_id: 'task-diffwarn', revisions: [] })
      if (u.includes('/api/tasks/task-diffwarn/plan/jobs')) return jsonResponse({ jobs: [] })
      if (u.includes('/api/tasks/task-diffwarn/workdoc')) return jsonResponse({ task_id: 'task-diffwarn', content: '', exists: false })
      if (u.includes('/api/settings')) {
        return jsonResponse({
          orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
          agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
          defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 }, dependency_policy: 'prudent' },
          workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
          project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
        })
      }
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Low-confidence diff task')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Low-confidence diff task'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Changes$/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Changes$/i }))

    await waitFor(() => {
      expect(screen.getByText(/Low-confidence diff:/i)).toBeInTheDocument()
    })
    expect(screen.getByText(/Provenance:/i)).toBeInTheDocument()
    expect(screen.getByText(/Base branch was inferred heuristically\./i)).toBeInTheDocument()
  })

  it('sends generation policy when approving before_generate_tasks gate', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    const task = {
      id: 'task-gen-1',
      title: 'Generate gate task',
      description: 'Decompose initiative',
      priority: 'P1',
      status: 'in_progress',
      task_type: 'initiative_plan',
      labels: [],
      blocked_by: [],
      blocks: [],
      pending_gate: 'before_generate_tasks',
      pipeline_template: ['analyze', 'initiative_plan', 'generate_tasks'],
      hitl_mode: 'supervised',
      metadata: {
        task_generation_defaults: {
          child_status: 'queued',
          child_hitl_mode: 'supervised',
          infer_deps: false,
        },
      },
    }
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({ columns: { backlog: [], queued: [], in_progress: [task], in_review: [], blocked: [], done: [], cancelled: [] } })
      }
      if (u.includes('/api/tasks/task-gen-1') && method === 'GET') return jsonResponse({ task })
      if (u.includes('/api/tasks/task-gen-1/approve-gate') && method === 'POST') {
        return jsonResponse({ task: { ...task, pending_gate: null }, message: 'Task generation approved. Task will resume shortly.' })
      }
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
      if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 0, in_progress: 1, draining: false, run_branch: null })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/review-queue')) return jsonResponse({ tasks: [] })
      if (u.includes('/api/phases')) return jsonResponse([])
      if (u.includes('/api/collaboration/presence')) return jsonResponse({ users: [] })
      if (u.includes('/api/collaboration/timeline/task-gen-1')) return jsonResponse({ events: [] })
      if (u.includes('/api/collaboration/feedback/task-gen-1')) return jsonResponse({ feedback: [] })
      if (u.includes('/api/collaboration/comments/task-gen-1')) return jsonResponse({ comments: [] })
      if (u.includes('/api/tasks/task-gen-1/plan')) return jsonResponse({ task_id: 'task-gen-1', revisions: [] })
      if (u.includes('/api/tasks/task-gen-1/plan/jobs')) return jsonResponse({ jobs: [] })
      if (u.includes('/api/tasks/task-gen-1/workdoc')) return jsonResponse({ task_id: 'task-gen-1', content: '', exists: false })
      if (u.includes('/api/settings')) {
        return jsonResponse({
          orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
          agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
          defaults: {
            quality_gate: { critical: 0, high: 0, medium: 0, low: 0 },
            dependency_policy: 'prudent',
            hitl_mode: 'autopilot',
          },
          workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
          project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
        })
      }
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Generate gate task')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Generate gate task'))
    await waitFor(() => {
      expect(screen.getByText(/Plan ready to generate tasks/i)).toBeInTheDocument()
    })
    expect(screen.queryByText(/Policy:/i)).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText(/^Start$/i), { target: { value: 'backlog' } })
    fireEvent.click(screen.getByRole('button', { name: /Supervised/i }))
    fireEvent.click(screen.getByRole('option', { name: /Review Only/i }))
    fireEvent.click(screen.getByRole('button', { name: /Generate tasks/i }))

    await waitFor(() => {
      const call = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/tasks/task-gen-1/approve-gate') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(call).toBeTruthy()
      const body = JSON.parse(String((call?.[1] as RequestInit).body))
      expect(body.generation_policy.child_status).toBe('backlog')
      expect(body.generation_policy.child_hitl_mode).toBe('review_only')
      expect(typeof body.generation_policy.infer_deps).toBe('boolean')
      expect(body.save_generation_policy_as_default).toBe(false)
    })
  })

  it('does not render generation controls for incompatible before_generate_tasks gate tasks', async () => {
    const jsonResponse = (payload: unknown) => Promise.resolve({ ok: true, json: async () => payload })
    const task = {
      id: 'task-gen-2',
      title: 'Incompatible generate gate',
      description: 'Feature task with inconsistent gate',
      priority: 'P2',
      status: 'in_progress',
      task_type: 'feature',
      labels: [],
      blocked_by: [],
      blocks: [],
      pending_gate: 'before_generate_tasks',
      pipeline_template: ['plan', 'implement', 'verify', 'review', 'commit'],
      hitl_mode: 'autopilot',
    }
    const mockedFetch = vi.fn().mockImplementation((url, init) => {
      const u = String(url)
      const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()
      if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
      if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
      if (u.includes('/api/tasks/board')) {
        return jsonResponse({ columns: { backlog: [], queued: [], in_progress: [task], in_review: [], blocked: [], done: [], cancelled: [] } })
      }
      if (u.includes('/api/tasks/task-gen-2') && method === 'GET') return jsonResponse({ task })
      if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
      if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 0, in_progress: 1, draining: false, run_branch: null })
      if (u.includes('/api/metrics')) return jsonResponse({})
      if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
      if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
      if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
      if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
      if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
      if (u.includes('/api/review-queue')) return jsonResponse({ tasks: [] })
      if (u.includes('/api/phases')) return jsonResponse([])
      if (u.includes('/api/collaboration/presence')) return jsonResponse({ users: [] })
      if (u.includes('/api/collaboration/timeline/task-gen-2')) return jsonResponse({ events: [] })
      if (u.includes('/api/collaboration/feedback/task-gen-2')) return jsonResponse({ feedback: [] })
      if (u.includes('/api/collaboration/comments/task-gen-2')) return jsonResponse({ comments: [] })
      if (u.includes('/api/tasks/task-gen-2/plan')) return jsonResponse({ task_id: 'task-gen-2', revisions: [] })
      if (u.includes('/api/tasks/task-gen-2/plan/jobs')) return jsonResponse({ jobs: [] })
      if (u.includes('/api/tasks/task-gen-2/workdoc')) return jsonResponse({ task_id: 'task-gen-2', content: '', exists: false })
      if (u.includes('/api/settings')) {
        return jsonResponse({
          orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 10, step_timeout_seconds: 600 },
          agent_routing: { default_role: 'general', task_type_roles: {}, role_provider_overrides: {} },
          defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 }, dependency_policy: 'prudent', hitl_mode: 'autopilot' },
          workers: { default: 'codex', default_model: '', routing: {}, providers: { codex: { type: 'codex', command: 'codex exec' } } },
          project: { commands: {}, prompt_overrides: {}, prompt_injections: {}, prompt_defaults: {} },
        })
      }
      return jsonResponse({})
    })
    global.fetch = mockedFetch as unknown as typeof fetch

    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Incompatible generate gate')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Incompatible generate gate'))
    await waitFor(() => {
      expect(screen.getByText(/This pipeline does not support task generation\./i)).toBeInTheDocument()
    })
    expect(screen.queryByText(/Save these settings as this task's generation defaults/i)).not.toBeInTheDocument()
  })
})
