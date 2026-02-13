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

function installFetchMock() {
  const jsonResponse = (payload: unknown) =>
    Promise.resolve({
      ok: true,
      json: async () => payload,
    })

  const task = {
    id: 'task-1',
    title: 'Task 1',
    description: 'Ship task controls',
    priority: 'P2',
    status: 'ready',
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
    approval_mode: 'human_review',
    hitl_mode: 'autopilot',
  }

  const settingsPayload = {
    orchestrator: { concurrency: 2, auto_deps: true, max_review_attempts: 3 },
    agent_routing: {
      default_role: 'general',
      task_type_roles: {},
      role_provider_overrides: {},
    },
    defaults: { quality_gate: { critical: 0, high: 0, medium: 0, low: 0 } },
    workers: {
      default: 'codex',
      routing: {},
      providers: { codex: { type: 'codex', command: 'codex' } },
    },
    project: {
      commands: {},
    },
  }

  const quickActions = [
    {
      id: 'qa-1',
      prompt: 'Summarize deploy logs',
      status: 'completed',
      kind: 'agent',
      result_summary: 'All checks passed.',
      promoted_task_id: null,
      started_at: '2026-02-13T00:00:00Z',
      finished_at: '2026-02-13T00:00:01Z',
    },
  ]

  const mockedFetch = vi.fn().mockImplementation((url, init) => {
    const u = String(url)
    const method = String((init as RequestInit | undefined)?.method || 'GET').toUpperCase()

    if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
    if (u.includes('/api/v3/collaboration/modes')) return jsonResponse({ modes: [] })

    if (u.includes('/api/v3/tasks/task-1/run') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1/retry') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1/cancel') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1/transition') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1/dependencies/task-0') && method === 'DELETE') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1/dependencies') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/analyze-dependencies') && method === 'POST') {
      return jsonResponse({ edges: [{ from: 'task-0', to: 'task-1' }] })
    }
    if (u.includes('/api/v3/tasks/task-1/reset-dep-analysis') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1/approve-gate') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks/task-1') && method === 'PATCH') return jsonResponse({ task })

    if (u.includes('/api/v3/orchestrator/control') && method === 'POST') {
      return jsonResponse({ status: 'running', queue_depth: 1, in_progress: 0, draining: false, run_branch: null })
    }
    if (u.includes('/api/v3/review/task-r1/approve') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/review/task-r1/request-changes') && method === 'POST') return jsonResponse({ task })
    if (u.includes('/api/v3/agents/spawn') && method === 'POST') {
      return jsonResponse({ agent: { id: 'agent-2', role: 'general', status: 'running' } })
    }
    if (u.includes('/api/v3/agents/agent-1/terminate') && method === 'POST') {
      return jsonResponse({ agent: { id: 'agent-1', role: 'general', status: 'terminated' } })
    }

    if (u.includes('/api/v3/settings') && method === 'GET') return jsonResponse(settingsPayload)
    if (u.includes('/api/v3/settings') && method === 'PATCH') return jsonResponse(settingsPayload)
    if (u.includes('/api/v3/projects/pinned/pinned-1') && method === 'DELETE') return jsonResponse({ removed: true })
    if (u.includes('/api/v3/projects/pinned') && method === 'GET') {
      return jsonResponse({ items: [{ id: 'pinned-1', path: '/tmp/repo-alpha', source: 'pinned', is_git: true }] })
    }
    if (u.includes('/api/v3/projects/pinned') && method === 'POST') {
      return jsonResponse({ project: { id: 'pinned-2', path: '/tmp/repo-beta', source: 'pinned', is_git: true } })
    }

    if (u.includes('/api/v3/quick-actions/qa-1/promote') && method === 'POST') return jsonResponse({ task, already_promoted: false })
    if (u.includes('/api/v3/quick-actions/qa-2/promote') && method === 'POST') return jsonResponse({ task, already_promoted: false })
    if (u.includes('/api/v3/quick-actions/') && method === 'GET') {
      const quickActionId = u.split('/api/v3/quick-actions/')[1]?.split('?')[0] || 'qa-1'
      return jsonResponse({
        quick_action: {
          id: quickActionId,
          prompt: 'triage errors',
          status: 'completed',
          result_summary: 'done',
          promoted_task_id: null,
          kind: 'agent',
        },
      })
    }
    if (u.includes('/api/v3/quick-actions') && method === 'POST') {
      return jsonResponse({
        quick_action: {
          id: 'qa-2',
          prompt: 'triage errors',
          status: 'queued',
          result_summary: null,
          promoted_task_id: null,
        },
      })
    }
    if (u.includes('/api/v3/quick-actions') && method === 'GET') return jsonResponse({ quick_actions: quickActions })

    if (u.includes('/api/v3/import/prd/preview') && method === 'POST') {
      return jsonResponse({
        job_id: 'job-1',
        preview: {
          nodes: [{ id: 'task-a', title: 'Task A', priority: 'P2' }],
          edges: [],
        },
      })
    }
    if (u.includes('/api/v3/import/prd/commit') && method === 'POST') {
      return jsonResponse({ created_task_ids: ['task-a'] })
    }
    if (u.includes('/api/v3/import/job-1') && method === 'GET') {
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

    if (u.includes('/api/v3/tasks/board')) {
      return jsonResponse({
        columns: {
          backlog: [task],
          ready: [],
          in_progress: [],
          in_review: [],
          blocked: [],
          done: [],
        },
      })
    }
    if (u.includes('/api/v3/tasks/execution-order')) return jsonResponse({ batches: [['task-1']] })
    if (u.includes('/api/v3/tasks/task-1') && method === 'GET') return jsonResponse({ task })
    if (u.includes('/api/v3/tasks') && !u.includes('/api/v3/tasks/')) return jsonResponse({ tasks: [task] })
    if (u.includes('/api/v3/orchestrator/status')) {
      return jsonResponse({ status: 'running', queue_depth: 1, in_progress: 0, draining: false, run_branch: null })
    }
    if (u.includes('/api/v3/review-queue')) {
      return jsonResponse({
        tasks: [{ id: 'task-r1', title: 'Review me', priority: 'P2', status: 'in_review', task_type: 'feature' }],
      })
    }
    if (u.includes('/api/v3/agents/types')) {
      return jsonResponse({ types: [{ role: 'general', display_name: 'General', task_type_affinity: [], allowed_steps: [] }] })
    }
    if (u.includes('/api/v3/agents') && method === 'GET') {
      return jsonResponse({ agents: [{ id: 'agent-1', role: 'general', status: 'running' }] })
    }
    if (u.includes('/api/v3/projects') && method === 'GET') {
      return jsonResponse({ projects: [{ id: 'repo-alpha', path: '/tmp/repo-alpha', source: 'workspace', is_git: true }] })
    }
    if (u.includes('/api/v3/phases')) return jsonResponse([])
    if (u.includes('/api/v3/collaboration/presence')) return jsonResponse({ users: [] })
    if (u.includes('/api/v3/metrics')) {
      return jsonResponse({ api_calls: 1, wall_time_seconds: 1, phases_completed: 0, phases_total: 0, tokens_used: 10, estimated_cost_usd: 0.01 })
    }
    if (u.includes('/api/v3/collaboration/timeline/task-1')) {
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
    if (u.includes('/api/v3/collaboration/feedback/task-1')) return jsonResponse({ feedback: [] })
    if (u.includes('/api/v3/collaboration/comments/task-1')) return jsonResponse({ comments: [] })

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
    MockWebSocket.instances = []
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket
  })

  it('executes task detail controls from the board route', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /^Run$/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Approve gate/i })).toBeInTheDocument()
      expect(screen.getByText('Need production API token')).toBeInTheDocument()
    })
    // Collaboration timeline loads via a separate async request; give it its own waitFor window
    await waitFor(() => {
      expect(screen.getByText(/Required human input/i)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /^Run$/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/v3/tasks/task-1/run') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })

    const transitionButton = screen.getByRole('button', { name: /Transition/i })
    const transitionSelect = transitionButton.closest('div')?.querySelector('select') as HTMLSelectElement | null
    expect(transitionSelect).toBeTruthy()
    fireEvent.change(transitionSelect as HTMLSelectElement, { target: { value: 'in_progress' } })
    fireEvent.click(screen.getByRole('button', { name: /Transition/i }))
    await waitFor(() => {
      const transitionCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/tasks/task-1/transition') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(transitionCall).toBeTruthy()
      const body = JSON.parse(String((transitionCall?.[1] as RequestInit).body))
      expect(body.status).toBe('in_progress')
    })

    fireEvent.change(screen.getByLabelText(/Add blocker task ID/i), { target: { value: 'task-99' } })
    fireEvent.click(screen.getByRole('button', { name: /Add dependency/i }))
    await waitFor(() => {
      const addDepCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/tasks/task-1/dependencies') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(addDepCall).toBeTruthy()
      const body = JSON.parse(String((addDepCall?.[1] as RequestInit).body))
      expect(body.depends_on).toBe('task-99')
    })

    fireEvent.click(screen.getByRole('button', { name: /Remove/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/v3/tasks/task-1/dependencies/task-0') && (init as RequestInit | undefined)?.method === 'DELETE'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Analyze dependencies/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/v3/tasks/analyze-dependencies') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Reset inferred deps/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/v3/tasks/task-1/reset-dep-analysis') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Approve gate/i }))
    await waitFor(() => {
      const gateCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/tasks/task-1/approve-gate') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(gateCall).toBeTruthy()
      const body = JSON.parse(String((gateCall?.[1] as RequestInit).body))
      expect(body.gate).toBe('human_review')
    })

    fireEvent.change(screen.getByLabelText(/Edit title/i), { target: { value: 'Task 1 revised' } })
    fireEvent.click(screen.getByRole('button', { name: /Save edits/i }))
    await waitFor(() => {
      const editCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/tasks/task-1') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(editCall).toBeTruthy()
      const body = JSON.parse(String((editCall?.[1] as RequestInit).body))
      expect(body.title).toBe('Task 1 revised')
    })
  }, 15000)

  it('executes execution, review, and agent control actions', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Execution/i })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Execution/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Execution/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: /^Pause$/i }))

    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) => {
          if (!String(url).includes('/api/v3/orchestrator/control')) return false
          if ((init as RequestInit | undefined)?.method !== 'POST') return false
          const body = JSON.parse(String((init as RequestInit).body))
          return body.action === 'pause'
        })
      ).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: /Review/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Review Queue/i })).toBeInTheDocument()
    })
    fireEvent.change(screen.getByLabelText(/Optional review guidance/i), { target: { value: 'Looks solid.' } })
    fireEvent.click(screen.getByRole('button', { name: /^Approve$/i }))

    await waitFor(() => {
      const reviewCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/review/task-r1/approve') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(reviewCall).toBeTruthy()
      const body = JSON.parse(String((reviewCall?.[1] as RequestInit).body))
      expect(body.guidance).toBe('Looks solid.')
    })

    fireEvent.click(screen.getByRole('button', { name: /Agents/i }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /^Agents$/i })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText(/Spawn capacity/i), { target: { value: '2' } })
    fireEvent.change(screen.getByLabelText(/Provider override/i), { target: { value: 'openai' } })
    fireEvent.click(screen.getByRole('button', { name: /Spawn agent/i }))

    await waitFor(() => {
      const spawnCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/agents/spawn') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(spawnCall).toBeTruthy()
      const body = JSON.parse(String((spawnCall?.[1] as RequestInit).body))
      expect(body.capacity).toBe(2)
      expect(body.override_provider).toBe('openai')
    })

    fireEvent.click(screen.getByRole('button', { name: /^Terminate$/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/v3/agents/agent-1/terminate') && (init as RequestInit | undefined)?.method === 'POST'
        )
      ).toBe(true)
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
    fireEvent.change(screen.getByLabelText(/Worker routing map/i), { target: { value: '{"review":"codex"}' } })
    fireEvent.change(
      screen.getByLabelText(/Worker providers/i),
      { target: { value: '{"codex":{"type":"codex","command":"codex"}}' } }
    )
    fireEvent.change(
      screen.getByLabelText(/Project commands by language/i),
      { target: { value: '{"python":{"test":"pytest -n auto","lint":"ruff check ."}}' } }
    )
    fireEvent.change(screen.getByLabelText(/Quality gate critical/i), { target: { value: '1' } })
    fireEvent.change(screen.getByLabelText(/Quality gate high/i), { target: { value: '2' } })
    fireEvent.change(screen.getByLabelText(/Quality gate medium/i), { target: { value: '3' } })
    fireEvent.change(screen.getByLabelText(/Quality gate low/i), { target: { value: '4' } })
    fireEvent.click(screen.getByRole('button', { name: /Save settings/i }))

    await waitFor(() => {
      const settingsCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/settings') && (init as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(settingsCall).toBeTruthy()
      const body = JSON.parse(String((settingsCall?.[1] as RequestInit).body))
      expect(body.orchestrator.concurrency).toBe(4)
      expect(body.orchestrator.auto_deps).toBe(false)
      expect(body.orchestrator.max_review_attempts).toBe(5)
      expect(body.defaults.quality_gate).toEqual({ critical: 1, high: 2, medium: 3, low: 4 })
      expect(body.agent_routing.task_type_roles).toEqual({ bug: 'debugger' })
      expect(body.workers.routing).toEqual({ review: 'codex' })
      expect(body.project.commands.python.test).toBe('pytest -n auto')
      expect(body.project.commands.python.lint).toBe('ruff check .')
    })

    fireEvent.click(screen.getByRole('button', { name: /Unpin/i }))
    await waitFor(() => {
      expect(
        mockedFetch.mock.calls.some(([url, init]) =>
          String(url).includes('/api/v3/projects/pinned/pinned-1') && (init as RequestInit | undefined)?.method === 'DELETE'
        )
      ).toBe(true)
    })
  })

  it('runs quick action and import modal workflows', async () => {
    const mockedFetch = installFetchMock()
    render(<App />)

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /^Create Work$/i }).length).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getAllByRole('button', { name: /^Create Work$/i })[0])
    fireEvent.click(screen.getByRole('button', { name: /Quick Action/i }))

    fireEvent.click(screen.getByRole('button', { name: /^Promote$/i }))
    await waitFor(() => {
      const promoteCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/quick-actions/qa-1/promote') &&
        (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(promoteCall).toBeTruthy()
      const body = JSON.parse(String((promoteCall?.[1] as RequestInit).body))
      expect(body.priority).toBe('P2')
    })

    fireEvent.change(screen.getByLabelText(/Prompt/i), { target: { value: 'triage errors' } })
    fireEvent.click(screen.getByRole('button', { name: /Run Quick Action/i }))

    await waitFor(() => {
      const quickActionCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/quick-actions') &&
        !String(url).includes('/promote') &&
        (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(quickActionCall).toBeTruthy()
      const body = JSON.parse(String((quickActionCall?.[1] as RequestInit).body))
      expect(body.prompt).toBe('triage errors')
    })

    fireEvent.click(screen.getAllByRole('button', { name: /^Create Work$/i })[0])
    fireEvent.click(screen.getByRole('button', { name: /Import PRD/i }))
    fireEvent.change(screen.getByLabelText(/PRD text/i), { target: { value: '- Task A' } })
    fireEvent.click(screen.getByRole('button', { name: /^Preview$/i }))

    await waitFor(() => {
      const previewCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/import/prd/preview') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(previewCall).toBeTruthy()
      const body = JSON.parse(String((previewCall?.[1] as RequestInit).body))
      expect(body.content).toBe('- Task A')
    })

    fireEvent.click(screen.getByRole('button', { name: /Commit to board/i }))
    await waitFor(() => {
      const commitCall = mockedFetch.mock.calls.find(([url, init]) =>
        String(url).includes('/api/v3/import/prd/commit') && (init as RequestInit | undefined)?.method === 'POST'
      )
      expect(commitCall).toBeTruthy()
      const body = JSON.parse(String((commitCall?.[1] as RequestInit).body))
      expect(body.job_id).toBe('job-1')
    })
  })
})
