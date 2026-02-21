import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

function installFetchMock(
  responder: (url: string) => Record<string, unknown>,
): ReturnType<typeof vi.fn> {
  const task = {
    id: 'task-1',
    title: 'Task 1',
    description: 'Investigate logs',
    priority: 'P2',
    status: 'in_progress',
    task_type: 'feature',
    blocked_by: [],
    blocks: [],
  }

  const jsonResponse = (payload: unknown) =>
    Promise.resolve({
      ok: true,
      json: async () => payload,
    })

  const mockedFetch = vi.fn().mockImplementation((url) => {
    const u = String(url)
    if (u === '/' || u.startsWith('/?')) return jsonResponse({ project_id: 'repo-alpha' })
    if (u.includes('/api/collaboration/modes')) return jsonResponse({ modes: [] })
    if (u.includes('/api/tasks/task-1/logs')) return jsonResponse(responder(u))
    if (u.includes('/api/collaboration/timeline/task-1')) return jsonResponse({ events: [] })
    if (u.includes('/api/tasks/task-1')) return jsonResponse({ task })
    if (u.includes('/api/tasks/board')) {
      return jsonResponse({
        columns: {
          backlog: [task],
          queued: [],
          in_progress: [],
          in_review: [],
          blocked: [],
          done: [],
        },
      })
    }
    if (u.includes('/api/tasks') && !u.includes('/api/tasks/')) return jsonResponse({ tasks: [task] })
    if (u.includes('/api/orchestrator/status')) return jsonResponse({ status: 'running', queue_depth: 0, in_progress: 1, draining: false, run_branch: null })
    if (u.includes('/api/review-queue')) return jsonResponse({ tasks: [] })
    if (u.includes('/api/agents')) return jsonResponse({ agents: [] })
    if (u.includes('/api/projects/pinned')) return jsonResponse({ items: [] })
    if (u.includes('/api/projects')) return jsonResponse({ projects: [] })
    if (u.includes('/api/workers/health')) return jsonResponse({ providers: [] })
    if (u.includes('/api/workers/routing')) return jsonResponse({ default: 'codex', rows: [] })
    if (u.includes('/api/tasks/execution-order')) return jsonResponse({ batches: [] })
    if (u.includes('/api/phases')) return jsonResponse([])
    if (u.includes('/api/metrics')) return jsonResponse({})
    return jsonResponse({})
  })

  global.fetch = mockedFetch as unknown as typeof fetch
  return mockedFetch
}

async function openTaskLogsTab(): Promise<void> {
  await waitFor(() => {
    expect(screen.getByRole('button', { name: /task 1/i })).toBeEnabled()
  }, { timeout: 5_000 })
  const taskButton = screen.getByRole('button', { name: /task 1/i })
  fireEvent.click(taskButton)
  let detailDialog: HTMLElement
  try {
    detailDialog = await screen.findByRole('dialog', { name: /task detail/i }, { timeout: 5_000 })
  } catch {
    // Retry once to reduce cross-test timing flakiness in full-suite runs.
    fireEvent.click(taskButton)
    detailDialog = await screen.findByRole('dialog', { name: /task detail/i }, { timeout: 5_000 })
  }
  fireEvent.click(within(detailDialog).getByRole('button', { name: /^logs$/i }))
}

describe('Task logs loading', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    window.location.hash = ''
    ;(globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket = MockWebSocket as unknown as typeof WebSocket
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('fetches backfill in multiple chunks and keeps raw stdout', async () => {
    const logCalls: string[] = []
    installFetchMock((url) => {
      logCalls.push(url)
      const parsed = new URL(url, 'http://localhost')
      const offset = Number(parsed.searchParams.get('stdout_offset') || '0')
      const backfill = parsed.searchParams.get('backfill') === 'true'
      if (!backfill) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: 'TAIL-END\n',
          stderr: '',
          stdout_offset: 450000,
          stderr_offset: 0,
          stdout_tail_start: 400000,
          stderr_tail_start: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-a',
          stdout_chunk_start: 400000,
          stderr_chunk_start: 0,
        }
      }
      if (offset === 200000) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: 'CHUNK-B\n',
          stderr: '',
          stdout_offset: 400000,
          stderr_offset: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-a',
          stdout_chunk_start: 200000,
          stderr_chunk_start: 0,
        }
      }
      if (offset === 0) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: 'CHUNK-A\n',
          stderr: '',
          stdout_offset: 200000,
          stderr_offset: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-a',
          stdout_chunk_start: 0,
          stderr_chunk_start: 0,
        }
      }
      return {
        mode: 'active',
        step: 'implement',
        stdout: '',
        stderr: '',
        stdout_offset: 450000,
        stderr_offset: 0,
        started_at: '2026-02-16T00:00:00Z',
        log_id: 'run-a',
        stdout_chunk_start: 0,
        stderr_chunk_start: 0,
      }
    })

    render(<App />)

    await openTaskLogsTab()

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      expect(panes.length).toBeGreaterThan(0)
      expect((panes[0]?.textContent || '')).toContain('TAIL-END')
    })

    await new Promise((resolve) => window.setTimeout(resolve, 2_100))
    await new Promise((resolve) => window.setTimeout(resolve, 2_100))

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      expect(stdout).toContain('CHUNK-A')
      expect(stdout).toContain('CHUNK-B')
      expect(stdout).toContain('TAIL-END')
      expect(stdout.indexOf('CHUNK-A')).toBeLessThan(stdout.indexOf('CHUNK-B'))
      expect(stdout.indexOf('CHUNK-B')).toBeLessThan(stdout.indexOf('TAIL-END'))
    })

    expect(logCalls.some((url) => url.includes('backfill=true') && url.includes('stdout_offset=200000') && url.includes('stdout_read_to=400000'))).toBe(true)
    expect(logCalls.some((url) => url.includes('backfill=true') && url.includes('stdout_offset=0') && url.includes('stdout_read_to=200000'))).toBe(true)
  }, 20_000)

  it('keeps rendered stdout suffix stable while backfill prepends older transcript chunks', async () => {
    const logCalls: string[] = []
    installFetchMock((url) => {
      logCalls.push(url)
      const parsed = new URL(url, 'http://localhost')
      const offset = Number(parsed.searchParams.get('stdout_offset') || '0')
      const backfill = parsed.searchParams.get('backfill') === 'true'
      if (!backfill) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: `${JSON.stringify({ type: 'assistant', message: { content: [{ type: 'text', text: 'Tail message\n' }] } })}\n`,
          stderr: '',
          stdout_offset: 450000,
          stderr_offset: 0,
          stdout_tail_start: 400000,
          stderr_tail_start: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-render',
          stdout_chunk_start: 400000,
          stderr_chunk_start: 0,
        }
      }
      if (offset === 200000) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: [
            JSON.stringify({ type: 'stream_event', event: { type: 'content_block_start', content_block: { type: 'tool_use', name: 'Write' } } }),
            JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Middle line\n' } } }),
          ].join('\n') + '\n',
          stderr: '',
          stdout_offset: 400000,
          stderr_offset: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-render',
          stdout_chunk_start: 200000,
          stderr_chunk_start: 0,
        }
      }
      if (offset === 0) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: `${JSON.stringify({ type: 'user', message: { content: [{ type: 'tool_result', is_error: true, content: 'denied' }] } })}\n`,
          stderr: '',
          stdout_offset: 200000,
          stderr_offset: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-render',
          stdout_chunk_start: 0,
          stderr_chunk_start: 0,
        }
      }
      return {
        mode: 'active',
        step: 'implement',
        stdout: '',
        stderr: '',
        stdout_offset: 450000,
        stderr_offset: 0,
        started_at: '2026-02-16T00:00:00Z',
        log_id: 'run-render',
        stdout_chunk_start: 0,
        stderr_chunk_start: 0,
      }
    })

    render(<App />)

    await openTaskLogsTab()

    let initialText = ''
    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      expect(stdout).toContain('Tail message')
      initialText = stdout
    })

    await new Promise((resolve) => window.setTimeout(resolve, 2_100))

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      expect(stdout).toContain('Tools used: Write')
      expect(stdout).toContain('Middle line')
      expect(stdout).toContain('Tool error: denied')
      expect(stdout.endsWith(initialText)).toBe(true)
    })

    expect(logCalls.some((url) => url.includes('backfill=true'))).toBe(true)
  }, 20_000)

  it('resets accumulator when log_id changes to a new run', async () => {
    let runBCaptured = false
    installFetchMock((url) => {
      const parsed = new URL(url, 'http://localhost')
      const offset = Number(parsed.searchParams.get('stdout_offset') || '0')
      const hasOffset = parsed.searchParams.has('stdout_offset')
      if (!hasOffset) {
        if (runBCaptured) {
          return {
            mode: 'active',
            step: 'implement',
            stdout: 'RUN-B\n',
            stderr: '',
            stdout_offset: 6,
            stderr_offset: 0,
            stdout_tail_start: 0,
            stderr_tail_start: 0,
            started_at: '2026-02-16T00:01:00Z',
            log_id: 'run-b',
          }
        }
        return {
          mode: 'active',
          step: 'implement',
          stdout: 'RUN-A\n',
          stderr: '',
          stdout_offset: 6,
          stderr_offset: 0,
          stdout_tail_start: 0,
          stderr_tail_start: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-a',
        }
      }

      if (offset >= 6 && !runBCaptured) {
        runBCaptured = true
        return {
          mode: 'active',
          step: 'implement',
          stdout: '',
          stderr: '',
          stdout_offset: 0,
          stderr_offset: 0,
          started_at: '2026-02-16T00:01:00Z',
          log_id: 'run-b',
        }
      }

      return {
        mode: 'active',
        step: 'implement',
        stdout: '',
        stderr: '',
        stdout_offset: 6,
        stderr_offset: 0,
        started_at: '2026-02-16T00:01:00Z',
        log_id: 'run-b',
      }
    })

    render(<App />)

    await openTaskLogsTab()

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      expect(stdout === '' || stdout.includes('RUN-A') || stdout.includes('RUN-B')).toBe(true)
    })

    await new Promise((resolve) => window.setTimeout(resolve, 2_100))

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      expect(stdout).toContain('RUN-B')
      expect(stdout).not.toContain('RUN-A')
    })
  }, 20_000)

  it('does not duplicate the same tool error emitted via multiple fields in one JSON record', async () => {
    let emitted = false
    installFetchMock(() => {
      if (emitted) {
        return {
          mode: 'active',
          step: 'implement',
          stdout: '',
          stderr: '',
          stdout_offset: 200,
          stderr_offset: 0,
          stdout_tail_start: 0,
          stderr_tail_start: 0,
          started_at: '2026-02-16T00:00:00Z',
          log_id: 'run-dedupe',
          stdout_chunk_start: 200,
          stderr_chunk_start: 0,
        }
      }
      emitted = true
      return {
        mode: 'active',
        step: 'implement',
        stdout: `${JSON.stringify({
          type: 'user',
          tool_use_result: 'duplicate-permission-error',
          message: {
            content: [{ type: 'tool_result', is_error: true, content: 'duplicate-permission-error' }],
          },
        })}\n`,
        stderr: '',
        stdout_offset: 200,
        stderr_offset: 0,
        stdout_tail_start: 0,
        stderr_tail_start: 0,
        started_at: '2026-02-16T00:00:00Z',
        log_id: 'run-dedupe',
        stdout_chunk_start: 0,
        stderr_chunk_start: 0,
      }
    })

    render(<App />)

    await openTaskLogsTab()

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      const count = (stdout.match(/Tool error: duplicate-permission-error/g) || []).length
      expect(count).toBe(1)
    })
  }, 20_000)

  it('keeps mixed plain-text and JSON logs visible for verify/review payloads', async () => {
    installFetchMock((url) => {
      const parsed = new URL(url, 'http://localhost')
      const hasOffset = parsed.searchParams.has('stdout_offset') || parsed.searchParams.has('stderr_offset')
      if (!hasOffset) {
        return {
          mode: 'history',
          step: 'verify',
          stdout: '{"status":"pass","reason_code":"unknown","summary":"all good"}\n',
          stderr: 'Reading prompt from stdin...\n{"findings":[]}\n',
          stdout_offset: 65,
          stderr_offset: 43,
          stdout_tail_start: 0,
          stderr_tail_start: 0,
          started_at: '2026-02-21T13:13:13Z',
          finished_at: '2026-02-21T13:13:54Z',
          log_id: 'run-verify-json',
          stdout_chunk_start: 0,
          stderr_chunk_start: 0,
        }
      }
      return {
        mode: 'history',
        step: 'verify',
        stdout: '',
        stderr: '',
        stdout_offset: 65,
        stderr_offset: 43,
        stdout_tail_start: 0,
        stderr_tail_start: 0,
        started_at: '2026-02-21T13:13:13Z',
        finished_at: '2026-02-21T13:13:54Z',
        log_id: 'run-verify-json',
        stdout_chunk_start: 65,
        stderr_chunk_start: 43,
      }
    })

    render(<App />)
    await openTaskLogsTab()

    await waitFor(() => {
      const panes = document.querySelectorAll('pre.task-log-output')
      const stdout = panes[0]?.textContent || ''
      const stderr = panes[1]?.textContent || ''
      expect(stdout).toContain('"status":"pass"')
      expect(stderr).toContain('Reading prompt from stdin...')
      expect(stderr).toContain('{"findings":[]}')
    })
  }, 20_000)
})
