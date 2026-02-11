import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import ReasoningViewer from './ReasoningViewer'

// Mock WebSocketContext
vi.mock('../../contexts/WebSocketContext', () => ({
  useChannel: vi.fn(),
  useWebSocket: vi.fn(() => ({
    status: 'disconnected',
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
    send: vi.fn(),
    addHandler: vi.fn(() => vi.fn()),
    lastEventId: 0,
  })),
}))

// Mock the api module
vi.mock('../../api', () => ({
  buildApiUrl: (path: string, _projectDir?: string, _query?: Record<string, unknown>) => path,
  buildAuthHeaders: (extra: Record<string, string> = {}) => ({ ...extra }),
}))

describe('ReasoningViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading state initially', () => {
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {})) // never resolves

    render(<ReasoningViewer taskId="task-1" />)

    expect(screen.getByText('Loading agent reasoning...')).toBeInTheDocument()
  })

  it('renders empty state when no reasoning available', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ reasonings: [] }),
    })

    render(<ReasoningViewer taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText(/no agent reasoning available yet/i)).toBeInTheDocument()
    })
  })

  it('renders reasoning steps for an agent', async () => {
    const mockReasonings = [
      {
        agent_id: 'agent-abcdef12',
        agent_role: 'Implementer',
        task_id: 'task-1',
        current_step: 'coding',
        steps: [
          {
            step_name: 'planning',
            status: 'completed',
            reasoning: 'Analyzed the requirements',
            output: 'Plan: implement feature X',
            duration_ms: 1200,
          },
          {
            step_name: 'coding',
            status: 'running',
            reasoning: 'Writing implementation',
          },
          {
            step_name: 'testing',
            status: 'pending',
          },
        ],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ reasonings: mockReasonings }),
    })

    render(<ReasoningViewer taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('Agent Reasoning')).toBeInTheDocument()
    })

    // Agent role is shown
    expect(screen.getByText('Implementer')).toBeInTheDocument()
    // Step names are rendered (single agent => steps auto-expanded)
    expect(screen.getByText('planning')).toBeInTheDocument()
    expect(screen.getByText('coding')).toBeInTheDocument()
    expect(screen.getByText('testing')).toBeInTheDocument()
    // Progress count shown
    expect(screen.getByText('1/3 steps')).toBeInTheDocument()
    // Duration is shown for completed step
    expect(screen.getByText('1.2s')).toBeInTheDocument()
  })

  it('expands and collapses a step to show reasoning details', async () => {
    const mockReasonings = [
      {
        agent_id: 'agent-abcdef12',
        agent_role: 'Implementer',
        task_id: 'task-1',
        steps: [
          {
            step_name: 'planning',
            status: 'completed',
            reasoning: 'Analyzed the task requirements carefully',
            output: 'Decided to use pattern X',
            duration_ms: 500,
          },
        ],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ reasonings: mockReasonings }),
    })

    render(<ReasoningViewer taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('planning')).toBeInTheDocument()
    })

    // Reasoning detail should not be visible initially
    expect(screen.queryByText('Analyzed the task requirements carefully')).not.toBeInTheDocument()

    // Click the step header to expand
    const stepHeader = screen.getByText('planning').closest('.reasoning-step-header')!
    fireEvent.click(stepHeader)

    // Now reasoning and output should be visible
    expect(screen.getByText('Analyzed the task requirements carefully')).toBeInTheDocument()
    expect(screen.getByText('Decided to use pattern X')).toBeInTheDocument()

    // Click again to collapse
    fireEvent.click(stepHeader)

    expect(screen.queryByText('Analyzed the task requirements carefully')).not.toBeInTheDocument()
  })

  it('shows multiple agents with expand/collapse on agent headers', async () => {
    const mockReasonings = [
      {
        agent_id: 'agent-aaaa1111',
        agent_role: 'Planner',
        task_id: 'task-1',
        steps: [
          { step_name: 'analyze', status: 'completed' },
        ],
      },
      {
        agent_id: 'agent-bbbb2222',
        agent_role: 'Reviewer',
        task_id: 'task-1',
        steps: [
          { step_name: 'review', status: 'running' },
        ],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ reasonings: mockReasonings }),
    })

    render(<ReasoningViewer taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('Planner')).toBeInTheDocument()
    })

    expect(screen.getByText('Reviewer')).toBeInTheDocument()

    // With multiple agents, steps are collapsed by default.
    // Click on the Planner header to expand it.
    const plannerHeader = screen.getByText('Planner').closest('.reasoning-agent-header')!
    fireEvent.click(plannerHeader)

    // Now the Planner steps should be visible
    expect(screen.getByText('analyze')).toBeInTheDocument()
  })
})
