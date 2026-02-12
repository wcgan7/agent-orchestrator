import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import ActivityTimeline from './ActivityTimeline'

// Mock the api module
vi.mock('../../api', () => ({
  buildApiUrl: (path: string, _projectDir?: string, _query?: Record<string, unknown>) => path,
  buildAuthHeaders: (extra: Record<string, string> = {}) => ({ ...extra }),
}))

describe('ActivityTimeline', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading state initially', () => {
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {})) // never resolves

    render(<ActivityTimeline taskId="task-1" />)

    expect(screen.getByText('Loading activity...')).toBeInTheDocument()
  })

  it('renders empty state when no events', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ events: [] }),
    })

    render(<ActivityTimeline taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('No activity recorded yet.')).toBeInTheDocument()
    })

    expect(screen.getByText('Activity')).toBeInTheDocument()
  })

  it('renders timeline events', async () => {
    const mockEvents = [
      {
        id: 'ev-1',
        type: 'status_change',
        timestamp: '2024-06-01T10:00:00Z',
        actor: 'system',
        actor_type: 'system',
        summary: 'Task started',
        details: 'Pipeline execution began',
      },
      {
        id: 'ev-2',
        type: 'comment',
        timestamp: '2024-06-01T10:05:00Z',
        actor: 'alice',
        actor_type: 'human',
        summary: 'Added a review comment',
      },
      {
        id: 'ev-3',
        type: 'agent_output',
        timestamp: '2024-06-01T10:10:00Z',
        actor: 'agent-1234',
        actor_type: 'agent',
        summary: 'Generated implementation code',
        details: 'Created 3 files with 150 lines',
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ events: mockEvents }),
    })

    render(<ActivityTimeline taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('Task started')).toBeInTheDocument()
    })

    expect(screen.getByText('Added a review comment')).toBeInTheDocument()
    expect(screen.getByText('Generated implementation code')).toBeInTheDocument()
    // Actor names are displayed
    expect(screen.getByText('system')).toBeInTheDocument()
    expect(screen.getByText('alice')).toBeInTheDocument()
    expect(screen.getByText('agent-1234')).toBeInTheDocument()
  })

  it('expands event to show details on click', async () => {
    const mockEvents = [
      {
        id: 'ev-1',
        type: 'status_change',
        timestamp: '2024-06-01T10:00:00Z',
        actor: 'system',
        actor_type: 'system',
        summary: 'Task started',
        details: 'Pipeline execution began with config v2',
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ events: mockEvents }),
    })

    render(<ActivityTimeline taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('Task started')).toBeInTheDocument()
    })

    // Details should not be visible initially
    expect(screen.queryByText('Pipeline execution began with config v2')).not.toBeInTheDocument()

    // Click the event
    const eventEl = screen.getByText('Task started').closest('.timeline-event')!
    fireEvent.click(eventEl)

    // Details should now be visible
    expect(screen.getByText('Pipeline execution began with config v2')).toBeInTheDocument()

    // Click again to collapse
    fireEvent.click(eventEl)
    expect(screen.queryByText('Pipeline execution began with config v2')).not.toBeInTheDocument()
  })

  it('loads timeline data from API with correct URL', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ events: [] }),
    })

    render(<ActivityTimeline taskId="task-42" projectDir="/home/project" />)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(1)
    })

    // Our mock buildApiUrl just returns the path, verify fetch was called
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v3/collaboration/timeline/task-42'),
      expect.any(Object)
    )
  })
})
