import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import FeedbackPanel from './FeedbackPanel'

// Mock the api module
vi.mock('../../api', () => ({
  buildApiUrl: (path: string, _projectDir?: string, _query?: Record<string, unknown>) => path,
  buildAuthHeaders: (extra: Record<string, string> = {}) => ({ ...extra }),
}))

describe('FeedbackPanel', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    user = userEvent.setup()
  })

  it('renders empty state when no feedback', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ feedback: [] }),
    })

    render(<FeedbackPanel taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('No active feedback for this task.')).toBeInTheDocument()
    })

    // Title and count are shown
    expect(screen.getByText('Feedback')).toBeInTheDocument()
    expect(screen.getByText('0 active')).toBeInTheDocument()
  })

  it('renders a list of active feedback items', async () => {
    const mockFeedback = [
      {
        id: 'fb-1',
        task_id: 'task-1',
        feedback_type: 'bug_report',
        priority: 'must',
        status: 'active',
        summary: 'Fix the login bug',
        details: 'Users cannot log in when session expires',
        target_file: 'src/auth.ts',
        action: 'fix',
        created_by: 'alice',
        created_at: '2024-06-01T10:00:00Z',
        agent_response: null,
      },
      {
        id: 'fb-2',
        task_id: 'task-1',
        feedback_type: 'style_preference',
        priority: 'suggestion',
        status: 'active',
        summary: 'Use camelCase for variables',
        details: '',
        target_file: null,
        action: 'apply',
        created_by: 'bob',
        created_at: '2024-06-01T11:00:00Z',
        agent_response: null,
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ feedback: mockFeedback }),
    })

    render(<FeedbackPanel taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('Fix the login bug')).toBeInTheDocument()
    })

    expect(screen.getByText('Use camelCase for variables')).toBeInTheDocument()
    expect(screen.getByText('2 active')).toBeInTheDocument()
    // Details and target file shown
    expect(screen.getByText('Users cannot log in when session expires')).toBeInTheDocument()
    expect(screen.getByText('src/auth.ts')).toBeInTheDocument()
  })

  it('submits new feedback via the form', async () => {
    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        // Initial load
        return Promise.resolve({
          ok: true,
          json: async () => ({ feedback: [] }),
        })
      }
      if (fetchCallCount === 2) {
        // POST submit
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: 'fb-new' }),
        })
      }
      // Refresh
      return Promise.resolve({
        ok: true,
        json: async () => ({
          feedback: [
            {
              id: 'fb-new',
              task_id: 'task-1',
              feedback_type: 'general',
              priority: 'should',
              status: 'active',
              summary: 'Improve error handling',
              details: '',
              target_file: null,
              action: '',
              created_by: 'me',
              created_at: '2024-06-01T12:00:00Z',
              agent_response: null,
            },
          ],
        }),
      })
    })

    render(<FeedbackPanel taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('No active feedback for this task.')).toBeInTheDocument()
    })

    // Open form
    const addBtn = screen.getByText('+ Add Feedback')
    await user.click(addBtn)

    // Fill in summary
    const summaryInput = screen.getByPlaceholderText('Summary (one line)')
    await user.type(summaryInput, 'Improve error handling')

    // Submit
    const submitBtn = screen.getByText('Submit Feedback')
    await user.click(submitBtn)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(3) // initial + POST + refresh
    })
  })

  it('dismisses a feedback item', async () => {
    const mockFeedback = [
      {
        id: 'fb-1',
        task_id: 'task-1',
        feedback_type: 'general',
        priority: 'should',
        status: 'active',
        summary: 'Do something',
        details: '',
        target_file: null,
        action: '',
        created_by: 'alice',
        created_at: '2024-06-01T10:00:00Z',
        agent_response: null,
      },
    ]

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ feedback: mockFeedback }),
        })
      }
      if (fetchCallCount === 2) {
        // POST dismiss
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        })
      }
      // Refresh
      return Promise.resolve({
        ok: true,
        json: async () => ({ feedback: [] }),
      })
    })

    render(<FeedbackPanel taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('Do something')).toBeInTheDocument()
    })

    const dismissBtn = screen.getByText('Dismiss')
    await user.click(dismissBtn)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(3)
    })
  })

  it('shows addressed feedback in a collapsible section', async () => {
    const mockFeedback = [
      {
        id: 'fb-1',
        task_id: 'task-1',
        feedback_type: 'general',
        priority: 'should',
        status: 'addressed',
        summary: 'Fixed the thing',
        details: '',
        target_file: null,
        action: '',
        created_by: 'alice',
        created_at: '2024-06-01T10:00:00Z',
        agent_response: 'Done!',
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ feedback: mockFeedback }),
    })

    render(<FeedbackPanel taskId="task-1" />)

    await waitFor(() => {
      expect(screen.getByText('1 addressed')).toBeInTheDocument()
    })

    expect(screen.getByText('Fixed the thing')).toBeInTheDocument()
    expect(screen.getByText('Agent: Done!')).toBeInTheDocument()
  })
})
