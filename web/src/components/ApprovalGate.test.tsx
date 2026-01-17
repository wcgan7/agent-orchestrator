import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ApprovalGate from './ApprovalGate'

describe('ApprovalGate', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    user = userEvent.setup()
  })

  it('renders loading state initially', () => {
    render(<ApprovalGate />)
    expect(screen.getByText(/loading approvals/i)).toBeInTheDocument()
  })

  it('displays empty state when no approvals', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText(/no pending approvals/i)).toBeInTheDocument()
    })
  })

  it('displays approval requests', async () => {
    const mockApprovals = [
      {
        request_id: 'req-1',
        gate_type: 'before_implement',
        message: 'Review implementation plan?',
        task_id: 'task-1',
        phase_id: 'phase-1',
        created_at: '2024-01-01T10:00:00Z',
        timeout: 300,
        context: {},
        show_diff: false,
        show_plan: true,
        show_tests: false,
        show_review: false,
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockApprovals,
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText(/review implementation plan/i)).toBeInTheDocument()
      expect(screen.getByText(/before_implement/i)).toBeInTheDocument()
      expect(screen.getByText(/task-1/i)).toBeInTheDocument()
    })
  })

  it('displays approval count badge', async () => {
    const mockApprovals = [
      {
        request_id: 'req-1',
        gate_type: 'before_implement',
        message: 'Approve this?',
        created_at: '2024-01-01T10:00:00Z',
        context: {},
        show_diff: false,
        show_plan: false,
        show_tests: false,
        show_review: false,
      },
      {
        request_id: 'req-2',
        gate_type: 'after_implement',
        message: 'Approve that?',
        created_at: '2024-01-01T10:05:00Z',
        context: {},
        show_diff: false,
        show_plan: false,
        show_tests: false,
        show_review: false,
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockApprovals,
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText('2')).toBeInTheDocument()
    })
  })

  it('handles approve action', async () => {
    const mockApprovals = [
      {
        request_id: 'req-1',
        gate_type: 'before_implement',
        message: 'Approve this?',
        created_at: '2024-01-01T10:00:00Z',
        context: {},
        show_diff: false,
        show_plan: false,
        show_tests: false,
        show_review: false,
      },
    ]

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        // Initial fetch
        return Promise.resolve({
          ok: true,
          json: async () => mockApprovals,
        })
      }
      if (fetchCallCount === 2) {
        // Approve response
        return Promise.resolve({
          ok: true,
          json: async () => ({
            success: true,
            message: 'Approval request approved',
          }),
        })
      }
      // Refresh fetch
      return Promise.resolve({
        ok: true,
        json: async () => [],
      })
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText(/approve this/i)).toBeInTheDocument()
    })

    const approveButton = screen.getByRole('button', { name: /approve/i })
    await user.click(approveButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(3) // initial + approve + refresh
    })
  })

  it('handles reject action', async () => {
    const mockApprovals = [
      {
        request_id: 'req-1',
        gate_type: 'before_implement',
        message: 'Approve this?',
        created_at: '2024-01-01T10:00:00Z',
        context: {},
        show_diff: false,
        show_plan: false,
        show_tests: false,
        show_review: false,
      },
    ]

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => mockApprovals,
        })
      }
      if (fetchCallCount === 2) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            success: true,
            message: 'Approval request rejected',
          }),
        })
      }
      return Promise.resolve({
        ok: true,
        json: async () => [],
      })
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText(/approve this/i)).toBeInTheDocument()
    })

    const rejectButton = screen.getByRole('button', { name: /reject/i })
    await user.click(rejectButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(3)
    })
  })

  it('allows adding feedback to approval', async () => {
    const mockApprovals = [
      {
        request_id: 'req-1',
        gate_type: 'before_implement',
        message: 'Approve this?',
        created_at: '2024-01-01T10:00:00Z',
        context: {},
        show_diff: false,
        show_plan: false,
        show_tests: false,
        show_review: false,
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockApprovals,
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText(/approve this/i)).toBeInTheDocument()
    })

    const feedbackInput = screen.getByPlaceholderText(/optional feedback/i)
    await user.type(feedbackInput, 'Looks good to me')

    expect(feedbackInput).toHaveValue('Looks good to me')
  })

  it('displays error message on fetch failure', async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument()
    })
  })

  it('includes authorization header when token is present', async () => {
    const mockGetItem = vi.fn().mockReturnValue('test-token')
    global.localStorage.getItem = mockGetItem

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    })

    render(<ApprovalGate />)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      )
    })
  })
})
