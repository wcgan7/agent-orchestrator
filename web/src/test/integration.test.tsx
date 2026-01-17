import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'

describe('Integration Tests', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    // Reset localStorage
    global.localStorage.clear()
    user = userEvent.setup()
  })

  describe('Approval Workflow', () => {
    it('completes full approval workflow from load to approval', async () => {
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

      const mockStatus = {
        project_dir: '',
        status: 'running',
        current_task_id: 'task-1',
        current_phase_id: 'phase-1',
        phases_completed: 0,
        phases_total: 1,
        tasks_ready: 0,
        tasks_running: 1,
        tasks_done: 0,
        tasks_blocked: 0,
      }

      let approvalsCallCount = 0
      global.fetch = vi.fn().mockImplementation((url) => {
        const urlString = url.toString()

        if (urlString.includes('/api/auth/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              enabled: false,
              authenticated: true,
              username: null,
            }),
          })
        }

        // Status endpoint
        if (urlString.includes('/api/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockStatus,
          })
        }

        // Approval requests endpoint
        if (urlString.includes('/api/approvals')) {
          approvalsCallCount++
          if (approvalsCallCount === 1) {
            return Promise.resolve({
              ok: true,
              json: async () => mockApprovals,
            })
          }
          return Promise.resolve({
            ok: true,
            json: async () => [],
          })
        }

        // Approve endpoint
        if (urlString.includes('/api/approvals/respond')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ success: true, message: 'Approved' }),
          })
        }

        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      // Wait for approval to load
      await waitFor(
        () => {
          expect(
            screen.getByText(/review implementation plan/i)
          ).toBeInTheDocument()
        },
        { timeout: 3000 }
      )

      // Approve the request
      const approveButton = screen.getByRole('button', { name: /approve/i })
      await user.click(approveButton)

      // Verify approval API was called
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/approvals/respond'),
          expect.objectContaining({
            method: 'POST',
            body: expect.stringContaining('"request_id":"req-1"'),
          })
        )
      })
    })

    it('handles rejection workflow with feedback', async () => {
      const mockApprovals = [
        {
          request_id: 'req-1',
          gate_type: 'after_implement',
          message: 'Review changes?',
          created_at: '2024-01-01T10:00:00Z',
          context: {},
          show_diff: false,
          show_plan: false,
          show_tests: false,
          show_review: false,
        },
      ]

      global.fetch = vi.fn().mockImplementation((url) => {
        if (url.toString().includes('/api/approvals/respond')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ success: true }),
          })
        }
        if (url.toString().includes('/api/approvals')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockApprovals,
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      await waitFor(() => {
        expect(screen.getByText(/review changes/i)).toBeInTheDocument()
      })

      // Add feedback
      const feedbackInput = screen.getByPlaceholderText(/optional feedback/i)
      await user.type(feedbackInput, 'Needs more tests')

      // Reject
      const rejectButton = screen.getByRole('button', { name: /reject/i })
      await user.click(rejectButton)

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/approvals/respond'),
          expect.objectContaining({
            method: 'POST',
            body: expect.stringContaining('Needs more tests'),
          })
        )
      })
    })
  })

  describe('Chat Workflow', () => {
    it('completes full chat workflow from open to send message', async () => {
      const mockMessages = [
        {
          id: 'msg-1',
          type: 'guidance',
          content: 'Initial message',
          timestamp: '2024-01-01T10:00:00Z',
          from_human: true,
          metadata: {},
        },
      ]

      global.fetch = vi.fn().mockImplementation((url, options) => {
        const urlString = url.toString()

        if (urlString.includes('/api/auth/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              enabled: false,
              authenticated: true,
              username: null,
            }),
          })
        }

        if (urlString.includes('/api/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              project_dir: '',
              status: 'running',
              run_id: 'run-123',
              phases_completed: 0,
              phases_total: 1,
              tasks_ready: 0,
              tasks_running: 0,
              tasks_done: 0,
              tasks_blocked: 0,
            }),
          })
        }

        // GET messages
        if (
          urlString.includes('/api/messages') &&
          (!options || options.method !== 'POST')
        ) {
          return Promise.resolve({
            ok: true,
            json: async () => mockMessages,
          })
        }

        // POST message
        if (
          urlString.includes('/api/messages') &&
          options?.method === 'POST'
        ) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ success: true }),
          })
        }

        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      // Open chat
      const chatToggle = await screen.findByRole('button', {
        name: /open chat/i,
      })
      await user.click(chatToggle)

      // Wait for messages to load
      await waitFor(() => {
        expect(screen.getByText(/initial message/i)).toBeInTheDocument()
      })

      // Type and send message
      const input = screen.getByPlaceholderText(/type a message/i)
      await user.type(input, 'New guidance message')

      const sendButton = screen.getByRole('button', { name: /send/i })
      await user.click(sendButton)

      // Verify message was sent
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/messages'),
          expect.objectContaining({
            method: 'POST',
            body: expect.stringContaining('New guidance message'),
          })
        )
      })
    })
  })

  describe('File Review Workflow', () => {
    it('completes full file review workflow', async () => {
      const mockFiles = [
        {
          file_path: 'src/app.ts',
          status: 'modified',
          additions: 10,
          deletions: 5,
          diff: '--- a/src/app.ts\n+++ b/src/app.ts\n@@ -1,3 +1,4 @@\n unchanged\n-deleted\n+added',
          approved: null,
          comments: [],
        },
        {
          file_path: 'src/test.ts',
          status: 'added',
          additions: 20,
          deletions: 0,
          diff: '+++ b/src/test.ts\n@@ -0,0 +1,20 @@\n+new file content',
          approved: null,
          comments: [],
        },
      ]

      global.fetch = vi.fn().mockImplementation((url, options) => {
        // GET file changes
        if (
          url.toString().includes('/api/file-changes') &&
          (!options || options.method !== 'POST')
        ) {
          return Promise.resolve({
            ok: true,
            json: async () => mockFiles,
          })
        }

        // POST file review
        if (
          url.toString().includes('/api/file-review') &&
          options?.method === 'POST'
        ) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ success: true }),
          })
        }

        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      // Wait for files to load
      await waitFor(() => {
        const fileList = document.querySelector('.file-review .file-list')
        expect(fileList).not.toBeNull()
        expect(within(fileList as HTMLElement).getByText('src/app.ts')).toBeInTheDocument()
      })

      // Add comment to first file
      const commentInput = screen.getByPlaceholderText(/add optional comment/i)
      await user.type(commentInput, 'Looks good')

      // Approve first file
      const approveButton = screen.getByRole('button', { name: /✓ approve/i })
      await user.click(approveButton)

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/file-review'),
          expect.objectContaining({
            method: 'POST',
            body: expect.stringContaining('src/app.ts'),
          })
        )
      })

      // Navigate to next file
      const nextButton = screen.getByRole('button', { name: /next →/i })
      await user.click(nextButton)

      await waitFor(() => {
        const fileList = document.querySelector('.file-review .file-list')
        expect(fileList).not.toBeNull()
        expect(within(fileList as HTMLElement).getByText('src/test.ts')).toBeInTheDocument()
        expect(screen.getByText(/file 2 of 2/i)).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('handles API errors gracefully across components', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('API Error'))

      render(<App />)

      // Check that error states are displayed
      await waitFor(() => {
        const errorMessages = screen.getAllByText(/error/i)
        expect(errorMessages.length).toBeGreaterThan(0)
      })
    })

    it('recovers from temporary network errors', async () => {
      let callCount = 0
      global.fetch = vi.fn().mockImplementation((url) => {
        const urlString = url.toString()

        if (urlString.includes('/api/auth/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              enabled: false,
              authenticated: true,
              username: null,
            }),
          })
        }

        if (urlString.includes('/api/status')) {
          callCount++
          if (callCount === 1) {
            return Promise.reject(new Error('Network error'))
          }
          return Promise.resolve({
            ok: true,
            json: async () => ({
              project_dir: '',
              status: 'running',
              phases_completed: 0,
              phases_total: 1,
              tasks_ready: 0,
              tasks_running: 0,
              tasks_done: 0,
              tasks_blocked: 0,
            }),
          })
        }

        if (urlString.includes('/api/approvals')) {
          return Promise.resolve({
            ok: true,
            json: async () => [],
          })
        }

        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      // First call fails
      await waitFor(() => {
        expect(
          screen.getByRole('heading', { name: /^error$/i })
        ).toBeInTheDocument()
      })

      // Manual retry (more deterministic than waiting for the 5s poll)
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await user.click(retryButton)

      await waitFor(() => {
        expect(screen.getByText(/no pending approvals/i)).toBeInTheDocument()
      })
    })
  })

  describe('Authentication', () => {
    it('includes auth token in all API requests when present', async () => {
      const mockToken = 'test-auth-token-123'
      global.localStorage.setItem('feature-prd-runner-auth-token', mockToken)

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => [],
      })

      render(<App />)

      await waitFor(() => {
        // Check that all fetch calls include the auth header
        const calls = (global.fetch as any).mock.calls
        const callsWithAuth = calls.filter(
          (call: any) =>
            call[1]?.headers?.Authorization === `Bearer ${mockToken}`
        )
        expect(callsWithAuth.length).toBeGreaterThan(0)
      })
    })

    it('works without auth token', async () => {
      global.localStorage.removeItem('feature-prd-runner-auth-token')

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => [],
      })

      render(<App />)

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled()
      })

      // Should not crash, just make requests without auth header
      const calls = (global.fetch as any).mock.calls
      const callsWithAuth = calls.filter(
        (call: any) => call[1]?.headers?.Authorization
      )
      expect(callsWithAuth.length).toBe(0)
    })
  })

  describe('Component Interaction', () => {
    it('updates approval count badge when approvals change', async () => {
      const mockApprovals = [
        {
          request_id: 'req-1',
          gate_type: 'before_implement',
          message: 'Approve 1',
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
          message: 'Approve 2',
          created_at: '2024-01-01T10:05:00Z',
          context: {},
          show_diff: false,
          show_plan: false,
          show_tests: false,
          show_review: false,
        },
      ]

      let approvalCount = 2
      global.fetch = vi.fn().mockImplementation((url, options) => {
        if (url.toString().includes('/api/approvals/respond')) {
          approvalCount--
          return Promise.resolve({
            ok: true,
            json: async () => ({ success: true }),
          })
        }
        if (url.toString().includes('/api/approvals')) {
          return Promise.resolve({
            ok: true,
            json: async () => (approvalCount === 2 ? mockApprovals : [mockApprovals[1]]),
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      // Wait for both approvals to load
      await waitFor(() => {
        expect(screen.getByText('2')).toBeInTheDocument()
      })

      // Approve one
      const approveButtons = screen.getAllByRole('button', { name: /approve/i })
      await user.click(approveButtons[0])

      // Badge should update to 1
      await waitFor(
        () => {
          expect(screen.getByText('1')).toBeInTheDocument()
        },
        { timeout: 5000 }
      )
    })

    it('maintains separate state for different components', async () => {
      const mockApprovals = [
        {
          request_id: 'req-1',
          gate_type: 'before_implement',
          message: 'Approval message',
          created_at: '2024-01-01T10:00:00Z',
          context: {},
          show_diff: false,
          show_plan: false,
          show_tests: false,
          show_review: false,
        },
      ]

      const mockFiles = [
        {
          file_path: 'test.ts',
          status: 'modified',
          additions: 5,
          deletions: 2,
          diff: 'test diff',
          approved: null,
          comments: [],
        },
      ]

      const mockMessages = [
        {
          id: 'msg-1',
          type: 'guidance',
          content: 'Chat message',
          timestamp: '2024-01-01T10:00:00Z',
          from_human: true,
          metadata: {},
        },
      ]

      global.fetch = vi.fn().mockImplementation((url) => {
        const urlString = url.toString()
        if (urlString.includes('/api/auth/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              enabled: false,
              authenticated: true,
              username: null,
            }),
          })
        }
        if (urlString.includes('/api/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              project_dir: '',
              status: 'running',
              run_id: 'run-123',
              current_task_id: 'task-1',
              phases_completed: 0,
              phases_total: 1,
              tasks_ready: 0,
              tasks_running: 0,
              tasks_done: 0,
              tasks_blocked: 0,
            }),
          })
        }

        if (urlString.includes('/api/approvals')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockApprovals,
          })
        }
        if (urlString.includes('/api/file-changes')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockFiles,
          })
        }
        if (urlString.includes('/api/messages')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockMessages,
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      // All components should load independently
      await waitFor(() => {
        expect(screen.getByText(/approval message/i)).toBeInTheDocument()
        const fileList = document.querySelector('.file-review .file-list')
        expect(fileList).not.toBeNull()
        expect(within(fileList as HTMLElement).getByText('test.ts')).toBeInTheDocument()
      })

      // Open chat
      const chatToggle = screen.getByRole('button', { name: /open chat/i })
      await user.click(chatToggle)

      await waitFor(() => {
        expect(screen.getByText(/chat message/i)).toBeInTheDocument()
      })

      // All three components should be visible and functional
      expect(screen.getByText(/approval message/i)).toBeInTheDocument()
      {
        const fileList = document.querySelector('.file-review .file-list')
        expect(fileList).not.toBeNull()
        expect(within(fileList as HTMLElement).getByText('test.ts')).toBeInTheDocument()
      }
      expect(screen.getByText(/chat message/i)).toBeInTheDocument()
    })
  })

  describe('Real-time Updates', () => {
    it('polls for new approval requests', async () => {
      let callCount = 0
      const originalSetInterval = global.setInterval
      const originalClearInterval = global.clearInterval
      global.setInterval = ((cb: any) => {
        if (typeof cb === 'function') cb()
        return 0 as any
      }) as any
      global.clearInterval = (() => {}) as any

      global.fetch = vi.fn().mockImplementation((url) => {
        const urlString = url.toString()
        if (urlString.includes('/api/auth/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              enabled: false,
              authenticated: true,
              username: null,
            }),
          })
        }

        if (urlString.includes('/api/status')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              project_dir: '',
              status: 'running',
              phases_completed: 0,
              phases_total: 1,
              tasks_ready: 0,
              tasks_running: 0,
              tasks_done: 0,
              tasks_blocked: 0,
            }),
          })
        }

        if (urlString.includes('/api/approvals')) {
          callCount++
          if (callCount === 1) {
            return Promise.resolve({
              ok: true,
              json: async () => [],
            })
          }
          return Promise.resolve({
            ok: true,
            json: async () => [
              {
                request_id: 'req-new',
                gate_type: 'before_implement',
                message: 'New approval',
                created_at: '2024-01-01T10:00:00Z',
                context: {},
                show_diff: false,
                show_plan: false,
                show_tests: false,
                show_review: false,
              },
            ],
          })
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({}),
        })
      })

      render(<App />)

      await waitFor(() => {
        expect(screen.getByText(/new approval/i)).toBeInTheDocument()
      })

      global.setInterval = originalSetInterval
      global.clearInterval = originalClearInterval
    })
  })
})
