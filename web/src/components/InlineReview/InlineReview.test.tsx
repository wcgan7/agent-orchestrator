import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import InlineReview from './InlineReview'

// Mock the api module
vi.mock('../../api', () => ({
  buildApiUrl: (path: string, _projectDir?: string, _query?: Record<string, unknown>) => path,
  buildAuthHeaders: (extra: Record<string, string> = {}) => ({ ...extra }),
}))

const SAMPLE_DIFF = `@@ -1,3 +1,4 @@
 import React from 'react'
-const old = true
+const updated = true
+const added = 'new line'
 export default {}
`

describe('InlineReview', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    user = userEvent.setup()
  })

  it('renders with empty state (no comments, shows diff)', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ comments: [] }),
    })

    render(
      <InlineReview
        taskId="task-1"
        filePath="src/index.ts"
        diff={SAMPLE_DIFF}
      />
    )

    // File path is shown in the header
    expect(screen.getByText('src/index.ts')).toBeInTheDocument()
    // Shows open comments count
    expect(screen.getByText('0 open comments')).toBeInTheDocument()
    // View toggle button exists
    expect(screen.getByText('Split')).toBeInTheDocument()
  })

  it('renders comments on the diff', async () => {
    const mockComments = [
      {
        id: 'c-1',
        file_path: 'src/index.ts',
        line_number: 2,
        body: 'Nice change here!',
        author: 'alice',
        author_type: 'human',
        resolved: false,
        created_at: '2024-06-01T12:00:00Z',
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ comments: mockComments }),
    })

    render(
      <InlineReview
        taskId="task-1"
        filePath="src/index.ts"
        diff={SAMPLE_DIFF}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('Nice change here!')).toBeInTheDocument()
    })

    // Author is displayed
    expect(screen.getByText('alice')).toBeInTheDocument()
    // Resolve button is present for unresolved comment
    expect(screen.getByText('Resolve')).toBeInTheDocument()
    // Shows 1 open comment
    expect(screen.getByText('1 open comments')).toBeInTheDocument()
  })

  it('shows the add comment form when clicking a diff line', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ comments: [] }),
    })

    render(
      <InlineReview
        taskId="task-1"
        filePath="src/index.ts"
        diff={SAMPLE_DIFF}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('0 open comments')).toBeInTheDocument()
    })

    // Click on an added diff line (the '+' lines in the diff)
    const addedLines = document.querySelectorAll('.diff-added')
    expect(addedLines.length).toBeGreaterThan(0)
    fireEvent.click(addedLines[0])

    // The comment form should now appear
    expect(screen.getByPlaceholderText('Write a review comment...')).toBeInTheDocument()
    expect(screen.getByText('Add Comment')).toBeInTheDocument()
    expect(screen.getByText('Cancel')).toBeInTheDocument()
  })

  it('submits a new comment via the form', async () => {
    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        // Initial fetch for comments
        return Promise.resolve({
          ok: true,
          json: async () => ({ comments: [] }),
        })
      }
      if (fetchCallCount === 2) {
        // POST new comment
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: 'c-new' }),
        })
      }
      // Refresh after submit
      return Promise.resolve({
        ok: true,
        json: async () => ({
          comments: [
            {
              id: 'c-new',
              file_path: 'src/index.ts',
              line_number: 2,
              body: 'My review comment',
              author: 'me',
              author_type: 'human',
              resolved: false,
              created_at: '2024-06-01T12:01:00Z',
            },
          ],
        }),
      })
    })

    render(
      <InlineReview
        taskId="task-1"
        filePath="src/index.ts"
        diff={SAMPLE_DIFF}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('0 open comments')).toBeInTheDocument()
    })

    // Click on an added diff line
    const addedLines = document.querySelectorAll('.diff-added')
    fireEvent.click(addedLines[0])

    const textarea = screen.getByPlaceholderText('Write a review comment...')
    await user.type(textarea, 'My review comment')

    const submitBtn = screen.getByText('Add Comment')
    await user.click(submitBtn)

    await waitFor(() => {
      // POST was called (fetchCallCount >= 2)
      expect(global.fetch).toHaveBeenCalledTimes(3) // initial + POST + refresh
    })
  })

  it('resolves a comment', async () => {
    const mockComments = [
      {
        id: 'c-1',
        file_path: 'src/index.ts',
        line_number: 2,
        body: 'Needs fix',
        author: 'bob',
        author_type: 'human',
        resolved: false,
        created_at: '2024-06-01T12:00:00Z',
      },
    ]

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ comments: mockComments }),
        })
      }
      if (fetchCallCount === 2) {
        // POST resolve
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        })
      }
      // Refresh after resolve
      return Promise.resolve({
        ok: true,
        json: async () => ({
          comments: [{ ...mockComments[0], resolved: true }],
        }),
      })
    })

    render(
      <InlineReview
        taskId="task-1"
        filePath="src/index.ts"
        diff={SAMPLE_DIFF}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('Needs fix')).toBeInTheDocument()
    })

    const resolveBtn = screen.getByText('Resolve')
    await user.click(resolveBtn)

    await waitFor(() => {
      // Resolve POST + refresh were called
      expect(global.fetch).toHaveBeenCalledTimes(3)
    })
  })

  it('toggles between unified and split view', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ comments: [] }),
    })

    render(
      <InlineReview
        taskId="task-1"
        filePath="src/index.ts"
        diff={SAMPLE_DIFF}
      />
    )

    // Initially unified, toggle button says "Split"
    const toggleBtn = screen.getByText('Split')
    await user.click(toggleBtn)

    // After click, button should say "Unified"
    expect(screen.getByText('Unified')).toBeInTheDocument()
  })
})
