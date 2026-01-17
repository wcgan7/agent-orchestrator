import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import FileReview from './FileReview'

describe('FileReview', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    user = userEvent.setup()
  })

  it('renders loading state initially', () => {
    render(<FileReview />)
    expect(screen.getByText(/loading file changes/i)).toBeInTheDocument()
  })

  it('displays empty state when no file changes', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    })

    render(<FileReview />)

    await waitFor(() => {
      expect(screen.getByText(/no file changes to review/i)).toBeInTheDocument()
    })
  })

  it('displays error message on fetch failure', async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

    render(<FileReview />)

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument()
      expect(screen.getByText(/network error/i)).toBeInTheDocument()
    })
  })

  it('displays file list with status indicators', async () => {
    const mockFiles = [
      {
        file_path: 'src/app.ts',
        status: 'modified',
        additions: 10,
        deletions: 5,
        diff: 'diff content',
        approved: null,
        comments: [],
      },
      {
        file_path: 'src/new.ts',
        status: 'added',
        additions: 20,
        deletions: 0,
        diff: 'diff content',
        approved: null,
        comments: [],
      },
      {
        file_path: 'src/old.ts',
        status: 'deleted',
        additions: 0,
        deletions: 15,
        diff: 'diff content',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    const { container } = render(<FileReview />)

    await waitFor(() => {
      const list = container.querySelector('.file-list')
      expect(list).not.toBeNull()
      const listQueries = within(list as HTMLElement)
      expect(listQueries.getByText('src/app.ts')).toBeInTheDocument()
      expect(listQueries.getByText('src/new.ts')).toBeInTheDocument()
      expect(listQueries.getByText('src/old.ts')).toBeInTheDocument()
      expect(screen.getAllByText('+10').length).toBeGreaterThan(0)
      expect(screen.getAllByText('-5').length).toBeGreaterThan(0)
      expect(screen.getAllByText('+20').length).toBeGreaterThan(0)
      expect(screen.getAllByText('-15').length).toBeGreaterThan(0)
    })
  })

  it('displays review statistics', async () => {
    const mockFiles = [
      {
        file_path: 'approved.ts',
        status: 'modified',
        additions: 5,
        deletions: 2,
        diff: 'diff',
        approved: true,
        comments: [],
      },
      {
        file_path: 'rejected.ts',
        status: 'modified',
        additions: 3,
        deletions: 1,
        diff: 'diff',
        approved: false,
        comments: [],
      },
      {
        file_path: 'pending.ts',
        status: 'modified',
        additions: 4,
        deletions: 2,
        diff: 'diff',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    render(<FileReview />)

    await waitFor(() => {
      expect(screen.getByText(/✓ 1/)).toBeInTheDocument() // approved count
      expect(screen.getByText(/✗ 1/)).toBeInTheDocument() // rejected count
      expect(screen.getByText(/⏳ 1/)).toBeInTheDocument() // pending count
      expect(screen.getByText(/total: 3/i)).toBeInTheDocument()
    })
  })

  it('allows file selection and displays selected file', async () => {
    const mockFiles = [
      {
        file_path: 'first.ts',
        status: 'modified',
        additions: 5,
        deletions: 2,
        diff: 'first diff',
        approved: null,
        comments: [],
      },
      {
        file_path: 'second.ts',
        status: 'added',
        additions: 10,
        deletions: 0,
        diff: 'second diff',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    const { container } = render(<FileReview />)

    await waitFor(() => {
      const list = container.querySelector('.file-list')
      expect(list).not.toBeNull()
      const listQueries = within(list as HTMLElement)
      expect(listQueries.getByText('first.ts')).toBeInTheDocument()
    })

    // Click on second file
    const list = container.querySelector('.file-list') as HTMLElement
    const secondFile = within(list).getByText('second.ts')
    await user.click(secondFile)

    // Should display second file's diff
    await waitFor(() => {
      expect(screen.getByText('second diff')).toBeInTheDocument()
    })
  })

  it('renders diff with syntax highlighting', async () => {
    const mockFiles = [
      {
        file_path: 'test.ts',
        status: 'modified',
        additions: 2,
        deletions: 1,
        diff: '--- a/test.ts\n+++ b/test.ts\n@@ -1,3 +1,4 @@\n unchanged line\n-deleted line\n+added line',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    render(<FileReview />)

    await waitFor(() => {
      const diffView = screen.getByText('-deleted line')
      expect(diffView).toBeInTheDocument()
      expect(diffView).toHaveClass('diff-line', 'deletion')

      const addedLine = screen.getByText('+added line')
      expect(addedLine).toHaveClass('diff-line', 'addition')

      const hunkLine = screen.getByText('@@ -1,3 +1,4 @@')
      expect(hunkLine).toHaveClass('diff-line', 'hunk')
    })
  })

  it('handles approve action', async () => {
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

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => mockFiles,
        })
      } else {
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        })
      }
    })

    render(<FileReview />)

    await waitFor(() => {
      const list = document.querySelector('.file-list')
      expect(list).not.toBeNull()
      expect(within(list as HTMLElement).getByText('test.ts')).toBeInTheDocument()
    })

    const approveButton = screen.getByRole('button', { name: /✓ approve/i })
    await user.click(approveButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/file-review'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"approved":true'),
        })
      )
    })
  })

  it('handles reject action', async () => {
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

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => mockFiles,
        })
      } else {
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        })
      }
    })

    render(<FileReview />)

    await waitFor(() => {
      const list = document.querySelector('.file-list')
      expect(list).not.toBeNull()
      expect(within(list as HTMLElement).getByText('test.ts')).toBeInTheDocument()
    })

    const rejectButton = screen.getByRole('button', { name: /✗ reject/i })
    await user.click(rejectButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/file-review'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"approved":false'),
        })
      )
    })
  })

  it('allows adding comments to reviews', async () => {
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

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    render(<FileReview />)

    await waitFor(() => {
      const list = document.querySelector('.file-list')
      expect(list).not.toBeNull()
      expect(within(list as HTMLElement).getByText('test.ts')).toBeInTheDocument()
    })

    const commentInput = screen.getByPlaceholderText(/add optional comment/i)
    await user.type(commentInput, 'This needs refactoring')

    expect(commentInput).toHaveValue('This needs refactoring')
  })

  it('includes comment in review submission', async () => {
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

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => mockFiles,
        })
      } else {
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        })
      }
    })

    render(<FileReview />)

    await waitFor(() => {
      const list = document.querySelector('.file-list')
      expect(list).not.toBeNull()
      expect(within(list as HTMLElement).getByText('test.ts')).toBeInTheDocument()
    })

    const commentInput = screen.getByPlaceholderText(/add optional comment/i)
    await user.type(commentInput, 'Needs work')

    const approveButton = screen.getByRole('button', { name: /✓ approve/i })
    await user.click(approveButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/file-review'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Needs work'),
        })
      )
    })
  })

  it('clears comment after successful review submission', async () => {
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

    let fetchCallCount = 0
    global.fetch = vi.fn().mockImplementation(() => {
      fetchCallCount++
      if (fetchCallCount === 1) {
        return Promise.resolve({
          ok: true,
          json: async () => mockFiles,
        })
      } else {
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true }),
        })
      }
    })

    render(<FileReview />)

    await waitFor(() => {
      const list = document.querySelector('.file-list')
      expect(list).not.toBeNull()
      expect(within(list as HTMLElement).getByText('test.ts')).toBeInTheDocument()
    })

    const commentInput = screen.getByPlaceholderText(
      /add optional comment/i
    ) as HTMLTextAreaElement
    await user.type(commentInput, 'Test comment')
    expect(commentInput.value).toBe('Test comment')

    const approveButton = screen.getByRole('button', { name: /✓ approve/i })
    await user.click(approveButton)

    await waitFor(() => {
      expect(commentInput.value).toBe('')
    })
  })

  it('disables buttons during review submission', async () => {
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

    global.fetch = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockFiles,
      })
      .mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(() => {
              resolve({
                ok: true,
                json: async () => ({ success: true }),
              })
            }, 100)
          })
      )

    render(<FileReview />)

    await waitFor(() => {
      const list = document.querySelector('.file-list')
      expect(list).not.toBeNull()
      expect(within(list as HTMLElement).getByText('test.ts')).toBeInTheDocument()
    })

    const approveButton = screen.getByRole('button', { name: /✓ approve/i })
    const rejectButton = screen.getByRole('button', { name: /✗ reject/i })

    await user.click(approveButton)

    // Buttons should be disabled during submission
    expect(approveButton).toBeDisabled()
    expect(rejectButton).toBeDisabled()
    expect(screen.getAllByText(/processing/i).length).toBeGreaterThan(0)
  })

  it('navigates between files with previous/next buttons', async () => {
    const mockFiles = [
      {
        file_path: 'first.ts',
        status: 'modified',
        additions: 5,
        deletions: 2,
        diff: 'first diff',
        approved: null,
        comments: [],
      },
      {
        file_path: 'second.ts',
        status: 'modified',
        additions: 3,
        deletions: 1,
        diff: 'second diff',
        approved: null,
        comments: [],
      },
      {
        file_path: 'third.ts',
        status: 'modified',
        additions: 7,
        deletions: 4,
        diff: 'third diff',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    render(<FileReview />)

    await waitFor(() => {
      expect(screen.getByText(/file 1 of 3/i)).toBeInTheDocument()
    })

    // Previous button should be disabled on first file
    const prevButton = screen.getByRole('button', { name: /← previous/i })
    expect(prevButton).toBeDisabled()

    // Click next
    const nextButton = screen.getByRole('button', { name: /next →/i })
    await user.click(nextButton)

    await waitFor(() => {
      expect(screen.getByText(/file 2 of 3/i)).toBeInTheDocument()
      expect(screen.getByText('second diff')).toBeInTheDocument()
    })

    // Click next again
    await user.click(nextButton)

    await waitFor(() => {
      expect(screen.getByText(/file 3 of 3/i)).toBeInTheDocument()
      expect(screen.getByText('third diff')).toBeInTheDocument()
    })

    // Next button should be disabled on last file
    expect(nextButton).toBeDisabled()

    // Click previous
    await user.click(prevButton)

    await waitFor(() => {
      expect(screen.getByText(/file 2 of 3/i)).toBeInTheDocument()
    })
  })

  it('includes authorization header when token is present', async () => {
    const mockGetItem = vi.fn().mockReturnValue('test-token')
    global.localStorage.getItem = mockGetItem

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    })

    render(<FileReview />)

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

  it('passes project directory and task ID as query parameters', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    })

    render(<FileReview taskId="task-123" projectDir="/path/to/project" />)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('project_dir=%2Fpath%2Fto%2Fproject'),
        expect.any(Object)
      )
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('task_id=task-123'),
        expect.any(Object)
      )
    })
  })

  it('displays approval status badges on reviewed files', async () => {
    const mockFiles = [
      {
        file_path: 'approved.ts',
        status: 'modified',
        additions: 5,
        deletions: 2,
        diff: 'diff',
        approved: true,
        comments: [],
      },
      {
        file_path: 'rejected.ts',
        status: 'modified',
        additions: 3,
        deletions: 1,
        diff: 'diff',
        approved: false,
        comments: [],
      },
      {
        file_path: 'pending.ts',
        status: 'modified',
        additions: 4,
        deletions: 2,
        diff: 'diff',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    const { container } = render(<FileReview />)

    await waitFor(() => {
      const fileItems = container.querySelectorAll('.file-item')
      expect(fileItems.length).toBeGreaterThanOrEqual(3)
      expect(fileItems[0]).toHaveClass('approved')
      expect(fileItems[1]).toHaveClass('rejected')
      expect(fileItems[2]).not.toHaveClass('approved')
      expect(fileItems[2]).not.toHaveClass('rejected')
    })
  })

  it('displays no diff message when diff is empty', async () => {
    const mockFiles = [
      {
        file_path: 'empty.ts',
        status: 'modified',
        additions: 0,
        deletions: 0,
        diff: '',
        approved: null,
        comments: [],
      },
    ]

    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockFiles,
    })

    render(<FileReview />)

    await waitFor(() => {
      expect(screen.getByText(/no diff available/i)).toBeInTheDocument()
    })
  })
})
