import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ToastProvider } from '../contexts/ToastContext'
import TaskLauncher from './TaskLauncher'

const renderWithProvider = (ui: React.ReactElement) => {
  return render(<ToastProvider>{ui}</ToastProvider>)
}

describe('TaskLauncher', () => {
  const mockOnRunStarted = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    global.fetch = vi.fn()
    // Mock localStorage
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: vi.fn(),
        setItem: vi.fn(),
        removeItem: vi.fn(),
      },
      writable: true,
    })
  })

  it('renders with default quick_task mode', () => {
    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    expect(screen.getByText('Launch New Run')).toBeInTheDocument()
    expect(screen.getByText('Quick Action')).toBeInTheDocument()
    expect(screen.getByText('Quick Prompt')).toBeInTheDocument()
    expect(screen.getByText('Full PRD')).toBeInTheDocument()

    // Quick task mode should be active by default
    const quickTaskButton = screen.getByRole('button', { name: /^quick action$/i })
    expect(quickTaskButton).toHaveClass('active')
  })

  it('switches between modes', () => {
    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const fullPrdButton = screen.getByRole('button', { name: /full prd/i })
    fireEvent.click(fullPrdButton)

    expect(fullPrdButton).toHaveClass('active')
  })

  it('disables submit button when content is empty', () => {
    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const submitButton = screen.getByRole('button', { name: /run quick action/i })
    expect(submitButton).toBeDisabled()
  })

  it('enables submit button when content is provided', () => {
    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const textarea = screen.getByLabelText(/action prompt/i)
    fireEvent.change(textarea, { target: { value: 'Test prompt' } })

    const submitButton = screen.getByRole('button', { name: /run quick action/i })
    expect(submitButton).not.toBeDisabled()
  })

  it('submits quick prompt and calls API correctly', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      json: async () => ({
        success: true,
        run_id: 'run-123',
        message: 'Run started successfully',
        prd_path: '/path/to/prd.md',
      }),
    })
    global.fetch = mockFetch

    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    // Switch to quick_prompt mode
    const quickPromptButton = screen.getByRole('button', { name: /quick prompt/i })
    fireEvent.click(quickPromptButton)

    const textarea = screen.getByLabelText(/feature prompt/i)
    fireEvent.change(textarea, { target: { value: 'Add user authentication' } })

    const submitButton = screen.getByRole('button', { name: /start run/i })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalled()
    })

    const callArgs = mockFetch.mock.calls[0]
    expect(callArgs[0]).toContain('/api/runs/start')

    const body = JSON.parse(callArgs[1].body)
    expect(body.mode).toBe('quick_prompt')
    expect(body.content).toBe('Add user authentication')
    expect(body.verification_profile).toBe('none')

    // Verify callback was invoked with run ID
    await waitFor(() => {
      expect(mockOnRunStarted).toHaveBeenCalledWith('run-123')
    })

    // Verify input was cleared after success
    expect(textarea).toHaveValue('')
  })

  it('promotes quick action to task when enabled', async () => {
    const mockFetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          message: 'Quick action executed successfully',
          quick_run: { id: 'qrun-123', status: 'completed' },
          error: null,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          task_id: 'task-123',
        }),
      })

    global.fetch = mockFetch

    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const textarea = screen.getByLabelText(/action prompt/i)
    fireEvent.change(textarea, { target: { value: 'Add audit logging for API errors' } })

    const promoteCheckbox = screen.getByRole('checkbox', { name: /save result as task/i })
    fireEvent.click(promoteCheckbox)

    const submitButton = screen.getByRole('button', { name: /run quick action/i })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2)
    })

    expect(mockFetch.mock.calls[0][0]).toContain('/api/v3/quick-runs')
    expect(mockFetch.mock.calls[1][0]).toContain('/api/v3/quick-runs/qrun-123/promote')

    const promoteBody = JSON.parse(mockFetch.mock.calls[1][1].body)
    expect(promoteBody.title).toContain('Add audit logging for API errors')
    expect(promoteBody.task_type).toBe('feature')
    expect(promoteBody.priority).toBe('P2')
  })

  it('handles API errors gracefully', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      json: async () => ({
        success: false,
        message: 'Failed to start run: API error',
        run_id: null,
        prd_path: null,
      }),
    })
    global.fetch = mockFetch

    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    // Switch to quick_prompt mode
    const quickPromptButton = screen.getByRole('button', { name: /quick prompt/i })
    fireEvent.click(quickPromptButton)

    const textarea = screen.getByLabelText(/feature prompt/i)
    fireEvent.change(textarea, { target: { value: 'Test prompt' } })

    const submitButton = screen.getByRole('button', { name: /start run/i })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText(/failed to start run: api error/i)).toBeInTheDocument()
    })

    expect(mockOnRunStarted).not.toHaveBeenCalled()
  })

  it('disables submit button while submitting', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      json: async () => ({ success: true, run_id: 'run-123', message: 'Success' }),
    })
    global.fetch = mockFetch

    renderWithProvider(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    // Switch to quick_prompt mode
    const quickPromptButton = screen.getByRole('button', { name: /quick prompt/i })
    fireEvent.click(quickPromptButton)

    const textarea = screen.getByLabelText(/feature prompt/i)
    fireEvent.change(textarea, { target: { value: 'Test prompt' } })

    const submitButton = screen.getByRole('button', { name: /start run/i })
    fireEvent.click(submitButton)

    // Button should be disabled during submission
    await waitFor(() => {
      expect(submitButton).toBeDisabled()
    })

    // Wait for submission to complete and verify input was cleared
    await waitFor(() => {
      expect(textarea).toHaveValue('')
    })

    // Button stays disabled because content is now empty (correct behavior)
    expect(submitButton).toBeDisabled()
  })
})
