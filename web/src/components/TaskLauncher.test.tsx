import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import TaskLauncher from './TaskLauncher'

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

  it('renders with default quick_prompt mode', () => {
    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    expect(screen.getByText('Launch New Run')).toBeInTheDocument()
    expect(screen.getByText('Quick Prompt')).toBeInTheDocument()
    expect(screen.getByText('Full PRD')).toBeInTheDocument()

    // Quick prompt mode should be active by default
    const quickButton = screen.getByRole('button', { name: /quick prompt/i })
    expect(quickButton).toHaveClass('active')
  })

  it('switches between modes', () => {
    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const fullPrdButton = screen.getByRole('button', { name: /full prd/i })
    fireEvent.click(fullPrdButton)

    expect(fullPrdButton).toHaveClass('active')
  })

  it('disables submit button when content is empty', () => {
    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const submitButton = screen.getByRole('button', { name: /start run/i })
    expect(submitButton).toBeDisabled()
  })

  it('enables submit button when content is provided', () => {
    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const textarea = screen.getByLabelText(/feature prompt/i)
    fireEvent.change(textarea, { target: { value: 'Test prompt' } })

    const submitButton = screen.getByRole('button', { name: /start run/i })
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

    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

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

    await waitFor(() => {
      expect(screen.getByText(/run started successfully/i)).toBeInTheDocument()
    })

    expect(mockOnRunStarted).toHaveBeenCalledWith('run-123')
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

    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

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

    render(<TaskLauncher projectDir="/test/project" onRunStarted={mockOnRunStarted} />)

    const textarea = screen.getByLabelText(/feature prompt/i)
    fireEvent.change(textarea, { target: { value: 'Test prompt' } })

    const submitButton = screen.getByRole('button', { name: /start run/i })
    fireEvent.click(submitButton)

    // Button should be disabled during submission
    await waitFor(() => {
      expect(submitButton).toBeDisabled()
    })

    // Wait for success message
    await waitFor(() => {
      expect(screen.getByText(/success/i)).toBeInTheDocument()
    })
  })
})
