import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Chat from './Chat'

describe('Chat', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    user = userEvent.setup()
  })

  it('renders chat toggle button when closed', () => {
    render(<Chat />)
    expect(screen.getByRole('button', { name: /open chat/i })).toBeInTheDocument()
  })

  it('opens chat panel when toggle clicked', async () => {
    render(<Chat />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    expect(screen.getByText(/live collaboration chat/i)).toBeInTheDocument()
  })

  it('displays no active run message when runId is not provided', async () => {
    render(<Chat />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    expect(screen.getByText(/no active run/i)).toBeInTheDocument()
    expect(screen.getByText(/chat is available when a run is active/i)).toBeInTheDocument()
  })

  it('fetches and displays messages when run is active', async () => {
    const mockMessages = [
      {
        id: 'msg-1',
        type: 'guidance',
        content: 'Focus on error handling',
        timestamp: '2024-01-01T10:00:00Z',
        from_human: true,
        metadata: {},
      },
      {
        id: 'msg-2',
        type: 'clarification_response',
        content: 'Adding error handling now',
        timestamp: '2024-01-01T10:01:00Z',
        from_human: false,
        metadata: {},
      },
    ]

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockMessages,
    })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByText(/focus on error handling/i)).toBeInTheDocument()
      expect(screen.getByText(/adding error handling now/i)).toBeInTheDocument()
    })
  })

  it('distinguishes between human and worker messages', async () => {
    const mockMessages = [
      {
        id: 'msg-1',
        type: 'guidance',
        content: 'Human message',
        timestamp: '2024-01-01T10:00:00Z',
        from_human: true,
        metadata: {},
      },
      {
        id: 'msg-2',
        type: 'clarification_response',
        content: 'Worker message',
        timestamp: '2024-01-01T10:01:00Z',
        from_human: false,
        metadata: {},
      },
    ]

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockMessages,
    })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      const messages = document.querySelectorAll('.chat-message')
      expect(messages.length).toBeGreaterThanOrEqual(2)
      expect(messages[0]).toHaveClass('human')
      expect(messages[1]).toHaveClass('worker')
    })
  })

  it('sends message when send button is clicked', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument()
    })

    const input = screen.getByPlaceholderText(/type a message/i)
    await user.type(input, 'Test message')

    const sendButton = screen.getByRole('button', { name: /send/i })
    await user.click(sendButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/messages'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Test message'),
        })
      )
    })
  })

  it('sends requirement messages with metadata', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })

    render(<Chat runId="run-123" projectDir="/tmp/project" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    const typeSelect = await screen.findByLabelText(/message type/i)
    await user.selectOptions(typeSelect, 'requirement')

    const requirementTaskId = screen.getByLabelText(/requirement task id/i)
    await user.type(requirementTaskId, 'phase-1')

    const prioritySelect = screen.getByLabelText(/requirement priority/i)
    await user.selectOptions(prioritySelect, 'high')

    const input = screen.getByPlaceholderText(/type a message/i)
    await user.type(input, 'Must add rate limiting')

    const sendButton = screen.getByRole('button', { name: /send/i })
    await user.click(sendButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/messages'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"type":"requirement"'),
        })
      )
    })

    const [, postCall] = (global.fetch as any).mock.calls
    expect(postCall[0]).toContain('project_dir=')
    expect(postCall[1].body).toContain('"priority":"high"')
    expect(postCall[1].body).toContain('"task_id":"phase-1"')
  })

  it('requires a task id for correction messages', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })

    render(<Chat runId="run-123" projectDir="/tmp/project" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    const typeSelect = await screen.findByLabelText(/message type/i)
    await user.selectOptions(typeSelect, 'correction')

    const input = screen.getByPlaceholderText(/type a message/i)
    await user.type(input, 'Missing input validation')

    const sendButton = screen.getByRole('button', { name: /send/i })
    expect(sendButton).toBeDisabled()

    const correctionTaskId = screen.getByLabelText(/correction task id/i)
    await user.type(correctionTaskId, 'phase-2')
    expect(sendButton).not.toBeDisabled()

    const correctionFile = screen.getByLabelText(/correction file path/i)
    await user.type(correctionFile, 'src/app.py')

    const correctionFix = screen.getByLabelText(/correction suggested fix/i)
    await user.type(correctionFix, 'Add validation for empty input')

    await user.click(sendButton)

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/messages'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"type":"correction"'),
        })
      )
    })

    const [, postCall] = (global.fetch as any).mock.calls
    expect(postCall[1].body).toContain('"task_id":"phase-2"')
    expect(postCall[1].body).toContain('"file":"src/app.py"')
    expect(postCall[1].body).toContain('"suggested_fix":"Add validation for empty input"')
    expect(postCall[1].body).toContain('"issue":"Missing input validation"')
  })

  it('clears input after sending message', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument()
    })

    const input = screen.getByPlaceholderText(/type a message/i) as HTMLTextAreaElement
    await user.type(input, 'Test message')
    expect(input.value).toBe('Test message')

    const sendButton = screen.getByRole('button', { name: /send/i })
    await user.click(sendButton)

    await waitFor(() => {
      expect(input.value).toBe('')
    })
  })

  it('sends message on Enter key press', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument()
    })

    const input = screen.getByPlaceholderText(/type a message/i)
    await user.type(input, 'Test message{Enter}')

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/messages'),
        expect.objectContaining({
          method: 'POST',
        })
      )
    })
  })

  it('disables send button when input is empty', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      const sendButton = screen.getByRole('button', { name: /send/i })
      expect(sendButton).toBeDisabled()
    })
  })

  it('displays error message on fetch failure', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network error'))

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument()
    })
  })

  it('closes chat panel when close button is clicked', async () => {
    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByText(/live collaboration chat/i)).toBeInTheDocument()
    })

    const closeButton = screen.getByRole('button', { name: /close chat/i })
    await user.click(closeButton)

    expect(screen.queryByText(/live collaboration chat/i)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /open chat/i })).toBeInTheDocument()
  })

  it('displays usage tips', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })

    render(<Chat runId="run-123" />)

    const toggleButton = screen.getByRole('button', { name: /open chat/i })
    await user.click(toggleButton)

    await waitFor(() => {
      expect(screen.getByText(/provide guidance/i)).toBeInTheDocument()
      expect(screen.getByText(/ask questions/i)).toBeInTheDocument()
    })
  })
})
