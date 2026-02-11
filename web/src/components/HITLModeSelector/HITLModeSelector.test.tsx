import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import HITLModeSelector from './HITLModeSelector'

// Mock the api module
vi.mock('../../api', () => ({
  buildApiUrl: (path: string, _projectDir?: string, _query?: Record<string, unknown>) => path,
  buildAuthHeaders: (extra: Record<string, string> = {}) => ({ ...extra }),
}))

const MOCK_MODES = [
  {
    mode: 'autopilot',
    display_name: 'Autopilot',
    description: 'Agents run freely.',
    approve_before_plan: false,
    approve_before_implement: false,
    approve_before_commit: false,
    approve_after_implement: false,
    allow_unattended: true,
    require_reasoning: false,
  },
  {
    mode: 'supervised',
    display_name: 'Supervised',
    description: 'Approve each step.',
    approve_before_plan: true,
    approve_before_implement: true,
    approve_before_commit: true,
    approve_after_implement: false,
    allow_unattended: false,
    require_reasoning: true,
  },
  {
    mode: 'collaborative',
    display_name: 'Collaborative',
    description: 'Work together with agents.',
    approve_before_plan: false,
    approve_before_implement: false,
    approve_before_commit: true,
    approve_after_implement: true,
    allow_unattended: false,
    require_reasoning: true,
  },
]

describe('HITLModeSelector', () => {
  let onModeChange: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.clearAllMocks()
    onModeChange = vi.fn()
  })

  it('renders the current mode', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ modes: MOCK_MODES }),
    })

    render(
      <HITLModeSelector currentMode="autopilot" onModeChange={onModeChange} />
    )

    await waitFor(() => {
      expect(screen.getByText('Autopilot')).toBeInTheDocument()
    })

    expect(screen.getByText('Agents run freely.')).toBeInTheDocument()
  })

  it('shows mode options when expanded', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ modes: MOCK_MODES }),
    })

    render(
      <HITLModeSelector currentMode="autopilot" onModeChange={onModeChange} />
    )

    await waitFor(() => {
      expect(screen.getByText('Autopilot')).toBeInTheDocument()
    })

    // Click to expand
    const currentMode = screen.getByText('Autopilot').closest('.hitl-current')!
    fireEvent.click(currentMode)

    // All mode options should be visible
    expect(screen.getByText('Supervised')).toBeInTheDocument()
    expect(screen.getByText('Collaborative')).toBeInTheDocument()
    expect(screen.getByText('Approve each step.')).toBeInTheDocument()
    expect(screen.getByText('Work together with agents.')).toBeInTheDocument()
  })

  it('calls onModeChange when selecting a mode', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ modes: MOCK_MODES }),
    })

    render(
      <HITLModeSelector currentMode="autopilot" onModeChange={onModeChange} />
    )

    await waitFor(() => {
      expect(screen.getByText('Autopilot')).toBeInTheDocument()
    })

    // Expand
    const currentMode = screen.getByText('Autopilot').closest('.hitl-current')!
    fireEvent.click(currentMode)

    // Select supervised mode
    const supervisedOption = screen.getByText('Supervised').closest('.hitl-option')!
    fireEvent.click(supervisedOption)

    expect(onModeChange).toHaveBeenCalledWith('supervised')
  })

  it('shows approval gate badges for modes', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ modes: MOCK_MODES }),
    })

    render(
      <HITLModeSelector currentMode="supervised" onModeChange={onModeChange} />
    )

    await waitFor(() => {
      expect(screen.getByText('Supervised')).toBeInTheDocument()
    })

    // Expand to see gate badges
    const currentMode = screen.getByText('Supervised').closest('.hitl-current')!
    fireEvent.click(currentMode)

    // Supervised has Plan, Impl, Commit gates
    // Note: 'Commit' appears in multiple modes so use getAllByText
    expect(screen.getByText('Plan')).toBeInTheDocument()
    expect(screen.getByText('Impl')).toBeInTheDocument()
    expect(screen.getAllByText('Commit').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Active badge on the current mode', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ modes: MOCK_MODES }),
    })

    render(
      <HITLModeSelector currentMode="autopilot" onModeChange={onModeChange} />
    )

    await waitFor(() => {
      expect(screen.getByText('Autopilot')).toBeInTheDocument()
    })

    // Expand
    const currentMode = screen.getByText('Autopilot').closest('.hitl-current')!
    fireEvent.click(currentMode)

    expect(screen.getByText('Active')).toBeInTheDocument()
  })

  it('falls back to default modes on fetch failure', async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new Error('Network error'))

    render(
      <HITLModeSelector currentMode="autopilot" onModeChange={onModeChange} />
    )

    await waitFor(() => {
      expect(screen.getByText('Autopilot')).toBeInTheDocument()
    })

    // Expand to verify default modes were loaded
    const currentMode = screen.getByText('Autopilot').closest('.hitl-current')!
    fireEvent.click(currentMode)

    expect(screen.getByText('Supervised')).toBeInTheDocument()
    expect(screen.getByText('Collaborative')).toBeInTheDocument()
  })
})
