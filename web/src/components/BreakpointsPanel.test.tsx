import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import BreakpointsPanel from './BreakpointsPanel'

describe('BreakpointsPanel', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    user = userEvent.setup()
  })

  it('fetches breakpoints with project_dir and shows empty state', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    })

    render(<BreakpointsPanel projectDir="/tmp/project" />)

    await waitFor(() => {
      expect(screen.getByText(/no breakpoints set/i)).toBeInTheDocument()
    })

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/breakpoints?project_dir=%2Ftmp%2Fproject'),
      expect.any(Object)
    )
  })

  it('creates a breakpoint and refreshes the list', async () => {
    const created = {
      id: 'bp-123',
      trigger: 'before_step',
      target: 'verify',
      task_id: null,
      condition: null,
      action: 'pause',
      enabled: true,
      hit_count: 0,
      created_at: '2024-01-01T10:00:00Z',
    }

    let getCount = 0
    global.fetch = vi.fn().mockImplementation((url: any, options?: any) => {
      const urlString = url.toString()
      const method = options?.method || 'GET'

      if (urlString.includes('/api/breakpoints') && method === 'GET') {
        getCount++
        return Promise.resolve({
          ok: true,
          json: async () => (getCount === 1 ? [] : [created]),
        })
      }

      if (urlString.includes('/api/breakpoints') && method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ success: true, message: 'Breakpoint created', data: created }),
        })
      }

      return Promise.resolve({
        ok: true,
        json: async () => ({}),
      })
    })

    render(<BreakpointsPanel projectDir="/tmp/project" />)

    await waitFor(() => {
      expect(screen.getByText(/no breakpoints set/i)).toBeInTheDocument()
    })

    const createButton = screen.getByRole('button', { name: /create/i })
    await user.click(createButton)

    await waitFor(() => {
      expect(screen.getByText('bp-123')).toBeInTheDocument()
    })

    const postCalls = (global.fetch as any).mock.calls.filter((c: any[]) => c[1]?.method === 'POST')
    expect(postCalls.length).toBeGreaterThanOrEqual(1)
    expect(postCalls[0][0]).toContain('project_dir=%2Ftmp%2Fproject')
    expect(postCalls[0][1].body).toContain('"trigger":"before_step"')
    expect(postCalls[0][1].body).toContain('"target":"verify"')
  })
})
