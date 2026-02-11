import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import AppShell from './AppShell'
import type { AppNavSection } from '../../types/ui'

const sections: AppNavSection[] = [
  { id: 'overview', label: 'Overview', description: 'Summary' },
  { id: 'execution', label: 'Execution', description: 'Run controls' },
  { id: 'tasks', label: 'Tasks', description: 'Task board' },
  { id: 'agents', label: 'Agents', description: 'Workers' },
  { id: 'diagnostics', label: 'Diagnostics', description: 'Health checks' },
]

function renderShell(onSectionChange = vi.fn()) {
  return render(
    <ThemeProvider theme={createTheme()}>
      <AppShell
        title="Feature PRD Runner"
        sections={sections}
        activeSection="overview"
        onSectionChange={onSectionChange}
        statusSummary={{ rawStatus: 'running', label: 'Running', severity: 'info', color: 'info', icon: 'play_arrow' }}
        commandHint="Cmd/Ctrl + K"
        onOpenCommandPalette={vi.fn()}
        commandBarCenter={<div>Project Selector</div>}
        commandBarRight={<button type="button">Right Action</button>}
        rightRail={<div>Rail Content</div>}
      >
        <div>Main Content</div>
      </AppShell>
    </ThemeProvider>
  )
}

describe('AppShell', () => {
  it('renders shell structure', () => {
    renderShell()

    expect(screen.getByText('Feature PRD Runner')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /command palette/i })).toBeInTheDocument()
    expect(screen.getByText('Rail Content')).toBeInTheDocument()
    expect(screen.getByText('Main Content')).toBeInTheDocument()
  })

  it('changes section when nav item clicked', async () => {
    const user = userEvent.setup()
    const onSectionChange = vi.fn()
    renderShell(onSectionChange)

    await user.click(screen.getByRole('button', { name: /execution run controls/i }))
    expect(onSectionChange).toHaveBeenCalledWith('execution')
  })
})
