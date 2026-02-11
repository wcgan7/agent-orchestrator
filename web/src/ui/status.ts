import type { GlobalStatusSummary } from '../types/ui'

export function mapStatusSummary(status: string | undefined | null): GlobalStatusSummary {
  const normalized = (status || 'unknown').toLowerCase()

  switch (normalized) {
    case 'running':
      return {
        rawStatus: normalized,
        label: 'Running',
        severity: 'info',
        color: 'info',
        icon: 'play_arrow',
      }
    case 'done':
    case 'completed':
    case 'idle':
      return {
        rawStatus: normalized,
        label: normalized === 'idle' ? 'Idle' : 'Completed',
        severity: 'success',
        color: 'success',
        icon: 'check_circle',
      }
    case 'blocked':
      return {
        rawStatus: normalized,
        label: 'Blocked',
        severity: 'error',
        color: 'error',
        icon: 'error',
      }
    case 'paused':
      return {
        rawStatus: normalized,
        label: 'Paused',
        severity: 'warning',
        color: 'warning',
        icon: 'pause_circle',
      }
    default:
      return {
        rawStatus: normalized,
        label: 'Unknown',
        severity: 'default',
        color: 'default',
        icon: 'help',
      }
  }
}
