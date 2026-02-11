import { describe, it, expect } from 'vitest'
import { mapStatusSummary } from './status'

describe('mapStatusSummary', () => {
  it('maps running status', () => {
    const result = mapStatusSummary('running')
    expect(result.label).toBe('Running')
    expect(result.color).toBe('info')
    expect(result.icon).toBe('play_arrow')
  })

  it('maps blocked status', () => {
    const result = mapStatusSummary('blocked')
    expect(result.label).toBe('Blocked')
    expect(result.color).toBe('error')
  })

  it('falls back to unknown', () => {
    const result = mapStatusSummary('mystery')
    expect(result.label).toBe('Unknown')
    expect(result.color).toBe('default')
  })
})
