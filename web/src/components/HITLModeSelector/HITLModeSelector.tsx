/**
 * HITL (Human-in-the-loop) mode selector — lets users choose how agents interact
 * with them during task execution.
 */

import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import './HITLModeSelector.css'

interface ModeConfig {
  mode: string
  display_name: string
  description: string
  approve_before_plan: boolean
  approve_before_implement: boolean
  approve_before_commit: boolean
  approve_after_implement: boolean
  allow_unattended: boolean
  require_reasoning: boolean
}

interface Props {
  currentMode: string
  onModeChange: (mode: string) => void
  projectDir?: string
}

const MODE_ICONS: Record<string, string> = {
  autopilot: '\u{1F680}',
  supervised: '\u{1F440}',
  collaborative: '\u{1F91D}',
  review_only: '\u{1F50D}',
}

export default function HITLModeSelector({ currentMode, onModeChange, projectDir }: Props) {
  const [modes, setModes] = useState<ModeConfig[]>([])
  const [expanded, setExpanded] = useState(false)

  const fetchModes = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/v2/collaboration/modes', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setModes(data.modes || [])
      }
    } catch {
      // Use defaults
      setModes([
        { mode: 'autopilot', display_name: 'Autopilot', description: 'Agents run freely.', approve_before_plan: false, approve_before_implement: false, approve_before_commit: false, approve_after_implement: false, allow_unattended: true, require_reasoning: false },
        { mode: 'supervised', display_name: 'Supervised', description: 'Approve each step.', approve_before_plan: true, approve_before_implement: true, approve_before_commit: true, approve_after_implement: false, allow_unattended: false, require_reasoning: true },
        { mode: 'collaborative', display_name: 'Collaborative', description: 'Work together with agents.', approve_before_plan: false, approve_before_implement: false, approve_before_commit: true, approve_after_implement: true, allow_unattended: false, require_reasoning: true },
        { mode: 'review_only', display_name: 'Review Only', description: 'Review all changes before commit.', approve_before_plan: false, approve_before_implement: false, approve_before_commit: true, approve_after_implement: true, allow_unattended: true, require_reasoning: false },
      ])
    }
  }, [projectDir])

  useEffect(() => {
    fetchModes()
  }, [fetchModes])

  const currentModeConfig = modes.find(m => m.mode === currentMode) || modes[0]

  const getGateBadges = (mode: ModeConfig) => {
    const gates: string[] = []
    if (mode.approve_before_plan) gates.push('Plan')
    if (mode.approve_before_implement) gates.push('Impl')
    if (mode.approve_after_implement) gates.push('Review')
    if (mode.approve_before_commit) gates.push('Commit')
    return gates
  }

  return (
    <div className="hitl-selector">
      <div className="hitl-current" onClick={() => setExpanded(!expanded)}>
        <span className="hitl-icon">{MODE_ICONS[currentMode] || '\u2699'}</span>
        <div className="hitl-current-info">
          <span className="hitl-current-name">{currentModeConfig?.display_name || currentMode}</span>
          <span className="hitl-current-desc">{currentModeConfig?.description || ''}</span>
        </div>
        <span className="hitl-expand">{expanded ? '\u25B2' : '\u25BC'}</span>
      </div>

      {expanded && (
        <div className="hitl-options">
          {modes.map(mode => {
            const gates = getGateBadges(mode)
            const isActive = mode.mode === currentMode

            return (
              <div
                key={mode.mode}
                className={`hitl-option ${isActive ? 'active' : ''}`}
                onClick={() => {
                  onModeChange(mode.mode)
                  setExpanded(false)
                }}
              >
                <div className="hitl-option-header">
                  <span className="hitl-option-icon">{MODE_ICONS[mode.mode] || '\u2699'}</span>
                  <span className="hitl-option-name">{mode.display_name}</span>
                  {isActive && <span className="hitl-active-badge">Active</span>}
                </div>
                <div className="hitl-option-desc">{mode.description}</div>
                {gates.length > 0 && (
                  <div className="hitl-option-gates">
                    <span className="gates-label">Approval gates:</span>
                    {gates.map(g => (
                      <span key={g} className="gate-badge">{g}</span>
                    ))}
                  </div>
                )}
                <div className="hitl-option-flags">
                  {mode.allow_unattended && <span className="flag-badge flag-unattended">Unattended</span>}
                  {mode.require_reasoning && <span className="flag-badge flag-reasoning">Shows Reasoning</span>}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
