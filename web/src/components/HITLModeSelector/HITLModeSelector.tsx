/**
 * HITL (Human-in-the-loop) mode selector — lets users choose how agents interact
 * with them during task execution.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'

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

const DEFAULT_MODES: ModeConfig[] = [
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
  {
    mode: 'review_only',
    display_name: 'Review Only',
    description: 'Review all changes before commit.',
    approve_before_plan: false,
    approve_before_implement: false,
    approve_before_commit: true,
    approve_after_implement: true,
    allow_unattended: true,
    require_reasoning: false,
  },
]

const HITL_SELECTOR_STYLES = `
.hitl-selector {
  position: relative;
}

.hitl-current {
  display: flex;
  align-items: center;
  gap: var(--spacing-2);
  width: 100%;
  text-align: left;
  appearance: none;
  padding: var(--spacing-2) var(--spacing-3);
  min-height: 44px;
  background: var(--color-bg-secondary);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: border-color var(--transition-fast);
}

.hitl-current:hover {
  border-color: var(--color-primary-400);
}

.hitl-current:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
}

.hitl-icon {
  font-size: var(--text-base);
}

.hitl-current-info {
  flex: 1;
  min-width: 0;
}

.hitl-current-name {
  display: block;
  font-size: var(--text-sm);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
}

.hitl-current-desc {
  display: block;
  font-size: var(--text-xs);
  color: var(--color-text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.hitl-expand {
  font-size: 10px;
  color: var(--color-text-muted);
}

.hitl-options {
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  z-index: 1400;
  background: var(--color-bg-primary);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  padding: var(--spacing-1);
  max-width: 360px;
  min-width: 280px;
}

.hitl-option {
  display: block;
  width: 100%;
  text-align: left;
  appearance: none;
  border: 0;
  background: transparent;
  padding: var(--spacing-3);
  min-height: 44px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background var(--transition-fast);
}

.hitl-option:hover {
  background: var(--color-bg-secondary);
}

.hitl-option.active {
  background: color-mix(in srgb, var(--color-primary-100) 50%, transparent);
  border-left: 3px solid var(--color-primary-500);
}

.hitl-option:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 1px;
}

.hitl-option-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-2);
  margin-bottom: var(--spacing-0-5);
}

.hitl-option-icon {
  font-size: var(--text-base);
}

.hitl-option-name {
  font-size: var(--text-sm);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
}

.hitl-active-badge {
  margin-left: auto;
  font-size: 10px;
  padding: 1px var(--spacing-1-5);
  background: var(--color-primary-500);
  color: white;
  border-radius: var(--radius-sm);
}

.hitl-option-desc {
  font-size: var(--text-xs);
  color: var(--color-text-secondary);
  margin-bottom: var(--spacing-1);
}

.hitl-option-gates {
  display: flex;
  align-items: center;
  gap: var(--spacing-1);
  flex-wrap: wrap;
  margin-bottom: var(--spacing-1);
}

.gates-label {
  font-size: 10px;
  color: var(--color-text-muted);
}

.gate-badge {
  font-size: 10px;
  padding: 1px var(--spacing-1);
  background: var(--color-warning-100);
  color: var(--color-warning-700);
  border-radius: var(--radius-sm);
}

.hitl-option-flags {
  display: flex;
  gap: var(--spacing-1);
}

.flag-badge {
  font-size: 10px;
  padding: 1px var(--spacing-1);
  border-radius: var(--radius-sm);
}

.flag-unattended {
  background: var(--color-success-100);
  color: var(--color-success-700);
}

.flag-reasoning {
  background: var(--color-info-100);
  color: var(--color-info-700);
}

@media (max-width: 720px) {
  .hitl-options {
    position: fixed;
    left: 1rem;
    right: 1rem;
    top: auto;
    bottom: 1rem;
    max-width: none;
    min-width: 0;
    max-height: 70vh;
    overflow: auto;
  }
}
`

export default function HITLModeSelector({ currentMode, onModeChange, projectDir }: Props) {
  const [modes, setModes] = useState<ModeConfig[]>(DEFAULT_MODES)
  const [expanded, setExpanded] = useState(false)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const listboxIdRef = useRef(`hitl-listbox-${Math.random().toString(36).slice(2, 10)}`)

  const fetchModes = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/collaboration/modes', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (!resp.ok) {
        setModes(DEFAULT_MODES)
        return
      }
      const data = await resp.json() as { modes?: ModeConfig[] }
      if (Array.isArray(data.modes) && data.modes.length > 0) {
        setModes(data.modes)
      } else {
        setModes(DEFAULT_MODES)
      }
    } catch {
      setModes(DEFAULT_MODES)
    }
  }, [projectDir])

  useEffect(() => {
    fetchModes()
  }, [fetchModes])

  useEffect(() => {
    if (!expanded) return
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target instanceof Node ? event.target : null
      if (!target) return
      if (!containerRef.current?.contains(target)) {
        setExpanded(false)
      }
    }
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setExpanded(false)
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onEscape)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onEscape)
    }
  }, [expanded])

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
    <div className="hitl-selector" ref={containerRef}>
      <style>{HITL_SELECTOR_STYLES}</style>
      <button
        type="button"
        className="hitl-current"
        aria-haspopup="listbox"
        aria-expanded={expanded}
        aria-controls={listboxIdRef.current}
        onClick={() => setExpanded((value) => !value)}
      >
        <span className="hitl-icon">{MODE_ICONS[currentMode] || '\u2699'}</span>
        <div className="hitl-current-info">
          <span className="hitl-current-name">{currentModeConfig?.display_name || currentMode}</span>
          <span className="hitl-current-desc">{currentModeConfig?.description || ''}</span>
        </div>
        <span className="hitl-expand">{expanded ? '\u25B2' : '\u25BC'}</span>
      </button>

      {expanded && (
        <div className="hitl-options" role="listbox" id={listboxIdRef.current}>
          {modes.map(mode => {
            const gates = getGateBadges(mode)
            const isActive = mode.mode === currentMode

            return (
              <button
                type="button"
                key={mode.mode}
                className={`hitl-option ${isActive ? 'active' : ''}`}
                role="option"
                aria-selected={isActive}
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
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
