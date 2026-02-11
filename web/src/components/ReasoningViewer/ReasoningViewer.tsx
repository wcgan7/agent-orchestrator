/**
 * Agent reasoning viewer — shows step-by-step agent thinking for transparency.
 */

import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'
import './ReasoningViewer.css'

interface ReasoningStep {
  step_name: string
  status: string       // pending | running | completed | failed | skipped
  reasoning?: string
  output?: string
  started_at?: string
  completed_at?: string
  duration_ms?: number
}

interface AgentReasoning {
  agent_id: string
  agent_role: string
  task_id: string
  pipeline_id?: string
  steps: ReasoningStep[]
  current_step?: string
}

interface Props {
  taskId: string
  projectDir?: string
}

export default function ReasoningViewer({ taskId, projectDir }: Props) {
  const [reasonings, setReasonings] = useState<AgentReasoning[]>([])
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)

  const fetchReasoning = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl(`/api/v2/agents/reasoning/${taskId}`, projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setReasonings(data.reasonings || [])
      }
    } catch {
      // Not available yet — show placeholder
    } finally {
      setLoading(false)
    }
  }, [taskId, projectDir])

  useEffect(() => {
    fetchReasoning()
  }, [fetchReasoning])

  // Real-time updates via WebSocket instead of polling
  useChannel('agents', useCallback((_event: string, _data: any) => {
    fetchReasoning()
  }, [fetchReasoning]))

  const toggleStep = (agentId: string, stepName: string) => {
    const key = `${agentId}:${stepName}`
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '\u2713'
      case 'running': return '\u25B6'
      case 'failed': return '\u2717'
      case 'skipped': return '\u2192'
      default: return '\u25CB'
    }
  }

  if (loading) {
    return <div className="reasoning-loading">Loading agent reasoning...</div>
  }

  if (reasonings.length === 0) {
    return (
      <div className="reasoning-empty">
        No agent reasoning available yet. Reasoning will appear when agents are working on this task.
      </div>
    )
  }

  return (
    <div className="reasoning-viewer">
      <h3 className="reasoning-title">Agent Reasoning</h3>

      {reasonings.map(r => (
        <div key={r.agent_id} className="reasoning-agent">
          <div
            className="reasoning-agent-header"
            onClick={() => setExpandedAgent(expandedAgent === r.agent_id ? null : r.agent_id)}
          >
            <span className="reasoning-agent-role">{r.agent_role}</span>
            <span className="reasoning-agent-id">{r.agent_id.slice(-8)}</span>
            {r.current_step && (
              <span className="reasoning-current-step">
                {'\u25B6'} {r.current_step}
              </span>
            )}
            <span className="reasoning-progress">
              {r.steps.filter(s => s.status === 'completed').length}/{r.steps.length} steps
            </span>
          </div>

          {(expandedAgent === r.agent_id || reasonings.length === 1) && (
            <div className="reasoning-steps">
              {r.steps.map(step => {
                const stepKey = `${r.agent_id}:${step.step_name}`
                const isExpanded = expandedSteps.has(stepKey)

                return (
                  <div key={step.step_name} className={`reasoning-step status-${step.status}`}>
                    <div
                      className="reasoning-step-header"
                      onClick={() => toggleStep(r.agent_id, step.step_name)}
                    >
                      <span className={`step-status-icon status-${step.status}`}>
                        {getStepStatusIcon(step.status)}
                      </span>
                      <span className="step-name">{step.step_name}</span>
                      {step.duration_ms !== undefined && (
                        <span className="step-duration">
                          {step.duration_ms < 1000
                            ? `${step.duration_ms}ms`
                            : `${(step.duration_ms / 1000).toFixed(1)}s`
                          }
                        </span>
                      )}
                    </div>

                    {isExpanded && (
                      <div className="reasoning-step-detail">
                        {step.reasoning && (
                          <div className="step-reasoning">
                            <div className="step-detail-label">Reasoning</div>
                            <div className="step-detail-content">{step.reasoning}</div>
                          </div>
                        )}
                        {step.output && (
                          <div className="step-output">
                            <div className="step-detail-label">Output</div>
                            <pre className="step-detail-content step-detail-pre">{step.output}</pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
