/**
 * Agent reasoning viewer — shows step-by-step agent thinking for transparency.
 */

import { useState, useEffect, useCallback } from 'react'
import { Box, Card, Chip, Stack, Typography } from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'

interface ReasoningStep {
  step_name: string
  status: string
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

const STATUS_COLORS: Record<string, string> = {
  pending: '#9ca3af',
  running: '#2563eb',
  completed: '#16a34a',
  failed: '#dc2626',
  skipped: '#6b7280',
}

export default function ReasoningViewer({ taskId, projectDir }: Props) {
  const [reasonings, setReasonings] = useState<AgentReasoning[]>([])
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)

  const fetchReasoning = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl(`/api/v3/agents/reasoning/${taskId}`, projectDir),
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
    return <Typography className="reasoning-loading" color="text.secondary">Loading agent reasoning...</Typography>
  }

  if (reasonings.length === 0) {
    return (
      <Typography className="reasoning-empty" color="text.secondary">
        No agent reasoning available yet. Reasoning will appear when agents are working on this task.
      </Typography>
    )
  }

  return (
    <Box className="reasoning-viewer" sx={{ p: 1.5 }}>
      <Typography className="reasoning-title" variant="h6" sx={{ fontSize: '1rem', mb: 1.5 }}>
        Agent Reasoning
      </Typography>

      <Stack spacing={1}>
        {reasonings.map(r => (
          <Card key={r.agent_id} className="reasoning-agent" variant="outlined">
            <Stack
              className="reasoning-agent-header"
              direction="row"
              spacing={1}
              alignItems="center"
              useFlexGap
              flexWrap="wrap"
              onClick={() => setExpandedAgent(expandedAgent === r.agent_id ? null : r.agent_id)}
              sx={{ p: 1.25, cursor: 'pointer', borderBottom: 1, borderColor: 'divider' }}
            >
              <Typography className="reasoning-agent-role" variant="body2" sx={{ fontWeight: 700 }}>
                {r.agent_role}
              </Typography>
              <Typography className="reasoning-agent-id" variant="caption" color="text.secondary" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                {r.agent_id.slice(-8)}
              </Typography>
              {r.current_step && (
                <Chip className="reasoning-current-step" size="small" label={`\u25B6 ${r.current_step}`} color="info" variant="outlined" />
              )}
              <Typography className="reasoning-progress" variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                {r.steps.filter(s => s.status === 'completed').length}/{r.steps.length} steps
              </Typography>
            </Stack>

            {(expandedAgent === r.agent_id || reasonings.length === 1) && (
              <Stack className="reasoning-steps" spacing={0.75} sx={{ p: 1.25 }}>
                {r.steps.map(step => {
                  const stepKey = `${r.agent_id}:${step.step_name}`
                  const isExpanded = expandedSteps.has(stepKey)

                  return (
                    <Box key={step.step_name} className={`reasoning-step status-${step.status}`} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                      <Stack
                        className="reasoning-step-header"
                        direction="row"
                        spacing={1}
                        alignItems="center"
                        onClick={() => toggleStep(r.agent_id, step.step_name)}
                        sx={{ p: 1, cursor: 'pointer' }}
                      >
                        <Typography className={`step-status-icon status-${step.status}`} sx={{ color: STATUS_COLORS[step.status] || 'text.secondary', fontSize: '0.875rem' }}>
                          {getStepStatusIcon(step.status)}
                        </Typography>
                        <Typography className="step-name" variant="body2" sx={{ fontWeight: 600 }}>
                          {step.step_name}
                        </Typography>
                        {step.duration_ms !== undefined && (
                          <Typography className="step-duration" variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                            {step.duration_ms < 1000 ? `${step.duration_ms}ms` : `${(step.duration_ms / 1000).toFixed(1)}s`}
                          </Typography>
                        )}
                      </Stack>

                      {isExpanded && (
                        <Box className="reasoning-step-detail" sx={{ p: 1, pt: 0, borderTop: '1px solid', borderColor: 'divider' }}>
                          {step.reasoning && (
                            <Box className="step-reasoning" sx={{ mb: 1 }}>
                              <Typography className="step-detail-label" variant="caption" color="text.secondary">Reasoning</Typography>
                              <Typography className="step-detail-content" variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{step.reasoning}</Typography>
                            </Box>
                          )}
                          {step.output && (
                            <Box className="step-output">
                              <Typography className="step-detail-label" variant="caption" color="text.secondary">Output</Typography>
                              <Box
                                component="pre"
                                className="step-detail-content step-detail-pre"
                                sx={{
                                  m: 0,
                                  mt: 0.25,
                                  p: 1,
                                  borderRadius: 1,
                                  bgcolor: 'background.default',
                                  border: '1px solid',
                                  borderColor: 'divider',
                                  whiteSpace: 'pre-wrap',
                                  fontFamily: '"IBM Plex Mono", monospace',
                                  fontSize: '0.75rem',
                                }}
                              >
                                {step.output}
                              </Box>
                            </Box>
                          )}
                        </Box>
                      )}
                    </Box>
                  )
                })}
              </Stack>
            )}
          </Card>
        ))}
      </Stack>
    </Box>
  )
}
