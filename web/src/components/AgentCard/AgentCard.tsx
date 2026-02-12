/**
 * Live agent status card — shows what each agent is currently doing.
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'
import { AgentStream } from './AgentStream'
import { AgentControls } from './AgentControls'

interface AgentData {
  id: string
  agent_type: string
  display_name: string
  status: string
  task_id: string | null
  current_step: string | null
  current_file: string | null
  tokens_used: number
  cost_usd: number
  elapsed_seconds: number
  retries: number
  started_at: string | null
  last_heartbeat: string | null
  output_tail: string[]
}

interface AgentTypeInfo {
  role: string
  display_name: string
  description: string
  task_type_affinity: string[]
  allowed_steps: string[]
  limits: { max_tokens: number; max_time_seconds: number; max_cost_usd: number }
}

interface Props {
  projectDir?: string
}

export default function AgentPanel({ projectDir }: Props) {
  const [agents, setAgents] = useState<AgentData[]>([])
  const [agentTypes, setAgentTypes] = useState<AgentTypeInfo[]>([])
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
  const [showSpawn, setShowSpawn] = useState(false)
  const [spawnRole, setSpawnRole] = useState('implementer')

  const fetchAgents = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/v3/agents', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setAgents(data.agents || [])
      }
    } catch {
      // retry on next cycle
    }
  }, [projectDir])

  const fetchTypes = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/v3/agents/types', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setAgentTypes(data.types || [])
      }
    } catch {
      // ignore
    }
  }, [projectDir])

  useEffect(() => {
    fetchAgents()
    fetchTypes()
  }, [fetchAgents, fetchTypes])

  // Real-time updates
  useChannel('agents', useCallback((_event: string, _data: any) => {
    fetchAgents()
  }, [fetchAgents]))

  const handleSpawn = async () => {
    try {
      await fetch(
        buildApiUrl('/api/v3/agents/spawn', projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ role: spawnRole }),
        }
      )
      setShowSpawn(false)
      fetchAgents()
    } catch {
      // error
    }
  }

  const handleAction = async (agentId: string, action: string) => {
    try {
      await fetch(
        buildApiUrl(`/api/v3/agents/${agentId}/${action}`, projectDir),
        { method: 'POST', headers: buildAuthHeaders() }
      )
      fetchAgents()
    } catch {
      // error
    }
  }

  const activeCount = agents.filter(a => a.status === 'running' || a.status === 'paused').length
  const idleCount = agents.filter(a => a.status === 'idle').length

  const statusIcon = (status: string) => {
    switch (status) {
      case 'running': return '\u25B6'
      case 'paused': return '\u23F8'
      case 'idle': return '\u25CB'
      case 'failed': return '\u2717'
      case 'terminated': return '\u25A0'
      default: return '\u2022'
    }
  }

  return (
    <Box sx={{ p: 2 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Stack direction="row" spacing={1} alignItems="baseline">
          <Typography variant="h5">Agents</Typography>
          <Typography variant="body2" color="text.secondary">
            {activeCount} active, {idleCount} idle
          </Typography>
        </Stack>
        <Button variant="contained" onClick={() => setShowSpawn(!showSpawn)}>
          Spawn Agent
        </Button>
      </Stack>

      {showSpawn && (
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mb: 2, p: 1.5, bgcolor: 'background.default', borderRadius: 2 }}>
          <TextField
            select
            size="small"
            value={spawnRole}
            onChange={(e) => setSpawnRole(e.target.value)}
            sx={{ minWidth: 220 }}
          >
            {agentTypes.map(t => (
              <MenuItem key={t.role} value={t.role}>{t.display_name}</MenuItem>
            ))}
          </TextField>
          <Button variant="contained" color="success" onClick={handleSpawn}>Spawn</Button>
          <Button variant="outlined" onClick={() => setShowSpawn(false)}>Cancel</Button>
        </Stack>
      )}

      <Stack spacing={1}>
        {agents.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ py: 6, textAlign: 'center' }}>
            No agents running. Spawn one to get started.
          </Typography>
        ) : (
          agents.map(agent => (
            <Card
              key={agent.id}
              variant="outlined"
              sx={{
                borderLeft: '3px solid',
                borderLeftColor: (() => {
                  switch (agent.status) {
                    case 'running': return 'success.main'
                    case 'paused': return 'warning.main'
                    case 'failed': return 'error.main'
                    case 'idle': return 'text.disabled'
                    default: return 'divider'
                  }
                })(),
                opacity: agent.status === 'terminated' ? 0.7 : 1,
              }}
            >
              <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                <Stack
                  direction="row"
                  alignItems="center"
                  spacing={1}
                  sx={{ cursor: 'pointer' }}
                  onClick={() => setExpandedAgent(expandedAgent === agent.id ? null : agent.id)}
                >
                  <Box sx={{ minWidth: 18, textAlign: 'center', fontSize: '0.75rem' }}>
                  {statusIcon(agent.status)}
                  </Box>
                  <Box sx={{ minWidth: 0 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{agent.display_name}</Typography>
                    <Typography variant="caption" color="text.secondary">{agent.agent_type}</Typography>
                  </Box>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ ml: 1, flex: 1 }}>
                  {agent.task_id && (
                    <Chip size="small" label={agent.task_id.slice(-8)} sx={{ fontFamily: '"IBM Plex Mono", monospace' }} />
                  )}
                  {agent.current_step && (
                    <Chip size="small" variant="outlined" label={agent.current_step} />
                  )}
                  </Stack>
                  <Stack direction="row" spacing={1}>
                    <Typography variant="caption" color="text.secondary" sx={{ fontFamily: '"IBM Plex Mono", monospace' }} title="Tokens">
                      {(agent.tokens_used / 1000).toFixed(0)}k
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ fontFamily: '"IBM Plex Mono", monospace' }} title="Cost">
                      ${agent.cost_usd.toFixed(2)}
                    </Typography>
                  </Stack>
                </Stack>

                {expandedAgent === agent.id && (
                  <Box sx={{ mt: 1.25, pt: 1.25, borderTop: 1, borderColor: 'divider' }}>
                  {agent.current_file && (
                    <Stack direction="row" spacing={1} sx={{ mb: 0.5 }}>
                      <Typography variant="body2" color="text.secondary" sx={{ minWidth: 60 }}>File:</Typography>
                      <Typography component="code" variant="body2">{agent.current_file}</Typography>
                    </Stack>
                  )}
                  <Stack direction="row" spacing={1} sx={{ mb: 0.5 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ minWidth: 60 }}>Runtime:</Typography>
                    <Typography variant="body2">
                      {Math.floor(agent.elapsed_seconds / 60)}m {Math.floor(agent.elapsed_seconds % 60)}s
                    </Typography>
                  </Stack>
                  <Stack direction="row" spacing={1} sx={{ mb: 0.5 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ minWidth: 60 }}>Retries:</Typography>
                    <Typography variant="body2">{agent.retries}</Typography>
                  </Stack>

                  <AgentStream outputTail={agent.output_tail} />

                  <AgentControls
                    agentId={agent.id}
                    status={agent.status}
                    onAction={handleAction}
                  />
                  </Box>
                )}
              </CardContent>
            </Card>
          ))
        )}
      </Stack>
    </Box>
  )
}
