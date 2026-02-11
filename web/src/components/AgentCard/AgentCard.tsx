/**
 * Live agent status card — shows what each agent is currently doing.
 */

import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'
import { AgentStream } from './AgentStream'
import { AgentControls } from './AgentControls'
import './AgentCard.css'

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
        buildApiUrl('/api/v2/agents', projectDir),
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
        buildApiUrl('/api/v2/agents/types', projectDir),
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
        buildApiUrl('/api/v2/agents/spawn', projectDir),
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
        buildApiUrl(`/api/v2/agents/${agentId}/${action}`, projectDir),
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
    <div className="agent-panel">
      <div className="agent-panel-header">
        <div className="agent-panel-title-group">
          <h2 className="agent-panel-title">Agents</h2>
          <span className="agent-panel-stats">
            {activeCount} active, {idleCount} idle
          </span>
        </div>
        <button className="agent-spawn-btn" onClick={() => setShowSpawn(!showSpawn)}>
          + Spawn Agent
        </button>
      </div>

      {showSpawn && (
        <div className="agent-spawn-form">
          <select
            className="agent-spawn-select"
            value={spawnRole}
            onChange={(e) => setSpawnRole(e.target.value)}
          >
            {agentTypes.map(t => (
              <option key={t.role} value={t.role}>{t.display_name}</option>
            ))}
          </select>
          <button className="agent-spawn-confirm" onClick={handleSpawn}>Spawn</button>
          <button className="agent-spawn-cancel" onClick={() => setShowSpawn(false)}>Cancel</button>
        </div>
      )}

      <div className="agent-cards">
        {agents.length === 0 ? (
          <div className="agent-empty">No agents running. Spawn one to get started.</div>
        ) : (
          agents.map(agent => (
            <div key={agent.id} className={`agent-card status-${agent.status}`}>
              <div className="agent-card-header" onClick={() => setExpandedAgent(
                expandedAgent === agent.id ? null : agent.id
              )}>
                <span className={`agent-status-dot status-${agent.status}`}>
                  {statusIcon(agent.status)}
                </span>
                <div className="agent-card-info">
                  <span className="agent-card-name">{agent.display_name}</span>
                  <span className="agent-card-role">{agent.agent_type}</span>
                </div>
                <div className="agent-card-meta">
                  {agent.task_id && (
                    <span className="agent-card-task">{agent.task_id.slice(-8)}</span>
                  )}
                  {agent.current_step && (
                    <span className="agent-card-step">{agent.current_step}</span>
                  )}
                </div>
                <div className="agent-card-stats">
                  <span className="agent-stat" title="Tokens">{(agent.tokens_used / 1000).toFixed(0)}k</span>
                  <span className="agent-stat" title="Cost">${agent.cost_usd.toFixed(2)}</span>
                </div>
              </div>

              {expandedAgent === agent.id && (
                <div className="agent-card-expanded">
                  {agent.current_file && (
                    <div className="agent-detail-row">
                      <span className="agent-detail-label">File:</span>
                      <code className="agent-detail-value">{agent.current_file}</code>
                    </div>
                  )}
                  <div className="agent-detail-row">
                    <span className="agent-detail-label">Runtime:</span>
                    <span className="agent-detail-value">
                      {Math.floor(agent.elapsed_seconds / 60)}m {Math.floor(agent.elapsed_seconds % 60)}s
                    </span>
                  </div>
                  <div className="agent-detail-row">
                    <span className="agent-detail-label">Retries:</span>
                    <span className="agent-detail-value">{agent.retries}</span>
                  </div>

                  <AgentStream outputTail={agent.output_tail} />

                  <AgentControls
                    agentId={agent.id}
                    status={agent.status}
                    onAction={handleAction}
                  />
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
