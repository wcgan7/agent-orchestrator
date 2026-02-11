/**
 * Agent overview grid — aggregate stats across all agents.
 *
 * Shows summary cards (total agents, running, idle, errored, total cost,
 * total tokens) and a per-type breakdown row. Designed to sit above or
 * alongside the individual AgentCard list.
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'
import './AgentOverview.css'

interface AgentData {
  id: string
  agent_type: string
  display_name: string
  status: string
  task_id: string | null
  tokens_used: number
  cost_usd: number
  elapsed_seconds: number
}

interface Props {
  projectDir?: string
}

export function AgentOverview({ projectDir }: Props) {
  const [agents, setAgents] = useState<AgentData[]>([])

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

  useEffect(() => {
    fetchAgents()
  }, [fetchAgents])

  useChannel('agents', useCallback((_event: string, _data: any) => {
    fetchAgents()
  }, [fetchAgents]))

  const stats = useMemo(() => {
    const total = agents.length
    const running = agents.filter(a => a.status === 'running').length
    const paused = agents.filter(a => a.status === 'paused').length
    const idle = agents.filter(a => a.status === 'idle').length
    const failed = agents.filter(a => a.status === 'failed').length
    const totalTokens = agents.reduce((sum, a) => sum + a.tokens_used, 0)
    const totalCost = agents.reduce((sum, a) => sum + a.cost_usd, 0)
    const totalTime = agents.reduce((sum, a) => sum + a.elapsed_seconds, 0)

    return { total, running, paused, idle, failed, totalTokens, totalCost, totalTime }
  }, [agents])

  const byType = useMemo(() => {
    const groups: Record<string, { count: number; running: number; tokens: number; cost: number }> = {}
    for (const a of agents) {
      if (!groups[a.agent_type]) {
        groups[a.agent_type] = { count: 0, running: 0, tokens: 0, cost: 0 }
      }
      groups[a.agent_type].count++
      if (a.status === 'running') groups[a.agent_type].running++
      groups[a.agent_type].tokens += a.tokens_used
      groups[a.agent_type].cost += a.cost_usd
    }
    return groups
  }, [agents])

  const formatTokens = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : `${(n / 1000).toFixed(0)}k`

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.floor(seconds)}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
  }

  return (
    <div className="agent-overview">
      <h3 className="agent-overview-title">Agent Overview</h3>

      {/* Summary cards */}
      <div className="agent-overview-grid">
        <div className="agent-overview-card">
          <div className="agent-overview-card-value">{stats.total}</div>
          <div className="agent-overview-card-label">Total Agents</div>
        </div>
        <div className="agent-overview-card card-running">
          <div className="agent-overview-card-value">{stats.running}</div>
          <div className="agent-overview-card-label">Running</div>
        </div>
        <div className="agent-overview-card card-idle">
          <div className="agent-overview-card-value">{stats.idle}</div>
          <div className="agent-overview-card-label">Idle</div>
        </div>
        <div className="agent-overview-card card-failed">
          <div className="agent-overview-card-value">{stats.failed}</div>
          <div className="agent-overview-card-label">Failed</div>
        </div>
        <div className="agent-overview-card">
          <div className="agent-overview-card-value">{formatTokens(stats.totalTokens)}</div>
          <div className="agent-overview-card-label">Total Tokens</div>
        </div>
        <div className="agent-overview-card">
          <div className="agent-overview-card-value">${stats.totalCost.toFixed(2)}</div>
          <div className="agent-overview-card-label">Total Cost</div>
        </div>
      </div>

      {/* Per-type breakdown */}
      {Object.keys(byType).length > 0 && (
        <div className="agent-overview-breakdown">
          <h4 className="agent-overview-breakdown-title">By Type</h4>
          <table className="agent-overview-table">
            <thead>
              <tr>
                <th>Role</th>
                <th>Count</th>
                <th>Running</th>
                <th>Tokens</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(byType)
                .sort(([, a], [, b]) => b.cost - a.cost)
                .map(([type, data]) => (
                  <tr key={type}>
                    <td className="agent-overview-role">{type}</td>
                    <td>{data.count}</td>
                    <td>{data.running}</td>
                    <td>{formatTokens(data.tokens)}</td>
                    <td>${data.cost.toFixed(2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {agents.length === 0 && (
        <div className="agent-overview-empty">No agents in pool. Spawn agents to see overview stats.</div>
      )}

      {/* Uptime */}
      {stats.totalTime > 0 && (
        <div className="agent-overview-uptime">
          Total agent uptime: {formatTime(stats.totalTime)}
        </div>
      )}
    </div>
  )
}
