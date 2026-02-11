import { useState, useEffect, useCallback, useMemo } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'
import './CostBreakdown.css'

interface Props {
  projectDir?: string
}

interface AgentCost {
  agent_id: string
  name: string
  role: string
  tokens_used: number
  cost_usd: number
  elapsed_seconds: number
  budget_usd?: number
}

interface TaskBoardItem {
  id: string
  task_type: string
  step: string
  tokens_used: number
  cost_usd: number
}

type TabId = 'agent' | 'task' | 'step'

const PIPELINE_STEPS = ['plan', 'implement', 'verify', 'review', 'commit'] as const

const BUDGET_WARNING_THRESHOLD = 0.8

const normalizeAgents = (value: unknown): AgentCost[] => {
  if (!Array.isArray(value)) return []
  const out: AgentCost[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object') continue
    const raw = item as Record<string, unknown>
    const agentId = typeof raw.agent_id === 'string' ? raw.agent_id : (typeof raw.id === 'string' ? raw.id : '')
    if (!agentId) continue
    out.push({
      agent_id: agentId,
      name: typeof raw.name === 'string' ? raw.name : agentId,
      role: typeof raw.role === 'string' ? raw.role : '',
      tokens_used: typeof raw.tokens_used === 'number' && Number.isFinite(raw.tokens_used) ? raw.tokens_used : 0,
      cost_usd: typeof raw.cost_usd === 'number' && Number.isFinite(raw.cost_usd) ? raw.cost_usd : 0,
      elapsed_seconds: typeof raw.elapsed_seconds === 'number' && Number.isFinite(raw.elapsed_seconds) ? raw.elapsed_seconds : 0,
      budget_usd: typeof raw.budget_usd === 'number' && Number.isFinite(raw.budget_usd) ? raw.budget_usd : undefined,
    })
  }
  return out
}

const normalizeTasks = (value: unknown): TaskBoardItem[] => {
  if (!value || typeof value !== 'object') return []

  // The board endpoint may return { columns: [...] } or an array directly
  let items: unknown[] = []
  if (Array.isArray(value)) {
    items = value
  } else {
    const raw = value as Record<string, unknown>
    if (Array.isArray(raw.columns)) {
      for (const col of raw.columns) {
        if (col && typeof col === 'object' && Array.isArray((col as Record<string, unknown>).tasks)) {
          items.push(...(col as Record<string, unknown>).tasks as unknown[])
        }
      }
    } else if (Array.isArray(raw.tasks)) {
      items = raw.tasks
    }
  }

  const out: TaskBoardItem[] = []
  for (const item of items) {
    if (!item || typeof item !== 'object') continue
    const raw = item as Record<string, unknown>
    const id = typeof raw.id === 'string' ? raw.id : ''
    if (!id) continue
    out.push({
      id,
      task_type: typeof raw.task_type === 'string' ? raw.task_type : (typeof raw.type === 'string' ? raw.type : 'unknown'),
      step: typeof raw.step === 'string' ? raw.step : '',
      tokens_used: typeof raw.tokens_used === 'number' && Number.isFinite(raw.tokens_used) ? raw.tokens_used : 0,
      cost_usd: typeof raw.cost_usd === 'number' && Number.isFinite(raw.cost_usd) ? raw.cost_usd : 0,
    })
  }
  return out
}

export default function CostBreakdown({ projectDir }: Props) {
  const [agents, setAgents] = useState<AgentCost[]>([])
  const [tasks, setTasks] = useState<TaskBoardItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<TabId>('agent')

  const fetchData = async () => {
    try {
      const [agentsRes, tasksRes] = await Promise.all([
        fetch(buildApiUrl('/api/v2/agents', projectDir), {
          headers: buildAuthHeaders(),
        }),
        fetch(buildApiUrl('/api/v2/tasks/board', projectDir), {
          headers: buildAuthHeaders(),
        }),
      ])

      if (agentsRes.ok) {
        const agentsData = await agentsRes.json()
        setAgents(normalizeAgents(agentsData))
      }

      if (tasksRes.ok) {
        const tasksData = await tasksRes.json()
        setTasks(normalizeTasks(tasksData))
      }

      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch cost data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [projectDir])

  useChannel('agents', useCallback(() => {
    fetchData()
  }, [projectDir]))

  useChannel('metrics', useCallback(() => {
    fetchData()
  }, [projectDir]))

  // Totals
  const totalCost = useMemo(() => {
    return agents.reduce((sum, a) => sum + a.cost_usd, 0)
  }, [agents])

  const totalTokens = useMemo(() => {
    return agents.reduce((sum, a) => sum + a.tokens_used, 0)
  }, [agents])

  const avgCostPerTask = useMemo(() => {
    if (tasks.length === 0) return 0
    const taskTotal = tasks.reduce((sum, t) => sum + t.cost_usd, 0)
    return taskTotal / tasks.length
  }, [tasks])

  // Budget warnings
  const budgetWarnings = useMemo(() => {
    return agents.filter((a) => {
      if (!a.budget_usd || a.budget_usd <= 0) return false
      return a.cost_usd / a.budget_usd >= BUDGET_WARNING_THRESHOLD
    })
  }, [agents])

  // By task type aggregation
  const taskTypeBreakdown = useMemo(() => {
    const map: Record<string, { tokens: number; cost: number; count: number }> = {}
    for (const t of tasks) {
      const key = t.task_type || 'unknown'
      if (!map[key]) {
        map[key] = { tokens: 0, cost: 0, count: 0 }
      }
      map[key].tokens += t.tokens_used
      map[key].cost += t.cost_usd
      map[key].count += 1
    }
    return Object.entries(map)
      .map(([type, data]) => ({ type, ...data }))
      .sort((a, b) => b.cost - a.cost)
  }, [tasks])

  // By step aggregation
  const stepBreakdown = useMemo(() => {
    const map: Record<string, { tokens: number; cost: number; count: number }> = {}
    for (const step of PIPELINE_STEPS) {
      map[step] = { tokens: 0, cost: 0, count: 0 }
    }
    for (const t of tasks) {
      const step = t.step || 'unknown'
      if (!map[step]) {
        map[step] = { tokens: 0, cost: 0, count: 0 }
      }
      map[step].tokens += t.tokens_used
      map[step].cost += t.cost_usd
      map[step].count += 1
    }
    return Object.entries(map)
      .map(([step, data]) => ({ step, ...data }))
      .sort((a, b) => b.cost - a.cost)
  }, [tasks])

  // Max cost for bar chart scaling
  const maxAgentCost = useMemo(() => {
    return Math.max(...agents.map((a) => a.cost_usd), 0.01)
  }, [agents])

  const maxTaskTypeCost = useMemo(() => {
    return Math.max(...taskTypeBreakdown.map((t) => t.cost), 0.01)
  }, [taskTypeBreakdown])

  const maxStepCost = useMemo(() => {
    return Math.max(...stepBreakdown.map((s) => s.cost), 0.01)
  }, [stepBreakdown])

  const formatCost = (cost: number): string => {
    return `$${cost.toFixed(4)}`
  }

  const formatTokens = (tokens: number): string => {
    return tokens.toLocaleString()
  }

  const formatDuration = (seconds: number): string => {
    if (seconds === 0) return '0s'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    const parts = []
    if (hours > 0) parts.push(`${hours}h`)
    if (minutes > 0) parts.push(`${minutes}m`)
    if (secs > 0 || parts.length === 0) parts.push(`${secs}s`)
    return parts.join(' ')
  }

  const hasData = agents.length > 0 || tasks.length > 0

  const tabs: { id: TabId; label: string }[] = [
    { id: 'agent', label: 'By Agent' },
    { id: 'task', label: 'By Task Type' },
    { id: 'step', label: 'By Step' },
  ]

  return (
    <div className="card">
      <h2>Cost Breakdown</h2>

      {loading ? (
        <LoadingSpinner label="Loading cost data..." />
      ) : error ? (
        <EmptyState
          icon={<span>&#x26A0;</span>}
          title="Error loading cost data"
          description={error}
          size="sm"
        />
      ) : !hasData ? (
        <EmptyState
          icon={<span>&#x1F4B0;</span>}
          title="No cost data available"
          description="Cost data will appear once agents start processing tasks"
          size="sm"
        />
      ) : (
        <div className="cost-breakdown-content">
          {/* Summary cards */}
          <div className="cost-breakdown-summary">
            <div className="cost-breakdown-summary-card">
              <div className="cost-breakdown-summary-value">{formatCost(totalCost)}</div>
              <div className="cost-breakdown-summary-label">Total Cost</div>
            </div>
            <div className="cost-breakdown-summary-card">
              <div className="cost-breakdown-summary-value">{formatTokens(totalTokens)}</div>
              <div className="cost-breakdown-summary-label">Total Tokens</div>
            </div>
            <div className="cost-breakdown-summary-card">
              <div className="cost-breakdown-summary-value">{formatCost(avgCostPerTask)}</div>
              <div className="cost-breakdown-summary-label">Avg Cost / Task</div>
            </div>
          </div>

          {/* Budget warnings */}
          {budgetWarnings.length > 0 && (
            <div className="cost-breakdown-warnings">
              {budgetWarnings.map((agent) => (
                <div key={agent.agent_id} className="cost-breakdown-warning">
                  <span className="cost-breakdown-warning-icon">&#x26A0;</span>
                  <span>
                    <strong>{agent.name}</strong> has used{' '}
                    {agent.budget_usd
                      ? `${Math.round((agent.cost_usd / agent.budget_usd) * 100)}%`
                      : 'N/A'}{' '}
                    of its budget ({formatCost(agent.cost_usd)} / {formatCost(agent.budget_usd || 0)})
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Tab navigation */}
          <div className="cost-breakdown-tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`cost-breakdown-tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="cost-breakdown-tab-content">
            {activeTab === 'agent' && (
              <div className="cost-breakdown-list">
                {agents.length === 0 ? (
                  <div className="cost-breakdown-empty">No agent data available</div>
                ) : (
                  agents
                    .slice()
                    .sort((a, b) => b.cost_usd - a.cost_usd)
                    .map((agent) => (
                      <div key={agent.agent_id} className="cost-breakdown-row">
                        <div className="cost-breakdown-row-header">
                          <div className="cost-breakdown-row-name">
                            <span className="cost-breakdown-agent-name">{agent.name}</span>
                            {agent.role && (
                              <span className="cost-breakdown-agent-role">{agent.role}</span>
                            )}
                          </div>
                          <div className="cost-breakdown-row-stats">
                            <span className="cost-breakdown-stat-tokens">
                              {formatTokens(agent.tokens_used)} tokens
                            </span>
                            <span className="cost-breakdown-stat-cost">
                              {formatCost(agent.cost_usd)}
                            </span>
                            <span className="cost-breakdown-stat-time">
                              {formatDuration(agent.elapsed_seconds)}
                            </span>
                          </div>
                        </div>
                        <div className="cost-breakdown-bar-track">
                          <div
                            className={`cost-breakdown-bar-fill ${
                              agent.budget_usd && agent.cost_usd / agent.budget_usd >= BUDGET_WARNING_THRESHOLD
                                ? 'warning'
                                : ''
                            }`}
                            style={{
                              width: `${Math.min((agent.cost_usd / maxAgentCost) * 100, 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))
                )}
              </div>
            )}

            {activeTab === 'task' && (
              <div className="cost-breakdown-list">
                {taskTypeBreakdown.length === 0 ? (
                  <div className="cost-breakdown-empty">No task data available</div>
                ) : (
                  taskTypeBreakdown.map((entry) => (
                    <div key={entry.type} className="cost-breakdown-row">
                      <div className="cost-breakdown-row-header">
                        <div className="cost-breakdown-row-name">
                          <span className="cost-breakdown-type-name">{entry.type}</span>
                          <span className="cost-breakdown-type-count">
                            {entry.count} task{entry.count !== 1 ? 's' : ''}
                          </span>
                        </div>
                        <div className="cost-breakdown-row-stats">
                          <span className="cost-breakdown-stat-tokens">
                            {formatTokens(entry.tokens)} tokens
                          </span>
                          <span className="cost-breakdown-stat-cost">
                            {formatCost(entry.cost)}
                          </span>
                        </div>
                      </div>
                      <div className="cost-breakdown-bar-track">
                        <div
                          className="cost-breakdown-bar-fill"
                          style={{
                            width: `${Math.min((entry.cost / maxTaskTypeCost) * 100, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}

            {activeTab === 'step' && (
              <div className="cost-breakdown-list">
                {stepBreakdown.length === 0 ? (
                  <div className="cost-breakdown-empty">No step data available</div>
                ) : (
                  stepBreakdown.map((entry) => (
                    <div key={entry.step} className="cost-breakdown-row">
                      <div className="cost-breakdown-row-header">
                        <div className="cost-breakdown-row-name">
                          <span className="cost-breakdown-step-name">{entry.step}</span>
                          <span className="cost-breakdown-type-count">
                            {entry.count} task{entry.count !== 1 ? 's' : ''}
                          </span>
                        </div>
                        <div className="cost-breakdown-row-stats">
                          <span className="cost-breakdown-stat-tokens">
                            {formatTokens(entry.tokens)} tokens
                          </span>
                          <span className="cost-breakdown-stat-cost">
                            {formatCost(entry.cost)}
                          </span>
                        </div>
                      </div>
                      <div className="cost-breakdown-bar-track">
                        <div
                          className="cost-breakdown-bar-fill"
                          style={{
                            width: `${Math.min((entry.cost / maxStepCost) * 100, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
