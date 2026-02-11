import { useState, useEffect, useCallback } from 'react'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import './MetricsChart.css'

interface RunMetrics {
  tokens_used: number
  api_calls: number
  estimated_cost_usd: number
  wall_time_seconds: number
  phases_completed: number
  phases_total: number
  files_changed: number
  lines_added: number
  lines_removed: number
}

interface Props {
  projectDir?: string
}

export default function MetricsChart({ projectDir }: Props) {
  const [metrics, setMetrics] = useState<RunMetrics | null>(null)

  const normalizeMetrics = (value: unknown): RunMetrics | null => {
    if (!value || typeof value !== 'object') return null
    const raw = value as Record<string, unknown>

    const num = (key: keyof RunMetrics): number => {
      const v = raw[key as string]
      return typeof v === 'number' && Number.isFinite(v) ? v : 0
    }

    return {
      tokens_used: num('tokens_used'),
      api_calls: num('api_calls'),
      estimated_cost_usd: num('estimated_cost_usd'),
      wall_time_seconds: num('wall_time_seconds'),
      phases_completed: num('phases_completed'),
      phases_total: num('phases_total'),
      files_changed: num('files_changed'),
      lines_added: num('lines_added'),
      lines_removed: num('lines_removed'),
    }
  }

  useEffect(() => {
    fetchMetrics()
  }, [projectDir])

  useChannel('metrics', useCallback(() => {
    fetchMetrics()
  }, [projectDir]))

  const fetchMetrics = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/metrics', projectDir), {
        headers: buildAuthHeaders(),
      })
      if (response.ok) {
        const data = await response.json()
        setMetrics(normalizeMetrics(data))
      }
    } catch (err) {
      console.error('Failed to fetch metrics:', err)
    }
  }

  const hasAnyMetrics =
    !!metrics &&
    (metrics.api_calls > 0 ||
      metrics.tokens_used > 0 ||
      metrics.phases_total > 0 ||
      metrics.files_changed > 0 ||
      metrics.lines_added > 0 ||
      metrics.lines_removed > 0 ||
      metrics.wall_time_seconds > 0)

  if (!metrics || !hasAnyMetrics) {
    return (
      <div className="card">
        <h2>Metrics Visualization</h2>
        <div className="empty-state">
          <p>No metrics data available for visualization</p>
          <p className="metrics-chart-empty-subtitle">
            Charts will appear once runs generate metrics
          </p>
        </div>
      </div>
    )
  }

  // Prepare data for code changes chart
  const codeChangesData = [
    {
      name: 'Added',
      lines: metrics.lines_added,
      fill: '#22c55e',
    },
    {
      name: 'Removed',
      lines: metrics.lines_removed,
      fill: '#ef4444',
    },
  ]

  // Prepare data for phase progress pie chart
  const phaseProgressData = [
    {
      name: 'Completed',
      value: metrics.phases_completed,
      fill: '#22c55e',
    },
    {
      name: 'Remaining',
      value: metrics.phases_total - metrics.phases_completed,
      fill: '#e5e7eb',
    },
  ]

  // Prepare data for usage metrics
  const usageData = [
    {
      metric: 'API Calls',
      value: metrics.api_calls,
    },
    {
      metric: 'Tokens (K)',
      value: Math.round(metrics.tokens_used / 1000),
    },
    {
      metric: 'Files Changed',
      value: metrics.files_changed,
    },
  ]

  const formatTime = (seconds: number): string => {
    if (seconds === 0) return '0s'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)

    if (hours > 0) return `${hours}h ${minutes}m`
    if (minutes > 0) return `${minutes}m ${secs}s`
    return `${secs}s`
  }

  const COLORS = ['#22c55e', '#e5e7eb']

  return (
    <div className="card">
      <h2>Metrics Visualization</h2>

      <div className="metrics-chart-container">
        {/* Phase Progress Pie Chart */}
        {metrics.phases_total > 0 && (
          <div className="metrics-chart-section">
            <h3>Phase Progress</h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={phaseProgressData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {phaseProgressData.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="metrics-chart-caption">
              {metrics.phases_completed} of {metrics.phases_total} phases completed
            </div>
          </div>
        )}

        {/* Code Changes Bar Chart */}
        {(metrics.lines_added > 0 || metrics.lines_removed > 0) && (
          <div className="metrics-chart-section">
            <h3>Code Changes</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={codeChangesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="lines" fill="#8884d8">
                  {codeChangesData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Usage Metrics Bar Chart */}
        {metrics.api_calls > 0 && (
          <div className="metrics-chart-section">
            <h3>Usage Metrics</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={usageData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Summary Stats */}
        <div className="metrics-summary">
          <div className="metrics-summary-grid">
            <div>
              <div className="metrics-summary-item-value">
                ${metrics.estimated_cost_usd.toFixed(2)}
              </div>
              <div className="metrics-summary-item-label">Estimated Cost</div>
            </div>
            <div>
              <div className="metrics-summary-item-value">
                {formatTime(metrics.wall_time_seconds)}
              </div>
              <div className="metrics-summary-item-label">Wall Time</div>
            </div>
            <div>
              <div className="metrics-summary-item-value">
                {metrics.files_changed}
              </div>
              <div className="metrics-summary-item-label">Files Changed</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
