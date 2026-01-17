import { useState, useEffect } from 'react'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

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

export default function MetricsChart() {
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
    const interval = setInterval(fetchMetrics, 10000) // Poll every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/metrics')
      if (response.ok) {
        const data = await response.json()
        setMetrics(normalizeMetrics(data))
      }
    } catch (err) {
      console.error('Failed to fetch metrics:', err)
    }
  }

  if (!metrics || metrics.api_calls === 0) {
    return (
      <div className="card">
        <h2>Metrics Visualization</h2>
        <div className="empty-state">
          <p>No metrics data available for visualization</p>
          <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
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
      fill: '#4caf50',
    },
    {
      name: 'Removed',
      lines: metrics.lines_removed,
      fill: '#f44336',
    },
  ]

  // Prepare data for phase progress pie chart
  const phaseProgressData = [
    {
      name: 'Completed',
      value: metrics.phases_completed,
      fill: '#4caf50',
    },
    {
      name: 'Remaining',
      value: metrics.phases_total - metrics.phases_completed,
      fill: '#e0e0e0',
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

  const COLORS = ['#4caf50', '#e0e0e0']

  return (
    <div className="card">
      <h2>Metrics Visualization</h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', marginTop: '1rem' }}>
        {/* Phase Progress Pie Chart */}
        {metrics.phases_total > 0 && (
          <div>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem', color: '#666' }}>
              Phase Progress
            </h3>
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
                  {phaseProgressData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', marginTop: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
              {metrics.phases_completed} of {metrics.phases_total} phases completed
            </div>
          </div>
        )}

        {/* Code Changes Bar Chart */}
        {(metrics.lines_added > 0 || metrics.lines_removed > 0) && (
          <div>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem', color: '#666' }}>
              Code Changes
            </h3>
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
          <div>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem', color: '#666' }}>
              Usage Metrics
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={usageData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#2196f3" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Summary Stats */}
        <div style={{ padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                ${metrics.estimated_cost_usd.toFixed(2)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#666' }}>Estimated Cost</div>
            </div>
            <div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                {formatTime(metrics.wall_time_seconds)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#666' }}>Wall Time</div>
            </div>
            <div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                {metrics.files_changed}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#666' }}>Files Changed</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
