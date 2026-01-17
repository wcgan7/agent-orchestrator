import { useState, useEffect } from 'react'

interface ProjectStatus {
  project_dir: string
  status: string
  current_task_id?: string
  current_phase_id?: string
  run_id?: string
  last_error?: string
  phases_completed: number
  phases_total: number
  tasks_ready: number
  tasks_running: number
  tasks_done: number
  tasks_blocked: number
}

interface Props {
  status: ProjectStatus | null
}

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

export default function MetricsPanel({ status }: Props) {
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

  const formatNumber = (num: number): string => {
    return num.toLocaleString()
  }

  const formatCost = (cost: number): string => {
    return `$${cost.toFixed(2)}`
  }

  return (
    <div className="card">
      <h2>Metrics</h2>

      {!metrics || metrics.api_calls === 0 ? (
        <div className="empty-state">
          <p>No metrics available</p>
          <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
            Metrics will appear once runs start
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <div style={{ fontSize: '0.875rem', color: '#666', marginBottom: '0.5rem' }}>
              API Usage
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                  {formatNumber(metrics.api_calls)}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  API Calls
                </div>
              </div>
              <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                  {formatNumber(metrics.tokens_used)}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Tokens
                </div>
              </div>
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.875rem', color: '#666', marginBottom: '0.5rem' }}>
              Cost & Time
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                  {formatCost(metrics.estimated_cost_usd)}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Estimated Cost
                </div>
              </div>
              <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#333' }}>
                  {formatDuration(metrics.wall_time_seconds)}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                  Wall Time
                </div>
              </div>
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.875rem', color: '#666', marginBottom: '0.5rem' }}>
              Code Changes
            </div>
            <div style={{ padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#4caf50' }}>
                    +{formatNumber(metrics.lines_added)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>Added</div>
                </div>
                <div style={{ width: '1px', height: '40px', background: '#ddd' }} />
                <div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#f44336' }}>
                    -{formatNumber(metrics.lines_removed)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>Removed</div>
                </div>
                <div style={{ width: '1px', height: '40px', background: '#ddd' }} />
                <div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#333' }}>
                    {formatNumber(metrics.files_changed)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>Files</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
