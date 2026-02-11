import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders, getMetricsExportUrl } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'
import './MetricsPanel.css'

interface Props {
  projectDir?: string
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

export default function MetricsPanel({ projectDir }: Props) {
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

  const hasAnyMetrics =
    !!metrics &&
    (metrics.api_calls > 0 ||
      metrics.tokens_used > 0 ||
      metrics.phases_total > 0 ||
      metrics.files_changed > 0 ||
      metrics.lines_added > 0 ||
      metrics.lines_removed > 0 ||
      metrics.wall_time_seconds > 0)

  const [showExport, setShowExport] = useState(false)

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Metrics</h2>
        {hasAnyMetrics && (
          <div style={{ position: 'relative' }}>
            <button
              className="btn"
              style={{ fontSize: '0.8rem', padding: '0.25rem 0.5rem' }}
              onClick={() => setShowExport(!showExport)}
            >
              Export
            </button>
            {showExport && (
              <div style={{
                position: 'absolute', right: 0, top: '100%', marginTop: '0.25rem',
                background: 'var(--bg-primary, #fff)', border: '1px solid var(--border-color, #e5e7eb)',
                borderRadius: '6px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', zIndex: 10,
                minWidth: '100px',
              }}>
                <a
                  href={getMetricsExportUrl(projectDir, 'csv')}
                  style={{ display: 'block', padding: '0.5rem 0.75rem', fontSize: '0.8rem', textDecoration: 'none', color: 'inherit' }}
                  onClick={() => setShowExport(false)}
                >
                  CSV
                </a>
                <a
                  href={getMetricsExportUrl(projectDir, 'html')}
                  style={{ display: 'block', padding: '0.5rem 0.75rem', fontSize: '0.8rem', textDecoration: 'none', color: 'inherit', borderTop: '1px solid var(--border-color, #e5e7eb)' }}
                  onClick={() => setShowExport(false)}
                >
                  HTML
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {!metrics || !hasAnyMetrics ? (
        <EmptyState
          icon={<span>📊</span>}
          title="No metrics available"
          description="Metrics will appear once runs start"
          size="sm"
        />
      ) : (
        <div className="metrics-panel-content">
          <div>
            <div className="metrics-panel-section-title">API Usage</div>
            <div className="metrics-panel-grid">
              <div className="metrics-panel-stat">
                <div className="metrics-panel-stat-value">
                  {formatNumber(metrics.api_calls)}
                </div>
                <div className="metrics-panel-stat-label">API Calls</div>
              </div>
              <div className="metrics-panel-stat">
                <div className="metrics-panel-stat-value">
                  {formatNumber(metrics.tokens_used)}
                </div>
                <div className="metrics-panel-stat-label">Tokens</div>
              </div>
            </div>
          </div>

          <div>
            <div className="metrics-panel-section-title">Cost & Time</div>
            <div className="metrics-panel-grid">
              <div className="metrics-panel-stat">
                <div className="metrics-panel-stat-value">
                  {formatCost(metrics.estimated_cost_usd)}
                </div>
                <div className="metrics-panel-stat-label">Estimated Cost</div>
              </div>
              <div className="metrics-panel-stat">
                <div className="metrics-panel-stat-value">
                  {formatDuration(metrics.wall_time_seconds)}
                </div>
                <div className="metrics-panel-stat-label">Wall Time</div>
              </div>
            </div>
          </div>

          <div>
            <div className="metrics-panel-section-title">Code Changes</div>
            <div className="metrics-panel-changes">
              <div className="metrics-panel-changes-row">
                <div>
                  <div className="metrics-panel-change-item-value added">
                    +{formatNumber(metrics.lines_added)}
                  </div>
                  <div className="metrics-panel-change-item-label">Added</div>
                </div>
                <div className="metrics-panel-divider" />
                <div>
                  <div className="metrics-panel-change-item-value removed">
                    -{formatNumber(metrics.lines_removed)}
                  </div>
                  <div className="metrics-panel-change-item-label">Removed</div>
                </div>
                <div className="metrics-panel-divider" />
                <div>
                  <div className="metrics-panel-change-item-value">
                    {formatNumber(metrics.files_changed)}
                  </div>
                  <div className="metrics-panel-change-item-label">Files</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
