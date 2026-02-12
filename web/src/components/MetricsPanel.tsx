import { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Button,
  Chip,
  Menu,
  MenuItem,
  Stack,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders, getMetricsExportUrl } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'

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
  const [exportAnchor, setExportAnchor] = useState<HTMLElement | null>(null)

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
      const response = await fetch(buildApiUrl('/api/v3/metrics', projectDir), {
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

  const formatNumber = (num: number): string => num.toLocaleString()
  const formatCost = (cost: number): string => `$${cost.toFixed(2)}`

  const hasAnyMetrics =
    !!metrics &&
    (metrics.api_calls > 0 ||
      metrics.tokens_used > 0 ||
      metrics.phases_total > 0 ||
      metrics.files_changed > 0 ||
      metrics.lines_added > 0 ||
      metrics.lines_removed > 0 ||
      metrics.wall_time_seconds > 0)

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1.5 }}>
        <Typography variant="h2" sx={{ fontSize: '1.125rem' }}>Metrics</Typography>
        {hasAnyMetrics && (
          <>
            <Button
              className="btn"
              size="small"
              variant="outlined"
              onClick={(e) => setExportAnchor(e.currentTarget)}
            >
              Export
            </Button>
            <Menu
              open={Boolean(exportAnchor)}
              anchorEl={exportAnchor}
              onClose={() => setExportAnchor(null)}
            >
              <MenuItem component="a" href={getMetricsExportUrl(projectDir, 'csv')} onClick={() => setExportAnchor(null)}>
                CSV
              </MenuItem>
              <MenuItem component="a" href={getMetricsExportUrl(projectDir, 'html')} onClick={() => setExportAnchor(null)}>
                HTML
              </MenuItem>
            </Menu>
          </>
        )}
      </Stack>

      {!metrics || !hasAnyMetrics ? (
        <EmptyState
          icon={<span>📊</span>}
          title="No metrics available"
          description="Metrics will appear once runs start"
          size="sm"
        />
      ) : (
        <Stack spacing={1.5} className="metrics-panel-content">
          <Box>
            <Typography className="metrics-panel-section-title" variant="caption" color="text.secondary">API Usage</Typography>
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
              <Chip className="metrics-panel-stat" label={`API Calls ${formatNumber(metrics.api_calls)}`} variant="outlined" />
              <Chip className="metrics-panel-stat" label={`Tokens ${formatNumber(metrics.tokens_used)}`} variant="outlined" />
            </Stack>
            <Typography className="sr-only">{formatNumber(metrics.tokens_used)}</Typography>
          </Box>

          <Box>
            <Typography className="metrics-panel-section-title" variant="caption" color="text.secondary">Cost & Time</Typography>
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
              <Chip className="metrics-panel-stat" label={`Estimated Cost ${formatCost(metrics.estimated_cost_usd)}`} color="warning" variant="outlined" />
              <Chip className="metrics-panel-stat" label={`Wall Time ${formatDuration(metrics.wall_time_seconds)}`} color="info" variant="outlined" />
            </Stack>
          </Box>

          <Box>
            <Typography className="metrics-panel-section-title" variant="caption" color="text.secondary">Code Changes</Typography>
            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }} className="metrics-panel-changes-row">
              <Chip className="metrics-panel-change-item-value added" label={`+${formatNumber(metrics.lines_added)} Added`} color="success" variant="outlined" />
              <Chip className="metrics-panel-change-item-value removed" label={`-${formatNumber(metrics.lines_removed)} Removed`} color="error" variant="outlined" />
              <Chip className="metrics-panel-change-item-value" label={`${formatNumber(metrics.files_changed)} Files`} variant="outlined" />
            </Stack>
          </Box>
        </Stack>
      )}
    </Box>
  )
}
