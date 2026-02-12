import { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Chip,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import { humanizeLabel } from '../ui/labels'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'

interface Phase {
  id: string
  name: string
  description: string
  status: string
  deps: string[]
  progress: number
}

interface Props {
  projectDir?: string
}

export default function PhaseTimeline({ projectDir }: Props) {
  const [phases, setPhases] = useState<Phase[]>([])
  const [loading, setLoading] = useState(true)

  const normalizePhases = (value: unknown): Phase[] => {
    if (!Array.isArray(value)) return []
    const out: Phase[] = []
    for (const item of value) {
      if (!item || typeof item !== 'object') continue
      const raw = item as Record<string, unknown>
      const id = typeof raw.id === 'string' ? raw.id : ''
      if (!id) continue
      const depsRaw = raw.deps
      const deps = Array.isArray(depsRaw)
        ? depsRaw.map((d) => String(d)).filter(Boolean)
        : []
      const progress =
        typeof raw.progress === 'number' && Number.isFinite(raw.progress)
          ? raw.progress
          : 0
      out.push({
        id,
        name: typeof raw.name === 'string' ? raw.name : '',
        description: typeof raw.description === 'string' ? raw.description : '',
        status: typeof raw.status === 'string' ? raw.status : '',
        deps,
        progress,
      })
    }
    return out
  }

  useEffect(() => {
    fetchPhases()
  }, [projectDir])

  useChannel('phases', useCallback(() => {
    fetchPhases()
  }, [projectDir]))

  const fetchPhases = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/v3/phases', projectDir), {
        headers: buildAuthHeaders(),
      })
      if (response.ok) {
        const data = await response.json()
        setPhases(normalizePhases(data))
      }
    } catch (err) {
      console.error('Failed to fetch phases:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>Phase Timeline</Typography>

      {loading ? (
        <LoadingSpinner label="Loading phases..." />
      ) : phases.length === 0 ? (
        <EmptyState
          icon={<span>📋</span>}
          title="No phases found"
          description="Run the planner first to generate phases"
          size="sm"
        />
      ) : (
        <Stack spacing={1.25} className="phase-list">
          {phases.map((phase) => (
            <Box
              key={phase.id}
              className="phase-item"
              data-status={phase.status}
              sx={{
                border: 1,
                borderColor: 'divider',
                borderRadius: 1.5,
                p: 1.25,
                bgcolor: 'background.default',
              }}
            >
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.75 }} className="phase-header">
                <Typography className="phase-name" fontWeight={600}>
                  {phase.name || phase.id}
                </Typography>
                <Chip size="small" className="phase-status" label={humanizeLabel(phase.status || 'unknown')} variant="outlined" />
              </Stack>

              {phase.description && (
                <Typography className="phase-description" variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
                  {phase.description}
                </Typography>
              )}

              {phase.deps && phase.deps.length > 0 && (
                <Typography className="phase-dependencies" variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                  Dependencies: {phase.deps.join(', ')}
                </Typography>
              )}

              <LinearProgress
                className="progress-bar"
                variant="determinate"
                value={Math.max(0, Math.min(100, phase.progress * 100))}
              />
            </Box>
          ))}
        </Stack>
      )}
    </Box>
  )
}
