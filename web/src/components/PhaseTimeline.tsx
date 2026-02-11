import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
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
      const response = await fetch(buildApiUrl('/api/phases', projectDir), {
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
    <div className="card">
      <h2>Phase Timeline</h2>

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
        <div className="phase-list">
          {phases.map((phase) => (
            <div
              key={phase.id}
              className="phase-item"
              data-status={phase.status}
            >
              <div className="phase-header">
                <div className="phase-name">{phase.name || phase.id}</div>
                <div className="phase-status">{phase.status}</div>
              </div>

              {phase.description && (
                <div className="phase-description">{phase.description}</div>
              )}

              {phase.deps && phase.deps.length > 0 && (
                <div className="phase-dependencies">
                  Dependencies: {phase.deps.join(', ')}
                </div>
              )}

              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${phase.progress * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
