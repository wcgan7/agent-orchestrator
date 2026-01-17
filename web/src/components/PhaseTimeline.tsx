import { useState, useEffect } from 'react'

interface Phase {
  id: string
  name: string
  description: string
  status: string
  deps: string[]
  progress: number
}

export default function PhaseTimeline() {
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
    const interval = setInterval(fetchPhases, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchPhases = async () => {
    try {
      const response = await fetch('/api/phases')
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
        <div className="empty-state">
          <p>Loading phases...</p>
        </div>
      ) : phases.length === 0 ? (
        <div className="empty-state">
          <p>No phases found</p>
          <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
            Run the planner first to generate phases
          </p>
        </div>
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
                <div style={{
                  fontSize: '0.75rem',
                  color: '#999',
                  marginTop: '0.5rem'
                }}>
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
