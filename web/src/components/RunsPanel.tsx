import { useEffect, useMemo, useState, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'
import './RunsPanel.css'

interface RunInfo {
  run_id: string
  task_id: string
  phase: string
  step: string
  status: string
  started_at: string
  updated_at: string
}

interface RunDetail extends RunInfo {
  current_task_id?: string | null
  current_phase_id?: string | null
  last_error?: string | null
}

interface Props {
  projectDir?: string
  currentRunId?: string
}

const normalizeRuns = (value: unknown): RunInfo[] => {
  if (!Array.isArray(value)) return []
  const out: RunInfo[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object') continue
    const raw = item as Record<string, unknown>
    const run_id = typeof raw.run_id === 'string' ? raw.run_id : ''
    if (!run_id) continue
    out.push({
      run_id,
      task_id: typeof raw.task_id === 'string' ? raw.task_id : '',
      phase: typeof raw.phase === 'string' ? raw.phase : '',
      step: typeof raw.step === 'string' ? raw.step : '',
      status: typeof raw.status === 'string' ? raw.status : '',
      started_at: typeof raw.started_at === 'string' ? raw.started_at : '',
      updated_at: typeof raw.updated_at === 'string' ? raw.updated_at : '',
    })
  }
  return out
}

const normalizeRunDetail = (value: unknown): RunDetail | null => {
  if (!value || typeof value !== 'object') return null
  const raw = value as Record<string, unknown>
  const run_id = typeof raw.run_id === 'string' ? raw.run_id : ''
  if (!run_id) return null
  return {
    run_id,
    task_id: typeof raw.task_id === 'string' ? raw.task_id : '',
    phase: typeof raw.phase === 'string' ? raw.phase : '',
    step: typeof raw.step === 'string' ? raw.step : '',
    status: typeof raw.status === 'string' ? raw.status : '',
    started_at: typeof raw.started_at === 'string' ? raw.started_at : '',
    updated_at: typeof raw.updated_at === 'string' ? raw.updated_at : '',
    current_task_id: typeof raw.current_task_id === 'string' ? raw.current_task_id : null,
    current_phase_id: typeof raw.current_phase_id === 'string' ? raw.current_phase_id : null,
    last_error: typeof raw.last_error === 'string' ? raw.last_error : null,
  }
}

export default function RunsPanel({ projectDir, currentRunId }: Props) {
  const [runs, setRuns] = useState<RunInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)

  useEffect(() => {
    fetchRuns()
  }, [projectDir])

  useChannel('runs', useCallback(() => {
    fetchRuns()
  }, [projectDir]))

  const fetchRuns = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/runs', projectDir, { limit: 25 }), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setRuns(normalizeRuns(data))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch runs')
    } finally {
      setLoading(false)
    }
  }

  const fetchRunDetail = async (runId: string) => {
    setDetailError(null)
    try {
      const response = await fetch(buildApiUrl(`/api/runs/${runId}`, projectDir), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setRunDetail(normalizeRunDetail(data))
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : 'Failed to fetch run details')
      setRunDetail(null)
    }
  }

  const sortedRuns = useMemo(() => {
    return [...runs].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''))
  }, [runs])

  return (
    <div className="card">
      <h2>Recent Runs</h2>

      <div className="runs-panel-header">
        <button onClick={fetchRuns} className="runs-panel-btn">
          Refresh
        </button>
        {currentRunId && (
          <div className="runs-panel-active-run">
            Active run: <span className="runs-panel-active-run-id">{currentRunId}</span>
          </div>
        )}
      </div>

      {loading ? (
        <LoadingSpinner label="Loading runs..." />
      ) : error ? (
        <EmptyState
          icon={<span>⚠️</span>}
          title="Error loading runs"
          description={error}
          size="sm"
        />
      ) : sortedRuns.length === 0 ? (
        <EmptyState
          icon={<span>🚀</span>}
          title="No runs found"
          description="Runs will appear after the first execution."
          size="sm"
        />
      ) : (
        <>
          <div className="runs-panel-table-wrapper">
            <table className="runs-panel-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Task</th>
                  <th>Phase</th>
                  <th>Step</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {sortedRuns.map((run) => (
                  <tr
                    key={run.run_id}
                    className={run.run_id === currentRunId ? 'active' : ''}
                  >
                    <td className="runs-panel-table-id">{run.run_id}</td>
                    <td>{run.task_id || '-'}</td>
                    <td>{run.phase || '-'}</td>
                    <td>{run.step || '-'}</td>
                    <td>{run.status || '-'}</td>
                    <td>
                      <button
                        onClick={() => {
                          setSelectedRunId(run.run_id)
                          void fetchRunDetail(run.run_id)
                        }}
                        className="runs-panel-table-btn"
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {selectedRunId && (
            <div className="runs-panel-detail">
              <div className="runs-panel-detail-title">Run Details</div>
              {detailError ? (
                <div className="runs-panel-detail-error">Error: {detailError}</div>
              ) : runDetail ? (
                <>
                  <div className="runs-panel-detail-id">{runDetail.run_id}</div>
                  <div className="runs-panel-detail-status">
                    Status: <strong>{runDetail.status}</strong>
                  </div>
                  {runDetail.last_error && (
                    <div className="runs-panel-detail-last-error">
                      Last error: {runDetail.last_error}
                    </div>
                  )}
                  <div className="runs-panel-detail-meta">
                    Current task: {runDetail.current_task_id || '-'} • Current phase: {runDetail.current_phase_id || '-'}
                  </div>
                </>
              ) : (
                <div className="runs-panel-detail-loading">Loading...</div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
