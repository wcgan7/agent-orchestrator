import { useEffect, useMemo, useState, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'
import './BreakpointsPanel.css'

interface BreakpointInfo {
  id: string
  trigger: string
  target: string
  task_id?: string | null
  condition?: string | null
  action: string
  enabled: boolean
  hit_count: number
  created_at?: string | null
}

interface Props {
  projectDir?: string
}

const normalizeBreakpoints = (value: unknown): BreakpointInfo[] => {
  if (!Array.isArray(value)) return []
  const out: BreakpointInfo[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object') continue
    const raw = item as Record<string, unknown>
    const id = typeof raw.id === 'string' ? raw.id : ''
    if (!id) continue
    out.push({
      id,
      trigger: typeof raw.trigger === 'string' ? raw.trigger : '',
      target: typeof raw.target === 'string' ? raw.target : '',
      task_id: typeof raw.task_id === 'string' ? raw.task_id : null,
      condition: typeof raw.condition === 'string' ? raw.condition : null,
      action: typeof raw.action === 'string' ? raw.action : 'pause',
      enabled: typeof raw.enabled === 'boolean' ? raw.enabled : true,
      hit_count:
        typeof raw.hit_count === 'number' && Number.isFinite(raw.hit_count)
          ? raw.hit_count
          : 0,
      created_at: typeof raw.created_at === 'string' ? raw.created_at : null,
    })
  }
  return out
}

const stepTargets = ['plan_impl', 'implement', 'verify', 'review', 'commit']
const triggerOptions = [
  { value: 'before_step', label: 'Before Step' },
  { value: 'after_step', label: 'After Step' },
]

export default function BreakpointsPanel({ projectDir }: Props) {
  const [breakpoints, setBreakpoints] = useState<BreakpointInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState({
    trigger: 'before_step',
    target: 'verify',
    task_id: '',
    condition: '',
    action: 'pause',
  })

  useEffect(() => {
    fetchBreakpoints()
  }, [projectDir])

  useChannel('breakpoints', useCallback(() => {
    fetchBreakpoints()
  }, [projectDir]))

  const fetchBreakpoints = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/breakpoints', projectDir), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) throw new Error(`HTTP error ${response.status}`)
      const data = await response.json()
      setBreakpoints(normalizeBreakpoints(data))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch breakpoints')
    } finally {
      setLoading(false)
    }
  }

  const createBreakpoint = async () => {
    if (creating) return
    setCreating(true)
    try {
      const response = await fetch(buildApiUrl('/api/breakpoints', projectDir), {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          trigger: form.trigger,
          target: form.target,
          task_id: form.task_id.trim() || null,
          condition: form.condition.trim() || null,
          action: form.action,
        }),
      })
      const data = await response.json()
      if (!response.ok || !data?.success) {
        throw new Error(data?.message || `HTTP error ${response.status}`)
      }
      setForm((prev) => ({ ...prev, task_id: '', condition: '' }))
      await fetchBreakpoints()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create breakpoint')
    } finally {
      setCreating(false)
    }
  }

  const toggleBreakpoint = async (breakpointId: string) => {
    try {
      const response = await fetch(
        buildApiUrl(`/api/breakpoints/${breakpointId}/toggle`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders(),
        }
      )
      const data = await response.json()
      if (!response.ok || !data?.success) {
        throw new Error(data?.message || `HTTP error ${response.status}`)
      }
      await fetchBreakpoints()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle breakpoint')
    }
  }

  const deleteBreakpoint = async (breakpointId: string) => {
    try {
      const response = await fetch(buildApiUrl(`/api/breakpoints/${breakpointId}`, projectDir), {
        method: 'DELETE',
        headers: buildAuthHeaders(),
      })
      const data = await response.json()
      if (!response.ok || !data?.success) {
        throw new Error(data?.message || `HTTP error ${response.status}`)
      }
      await fetchBreakpoints()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete breakpoint')
    }
  }

  const clearAll = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/breakpoints', projectDir), {
        method: 'DELETE',
        headers: buildAuthHeaders(),
      })
      const data = await response.json()
      if (!response.ok || !data?.success) {
        throw new Error(data?.message || `HTTP error ${response.status}`)
      }
      await fetchBreakpoints()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear breakpoints')
    }
  }

  const sorted = useMemo(() => {
    return [...breakpoints].sort((a, b) => a.id.localeCompare(b.id))
  }, [breakpoints])

  return (
    <div className="card">
      <h2>Breakpoints</h2>

      <div className="breakpoints-actions">
        <button onClick={fetchBreakpoints} className="breakpoints-btn">
          Refresh
        </button>
        <button
          onClick={clearAll}
          disabled={breakpoints.length === 0}
          className="breakpoints-btn breakpoints-btn-danger"
        >
          Clear All
        </button>
      </div>

      {error && <div className="breakpoints-error">Error: {error}</div>}

      <div className="breakpoints-form">
        <div className="breakpoints-form-title">Create Breakpoint</div>

        <div className="breakpoints-form-row">
          <select
            value={form.trigger}
            onChange={(e) => setForm((prev) => ({ ...prev, trigger: e.target.value }))}
            className="breakpoints-form-select"
          >
            {triggerOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          <select
            value={form.target}
            onChange={(e) => setForm((prev) => ({ ...prev, target: e.target.value }))}
            className="breakpoints-form-select"
          >
            {stepTargets.map((step) => (
              <option key={step} value={step}>
                {step}
              </option>
            ))}
          </select>

          <input
            type="text"
            placeholder="Optional task_id"
            value={form.task_id}
            onChange={(e) => setForm((prev) => ({ ...prev, task_id: e.target.value }))}
            className="breakpoints-form-input breakpoints-form-input-task"
          />

          <input
            type="text"
            placeholder="Optional condition (e.g. files_changed > 10)"
            value={form.condition}
            onChange={(e) => setForm((prev) => ({ ...prev, condition: e.target.value }))}
            className="breakpoints-form-input breakpoints-form-input-condition"
          />

          <button
            onClick={createBreakpoint}
            disabled={creating}
            className="breakpoints-btn breakpoints-btn-primary"
          >
            {creating ? 'Creating...' : 'Create'}
          </button>
        </div>

        <div className="breakpoints-form-hint">
          Breakpoints pause the runner when hit. Use the Control Panel to resume the blocked task.
        </div>
      </div>

      {loading ? (
        <LoadingSpinner label="Loading breakpoints..." />
      ) : sorted.length === 0 ? (
        <EmptyState
          icon={<span>🔴</span>}
          title="No breakpoints set"
          description="Create a breakpoint above to pause execution at specific points."
          size="sm"
        />
      ) : (
        <div className="breakpoints-table-wrapper">
          <table className="breakpoints-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Trigger</th>
                <th>Target</th>
                <th>Task</th>
                <th>Condition</th>
                <th>Hits</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sorted.map((bp) => (
                <tr key={bp.id} className={bp.enabled ? '' : 'disabled'}>
                  <td className="breakpoints-table-id">{bp.id}</td>
                  <td>{bp.trigger}</td>
                  <td>{bp.target}</td>
                  <td>{bp.task_id || '-'}</td>
                  <td>{bp.condition || '-'}</td>
                  <td>{bp.hit_count}</td>
                  <td className="breakpoints-table-actions">
                    <button
                      onClick={() => toggleBreakpoint(bp.id)}
                      className="breakpoints-table-btn"
                    >
                      {bp.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button
                      onClick={() => deleteBreakpoint(bp.id)}
                      className="breakpoints-table-btn breakpoints-table-btn-delete"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
