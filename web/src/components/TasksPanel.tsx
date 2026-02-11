import { useEffect, useMemo, useState, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'
import './TasksPanel.css'

interface TaskInfo {
  id: string
  type: string
  phase_id?: string | null
  step: string
  lifecycle: string
  status: string
  branch?: string | null
  last_error?: string | null
  last_run_id?: string | null
  worker_attempts: number
}

interface Props {
  projectDir?: string
  currentTaskId?: string
}

const normalizeTasks = (value: unknown): TaskInfo[] => {
  if (!Array.isArray(value)) return []
  const out: TaskInfo[] = []
  for (const item of value) {
    if (!item || typeof item !== 'object') continue
    const raw = item as Record<string, unknown>
    const id = typeof raw.id === 'string' ? raw.id : ''
    if (!id) continue
    out.push({
      id,
      type: typeof raw.type === 'string' ? raw.type : '',
      phase_id: typeof raw.phase_id === 'string' ? raw.phase_id : null,
      step: typeof raw.step === 'string' ? raw.step : '',
      lifecycle: typeof raw.lifecycle === 'string' ? raw.lifecycle : '',
      status: typeof raw.status === 'string' ? raw.status : '',
      branch: typeof raw.branch === 'string' ? raw.branch : null,
      last_error: typeof raw.last_error === 'string' ? raw.last_error : null,
      last_run_id: typeof raw.last_run_id === 'string' ? raw.last_run_id : null,
      worker_attempts:
        typeof raw.worker_attempts === 'number' && Number.isFinite(raw.worker_attempts)
          ? raw.worker_attempts
          : 0,
    })
  }
  return out
}

export default function TasksPanel({ projectDir, currentTaskId }: Props) {
  const [tasks, setTasks] = useState<TaskInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')

  useEffect(() => {
    fetchTasks()
  }, [projectDir])

  useChannel('tasks', useCallback(() => {
    fetchTasks()
  }, [projectDir]))

  const fetchTasks = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/tasks', projectDir), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setTasks(normalizeTasks(data))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tasks')
    } finally {
      setLoading(false)
    }
  }

  const filteredTasks = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return tasks
    return tasks.filter((t) => {
      return (
        t.id.toLowerCase().includes(q) ||
        (t.phase_id || '').toLowerCase().includes(q) ||
        t.step.toLowerCase().includes(q) ||
        t.lifecycle.toLowerCase().includes(q) ||
        t.status.toLowerCase().includes(q)
      )
    })
  }, [tasks, query])

  const counts = useMemo(() => {
    const byLifecycle: Record<string, number> = {}
    for (const t of tasks) {
      byLifecycle[t.lifecycle] = (byLifecycle[t.lifecycle] || 0) + 1
    }
    return byLifecycle
  }, [tasks])

  return (
    <div className="card">
      <h2>Tasks</h2>

      <div className="tasks-panel-header">
        <input
          type="text"
          placeholder="Filter tasks..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="tasks-panel-search"
        />
        <button onClick={fetchTasks} className="tasks-panel-btn">
          Refresh
        </button>
      </div>

      {loading ? (
        <LoadingSpinner label="Loading tasks..." />
      ) : error ? (
        <EmptyState
          icon={<span>⚠️</span>}
          title="Error loading tasks"
          description={error}
          size="sm"
        />
      ) : tasks.length === 0 ? (
        <EmptyState
          icon={<span>📝</span>}
          title="No tasks found"
          description="Generate phases to create a task queue."
          size="sm"
        />
      ) : (
        <>
          <div className="tasks-panel-summary">
            Total: {tasks.length}
            {Object.keys(counts).length > 0 && (
              <>
                {' • '}
                {Object.entries(counts)
                  .map(([k, v]) => `${k}: ${v}`)
                  .join(' • ')}
              </>
            )}
          </div>

          <div className="tasks-panel-table-wrapper">
            <table className="tasks-panel-table">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Phase</th>
                  <th>Step</th>
                  <th>Lifecycle</th>
                  <th>Attempts</th>
                </tr>
              </thead>
              <tbody>
                {filteredTasks.map((task) => (
                  <tr
                    key={task.id}
                    className={task.id === currentTaskId ? 'active' : ''}
                  >
                    <td>
                      <span className="tasks-panel-task-id">{task.id}</span>
                      {task.last_error && (
                        <div className="tasks-panel-task-error">
                          {task.last_error}
                        </div>
                      )}
                    </td>
                    <td>{task.phase_id || '-'}</td>
                    <td>{task.step || '-'}</td>
                    <td>{task.lifecycle || '-'}</td>
                    <td>{task.worker_attempts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
