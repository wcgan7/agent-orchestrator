import { useState, useEffect } from 'react'
import { fetchExecutionOrder } from '../api'
import './ParallelPlanView.css'

interface Props {
  projectDir?: string
}

interface TaskItem {
  id: string
  title?: string
  status?: string
  task_type?: string
}

type BatchItem = TaskItem[] | { batch: number; tasks: TaskItem[] }

export default function ParallelPlanView({ projectDir }: Props) {
  const [batches, setBatches] = useState<TaskItem[][]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchExecutionOrder(projectDir)
      .then((data) => {
        // Normalize response — may be array of arrays or array of objects with batch/tasks
        const raw: BatchItem[] = Array.isArray(data) ? data : data.batches || data.order || []
        const normalized = raw.map((item) => {
          if (Array.isArray(item)) return item
          if (typeof item === 'object' && 'tasks' in item) return (item as any).tasks as TaskItem[]
          return [item] as TaskItem[]
        })
        setBatches(normalized)
      })
      .catch((err) => setError(err.message || 'Failed to load execution order'))
      .finally(() => setLoading(false))
  }, [projectDir])

  if (loading) return <div className="card parallel-plan-view"><h2>Parallel Execution Plan</h2><p>Loading...</p></div>
  if (error) return <div className="card parallel-plan-view"><h2>Parallel Execution Plan</h2><p className="parallel-plan-error">{error}</p></div>

  if (batches.length === 0) {
    return (
      <div className="card parallel-plan-view">
        <h2>Parallel Execution Plan</h2>
        <div className="parallel-plan-empty">No execution batches found</div>
      </div>
    )
  }

  return (
    <div className="card parallel-plan-view">
      <h2>Parallel Execution Plan</h2>
      <div className="parallel-plan-lanes">
        {batches.map((batch, waveIdx) => (
          <div className="parallel-plan-lane" key={waveIdx}>
            <div className="parallel-plan-lane-header">
              <span className="parallel-plan-wave-badge">Wave {waveIdx + 1}</span>
              <span className="parallel-plan-wave-count">
                {batch.length} task{batch.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div className="parallel-plan-tasks">
              {batch.map((task) => (
                <div className="parallel-plan-task" key={task.id}>
                  <div className="parallel-plan-task-id">{task.id.slice(-8)}</div>
                  {task.title && <div className="parallel-plan-task-title">{task.title}</div>}
                  {task.status && <div className="parallel-plan-task-status">{task.status}</div>}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
