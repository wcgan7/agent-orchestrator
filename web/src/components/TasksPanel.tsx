import { useEffect, useMemo, useState, useCallback } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
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
    return tasks.filter((t) => (
      t.id.toLowerCase().includes(q) ||
      (t.phase_id || '').toLowerCase().includes(q) ||
      t.step.toLowerCase().includes(q) ||
      t.lifecycle.toLowerCase().includes(q) ||
      t.status.toLowerCase().includes(q)
    ))
  }, [tasks, query])

  const counts = useMemo(() => {
    const byLifecycle: Record<string, number> = {}
    for (const t of tasks) {
      byLifecycle[t.lifecycle] = (byLifecycle[t.lifecycle] || 0) + 1
    }
    return byLifecycle
  }, [tasks])

  return (
    <Box>
      <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>Tasks</Typography>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} className="tasks-panel-header" sx={{ mb: 1.5 }}>
        <TextField
          size="small"
          placeholder="Filter tasks..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="tasks-panel-search"
          fullWidth
        />
        <Button variant="outlined" onClick={fetchTasks} className="tasks-panel-btn">
          Refresh
        </Button>
      </Stack>

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
          <Box className="tasks-panel-summary" sx={{ mb: 1.25 }}>
            <Typography variant="caption" color="text.secondary">
              Total: {tasks.length}
            </Typography>
            <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.75 }}>
              {Object.entries(counts).map(([k, v]) => (
                <Chip key={k} size="small" variant="outlined" label={`${k}: ${v}`} />
              ))}
            </Stack>
          </Box>

          <Box className="tasks-panel-table-wrapper" sx={{ overflowX: 'auto' }}>
            <Table className="tasks-panel-table" size="small" aria-label="Task table">
              <TableHead>
                <TableRow>
                  <TableCell>Task</TableCell>
                  <TableCell>Phase</TableCell>
                  <TableCell>Step</TableCell>
                  <TableCell>Lifecycle</TableCell>
                  <TableCell align="right">Attempts</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredTasks.map((task) => {
                  const isActive = task.id === currentTaskId
                  return (
                    <TableRow
                      key={task.id}
                      className={isActive ? 'active' : ''}
                      sx={isActive ? { bgcolor: 'action.selected' } : undefined}
                    >
                      <TableCell>
                        <Typography className="tasks-panel-task-id" fontWeight={600}>{task.id}</Typography>
                        {task.last_error && (
                          <Alert className="tasks-panel-task-error" severity="error" sx={{ mt: 0.5, py: 0 }}>
                            {task.last_error}
                          </Alert>
                        )}
                      </TableCell>
                      <TableCell>{task.phase_id || '-'}</TableCell>
                      <TableCell>{task.step || '-'}</TableCell>
                      <TableCell>{task.lifecycle || '-'}</TableCell>
                      <TableCell align="right">{task.worker_attempts}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </Box>
        </>
      )}
    </Box>
  )
}
