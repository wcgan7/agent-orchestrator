import { useState, useEffect } from 'react'
import { Box, Chip, Paper, Typography } from '@mui/material'
import { fetchExecutionOrder } from '../api'
import { humanizeLabel } from '../ui/labels'

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

  if (loading) {
    return (
      <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Parallel Execution Plan
        </Typography>
        <Typography variant="body2">Loading...</Typography>
      </Paper>
    )
  }

  if (error) {
    return (
      <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Parallel Execution Plan
        </Typography>
        <Typography variant="body2" sx={{ color: 'error.main' }}>
          {error}
        </Typography>
      </Paper>
    )
  }

  if (batches.length === 0) {
    return (
      <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Parallel Execution Plan
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center', py: 4 }}>
          No execution batches found
        </Typography>
      </Paper>
    )
  }

  return (
    <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Parallel Execution Plan
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {batches.map((batch, waveIdx) => (
          <Box
            key={waveIdx}
            sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5, bgcolor: 'action.hover' }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
              <Chip label={`Wave ${waveIdx + 1}`} size="small" color="primary" />
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {batch.length} task{batch.length !== 1 ? 's' : ''}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {batch.map((task) => (
                <Paper
                  key={task.id}
                  variant="outlined"
                  sx={{
                    p: 1,
                    fontSize: '0.8rem',
                    minWidth: 150,
                    flex: 1,
                    maxWidth: 250,
                  }}
                >
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
                    {task.id.slice(-8)}
                  </Typography>
                  {task.title && (
                    <Typography variant="body2" sx={{ mt: 0.25, fontWeight: 500 }} noWrap>
                      {task.title}
                    </Typography>
                  )}
                  {task.status && (
                    <Typography variant="caption" sx={{ mt: 0.25, display: 'block' }}>
                      {humanizeLabel(task.status)}
                    </Typography>
                  )}
                </Paper>
              ))}
            </Box>
          </Box>
        ))}
      </Box>
    </Paper>
  )
}
