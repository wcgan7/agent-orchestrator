import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material'
import RefreshIcon from '@mui/icons-material/Refresh'
import LaunchIcon from '@mui/icons-material/Launch'
import { buildApiUrl, buildAuthHeaders } from '../api'

interface QuickRun {
  id: string
  prompt: string
  status: 'running' | 'completed' | 'failed'
  started_at: string
  finished_at?: string | null
  result_summary?: string | null
  error?: string | null
  promoted_task_id?: string | null
}

interface Props {
  projectDir?: string
}

const statusColor = (status: QuickRun['status']): 'info' | 'success' | 'error' => {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'error'
  return 'info'
}

const trimPrompt = (prompt: string): string => {
  const compact = prompt.replace(/\s+/g, ' ').trim()
  return compact.length > 86 ? `${compact.slice(0, 83)}...` : compact
}

export default function QuickRunsPanel({ projectDir }: Props) {
  const [items, setItems] = useState<QuickRun[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [promoting, setPromoting] = useState<string | null>(null)

  const fetchQuickRuns = useCallback(async (isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true)
      const response = await fetch(buildApiUrl('/api/v3/quick-runs', projectDir, { limit: 25 }), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const data = await response.json()
      setItems(Array.isArray(data) ? data : [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load quick runs')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [projectDir])

  useEffect(() => {
    fetchQuickRuns()
    const timer = setInterval(() => {
      fetchQuickRuns()
    }, 10000)
    return () => clearInterval(timer)
  }, [fetchQuickRuns])

  const handlePromote = async (quickRunId: string) => {
    try {
      setPromoting(quickRunId)
      const response = await fetch(buildApiUrl(`/api/v3/quick-runs/${quickRunId}/promote`, projectDir), {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          task_type: 'feature',
          priority: 'P2',
        }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body?.detail || `HTTP ${response.status}`)
      }
      await fetchQuickRuns(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to promote quick run')
    } finally {
      setPromoting(null)
    }
  }

  const latest = useMemo(() => items[0], [items])

  if (loading) {
    return (
      <Stack spacing={1.5} alignItems="center" sx={{ py: 3 }}>
        <CircularProgress size={22} />
        <Typography variant="body2" color="text.secondary">Loading quick runs...</Typography>
      </Stack>
    )
  }

  return (
    <Stack spacing={1.5}>
      <Stack direction="row" alignItems="center" justifyContent="space-between">
        <Typography variant="h6">Quick Runs</Typography>
        <Button
          size="small"
          startIcon={<RefreshIcon />}
          variant="outlined"
          onClick={() => fetchQuickRuns(true)}
          disabled={refreshing}
        >
          Refresh
        </Button>
      </Stack>

      <Typography variant="body2" color="text.secondary">
        One-off actions are listed here. Promote any completed run to create a board task.
      </Typography>

      {error && <Alert severity="error">{error}</Alert>}

      {latest && latest.status === 'failed' && latest.error && (
        <Alert severity="warning">
          Latest quick run failed: {latest.error}
        </Alert>
      )}

      {items.length === 0 ? (
        <Alert severity="info">No quick runs yet. Run a Quick Action to populate this list.</Alert>
      ) : (
        <Box sx={{ overflowX: 'auto' }}>
          <Table size="small" aria-label="Quick runs table">
            <TableHead>
              <TableRow>
                <TableCell>Status</TableCell>
                <TableCell>Prompt</TableCell>
                <TableCell>Started</TableCell>
                <TableCell>Task</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {items.map((item) => (
                <TableRow key={item.id} hover>
                  <TableCell>
                    <Chip size="small" color={statusColor(item.status)} label={item.status} />
                  </TableCell>
                  <TableCell sx={{ maxWidth: 420 }}>
                    <Tooltip title={item.prompt}>
                      <Typography variant="body2" sx={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>
                        {trimPrompt(item.prompt)}
                      </Typography>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {new Date(item.started_at).toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {item.promoted_task_id ? (
                      <Chip size="small" variant="outlined" label={item.promoted_task_id} />
                    ) : (
                      <Typography variant="body2" color="text.secondary">Not promoted</Typography>
                    )}
                  </TableCell>
                  <TableCell align="right">
                    {item.status === 'completed' && !item.promoted_task_id ? (
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<LaunchIcon />}
                        onClick={() => handlePromote(item.id)}
                        disabled={promoting === item.id}
                      >
                        Promote
                      </Button>
                    ) : (
                      <Typography variant="caption" color="text.secondary">
                        {item.promoted_task_id ? 'Promoted' : 'N/A'}
                      </Typography>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}
    </Stack>
  )
}

