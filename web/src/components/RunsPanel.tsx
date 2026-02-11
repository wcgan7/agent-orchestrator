import { useEffect, useMemo, useState, useCallback } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
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

  const sortedRuns = useMemo(() => [...runs].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')), [runs])

  return (
    <Box>
      <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>Recent Runs</Typography>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }} justifyContent="space-between" className="runs-panel-header" sx={{ mb: 1.5 }}>
        <Button onClick={fetchRuns} variant="outlined" className="runs-panel-btn">Refresh</Button>
        {currentRunId && (
          <Typography className="runs-panel-active-run" variant="body2" color="text.secondary">
            Active run: <span className="runs-panel-active-run-id">{currentRunId}</span>
          </Typography>
        )}
      </Stack>

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
          <Box className="runs-panel-table-wrapper" sx={{ overflowX: 'auto' }}>
            <Table className="runs-panel-table" size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Run</TableCell>
                  <TableCell>Task</TableCell>
                  <TableCell>Phase</TableCell>
                  <TableCell>Step</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedRuns.map((run) => (
                  <TableRow
                    key={run.run_id}
                    className={run.run_id === currentRunId ? 'active' : ''}
                    sx={run.run_id === currentRunId ? { bgcolor: 'action.selected' } : undefined}
                  >
                    <TableCell className="runs-panel-table-id">{run.run_id}</TableCell>
                    <TableCell>{run.task_id || '-'}</TableCell>
                    <TableCell>{run.phase || '-'}</TableCell>
                    <TableCell>{run.step || '-'}</TableCell>
                    <TableCell><Chip size="small" variant="outlined" label={run.status || '-'} /></TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        onClick={() => {
                          setSelectedRunId(run.run_id)
                          void fetchRunDetail(run.run_id)
                        }}
                        className="runs-panel-table-btn"
                      >
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>

          {selectedRunId && (
            <Card className="runs-panel-detail" variant="outlined" sx={{ mt: 1.5 }}>
              <CardContent>
                <Typography className="runs-panel-detail-title" variant="subtitle2">Run Details</Typography>
                {detailError ? (
                  <Alert className="runs-panel-detail-error" severity="error" sx={{ mt: 1 }}>
                    Error: {detailError}
                  </Alert>
                ) : runDetail ? (
                  <Stack spacing={0.75} sx={{ mt: 1 }}>
                    <Typography className="runs-panel-detail-id" variant="body2">{runDetail.run_id}</Typography>
                    <Typography className="runs-panel-detail-status" variant="body2">
                      Status: <strong>{runDetail.status}</strong>
                    </Typography>
                    {runDetail.last_error && (
                      <Alert className="runs-panel-detail-last-error" severity="error">Last error: {runDetail.last_error}</Alert>
                    )}
                    <Typography className="runs-panel-detail-meta" variant="caption" color="text.secondary">
                      Current task: {runDetail.current_task_id || '-'} • Current phase: {runDetail.current_phase_id || '-'}
                    </Typography>
                  </Stack>
                ) : (
                  <Typography className="runs-panel-detail-loading" variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Loading...
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  )
}
