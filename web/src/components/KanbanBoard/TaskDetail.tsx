/**
 * Task detail side panel with collaboration features.
 */

import { useState, useEffect } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  Chip,
  Divider,
  Drawer,
  MenuItem,
  Stack,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders, fetchInspect, fetchExplain, fetchTaskLogs, fetchTrace } from '../../api'
import FeedbackPanel from '../FeedbackPanel/FeedbackPanel'
import ActivityTimeline from '../ActivityTimeline/ActivityTimeline'
import ReasoningViewer from '../ReasoningViewer/ReasoningViewer'
import CorrectionForm from '../CorrectionForm'

type DetailTab =
  | 'summary'
  | 'dependencies'
  | 'logs'
  | 'interventions'
  | 'activity'
  | 'reasoning'
  | 'inspect'
  | 'trace'

interface TaskData {
  id: string
  title: string
  description: string
  task_type: string
  priority: string
  status: string
  labels: string[]
  assignee: string | null
  assignee_type: string | null
  acceptance_criteria: string[]
  context_files: string[]
  blocked_by: string[]
  blocks: string[]
  children_ids: string[]
  effort: string | null
  error: string | null
  error_type: string | null
  created_at: string
  updated_at: string
  completed_at: string | null
  source: string
  created_by: string | null
  [key: string]: any
}

interface Props {
  task: TaskData
  projectDir?: string
  onClose: () => void
  onUpdated: () => void
  onNavigateTask?: (task: TaskData) => void
}

const STATUS_COLORS: Record<string, 'default' | 'info' | 'warning' | 'success' | 'error'> = {
  backlog: 'default',
  ready: 'info',
  in_progress: 'warning',
  in_review: 'info',
  done: 'success',
  blocked: 'error',
  cancelled: 'default',
}

const PRIORITY_COLORS: Record<string, string> = {
  P0: '#ef4444',
  P1: '#f97316',
  P2: '#3b82f6',
  P3: '#9ca3af',
}

export function TaskDetail({ task, projectDir, onClose, onUpdated, onNavigateTask }: Props) {
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(task.title)
  const [description, setDescription] = useState(task.description)
  const [priority, setPriority] = useState(task.priority)
  const [taskType, setTaskType] = useState(task.task_type)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<DetailTab>('summary')
  const [actionError, setActionError] = useState<string | null>(null)

  const handleSave = async () => {
    setSaving(true)
    try {
      await fetch(
        buildApiUrl(`/api/v3/tasks/${task.id}`, projectDir),
        {
          method: 'PATCH',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ title, description, priority, task_type: taskType }),
        }
      )
      setEditing(false)
      onUpdated()
    } finally {
      setSaving(false)
    }
  }

  const handleTransition = async (newStatus: string) => {
    try {
      const response = await fetch(
        buildApiUrl(`/api/v3/tasks/${task.id}/transition`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ status: newStatus }),
        }
      )
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setActionError(data?.detail || `Failed to transition task (${response.status})`)
        return
      }
      setActionError(null)
      onUpdated()
    } catch {
      setActionError('Transition failed due to a network or server error')
    }
  }

  const handleRunNow = async () => {
    try {
      const response = await fetch(
        buildApiUrl(`/api/v3/tasks/${task.id}/run`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ claimer: 'manual_ui', assignee_type: 'agent' }),
        }
      )
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setActionError(data?.detail || `Failed to start task (${response.status})`)
        return
      }
      setActionError(null)
      onUpdated()
    } catch {
      setActionError('Run request failed due to a network or server error')
    }
  }

  const handleRetry = async () => {
    try {
      const response = await fetch(
        buildApiUrl(`/api/v3/tasks/${task.id}/retry`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ reason: 'manual_retry_from_task_detail' }),
        }
      )
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setActionError(data?.detail || `Failed to retry task (${response.status})`)
        return
      }
      setActionError(null)
      onUpdated()
    } catch {
      setActionError('Retry request failed due to a network or server error')
    }
  }

  const handleCancelTask = async () => {
    try {
      const response = await fetch(
        buildApiUrl(`/api/v3/tasks/${task.id}/cancel`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ reason: 'manual_cancel_from_task_detail' }),
        }
      )
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setActionError(data?.detail || `Failed to cancel task (${response.status})`)
        return
      }
      setActionError(null)
      onUpdated()
    } catch {
      setActionError('Cancel request failed due to a network or server error')
    }
  }

  const handleDelete = async () => {
    if (!confirm('Delete this task?')) return
    await fetch(
      buildApiUrl(`/api/v3/tasks/${task.id}`, projectDir),
      { method: 'DELETE', headers: buildAuthHeaders() }
    )
    onUpdated()
  }

  return (
    <Drawer anchor="right" open onClose={onClose} PaperProps={{ sx: { width: { xs: '100%', sm: 560 } } }}>
      <Stack sx={{ height: '100%' }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
            <Chip size="small" label={task.task_type} sx={{ bgcolor: 'text.primary', color: 'background.paper' }} />
            <Typography variant="caption" color="text.secondary" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
              {task.id}
            </Typography>
            <Typography variant="caption" sx={{ fontWeight: 700, color: PRIORITY_COLORS[task.priority] || '#3b82f6' }}>
              {task.priority}
            </Typography>
          </Stack>
          <Button onClick={onClose}>Close</Button>
        </Stack>

        <Tabs
          value={activeTab}
          onChange={(_, value: DetailTab) => setActiveTab(value)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Summary" value="summary" />
          <Tab label="Dependencies" value="dependencies" />
          <Tab label="Logs" value="logs" />
          <Tab label="Interventions" value="interventions" />
          <Tab label="Activity" value="activity" />
          <Tab label="Reasoning" value="reasoning" />
          <Tab label="Inspect" value="inspect" />
          <Tab label="Trace" value="trace" />
        </Tabs>

        <Box sx={{ p: 2, overflowY: 'auto', flex: 1 }}>
          {activeTab === 'interventions' ? (
            <Stack spacing={2}>
              <FeedbackPanel taskId={task.id} projectDir={projectDir} />
              {(task.error || task.status === 'blocked' || task.blocked_by.length > 0) && (
                <CorrectionForm taskId={task.id} projectDir={projectDir} />
              )}
            </Stack>
          ) : activeTab === 'activity' ? (
            <ActivityTimeline taskId={task.id} projectDir={projectDir} />
          ) : activeTab === 'reasoning' ? (
            <ReasoningViewer taskId={task.id} projectDir={projectDir} />
          ) : activeTab === 'inspect' ? (
            <InspectTab taskId={task.id} projectDir={projectDir} />
          ) : activeTab === 'logs' ? (
            <LogsTab taskId={task.id} projectDir={projectDir} />
          ) : activeTab === 'trace' ? (
            <TraceTab taskId={task.id} projectDir={projectDir} />
          ) : editing ? (
            <Stack spacing={1.25}>
              <TextField value={title} onChange={(e) => setTitle(e.target.value)} label="Title" fullWidth />
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                <TextField select value={taskType} onChange={(e) => setTaskType(e.target.value)} label="Type" fullWidth>
                  <MenuItem value="feature">Feature</MenuItem>
                  <MenuItem value="bug">Bug</MenuItem>
                  <MenuItem value="refactor">Refactor</MenuItem>
                  <MenuItem value="research">Research</MenuItem>
                  <MenuItem value="test">Test</MenuItem>
                  <MenuItem value="docs">Docs</MenuItem>
                </TextField>
                <TextField select value={priority} onChange={(e) => setPriority(e.target.value)} label="Priority" fullWidth>
                  <MenuItem value="P0">P0 - Critical</MenuItem>
                  <MenuItem value="P1">P1 - High</MenuItem>
                  <MenuItem value="P2">P2 - Medium</MenuItem>
                  <MenuItem value="P3">P3 - Low</MenuItem>
                </TextField>
              </Stack>
              <TextField
                multiline
                minRows={6}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                label="Description"
                fullWidth
              />
              <Stack direction="row" spacing={1}>
                <Button variant="contained" onClick={handleSave} disabled={saving}>
                  {saving ? 'Saving...' : 'Save'}
                </Button>
                <Button variant="outlined" onClick={() => setEditing(false)}>
                  Cancel
                </Button>
              </Stack>
            </Stack>
          ) : (
            <>
              <Typography variant="h6" sx={{ mb: 1 }}>{task.title}</Typography>
              {task.description && (
                <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', mb: 1.5 }}>
                  {task.description}
                </Typography>
              )}
              <Button size="small" variant="outlined" onClick={() => setEditing(true)}>Edit</Button>
            </>
          )}

          {activeTab === 'summary' && (
            <Stack spacing={2} sx={{ mt: 2 }}>
              {actionError && (
                <Alert severity="error">{actionError}</Alert>
              )}
              <Box>
                <Typography variant="overline" color="text.secondary">Status</Typography>
                <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                  <Chip size="small" color={STATUS_COLORS[task.status] || 'default'} label={task.status} />
                  <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
                    {task.status === 'backlog' && (
                      <>
                        <Button size="small" variant="outlined" onClick={() => handleTransition('ready')}>Move to Ready</Button>
                        <Button size="small" variant="contained" onClick={handleRunNow}>Run Now</Button>
                      </>
                    )}
                    {task.status === 'ready' && (
                      <Button size="small" variant="contained" onClick={handleRunNow}>Run Now</Button>
                    )}
                    {task.status === 'in_progress' && (
                      <>
                        <Button size="small" variant="outlined" onClick={() => handleTransition('in_review')}>Send to Human Review</Button>
                      </>
                    )}
                    {task.status === 'blocked' && (
                      <>
                        <Button size="small" variant="contained" onClick={handleRunNow}>Run Now</Button>
                        <Button size="small" variant="outlined" onClick={handleRetry}>Retry</Button>
                        <Button size="small" color="error" variant="outlined" onClick={handleCancelTask}>Cancel</Button>
                      </>
                    )}
                    {task.status === 'in_review' && (
                      <>
                        <Button size="small" variant="outlined" onClick={() => handleTransition('done')}>Approve</Button>
                        <Button size="small" variant="outlined" onClick={() => handleTransition('in_progress')}>Request Changes</Button>
                        <Button size="small" variant="outlined" onClick={handleRetry}>Retry</Button>
                        <Button size="small" color="error" variant="outlined" onClick={handleCancelTask}>Cancel</Button>
                      </>
                    )}
                  </Stack>
                </Stack>
              </Box>

              {task.acceptance_criteria.length > 0 && (
                <Box>
                  <Typography variant="overline" color="text.secondary">Acceptance Criteria</Typography>
                  <Stack component="ul" spacing={0.5} sx={{ pl: 2, my: 0.5 }}>
                    {task.acceptance_criteria.map((ac, i) => (
                      <Typography key={i} component="li" variant="body2" color="text.secondary">{ac}</Typography>
                    ))}
                  </Stack>
                </Box>
              )}

              {task.labels.length > 0 && (
                <Box>
                  <Typography variant="overline" color="text.secondary">Labels</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                    {task.labels.map((l) => (
                      <Chip key={l} size="small" label={l} variant="outlined" />
                    ))}
                  </Stack>
                </Box>
              )}

              <Box>
                <Typography variant="overline" color="text.secondary">Metadata</Typography>
                <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">Created: {new Date(task.created_at).toLocaleString()}</Typography>
                  <Typography variant="caption" color="text.secondary">Updated: {new Date(task.updated_at).toLocaleString()}</Typography>
                  {task.completed_at && <Typography variant="caption" color="text.secondary">Completed: {new Date(task.completed_at).toLocaleString()}</Typography>}
                  {task.assignee && <Typography variant="caption" color="text.secondary">Assignee: {task.assignee} ({task.assignee_type})</Typography>}
                  <Typography variant="caption" color="text.secondary">Source: {task.source}</Typography>
                </Stack>
              </Box>

              <Divider />
              <Button color="error" variant="outlined" onClick={handleDelete}>Delete Task</Button>
            </Stack>
          )}

          {activeTab === 'dependencies' && (
            <Stack spacing={2} sx={{ mt: 2 }}>
              {(task.status === 'blocked' || task.blocked_by.length > 0) && (
                <BlockedTaskAssistant
                  task={task}
                  projectDir={projectDir}
                  onUpdated={onUpdated}
                  onNavigateTask={onNavigateTask}
                />
              )}
              {(task.blocked_by.length > 0 || task.blocks.length > 0) ? (
                <Box>
                  <Typography variant="overline" color="text.secondary">Dependencies</Typography>
                  {task.blocked_by.length > 0 && (
                    <Stack direction="row" spacing={0.5} alignItems="center" useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">Blocked by:</Typography>
                      {task.blocked_by.map((id) => (
                        <Chip key={id} size="small" variant="outlined" label={id.slice(-8)} sx={{ fontFamily: '"IBM Plex Mono", monospace' }} />
                      ))}
                    </Stack>
                  )}
                  {task.blocks.length > 0 && (
                    <Stack direction="row" spacing={0.5} alignItems="center" useFlexGap flexWrap="wrap" sx={{ mt: 0.75 }}>
                      <Typography variant="caption" color="text.secondary">Blocks:</Typography>
                      {task.blocks.map((id) => (
                        <Chip key={id} size="small" variant="outlined" label={id.slice(-8)} sx={{ fontFamily: '"IBM Plex Mono", monospace' }} />
                      ))}
                    </Stack>
                  )}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">No dependencies found for this task.</Typography>
              )}

              {task.context_files.length > 0 && (
                <Box>
                  <Typography variant="overline" color="text.secondary">Context Files</Typography>
                  <Stack component="ul" spacing={0.5} sx={{ pl: 2, my: 0.5 }}>
                    {task.context_files.map((f) => (
                      <Typography key={f} component="li" variant="body2">
                        <Box component="code" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>{f}</Box>
                      </Typography>
                    ))}
                  </Stack>
                </Box>
              )}

              {task.error && (
                <Alert severity="error">
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Error</Typography>
                  <Box component="pre" sx={{ whiteSpace: 'pre-wrap', m: 0, fontSize: '0.75rem' }}>{task.error}</Box>
                  <ExplainButton taskId={task.id} projectDir={projectDir} />
                </Alert>
              )}
            </Stack>
          )}
        </Box>
      </Stack>
    </Drawer>
  )
}

function BlockedTaskAssistant({
  task,
  projectDir,
  onUpdated,
  onNavigateTask,
}: {
  task: TaskData
  projectDir?: string
  onUpdated: () => void
  onNavigateTask?: (task: TaskData) => void
}) {
  const [blockerTasks, setBlockerTasks] = useState<TaskData[]>([])
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState<string | null>(null)
  const [loadingTaskId, setLoadingTaskId] = useState<string | null>(null)
  const [assistantError, setAssistantError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const loadBlockers = async () => {
      if (task.blocked_by.length === 0) {
        setBlockerTasks([])
        return
      }
      setLoading(true)
      setAssistantError(null)
      try {
        const results = await Promise.all(
          task.blocked_by.map(async (id) => {
            const response = await fetch(buildApiUrl(`/api/v3/tasks/${id}`, projectDir), {
              headers: buildAuthHeaders(),
            })
            if (!response.ok) return null
            const payload = await response.json()
            return payload?.task || null
          }),
        )
        if (!active) return
        setBlockerTasks(results.filter(Boolean))
      } catch {
        if (!active) return
        setAssistantError('Failed to load blocker details')
      } finally {
        if (active) setLoading(false)
      }
    }

    loadBlockers()
    return () => {
      active = false
    }
  }, [task.id, task.blocked_by, projectDir])

  const runBlocker = async (blockerId: string) => {
    setRunning(blockerId)
    setAssistantError(null)
    try {
      const response = await fetch(
        buildApiUrl(`/api/v3/tasks/${blockerId}/run`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ claimer: 'manual_ui', assignee_type: 'agent' }),
        },
      )
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setAssistantError(data?.detail || `Failed to run blocker (${response.status})`)
        return
      }
      onUpdated()
    } catch {
      setAssistantError('Failed to run blocker due to a network or server error')
    } finally {
      setRunning(null)
    }
  }

  const openBlocker = async (blockerId: string) => {
    if (!onNavigateTask) return
    setLoadingTaskId(blockerId)
    setAssistantError(null)
    try {
      const response = await fetch(buildApiUrl(`/api/v3/tasks/${blockerId}`, projectDir), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setAssistantError(data?.detail || `Failed to load blocker (${response.status})`)
        return
      }
      const payload = await response.json()
      if (payload?.task) onNavigateTask(payload.task)
    } catch {
      setAssistantError('Failed to load blocker due to a network or server error')
    } finally {
      setLoadingTaskId(null)
    }
  }

  const tryUnblockCurrent = async () => {
    setAssistantError(null)
    try {
      const response = await fetch(
        buildApiUrl(`/api/v3/tasks/${task.id}/retry`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ reason: 'manual_unblock_shortcut' }),
        },
      )
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        setAssistantError(data?.detail || `Failed to unblock task (${response.status})`)
        return
      }
      onUpdated()
    } catch {
      setAssistantError('Unblock request failed due to a network or server error')
    }
  }

  return (
    <Alert severity={task.status === 'blocked' ? 'warning' : 'info'} variant="outlined">
      <Stack spacing={1.25}>
        <Typography variant="subtitle2">
          {task.status === 'blocked' ? 'Task is blocked' : 'Dependency blockers detected'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Resolve or run blocker tasks, then use Try Unblock to move this task back to ready.
        </Typography>
        {assistantError && <Alert severity="error">{assistantError}</Alert>}
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
          <ExplainButton taskId={task.id} projectDir={projectDir} />
          <Button size="small" variant="contained" onClick={tryUnblockCurrent}>
            Try Unblock
          </Button>
        </Stack>

        {loading ? (
          <Typography variant="body2" color="text.secondary">Loading blocker details...</Typography>
        ) : blockerTasks.length > 0 ? (
          <Stack spacing={1}>
            {blockerTasks.map((blocker) => (
              <Card key={blocker.id} variant="outlined" sx={{ p: 1 }}>
                <Stack spacing={0.5}>
                  <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
                    <Typography variant="body2" sx={{ fontWeight: 700 }}>{blocker.title}</Typography>
                    <Chip size="small" label={blocker.status} color={STATUS_COLORS[blocker.status] || 'default'} />
                    <Typography variant="caption" color="text.secondary" sx={{ fontFamily: '"IBM Plex Mono", monospace' }}>
                      {blocker.id}
                    </Typography>
                  </Stack>
                  {blocker.error && (
                    <Typography variant="caption" color="error.main" sx={{ wordBreak: 'break-word' }}>
                      {blocker.error}
                    </Typography>
                  )}
                  <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => openBlocker(blocker.id)}
                      disabled={!onNavigateTask || loadingTaskId === blocker.id}
                    >
                      {loadingTaskId === blocker.id ? 'Opening...' : 'Open'}
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => runBlocker(blocker.id)}
                      disabled={running === blocker.id}
                    >
                      {running === blocker.id ? 'Running...' : 'Run Blocker'}
                    </Button>
                  </Stack>
                </Stack>
              </Card>
            ))}
          </Stack>
        ) : task.blocked_by.length > 0 ? (
          <Typography variant="body2" color="text.secondary">
            Blocker IDs: {task.blocked_by.join(', ')}
          </Typography>
        ) : null}
      </Stack>
    </Alert>
  )
}

function ExplainButton({ taskId, projectDir }: { taskId: string; projectDir?: string }) {
  const [explanation, setExplanation] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleClick = async () => {
    setLoading(true)
    try {
      const data = await fetchExplain(taskId, projectDir)
      setExplanation(data.explanation)
    } catch {
      setExplanation('Failed to load explanation')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ mt: 1 }}>
      {explanation ? (
        <Box component="pre" sx={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap', bgcolor: 'background.default', p: 1, borderRadius: 1, m: 0 }}>
          {explanation}
        </Box>
      ) : (
        <Button size="small" variant="outlined" onClick={handleClick} disabled={loading} sx={{ mt: 0.5 }}>
          {loading ? 'Loading...' : 'Why blocked?'}
        </Button>
      )}
    </Box>
  )
}

function InspectTab({ taskId, projectDir }: { taskId: string; projectDir?: string }) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchInspect(taskId, projectDir)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [taskId, projectDir])

  if (loading) return <Typography>Loading inspection data...</Typography>
  if (error) return <Alert severity="error">{error}</Alert>
  if (!data) return <Typography>No data available</Typography>

  return (
    <Stack spacing={1.5}>
      <Table size="small">
        <TableBody>
          {[
            ['Lifecycle', data.lifecycle],
            ['Step', data.step],
            ['Status', data.status],
            ['Worker Attempts', data.worker_attempts],
            ['Last Error', data.last_error || '-'],
            ['Error Type', data.last_error_type || '-'],
          ].map(([key, val]) => (
            <TableRow key={key}>
              <TableCell sx={{ fontWeight: 600, width: '40%' }}>{key}</TableCell>
              <TableCell sx={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '0.8rem' }}>{String(val)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {data.context && data.context.length > 0 && (
        <Box>
          <Typography variant="subtitle2">Context</Typography>
          <Stack component="ul" spacing={0.5} sx={{ pl: 2, my: 0.5 }}>
            {data.context.map((c: string, i: number) => <Typography key={i} component="li" variant="body2">{c}</Typography>)}
          </Stack>
        </Box>
      )}

      {data.metadata && Object.keys(data.metadata).length > 0 && (
        <Box>
          <Typography variant="subtitle2">Metadata</Typography>
          <Box component="pre" sx={{ fontSize: '0.8rem', bgcolor: 'background.default', p: 1, borderRadius: 1, overflow: 'auto', m: 0 }}>
            {JSON.stringify(data.metadata, null, 2)}
          </Box>
        </Box>
      )}
    </Stack>
  )
}

function LogsTab({ taskId, projectDir }: { taskId: string; projectDir?: string }) {
  const [logs, setLogs] = useState<Record<string, string[]>>({})
  const [step, setStep] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchTaskLogs(taskId, projectDir, step || undefined, 200)
      .then((data) => setLogs(data.logs || {}))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [taskId, projectDir, step])

  return (
    <Stack spacing={1.25}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="body2" sx={{ fontWeight: 500 }}>Step filter:</Typography>
        <TextField size="small" select value={step} onChange={(e) => setStep(e.target.value)} sx={{ minWidth: 180 }}>
          <MenuItem value="">All</MenuItem>
          <MenuItem value="plan_impl">Plan/Impl</MenuItem>
          <MenuItem value="implement">Implement</MenuItem>
          <MenuItem value="verify">Verify</MenuItem>
          <MenuItem value="review">Review</MenuItem>
          <MenuItem value="commit">Commit</MenuItem>
        </TextField>
      </Stack>

      {loading ? (
        <Typography>Loading logs...</Typography>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : Object.keys(logs).length === 0 ? (
        <Typography color="text.secondary">No logs found for this task</Typography>
      ) : (
        Object.entries(logs).map(([filename, lines]) => (
          <Box key={filename}>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>{filename}</Typography>
            <Box
              component="pre"
              sx={{
                m: 0,
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: '0.75rem',
                bgcolor: 'background.default',
                p: 1,
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
                overflow: 'auto',
                maxHeight: 300,
                whiteSpace: 'pre-wrap',
              }}
            >
              {lines.join('\n') || '(empty)'}
            </Box>
          </Box>
        ))
      )}
    </Stack>
  )
}

function TraceTab({ taskId, projectDir }: { taskId: string; projectDir?: string }) {
  const [events, setEvents] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchTrace(taskId, projectDir, 100)
      .then((data) => setEvents(Array.isArray(data) ? data : []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [taskId, projectDir])

  if (loading) return <Typography>Loading event history...</Typography>
  if (error) return <Alert severity="error">{error}</Alert>
  if (events.length === 0) return <Typography color="text.secondary">No events found for this task</Typography>

  return (
    <Stack spacing={1}>
      <Typography variant="caption" color="text.secondary">
        {events.length} event{events.length !== 1 ? 's' : ''}
      </Typography>
      {events.map((event, i) => {
        const eventType = event.event_type || 'unknown'
        const timestamp = event.timestamp || ''
        const isFail = eventType.includes('fail') || eventType.includes('error') || eventType.includes('violation')
        const isPass = eventType.includes('pass') || eventType === 'task_completed'
        return (
          <Box
            key={i}
            sx={{
              p: 1,
              borderLeft: '3px solid',
              borderLeftColor: isFail ? 'error.main' : isPass ? 'success.main' : 'info.main',
              bgcolor: 'background.default',
              borderRadius: '0 4px 4px 0',
            }}
          >
            <Stack direction="row" justifyContent="space-between" spacing={1}>
              <Typography variant="body2" sx={{ fontWeight: 700 }}>{eventType}</Typography>
              <Typography variant="caption" color="text.secondary">{timestamp}</Typography>
            </Stack>
            {event.run_id && <Typography variant="caption" color="text.secondary">Run: {event.run_id}</Typography>}
            {event.error_type && <Typography variant="caption" color="error.main">Error: {event.error_type}</Typography>}
            {event.error_detail && <Typography variant="caption" color="error.dark">{String(event.error_detail).slice(0, 150)}</Typography>}
            {event.block_reason && <Typography variant="caption">Reason: {event.block_reason}</Typography>}
            {event.passed !== undefined && (
              <Typography variant="caption" color={event.passed ? 'success.main' : 'error.main'}>
                {event.passed ? 'Passed' : 'Failed'}
              </Typography>
            )}
          </Box>
        )
      })}
    </Stack>
  )
}
