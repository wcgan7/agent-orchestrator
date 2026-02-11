/**
 * Kanban board for the dynamic task engine.
 * Tasks flow through columns: Backlog → Ready → In Progress → In Review → Done
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import {
  Box,
  Button,
  Card,
  Chip,
  MenuItem,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'
import { TaskCard } from './TaskCard'
import { TaskDetail } from './TaskDetail'
import { CreateTaskModal } from './CreateTaskModal'

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
  effort: string | null
  blocked_by: string[]
  blocks: string[]
  children_ids: string[]
  acceptance_criteria: string[]
  context_files: string[]
  created_at: string
  updated_at: string
  completed_at: string | null
  error: string | null
  error_type: string | null
  source: string
  created_by: string | null
  [key: string]: any
}

interface BoardData {
  columns: Record<string, TaskData[]>
}

interface SavedView {
  id: string
  name: string
  filterType: string
  filterPriority: string
  filterStatus: string
  searchQuery: string
}

const COLUMN_ORDER = ['backlog', 'ready', 'in_progress', 'in_review', 'done']
const COLUMN_LABELS: Record<string, string> = {
  backlog: 'Backlog',
  ready: 'Ready',
  in_progress: 'In Progress',
  in_review: 'In Review',
  blocked: 'Blocked',
  done: 'Done',
}

const STORAGE_KEY_SAVED_VIEWS = 'feature-prd-runner-kanban-saved-views'
const BUILTIN_VIEWS: SavedView[] = [
  {
    id: 'all',
    name: 'All Tasks',
    filterType: '',
    filterPriority: '',
    filterStatus: '',
    searchQuery: '',
  },
  {
    id: 'blocked',
    name: 'Blocked Tasks',
    filterType: '',
    filterPriority: '',
    filterStatus: 'blocked',
    searchQuery: '',
  },
  {
    id: 'high-priority',
    name: 'High Priority',
    filterType: '',
    filterPriority: 'P0',
    filterStatus: '',
    searchQuery: '',
  },
  {
    id: 'in-review',
    name: 'Needs Review',
    filterType: '',
    filterPriority: '',
    filterStatus: 'in_review',
    searchQuery: '',
  },
]

interface Props {
  projectDir?: string
}

export default function KanbanBoard({ projectDir }: Props) {
  const [board, setBoard] = useState<BoardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTask, setSelectedTask] = useState<TaskData | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [filterType, setFilterType] = useState<string>('')
  const [filterPriority, setFilterPriority] = useState<string>('')
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [savedViews, setSavedViews] = useState<SavedView[]>([])
  const [selectedViewId, setSelectedViewId] = useState('all')
  const [draggedTaskId, setDraggedTaskId] = useState<string | null>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_SAVED_VIEWS)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        setSavedViews(parsed.filter((v) => v && typeof v.id === 'string'))
      }
    } catch {
      // ignore invalid localStorage data
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_SAVED_VIEWS, JSON.stringify(savedViews))
  }, [savedViews])

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
        if (e.key === 'Escape') {
          ;(e.target as HTMLElement).blur()
        }
        return
      }

      if (e.key === 'n' || e.key === 'N') {
        e.preventDefault()
        setShowCreate(true)
      } else if (e.key === '/') {
        e.preventDefault()
        searchInputRef.current?.focus()
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault()
        fetchBoard()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  const fetchBoard = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/v3/tasks/board', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setBoard(data)
      }
    } catch {
      // silently retry on next cycle
    } finally {
      setLoading(false)
    }
  }, [projectDir])

  useEffect(() => {
    fetchBoard()
  }, [fetchBoard])

  useChannel('tasks', useCallback((_event: string, _data: any) => {
    fetchBoard()
  }, [fetchBoard]))

  const handleDragStart = (taskId: string) => {
    setDraggedTaskId(taskId)
  }

  const handleDrop = async (targetColumn: string) => {
    if (!draggedTaskId) return
    setDraggedTaskId(null)

    try {
      await fetch(
        buildApiUrl(`/api/v3/tasks/${draggedTaskId}/transition`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ status: targetColumn }),
        }
      )
      fetchBoard()
    } catch {
      // transition failed — board will refresh
    }
  }

  const handleTaskCreated = () => {
    setShowCreate(false)
    fetchBoard()
  }

  const handleTaskUpdated = () => {
    setSelectedTask(null)
    fetchBoard()
  }

  const allViews = useMemo(() => [...BUILTIN_VIEWS, ...savedViews], [savedViews])

  const applyView = useCallback((view: SavedView) => {
    setFilterType(view.filterType)
    setFilterPriority(view.filterPriority)
    setFilterStatus(view.filterStatus)
    setSearchQuery(view.searchQuery)
    setSelectedViewId(view.id)
  }, [])

  const saveCurrentView = () => {
    const name = prompt('Saved view name')
    if (!name?.trim()) return
    const id = `custom-${Date.now()}`
    const next: SavedView = {
      id,
      name: name.trim(),
      filterType,
      filterPriority,
      filterStatus,
      searchQuery,
    }
    setSavedViews((prev) => [next, ...prev.filter((v) => v.name !== next.name)].slice(0, 12))
    setSelectedViewId(id)
  }

  const clearFilters = () => {
    applyView(BUILTIN_VIEWS[0])
  }

  const filteredBoard = useMemo(() => {
    if (!board?.columns) return null
    const filtered: Record<string, TaskData[]> = {}
    for (const [col, tasks] of Object.entries(board.columns)) {
      filtered[col] = tasks.filter((t) => {
        if (filterType && t.task_type !== filterType) return false
        if (filterPriority && t.priority !== filterPriority) return false
        if (filterStatus && t.status !== filterStatus) return false
        if (searchQuery) {
          const q = searchQuery.toLowerCase()
          if (!t.title.toLowerCase().includes(q) && !t.id.toLowerCase().includes(q)) return false
        }
        return true
      })
    }
    return { columns: filtered }
  }, [board, filterType, filterPriority, filterStatus, searchQuery])

  const totalTasks = board?.columns
    ? Object.values(board.columns).reduce((sum, col) => sum + col.length, 0)
    : 0

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 220 }}>
        <Typography color="text.secondary">Loading board...</Typography>
      </Box>
    )
  }

  return (
    <Stack spacing={1} sx={{ height: '100%' }}>
      <Card variant="outlined" sx={{ p: 1.25 }}>
        <Stack
          direction={{ xs: 'column', lg: 'row' }}
          spacing={1}
          alignItems={{ lg: 'flex-start' }}
          sx={{
            flexWrap: { lg: 'wrap' },
            rowGap: 1.25,
            columnGap: 1,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flexShrink: 0 }}>
            <Typography variant="h6">Task Board</Typography>
            <Chip size="small" label={`${totalTasks} tasks`} variant="outlined" />
          </Stack>

          <Stack
            direction={{ xs: 'column', md: 'row' }}
            spacing={1}
            alignItems={{ md: 'center' }}
            sx={{
              flex: '1 1 720px',
              minWidth: 0,
              width: '100%',
            }}
          >
            <TextField
              inputRef={searchInputRef}
              size="small"
              placeholder="Search tasks... ( / )"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              sx={{ minWidth: { xs: '100%', md: 210 }, flex: { md: '1 1 240px' } }}
            />
            <TextField
              size="small"
              select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              SelectProps={{ displayEmpty: true }}
              sx={{ minWidth: { xs: '100%', md: 130 }, flex: { md: '1 1 140px' } }}
            >
              <MenuItem value="">All types</MenuItem>
              <MenuItem value="feature">Feature</MenuItem>
              <MenuItem value="bug">Bug</MenuItem>
              <MenuItem value="refactor">Refactor</MenuItem>
              <MenuItem value="research">Research</MenuItem>
              <MenuItem value="test">Test</MenuItem>
              <MenuItem value="docs">Docs</MenuItem>
            </TextField>
            <TextField
              size="small"
              select
              value={filterPriority}
              onChange={(e) => setFilterPriority(e.target.value)}
              SelectProps={{ displayEmpty: true }}
              sx={{ minWidth: { xs: '100%', md: 150 }, flex: { md: '1 1 150px' } }}
            >
              <MenuItem value="">All priorities</MenuItem>
              <MenuItem value="P0">P0 - Critical</MenuItem>
              <MenuItem value="P1">P1 - High</MenuItem>
              <MenuItem value="P2">P2 - Medium</MenuItem>
              <MenuItem value="P3">P3 - Low</MenuItem>
            </TextField>
            <TextField
              size="small"
              select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              SelectProps={{ displayEmpty: true }}
              sx={{ minWidth: { xs: '100%', md: 145 }, flex: { md: '1 1 145px' } }}
            >
              <MenuItem value="">All status</MenuItem>
              <MenuItem value="backlog">Backlog</MenuItem>
              <MenuItem value="ready">Ready</MenuItem>
              <MenuItem value="in_progress">In Progress</MenuItem>
              <MenuItem value="in_review">In Review</MenuItem>
              <MenuItem value="blocked">Blocked</MenuItem>
              <MenuItem value="done">Done</MenuItem>
            </TextField>
            <TextField
              size="small"
              select
              value={selectedViewId}
              onChange={(e) => {
                const next = allViews.find((v) => v.id === e.target.value)
                if (next) applyView(next)
              }}
              sx={{ minWidth: { xs: '100%', md: 145 }, flex: { md: '1 1 145px' } }}
            >
              {allViews.map((view) => (
                <MenuItem key={view.id} value={view.id}>{view.name}</MenuItem>
              ))}
            </TextField>
          </Stack>

          <Stack
            direction="row"
            spacing={0.75}
            useFlexGap
            flexWrap="wrap"
            sx={{
              width: { xs: '100%', lg: 'auto' },
              justifyContent: { xs: 'flex-start', lg: 'flex-end' },
              flexShrink: 0,
              mt: { xs: 0.5, lg: 0.25 },
            }}
          >
            <Tooltip title="Save current filters as a reusable view">
              <Button variant="outlined" onClick={saveCurrentView}>Save View</Button>
            </Tooltip>
            <Tooltip title="Reset all filters and search">
              <Button variant="outlined" onClick={clearFilters}>Clear</Button>
            </Tooltip>
            <Tooltip title="Reload board data from server">
              <Button variant="outlined" onClick={fetchBoard}>Refresh</Button>
            </Tooltip>
            <Tooltip title="Create a new task (shortcut: N)">
              <Button variant="contained" onClick={() => setShowCreate(true)}>New Task</Button>
            </Tooltip>
          </Stack>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Tip: press <strong>/</strong> to focus search, <strong>N</strong> to create task, and drag cards between columns.
        </Typography>
      </Card>

      <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
        <Button size="small" variant="outlined" onClick={() => applyView(BUILTIN_VIEWS[1])}>Blocked Only</Button>
        <Button size="small" variant="outlined" onClick={() => applyView(BUILTIN_VIEWS[2])}>Critical</Button>
        <Button size="small" variant="outlined" onClick={() => applyView(BUILTIN_VIEWS[3])}>Needs Review</Button>
      </Stack>

      <Box sx={{ display: 'flex', gap: 1, flex: 1, overflowX: 'auto', pb: 0.5 }}>
        {COLUMN_ORDER.map((col) => {
          const tasks = filteredBoard?.columns[col] || []
          const blockedTasks = col === 'ready' ? (filteredBoard?.columns.blocked || []) : []
          const allTasks = [...tasks, ...blockedTasks]

          return (
            <Card
              key={col}
              variant="outlined"
              sx={{
                minWidth: 250,
                maxWidth: 340,
                width: 320,
                display: 'flex',
                flexDirection: 'column',
                bgcolor: 'background.default',
                outline: draggedTaskId ? '2px dashed' : 'none',
                outlineColor: draggedTaskId ? 'info.main' : 'transparent',
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => handleDrop(col)}
            >
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ px: 1.25, py: 1, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="caption" sx={{ textTransform: 'uppercase', fontWeight: 700, letterSpacing: 0.5, color: 'text.secondary' }}>
                  {COLUMN_LABELS[col]}
                </Typography>
                <Chip size="small" label={allTasks.length} />
              </Stack>

              <Stack spacing={0.75} sx={{ p: 1, overflowY: 'auto', flex: 1 }}>
                {allTasks.map((task) => (
                  <TaskCard
                    key={task.id}
                    task={task}
                    projectDir={projectDir}
                    onClick={() => setSelectedTask(task)}
                    onDragStart={() => handleDragStart(task.id)}
                  />
                ))}
                {allTasks.length === 0 && (
                  <Typography variant="body2" color="text.disabled" sx={{ textAlign: 'center', py: 3 }}>
                    No tasks
                  </Typography>
                )}
              </Stack>
            </Card>
          )
        })}
      </Box>

      {selectedTask && (
        <TaskDetail
          task={selectedTask}
          projectDir={projectDir}
          onClose={() => setSelectedTask(null)}
          onUpdated={handleTaskUpdated}
          onNavigateTask={(nextTask) => setSelectedTask(nextTask)}
        />
      )}

      {showCreate && (
        <CreateTaskModal
          projectDir={projectDir}
          onCreated={handleTaskCreated}
          onClose={() => setShowCreate(false)}
        />
      )}
    </Stack>
  )
}
