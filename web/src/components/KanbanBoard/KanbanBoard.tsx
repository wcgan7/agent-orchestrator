/**
 * Kanban board for the dynamic task engine.
 * Tasks flow through columns: Backlog → Ready → In Progress → In Review → Done
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'
import { TaskCard } from './TaskCard'
import { TaskDetail } from './TaskDetail'
import { CreateTaskModal } from './CreateTaskModal'
import './KanbanBoard.css'

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

const COLUMN_ORDER = ['backlog', 'ready', 'in_progress', 'in_review', 'done']
const COLUMN_LABELS: Record<string, string> = {
  backlog: 'Backlog',
  ready: 'Ready',
  in_progress: 'In Progress',
  in_review: 'In Review',
  blocked: 'Blocked',
  done: 'Done',
}

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
  const [searchQuery, setSearchQuery] = useState('')
  const [draggedTaskId, setDraggedTaskId] = useState<string | null>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  // ---- Keyboard shortcuts ----
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Skip when user is typing in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
        // Allow Escape to blur search input
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
        buildApiUrl('/api/v2/tasks/board', projectDir),
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

  // Real-time updates via WebSocket
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
        buildApiUrl(`/api/v2/tasks/${draggedTaskId}/transition`, projectDir),
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

  // Filter logic
  const filteredBoard = useMemo(() => {
    if (!board?.columns) return null
    const filtered: Record<string, TaskData[]> = {}
    for (const [col, tasks] of Object.entries(board.columns)) {
      filtered[col] = tasks.filter((t) => {
        if (filterType && t.task_type !== filterType) return false
        if (filterPriority && t.priority !== filterPriority) return false
        if (searchQuery) {
          const q = searchQuery.toLowerCase()
          if (!t.title.toLowerCase().includes(q) && !t.id.toLowerCase().includes(q)) return false
        }
        return true
      })
    }
    return { columns: filtered }
  }, [board, filterType, filterPriority, searchQuery])

  const totalTasks = board?.columns
    ? Object.values(board.columns).reduce((sum, col) => sum + col.length, 0)
    : 0

  if (loading) {
    return <div className="kanban-loading">Loading board...</div>
  }

  return (
    <div className="kanban-container">
      {/* Toolbar */}
      <div className="kanban-toolbar">
        <div className="kanban-toolbar-left">
          <h2 className="kanban-title">Task Board</h2>
          <span className="kanban-count">{totalTasks} tasks</span>
        </div>
        <div className="kanban-toolbar-center">
          <input
            ref={searchInputRef}
            className="kanban-search"
            type="text"
            placeholder="Search tasks... ( / )"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <select
            className="kanban-filter"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
          >
            <option value="">All types</option>
            <option value="feature">Feature</option>
            <option value="bug">Bug</option>
            <option value="refactor">Refactor</option>
            <option value="research">Research</option>
            <option value="test">Test</option>
            <option value="docs">Docs</option>
          </select>
          <select
            className="kanban-filter"
            value={filterPriority}
            onChange={(e) => setFilterPriority(e.target.value)}
          >
            <option value="">All priorities</option>
            <option value="P0">P0 - Critical</option>
            <option value="P1">P1 - High</option>
            <option value="P2">P2 - Medium</option>
            <option value="P3">P3 - Low</option>
          </select>
        </div>
        <div className="kanban-toolbar-right">
          <button className="kanban-btn-refresh" onClick={fetchBoard} title="Refresh">
            &#x21bb;
          </button>
          <button className="kanban-btn-create" onClick={() => setShowCreate(true)} title="New task (N)">
            + New Task
          </button>
        </div>
      </div>

      {/* Board */}
      <div className="kanban-board">
        {COLUMN_ORDER.map((col) => {
          const tasks = filteredBoard?.columns[col] || []
          const blockedTasks = col === 'ready' ? (filteredBoard?.columns['blocked'] || []) : []
          const allTasks = [...tasks, ...blockedTasks]

          return (
            <div
              key={col}
              className={`kanban-column ${draggedTaskId ? 'kanban-column-drop-target' : ''}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => handleDrop(col)}
            >
              <div className="kanban-column-header">
                <span className="kanban-column-title">{COLUMN_LABELS[col]}</span>
                <span className="kanban-column-count">{allTasks.length}</span>
              </div>
              <div className="kanban-column-body">
                {allTasks.map((task) => (
                  <TaskCard
                    key={task.id}
                    task={task}
                    onClick={() => setSelectedTask(task)}
                    onDragStart={() => handleDragStart(task.id)}
                  />
                ))}
                {allTasks.length === 0 && (
                  <div className="kanban-empty">No tasks</div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Task detail slide-over */}
      {selectedTask && (
        <TaskDetail
          task={selectedTask}
          projectDir={projectDir}
          onClose={() => setSelectedTask(null)}
          onUpdated={handleTaskUpdated}
        />
      )}

      {/* Create modal */}
      {showCreate && (
        <CreateTaskModal
          projectDir={projectDir}
          onCreated={handleTaskCreated}
          onClose={() => setShowCreate(false)}
        />
      )}
    </div>
  )
}
