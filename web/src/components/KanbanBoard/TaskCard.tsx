/**
 * Compact task card for the Kanban board.
 */

import './KanbanBoard.css'

interface TaskData {
  id: string
  title: string
  task_type: string
  priority: string
  status: string
  labels: string[]
  assignee: string | null
  blocked_by: string[]
  effort: string | null
  error: string | null
}

const TYPE_ICONS: Record<string, string> = {
  feature: 'F',
  bug: 'B',
  refactor: 'R',
  research: '?',
  test: 'T',
  docs: 'D',
  security: 'S',
  performance: 'P',
  custom: 'C',
  review: 'V',
}

const PRIORITY_COLORS: Record<string, string> = {
  P0: '#ef4444',
  P1: '#f97316',
  P2: '#3b82f6',
  P3: '#9ca3af',
}

interface Props {
  task: TaskData
  onClick: () => void
  onDragStart: () => void
}

export function TaskCard({ task, onClick, onDragStart }: Props) {
  const isBlocked = task.blocked_by.length > 0
  const hasError = !!task.error
  const typeIcon = TYPE_ICONS[task.task_type] || '?'
  const priorityColor = PRIORITY_COLORS[task.priority] || PRIORITY_COLORS.P2

  return (
    <div
      className={`task-card ${isBlocked ? 'task-card-blocked' : ''} ${hasError ? 'task-card-error' : ''}`}
      draggable
      onClick={onClick}
      onDragStart={(e) => {
        e.dataTransfer.setData('text/plain', task.id)
        onDragStart()
      }}
    >
      <div className="task-card-priority-bar" style={{ backgroundColor: priorityColor }} />
      <div className="task-card-content">
        <div className="task-card-top">
          <span className={`task-card-type task-card-type-${task.task_type}`} title={task.task_type}>
            {typeIcon}
          </span>
          <span className="task-card-id">{task.id.slice(-8)}</span>
          {task.effort && <span className="task-card-effort">{task.effort}</span>}
        </div>
        <div className="task-card-title">{task.title}</div>
        <div className="task-card-bottom">
          <div className="task-card-labels">
            {task.labels.slice(0, 3).map((label) => (
              <span key={label} className="task-card-label">{label}</span>
            ))}
          </div>
          <div className="task-card-meta">
            {isBlocked && <span className="task-card-blocked-icon" title="Blocked">&#128274;</span>}
            {task.assignee && (
              <span className="task-card-assignee" title={task.assignee}>
                {task.assignee.slice(0, 2).toUpperCase()}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
