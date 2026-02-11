/**
 * Task detail slide-over panel with collaboration features.
 */

import { useState } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import FeedbackPanel from '../FeedbackPanel/FeedbackPanel'
import ActivityTimeline from '../ActivityTimeline/ActivityTimeline'
import ReasoningViewer from '../ReasoningViewer/ReasoningViewer'
import './KanbanBoard.css'

type DetailTab = 'details' | 'feedback' | 'activity' | 'reasoning'

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
}

export function TaskDetail({ task, projectDir, onClose, onUpdated }: Props) {
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(task.title)
  const [description, setDescription] = useState(task.description)
  const [priority, setPriority] = useState(task.priority)
  const [taskType, setTaskType] = useState(task.task_type)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<DetailTab>('details')

  const handleSave = async () => {
    setSaving(true)
    try {
      await fetch(
        buildApiUrl(`/api/v2/tasks/${task.id}`, projectDir),
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
      await fetch(
        buildApiUrl(`/api/v2/tasks/${task.id}/transition`, projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ status: newStatus }),
        }
      )
      onUpdated()
    } catch {
      // transition failed
    }
  }

  const handleDelete = async () => {
    if (!confirm('Delete this task?')) return
    await fetch(
      buildApiUrl(`/api/v2/tasks/${task.id}`, projectDir),
      { method: 'DELETE', headers: buildAuthHeaders() }
    )
    onUpdated()
  }

  return (
    <div className="task-detail-overlay" onClick={onClose}>
      <div className="task-detail-panel" onClick={(e) => e.stopPropagation()}>
        <div className="task-detail-header">
          <div className="task-detail-header-left">
            <span className={`task-detail-type type-${task.task_type}`}>{task.task_type}</span>
            <span className="task-detail-id">{task.id}</span>
            <span className={`task-detail-priority priority-${task.priority}`}>{task.priority}</span>
          </div>
          <button className="task-detail-close" onClick={onClose}>&times;</button>
        </div>

        {/* Collaboration tabs */}
        <div className="task-detail-tabs">
          <button
            className={`detail-tab ${activeTab === 'details' ? 'active' : ''}`}
            onClick={() => setActiveTab('details')}
          >
            Details
          </button>
          <button
            className={`detail-tab ${activeTab === 'feedback' ? 'active' : ''}`}
            onClick={() => setActiveTab('feedback')}
          >
            Feedback
          </button>
          <button
            className={`detail-tab ${activeTab === 'activity' ? 'active' : ''}`}
            onClick={() => setActiveTab('activity')}
          >
            Activity
          </button>
          <button
            className={`detail-tab ${activeTab === 'reasoning' ? 'active' : ''}`}
            onClick={() => setActiveTab('reasoning')}
          >
            Reasoning
          </button>
        </div>

        <div className="task-detail-body">
          {activeTab === 'feedback' ? (
            <FeedbackPanel taskId={task.id} projectDir={projectDir} />
          ) : activeTab === 'activity' ? (
            <ActivityTimeline taskId={task.id} projectDir={projectDir} />
          ) : activeTab === 'reasoning' ? (
            <ReasoningViewer taskId={task.id} projectDir={projectDir} />
          ) : editing ? (
            <div className="task-detail-edit">
              <input
                className="task-detail-title-input"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Task title"
              />
              <div className="task-detail-edit-row">
                <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                  <option value="feature">Feature</option>
                  <option value="bug">Bug</option>
                  <option value="refactor">Refactor</option>
                  <option value="research">Research</option>
                  <option value="test">Test</option>
                  <option value="docs">Docs</option>
                </select>
                <select value={priority} onChange={(e) => setPriority(e.target.value)}>
                  <option value="P0">P0 - Critical</option>
                  <option value="P1">P1 - High</option>
                  <option value="P2">P2 - Medium</option>
                  <option value="P3">P3 - Low</option>
                </select>
              </div>
              <textarea
                className="task-detail-desc-input"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Description..."
                rows={6}
              />
              <div className="task-detail-edit-actions">
                <button className="btn-save" onClick={handleSave} disabled={saving}>
                  {saving ? 'Saving...' : 'Save'}
                </button>
                <button className="btn-cancel" onClick={() => setEditing(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <>
              <h2 className="task-detail-title">{task.title}</h2>
              {task.description && (
                <div className="task-detail-description">{task.description}</div>
              )}
              <button className="btn-edit" onClick={() => setEditing(true)}>Edit</button>
            </>
          )}

          {activeTab === 'details' && (
            <>
              {/* Status & Actions */}
              <div className="task-detail-section">
                <h3>Status</h3>
                <div className="task-detail-status">
                  <span className={`status-badge status-${task.status}`}>{task.status}</span>
                  <div className="task-detail-transitions">
                    {task.status === 'backlog' && (
                      <button onClick={() => handleTransition('ready')}>Move to Ready</button>
                    )}
                    {task.status === 'ready' && (
                      <button onClick={() => handleTransition('in_progress')}>Start</button>
                    )}
                    {task.status === 'in_progress' && (
                      <>
                        <button onClick={() => handleTransition('in_review')}>Send to Review</button>
                        <button onClick={() => handleTransition('done')}>Mark Done</button>
                      </>
                    )}
                    {task.status === 'in_review' && (
                      <>
                        <button onClick={() => handleTransition('done')}>Approve</button>
                        <button onClick={() => handleTransition('in_progress')}>Request Changes</button>
                      </>
                    )}
                    {task.status === 'blocked' && (
                      <button onClick={() => handleTransition('ready')}>Unblock</button>
                    )}
                  </div>
                </div>
              </div>

              {/* Acceptance Criteria */}
              {task.acceptance_criteria.length > 0 && (
                <div className="task-detail-section">
                  <h3>Acceptance Criteria</h3>
                  <ul className="task-detail-criteria">
                    {task.acceptance_criteria.map((ac, i) => (
                      <li key={i}>{ac}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Labels */}
              {task.labels.length > 0 && (
                <div className="task-detail-section">
                  <h3>Labels</h3>
                  <div className="task-detail-labels">
                    {task.labels.map((l) => (
                      <span key={l} className="task-card-label">{l}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Dependencies */}
              {(task.blocked_by.length > 0 || task.blocks.length > 0) && (
                <div className="task-detail-section">
                  <h3>Dependencies</h3>
                  {task.blocked_by.length > 0 && (
                    <div className="task-detail-deps">
                      <span className="dep-label">Blocked by:</span>
                      {task.blocked_by.map((id) => (
                        <span key={id} className="dep-chip">{id.slice(-8)}</span>
                      ))}
                    </div>
                  )}
                  {task.blocks.length > 0 && (
                    <div className="task-detail-deps">
                      <span className="dep-label">Blocks:</span>
                      {task.blocks.map((id) => (
                        <span key={id} className="dep-chip">{id.slice(-8)}</span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Context Files */}
              {task.context_files.length > 0 && (
                <div className="task-detail-section">
                  <h3>Context Files</h3>
                  <ul className="task-detail-files">
                    {task.context_files.map((f) => (
                      <li key={f}><code>{f}</code></li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Error */}
              {task.error && (
                <div className="task-detail-section task-detail-error">
                  <h3>Error</h3>
                  <pre>{task.error}</pre>
                </div>
              )}

              {/* Metadata */}
              <div className="task-detail-section task-detail-meta">
                <div>Created: {new Date(task.created_at).toLocaleString()}</div>
                <div>Updated: {new Date(task.updated_at).toLocaleString()}</div>
                {task.completed_at && <div>Completed: {new Date(task.completed_at).toLocaleString()}</div>}
                {task.assignee && <div>Assignee: {task.assignee} ({task.assignee_type})</div>}
                <div>Source: {task.source}</div>
              </div>

              {/* Danger zone */}
              <div className="task-detail-section task-detail-danger">
                <button className="btn-delete" onClick={handleDelete}>Delete Task</button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
