/**
 * Modal for creating a new task.
 */

import { useState } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import './KanbanBoard.css'

interface Props {
  projectDir?: string
  onCreated: () => void
  onClose: () => void
}

export function CreateTaskModal({ projectDir, onCreated, onClose }: Props) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [taskType, setTaskType] = useState('feature')
  const [priority, setPriority] = useState('P2')
  const [labels, setLabels] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return

    setSaving(true)
    setError(null)

    try {
      const resp = await fetch(
        buildApiUrl('/api/v2/tasks', projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            title: title.trim(),
            description: description.trim(),
            task_type: taskType,
            priority,
            labels: labels.split(',').map((l) => l.trim()).filter(Boolean),
          }),
        }
      )
      if (resp.ok) {
        onCreated()
      } else {
        const data = await resp.json()
        setError(data.detail || 'Failed to create task')
      }
    } catch {
      setError('Network error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Create Task</h2>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <form onSubmit={handleSubmit} className="modal-body">
          <div className="form-group">
            <label>Title *</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Task title"
              autoFocus
              required
            />
          </div>
          <div className="form-group">
            <label>Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Task description..."
              rows={4}
            />
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Type</label>
              <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                <option value="feature">Feature</option>
                <option value="bug">Bug</option>
                <option value="refactor">Refactor</option>
                <option value="research">Research</option>
                <option value="test">Test</option>
                <option value="docs">Docs</option>
                <option value="security">Security</option>
                <option value="performance">Performance</option>
              </select>
            </div>
            <div className="form-group">
              <label>Priority</label>
              <select value={priority} onChange={(e) => setPriority(e.target.value)}>
                <option value="P0">P0 - Critical</option>
                <option value="P1">P1 - High</option>
                <option value="P2">P2 - Medium</option>
                <option value="P3">P3 - Low</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <label>Labels (comma-separated)</label>
            <input
              type="text"
              value={labels}
              onChange={(e) => setLabels(e.target.value)}
              placeholder="auth, urgent, frontend"
            />
          </div>
          {error && <div className="form-error">{error}</div>}
          <div className="modal-footer">
            <button type="button" className="btn-cancel" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn-create" disabled={saving || !title.trim()}>
              {saving ? 'Creating...' : 'Create Task'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
