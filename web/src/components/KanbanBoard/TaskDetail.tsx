/**
 * Task detail slide-over panel with collaboration features.
 */

import { useState, useEffect } from 'react'
import { buildApiUrl, buildAuthHeaders, fetchInspect, fetchExplain, fetchTaskLogs, fetchTrace } from '../../api'
import FeedbackPanel from '../FeedbackPanel/FeedbackPanel'
import ActivityTimeline from '../ActivityTimeline/ActivityTimeline'
import ReasoningViewer from '../ReasoningViewer/ReasoningViewer'
import CorrectionForm from '../CorrectionForm'
import './KanbanBoard.css'

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
}

export function TaskDetail({ task, projectDir, onClose, onUpdated }: Props) {
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(task.title)
  const [description, setDescription] = useState(task.description)
  const [priority, setPriority] = useState(task.priority)
  const [taskType, setTaskType] = useState(task.task_type)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<DetailTab>('summary')

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

        {/* Task detail tabs */}
        <div className="task-detail-tabs">
          <button
            className={`detail-tab ${activeTab === 'summary' ? 'active' : ''}`}
            onClick={() => setActiveTab('summary')}
          >
            Summary
          </button>
          <button
            className={`detail-tab ${activeTab === 'dependencies' ? 'active' : ''}`}
            onClick={() => setActiveTab('dependencies')}
          >
            Dependencies
          </button>
          <button
            className={`detail-tab ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            Logs
          </button>
          <button
            className={`detail-tab ${activeTab === 'interventions' ? 'active' : ''}`}
            onClick={() => setActiveTab('interventions')}
          >
            Interventions
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
          <button
            className={`detail-tab ${activeTab === 'inspect' ? 'active' : ''}`}
            onClick={() => setActiveTab('inspect')}
          >
            Inspect
          </button>
          <button
            className={`detail-tab ${activeTab === 'trace' ? 'active' : ''}`}
            onClick={() => setActiveTab('trace')}
          >
            Trace
          </button>
        </div>

        <div className="task-detail-body">
          {activeTab === 'interventions' ? (
            <div className="task-detail-section">
              <FeedbackPanel taskId={task.id} projectDir={projectDir} />
              {(task.error || task.status === 'blocked' || task.blocked_by.length > 0) && (
                <div style={{ marginTop: '1rem' }}>
                  <CorrectionForm taskId={task.id} projectDir={projectDir} />
                </div>
              )}
            </div>
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

          {activeTab === 'summary' && (
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

          {activeTab === 'dependencies' && (
            <>
              {(task.blocked_by.length > 0 || task.blocks.length > 0) ? (
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
              ) : (
                <div className="task-detail-section">
                  <p>No dependencies found for this task.</p>
                </div>
              )}

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

              {task.error && (
                <div className="task-detail-section task-detail-error">
                  <h3>Error</h3>
                  <pre>{task.error}</pre>
                  <ExplainButton taskId={task.id} projectDir={projectDir} />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// --- Sub-components for new tabs ---

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
    <div style={{ marginTop: '0.5rem' }}>
      {explanation ? (
        <pre style={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap', background: 'var(--bg-secondary, #f9fafb)', padding: '0.5rem', borderRadius: '4px' }}>
          {explanation}
        </pre>
      ) : (
        <button className="btn-edit" onClick={handleClick} disabled={loading}>
          {loading ? 'Loading...' : 'Why blocked?'}
        </button>
      )}
    </div>
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

  if (loading) return <div>Loading inspection data...</div>
  if (error) return <div style={{ color: '#dc2626' }}>{error}</div>
  if (!data) return <div>No data available</div>

  return (
    <div style={{ fontSize: '0.85rem' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <tbody>
          {[
            ['Lifecycle', data.lifecycle],
            ['Step', data.step],
            ['Status', data.status],
            ['Worker Attempts', data.worker_attempts],
            ['Last Error', data.last_error || '-'],
            ['Error Type', data.last_error_type || '-'],
          ].map(([key, val]) => (
            <tr key={key} style={{ borderBottom: '1px solid var(--border-color, #e5e7eb)' }}>
              <td style={{ padding: '0.5rem', fontWeight: 500, width: '40%' }}>{key}</td>
              <td style={{ padding: '0.5rem', fontFamily: 'var(--font-mono, monospace)', fontSize: '0.8rem' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.context && data.context.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <strong>Context</strong>
          <ul style={{ margin: '0.5rem 0', paddingLeft: '1.25rem' }}>
            {data.context.map((c: string, i: number) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}
      {data.metadata && Object.keys(data.metadata).length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <strong>Metadata</strong>
          <pre style={{ fontSize: '0.8rem', background: 'var(--bg-secondary, #f9fafb)', padding: '0.5rem', borderRadius: '4px', overflow: 'auto' }}>
            {JSON.stringify(data.metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
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
    <div style={{ fontSize: '0.85rem' }}>
      <div style={{ marginBottom: '0.75rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <label style={{ fontWeight: 500 }}>Step filter:</label>
        <select value={step} onChange={(e) => setStep(e.target.value)} style={{ padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid var(--border-color, #e5e7eb)' }}>
          <option value="">All</option>
          <option value="plan_impl">Plan/Impl</option>
          <option value="implement">Implement</option>
          <option value="verify">Verify</option>
          <option value="review">Review</option>
          <option value="commit">Commit</option>
        </select>
      </div>
      {loading ? (
        <div>Loading logs...</div>
      ) : error ? (
        <div style={{ color: '#dc2626' }}>{error}</div>
      ) : Object.keys(logs).length === 0 ? (
        <div style={{ color: 'var(--text-secondary, #6b7280)' }}>No logs found for this task</div>
      ) : (
        Object.entries(logs).map(([filename, lines]) => (
          <div key={filename} style={{ marginBottom: '1rem' }}>
            <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{filename}</div>
            <pre style={{
              fontFamily: 'var(--font-mono, monospace)',
              fontSize: '0.75rem',
              background: 'var(--bg-secondary, #f9fafb)',
              padding: '0.75rem',
              borderRadius: '6px',
              border: '1px solid var(--border-color, #e5e7eb)',
              overflow: 'auto',
              maxHeight: '300px',
              whiteSpace: 'pre-wrap',
            }}>
              {lines.join('\n') || '(empty)'}
            </pre>
          </div>
        ))
      )}
    </div>
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

  if (loading) return <div>Loading event history...</div>
  if (error) return <div style={{ color: '#dc2626' }}>{error}</div>
  if (events.length === 0) return <div style={{ color: 'var(--text-secondary, #6b7280)' }}>No events found for this task</div>

  return (
    <div style={{ fontSize: '0.85rem' }}>
      <div style={{ marginBottom: '0.5rem', color: 'var(--text-secondary, #6b7280)' }}>
        {events.length} event{events.length !== 1 ? 's' : ''}
      </div>
      {events.map((event, i) => {
        const eventType = event.event_type || 'unknown'
        const timestamp = event.timestamp || ''
        const isFail = eventType.includes('fail') || eventType.includes('error') || eventType.includes('violation')
        const isPass = eventType.includes('pass') || eventType === 'task_completed'
        return (
          <div key={i} style={{
            padding: '0.5rem 0.75rem',
            borderLeft: `3px solid ${isFail ? '#dc2626' : isPass ? '#16a34a' : '#3b82f6'}`,
            marginBottom: '0.5rem',
            background: 'var(--bg-secondary, #f9fafb)',
            borderRadius: '0 4px 4px 0',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
              <strong>{eventType}</strong>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #6b7280)' }}>{timestamp}</span>
            </div>
            {event.run_id && <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #6b7280)' }}>Run: {event.run_id}</div>}
            {event.error_type && <div style={{ fontSize: '0.8rem', color: '#dc2626' }}>Error: {event.error_type}</div>}
            {event.error_detail && <div style={{ fontSize: '0.75rem', color: '#991b1b', marginTop: '0.25rem' }}>{String(event.error_detail).slice(0, 150)}</div>}
            {event.block_reason && <div style={{ fontSize: '0.8rem' }}>Reason: {event.block_reason}</div>}
            {event.passed !== undefined && <div style={{ color: event.passed ? '#16a34a' : '#dc2626' }}>{event.passed ? 'Passed' : 'Failed'}</div>}
          </div>
        )
      })}
    </div>
  )
}
