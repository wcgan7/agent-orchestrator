/**
 * Structured feedback panel — allows humans to give actionable guidance to agents.
 */

import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import './FeedbackPanel.css'

interface FeedbackItem {
  id: string
  task_id: string
  feedback_type: string
  priority: string
  status: string
  summary: string
  details: string
  target_file: string | null
  action: string
  created_by: string
  created_at: string
  agent_response: string | null
}

interface Props {
  taskId: string
  projectDir?: string
}

const FEEDBACK_TYPES = [
  { value: 'general', label: 'General Guidance' },
  { value: 'approach_change', label: 'Change Approach' },
  { value: 'library_swap', label: 'Swap Library' },
  { value: 'file_restriction', label: 'File Restriction' },
  { value: 'style_preference', label: 'Style Preference' },
  { value: 'bug_report', label: 'Bug Report' },
]

const PRIORITIES = [
  { value: 'must', label: 'Must Follow' },
  { value: 'should', label: 'Should Follow' },
  { value: 'suggestion', label: 'Suggestion' },
]

export default function FeedbackPanel({ taskId, projectDir }: Props) {
  const [feedback, setFeedback] = useState<FeedbackItem[]>([])
  const [showForm, setShowForm] = useState(false)
  const [formType, setFormType] = useState('general')
  const [formPriority, setFormPriority] = useState('should')
  const [formSummary, setFormSummary] = useState('')
  const [formDetails, setFormDetails] = useState('')
  const [formFile, setFormFile] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const fetchFeedback = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl(`/api/v2/collaboration/feedback/${taskId}`, projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setFeedback(data.feedback || [])
      }
    } catch {
      // retry
    }
  }, [taskId, projectDir])

  useEffect(() => {
    fetchFeedback()
  }, [fetchFeedback])

  const handleSubmit = async () => {
    if (!formSummary.trim()) return
    setSubmitting(true)
    try {
      await fetch(
        buildApiUrl('/api/v2/collaboration/feedback', projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            task_id: taskId,
            feedback_type: formType,
            priority: formPriority,
            summary: formSummary,
            details: formDetails,
            target_file: formFile || null,
          }),
        }
      )
      setShowForm(false)
      setFormSummary('')
      setFormDetails('')
      setFormFile('')
      fetchFeedback()
    } finally {
      setSubmitting(false)
    }
  }

  const handleDismiss = async (feedbackId: string) => {
    await fetch(
      buildApiUrl(`/api/v2/collaboration/feedback/${feedbackId}/dismiss`, projectDir),
      { method: 'POST', headers: buildAuthHeaders() }
    )
    fetchFeedback()
  }

  const activeFeedback = feedback.filter(f => f.status === 'active')
  const addressedFeedback = feedback.filter(f => f.status === 'addressed')

  return (
    <div className="feedback-panel">
      <div className="feedback-header">
        <h3 className="feedback-title">Feedback</h3>
        <span className="feedback-count">{activeFeedback.length} active</span>
        <button className="feedback-add-btn" onClick={() => setShowForm(!showForm)}>
          + Add Feedback
        </button>
      </div>

      {showForm && (
        <div className="feedback-form">
          <div className="feedback-form-row">
            <select value={formType} onChange={(e) => setFormType(e.target.value)}>
              {FEEDBACK_TYPES.map(t => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
            <select value={formPriority} onChange={(e) => setFormPriority(e.target.value)}>
              {PRIORITIES.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>
          <input
            className="feedback-form-input"
            placeholder="Summary (one line)"
            value={formSummary}
            onChange={(e) => setFormSummary(e.target.value)}
          />
          <textarea
            className="feedback-form-textarea"
            placeholder="Details (optional)"
            value={formDetails}
            onChange={(e) => setFormDetails(e.target.value)}
            rows={3}
          />
          <input
            className="feedback-form-input"
            placeholder="Target file (optional)"
            value={formFile}
            onChange={(e) => setFormFile(e.target.value)}
          />
          <div className="feedback-form-actions">
            <button className="feedback-submit" onClick={handleSubmit} disabled={submitting}>
              {submitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
            <button className="feedback-cancel" onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </div>
      )}

      <div className="feedback-list">
        {activeFeedback.map(fb => (
          <div key={fb.id} className={`feedback-item priority-${fb.priority}`}>
            <div className="feedback-item-header">
              <span className={`feedback-type-badge type-${fb.feedback_type}`}>
                {fb.feedback_type.replace('_', ' ')}
              </span>
              <span className={`feedback-priority priority-${fb.priority}`}>
                {fb.priority}
              </span>
              <button className="feedback-dismiss" onClick={() => handleDismiss(fb.id)}>
                Dismiss
              </button>
            </div>
            <div className="feedback-item-summary">{fb.summary}</div>
            {fb.details && <div className="feedback-item-details">{fb.details}</div>}
            {fb.target_file && (
              <div className="feedback-item-file">
                <code>{fb.target_file}</code>
              </div>
            )}
          </div>
        ))}
        {activeFeedback.length === 0 && !showForm && (
          <div className="feedback-empty">No active feedback for this task.</div>
        )}
      </div>

      {addressedFeedback.length > 0 && (
        <details className="feedback-addressed-section">
          <summary className="feedback-addressed-toggle">
            {addressedFeedback.length} addressed
          </summary>
          <div className="feedback-list">
            {addressedFeedback.map(fb => (
              <div key={fb.id} className="feedback-item addressed">
                <div className="feedback-item-summary">{fb.summary}</div>
                {fb.agent_response && (
                  <div className="feedback-agent-response">
                    Agent: {fb.agent_response}
                  </div>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
