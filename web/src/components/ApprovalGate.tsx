import { useState, useEffect } from 'react'
import './ApprovalGate.css'

interface ApprovalGateInfo {
  request_id: string
  gate_type: string
  message: string
  task_id?: string
  phase_id?: string
  created_at: string
  timeout?: number
  context: Record<string, any>
  show_diff: boolean
  show_plan: boolean
  show_tests: boolean
  show_review: boolean
}

interface ApprovalGateProps {
  projectDir?: string
}

const ApprovalGate = ({ projectDir }: ApprovalGateProps) => {
  const [approvals, setApprovals] = useState<ApprovalGateInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [feedback, setFeedback] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState<string | null>(null)
  const [resultMessage, setResultMessage] = useState<string | null>(null)

  useEffect(() => {
    fetchApprovals()
    const interval = setInterval(fetchApprovals, 5000)
    return () => clearInterval(interval)
  }, [projectDir])

  const fetchApprovals = async () => {
    try {
      const url = projectDir
        ? `/api/approvals?project_dir=${encodeURIComponent(projectDir)}`
        : '/api/approvals'

      const headers: HeadersInit = {}
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(url, { headers })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setApprovals(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch approvals')
    } finally {
      setLoading(false)
    }
  }

  const handleRespond = async (requestId: string, approved: boolean) => {
    setSubmitting(requestId)
    setResultMessage(null)

    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch('/api/approvals/respond', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          request_id: requestId,
          approved,
          feedback: feedback[requestId] || null,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }

      const data = await response.json()
      setResultMessage(data.message)

      // Clear feedback for this request
      setFeedback((prev) => {
        const updated = { ...prev }
        delete updated[requestId]
        return updated
      })

      // Refresh approvals
      await fetchApprovals()
    } catch (err) {
      setResultMessage(
        `Error: ${err instanceof Error ? err.message : 'Failed to respond'}`
      )
    } finally {
      setSubmitting(null)
    }
  }

  const handleFeedbackChange = (requestId: string, value: string) => {
    setFeedback((prev) => ({ ...prev, [requestId]: value }))
  }

  const formatTimestamp = (timestamp: string): string => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleString()
    } catch {
      return timestamp
    }
  }

  const renderContext = (approval: ApprovalGateInfo) => {
    const contextSections: JSX.Element[] = []

    if (approval.show_diff && approval.context.diff) {
      contextSections.push(
        <div key="diff" className="context-section">
          <h4>Diff:</h4>
          <pre className="context-content">{approval.context.diff}</pre>
        </div>
      )
    }

    if (approval.show_plan && approval.context.plan) {
      contextSections.push(
        <div key="plan" className="context-section">
          <h4>Plan:</h4>
          <pre className="context-content">{approval.context.plan}</pre>
        </div>
      )
    }

    if (approval.show_tests && approval.context.tests) {
      contextSections.push(
        <div key="tests" className="context-section">
          <h4>Tests:</h4>
          <pre className="context-content">{approval.context.tests}</pre>
        </div>
      )
    }

    if (approval.show_review && approval.context.review) {
      contextSections.push(
        <div key="review" className="context-section">
          <h4>Review:</h4>
          <pre className="context-content">{approval.context.review}</pre>
        </div>
      )
    }

    // Show any other context data
    const otherContext = Object.entries(approval.context).filter(
      ([key]) => !['diff', 'plan', 'tests', 'review'].includes(key)
    )
    if (otherContext.length > 0) {
      contextSections.push(
        <div key="other" className="context-section">
          <h4>Additional Context:</h4>
          <pre className="context-content">
            {JSON.stringify(Object.fromEntries(otherContext), null, 2)}
          </pre>
        </div>
      )
    }

    return contextSections.length > 0 ? contextSections : null
  }

  if (loading) {
    return (
      <div className="approval-gate">
        <h2>Pending Approvals</h2>
        <div className="loading">Loading approvals...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="approval-gate">
        <h2>Pending Approvals</h2>
        <div className="error">Error: {error}</div>
      </div>
    )
  }

  if (approvals.length === 0) {
    return (
      <div className="approval-gate">
        <h2>Pending Approvals</h2>
        <div className="empty-state">No pending approvals</div>
      </div>
    )
  }

  return (
    <div className="approval-gate">
      <h2>
        Pending Approvals
        <span className="approval-count">{approvals.length}</span>
      </h2>

      {resultMessage && (
        <div className="result-message">
          {resultMessage}
          <button
            onClick={() => setResultMessage(null)}
            className="close-btn"
            aria-label="Close"
          >
            ×
          </button>
        </div>
      )}

      <div className="approvals-list">
        {approvals.map((approval) => (
          <div key={approval.request_id} className="approval-card">
            <div className="approval-header">
              <div className="approval-meta">
                <span className="gate-type">{approval.gate_type}</span>
                {approval.task_id && (
                  <span className="task-id">Task: {approval.task_id}</span>
                )}
                {approval.phase_id && (
                  <span className="phase-id">Phase: {approval.phase_id}</span>
                )}
              </div>
              <div className="approval-timestamp">
                {formatTimestamp(approval.created_at)}
              </div>
            </div>

            <div className="approval-message">{approval.message}</div>

            {approval.timeout && (
              <div className="approval-timeout">
                Timeout: {approval.timeout} seconds
              </div>
            )}

            {renderContext(approval)}

            <div className="approval-actions">
              <textarea
                className="feedback-input"
                placeholder="Optional feedback..."
                value={feedback[approval.request_id] || ''}
                onChange={(e) =>
                  handleFeedbackChange(approval.request_id, e.target.value)
                }
                disabled={submitting === approval.request_id}
                rows={2}
              />
              <div className="action-buttons">
                <button
                  className="approve-btn"
                  onClick={() => handleRespond(approval.request_id, true)}
                  disabled={submitting === approval.request_id}
                >
                  {submitting === approval.request_id ? 'Processing...' : 'Approve'}
                </button>
                <button
                  className="reject-btn"
                  onClick={() => handleRespond(approval.request_id, false)}
                  disabled={submitting === approval.request_id}
                >
                  {submitting === approval.request_id ? 'Processing...' : 'Reject'}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ApprovalGate
