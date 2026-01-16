import { useState } from 'react'

interface Props {
  currentTaskId?: string
  currentPhaseId?: string
  status?: string
}

type ControlAction = 'retry' | 'skip' | 'resume' | 'stop'

export default function ControlPanel({ currentTaskId, currentPhaseId, status }: Props) {
  const [loading, setLoading] = useState<ControlAction | null>(null)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [selectedStep, setSelectedStep] = useState('plan_impl')

  const steps = [
    { value: 'plan_impl', label: 'Plan Implementation' },
    { value: 'implement', label: 'Implement' },
    { value: 'verify', label: 'Verify' },
    { value: 'review', label: 'Review' },
    { value: 'commit', label: 'Commit' },
  ]

  const executeAction = async (action: ControlAction) => {
    if (!currentTaskId && action !== 'stop') {
      setMessage({ type: 'error', text: 'No active task to control' })
      return
    }

    setLoading(action)
    setMessage(null)

    try {
      const body: any = {
        action,
        task_id: currentTaskId || null,
        params: action === 'retry' ? { step: selectedStep } : null,
      }

      const response = await fetch('/api/control', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      })

      const data = await response.json()

      if (data.success) {
        setMessage({ type: 'success', text: data.message || `${action} completed successfully` })
      } else {
        setMessage({ type: 'error', text: data.message || `${action} failed` })
      }
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to execute action',
      })
    } finally {
      setLoading(null)
    }
  }

  return (
    <div className="card">
      <h2>Run Control</h2>

      {/* Status Display */}
      {currentTaskId && (
        <div style={{ marginBottom: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px' }}>
          <div style={{ fontSize: '0.875rem', color: '#666' }}>
            <strong>Current Task:</strong> {currentTaskId}
          </div>
          {currentPhaseId && (
            <div style={{ fontSize: '0.875rem', color: '#666', marginTop: '0.25rem' }}>
              <strong>Phase:</strong> {currentPhaseId}
            </div>
          )}
          {status && (
            <div style={{ fontSize: '0.875rem', color: '#666', marginTop: '0.25rem' }}>
              <strong>Status:</strong> {status}
            </div>
          )}
        </div>
      )}

      {/* Message Display */}
      {message && (
        <div
          style={{
            padding: '0.75rem',
            marginBottom: '1rem',
            borderRadius: '4px',
            backgroundColor: message.type === 'success' ? '#e8f5e9' : '#ffebee',
            color: message.type === 'success' ? '#2e7d32' : '#c62828',
            border: `1px solid ${message.type === 'success' ? '#4caf50' : '#f44336'}`,
          }}
        >
          {message.text}
        </div>
      )}

      {/* Step Selector for Retry */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem', color: '#333' }}>
          Retry from step:
        </label>
        <select
          value={selectedStep}
          onChange={(e) => setSelectedStep(e.target.value)}
          style={{
            width: '100%',
            padding: '0.5rem',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '0.875rem',
            backgroundColor: '#fff',
            cursor: 'pointer',
          }}
        >
          {steps.map((step) => (
            <option key={step.value} value={step.value}>
              {step.label}
            </option>
          ))}
        </select>
      </div>

      {/* Control Buttons */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
        <button
          onClick={() => executeAction('retry')}
          disabled={loading !== null || !currentTaskId}
          style={{
            padding: '0.75rem',
            border: 'none',
            borderRadius: '4px',
            fontSize: '0.875rem',
            fontWeight: 600,
            cursor: loading !== null || !currentTaskId ? 'not-allowed' : 'pointer',
            backgroundColor: loading === 'retry' ? '#1976d2' : '#2196f3',
            color: '#fff',
            opacity: loading !== null || !currentTaskId ? 0.6 : 1,
          }}
        >
          {loading === 'retry' ? 'Retrying...' : 'Retry Task'}
        </button>

        <button
          onClick={() => executeAction('skip')}
          disabled={loading !== null || !currentTaskId}
          style={{
            padding: '0.75rem',
            border: 'none',
            borderRadius: '4px',
            fontSize: '0.875rem',
            fontWeight: 600,
            cursor: loading !== null || !currentTaskId ? 'not-allowed' : 'pointer',
            backgroundColor: loading === 'skip' ? '#f57c00' : '#ff9800',
            color: '#fff',
            opacity: loading !== null || !currentTaskId ? 0.6 : 1,
          }}
        >
          {loading === 'skip' ? 'Skipping...' : 'Skip Step'}
        </button>

        <button
          onClick={() => executeAction('resume')}
          disabled={loading !== null || !currentTaskId}
          style={{
            padding: '0.75rem',
            border: 'none',
            borderRadius: '4px',
            fontSize: '0.875rem',
            fontWeight: 600,
            cursor: loading !== null || !currentTaskId ? 'not-allowed' : 'pointer',
            backgroundColor: loading === 'resume' ? '#388e3c' : '#4caf50',
            color: '#fff',
            opacity: loading !== null || !currentTaskId ? 0.6 : 1,
          }}
        >
          {loading === 'resume' ? 'Resuming...' : 'Resume Task'}
        </button>

        <button
          onClick={() => executeAction('stop')}
          disabled={loading !== null}
          style={{
            padding: '0.75rem',
            border: 'none',
            borderRadius: '4px',
            fontSize: '0.875rem',
            fontWeight: 600,
            cursor: loading !== null ? 'not-allowed' : 'pointer',
            backgroundColor: loading === 'stop' ? '#c62828' : '#f44336',
            color: '#fff',
            opacity: loading !== null ? 0.6 : 1,
          }}
        >
          {loading === 'stop' ? 'Stopping...' : 'Stop Run'}
        </button>
      </div>

      {/* Help Text */}
      <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px', fontSize: '0.75rem', color: '#666' }}>
        <strong>Controls:</strong>
        <ul style={{ margin: '0.5rem 0 0 1.25rem', padding: 0 }}>
          <li><strong>Retry:</strong> Restart task from selected step</li>
          <li><strong>Skip:</strong> Skip current step and move to next</li>
          <li><strong>Resume:</strong> Resume a blocked task</li>
          <li><strong>Stop:</strong> Send stop signal to running process</li>
        </ul>
      </div>
    </div>
  )
}
