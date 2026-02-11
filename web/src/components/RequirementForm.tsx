import { useState } from 'react'
import { sendRequirement } from '../api'
import './RequirementForm.css'

interface Props {
  projectDir?: string
  onSent?: () => void
}

export default function RequirementForm({ projectDir, onSent }: Props) {
  const [requirement, setRequirement] = useState('')
  const [taskId, setTaskId] = useState('')
  const [priority, setPriority] = useState<'high' | 'medium' | 'low'>('medium')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!requirement.trim()) return

    setSending(true)
    setError(null)
    try {
      await sendRequirement({
        requirement: requirement.trim(),
        task_id: taskId.trim() || undefined,
        priority,
      }, projectDir)
      setRequirement('')
      setTaskId('')
      onSent?.()
    } catch (err: any) {
      setError(err.message || 'Failed to send requirement')
    } finally {
      setSending(false)
    }
  }

  return (
    <form className="requirement-form" onSubmit={handleSubmit}>
      <h4>Add Requirement</h4>
      <textarea
        value={requirement}
        onChange={(e) => setRequirement(e.target.value)}
        placeholder="Describe the requirement..."
        disabled={sending}
      />
      <div className="requirement-form-row">
        <input
          value={taskId}
          onChange={(e) => setTaskId(e.target.value)}
          placeholder="Task ID (optional)"
          disabled={sending}
        />
        <select
          value={priority}
          onChange={(e) => setPriority(e.target.value as any)}
          disabled={sending}
        >
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
      </div>
      {error && <div style={{ color: '#dc2626', fontSize: '0.8rem' }}>{error}</div>}
      <div className="requirement-form-actions">
        <button type="submit" className="btn-send" disabled={!requirement.trim() || sending}>
          {sending ? 'Sending...' : 'Add Requirement'}
        </button>
      </div>
    </form>
  )
}
