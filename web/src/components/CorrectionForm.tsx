import { useState } from 'react'
import { sendCorrection } from '../api'
import './CorrectionForm.css'

interface Props {
  taskId: string
  projectDir?: string
  onSent?: () => void
}

export default function CorrectionForm({ taskId, projectDir, onSent }: Props) {
  const [issue, setIssue] = useState('')
  const [filePath, setFilePath] = useState('')
  const [suggestedFix, setSuggestedFix] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!issue.trim()) return

    setSending(true)
    setError(null)
    try {
      await sendCorrection(taskId, {
        issue: issue.trim(),
        file_path: filePath.trim() || undefined,
        suggested_fix: suggestedFix.trim() || undefined,
      }, projectDir)
      setIssue('')
      setFilePath('')
      setSuggestedFix('')
      onSent?.()
    } catch (err: any) {
      setError(err.message || 'Failed to send correction')
    } finally {
      setSending(false)
    }
  }

  return (
    <form className="correction-form" onSubmit={handleSubmit}>
      <h4>Send Correction</h4>
      <textarea
        value={issue}
        onChange={(e) => setIssue(e.target.value)}
        placeholder="Describe the issue..."
        disabled={sending}
      />
      <div className="correction-form-row">
        <input
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          placeholder="File path (optional)"
          disabled={sending}
        />
        <input
          value={suggestedFix}
          onChange={(e) => setSuggestedFix(e.target.value)}
          placeholder="Suggested fix (optional)"
          disabled={sending}
        />
      </div>
      {error && <div style={{ color: '#dc2626', fontSize: '0.8rem' }}>{error}</div>}
      <div className="correction-form-actions">
        <button type="submit" className="btn-send" disabled={!issue.trim() || sending}>
          {sending ? 'Sending...' : 'Send Correction'}
        </button>
      </div>
    </form>
  )
}
