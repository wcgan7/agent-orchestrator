import { useState, useEffect } from 'react'
import { fetchExplain } from '../api'
import './ExplainModal.css'

interface Props {
  taskId: string
  projectDir?: string
  onClose: () => void
}

export default function ExplainModal({ taskId, projectDir, onClose }: Props) {
  const [explanation, setExplanation] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchExplain(taskId, projectDir)
      .then((data) => {
        setExplanation(data.explanation)
      })
      .catch((err) => {
        setError(err.message || 'Failed to fetch explanation')
      })
      .finally(() => setLoading(false))
  }, [taskId, projectDir])

  return (
    <div className="explain-modal-overlay" onClick={onClose}>
      <div className="explain-modal" onClick={(e) => e.stopPropagation()}>
        <div className="explain-modal-header">
          <h3>Why is this task blocked?</h3>
          <button className="explain-modal-close" onClick={onClose}>&times;</button>
        </div>
        {loading ? (
          <div className="explain-modal-loading">Loading explanation...</div>
        ) : error ? (
          <div className="explain-modal-error">{error}</div>
        ) : (
          <div className="explain-modal-body">{explanation}</div>
        )}
      </div>
    </div>
  )
}
