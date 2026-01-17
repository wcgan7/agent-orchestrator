import { useState, useEffect } from 'react'
import './FileReview.css'

interface FileChange {
  file_path: string
  status: string
  additions: number
  deletions: number
  diff: string
  approved: boolean | null
  comments: string[]
}

interface FileReviewProps {
  taskId?: string
  projectDir?: string
}

const FileReview = ({ taskId, projectDir }: FileReviewProps) => {
  const [files, setFiles] = useState<FileChange[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<number>(0)
  const [comments, setComments] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState<string | null>(null)

  useEffect(() => {
    fetchFileChanges()
  }, [taskId, projectDir])

  const fetchFileChanges = async () => {
    try {
      const params = new URLSearchParams()
      if (projectDir) params.append('project_dir', projectDir)
      if (taskId) params.append('task_id', taskId)

      const headers: HeadersInit = {}
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`/api/file-changes?${params}`, { headers })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setFiles(Array.isArray(data) ? data : [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch file changes')
    } finally {
      setLoading(false)
    }
  }

  const handleReview = async (filePath: string, approved: boolean) => {
    setSubmitting(filePath)

    try {
      const params = new URLSearchParams()
      if (projectDir) params.append('project_dir', projectDir)

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`/api/file-review?${params}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          file_path: filePath,
          approved,
          comment: comments[filePath] || null,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }

      // Update local state
      setFiles(
        files.map((f) =>
          f.file_path === filePath ? { ...f, approved } : f
        )
      )

      // Clear comment
      setComments((prev) => {
        const updated = { ...prev }
        delete updated[filePath]
        return updated
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit review')
    } finally {
      setSubmitting(null)
    }
  }

  const handleCommentChange = (filePath: string, value: string) => {
    setComments((prev) => ({ ...prev, [filePath]: value }))
  }

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'added':
        return '#4caf50'
      case 'modified':
        return '#2196f3'
      case 'deleted':
        return '#f44336'
      default:
        return '#9e9e9e'
    }
  }

  const getStatusIcon = (status: string): string => {
    switch (status) {
      case 'added':
        return '+'
      case 'modified':
        return '~'
      case 'deleted':
        return '-'
      default:
        return '?'
    }
  }

  const renderDiff = (diff: string) => {
    if (!diff) return <div className="no-diff">No diff available</div>

    const lines = diff.split('\n')
    return (
      <div className="diff-view">
        {lines.map((line, idx) => {
          let className = 'diff-line'
          if (line.startsWith('+') && !line.startsWith('+++')) {
            className += ' addition'
          } else if (line.startsWith('-') && !line.startsWith('---')) {
            className += ' deletion'
          } else if (line.startsWith('@@')) {
            className += ' hunk'
          }

          return (
            <div key={idx} className={className}>
              {line || ' '}
            </div>
          )
        })}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="file-review">
        <h2>File Review</h2>
        <div className="loading">Loading file changes...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="file-review">
        <h2>File Review</h2>
        <div className="error">Error: {error}</div>
      </div>
    )
  }

  if (files.length === 0) {
    return (
      <div className="file-review">
        <h2>File Review</h2>
        <div className="empty-state">No file changes to review</div>
      </div>
    )
  }

  const currentFile = files[selectedFile]
  const approvedCount = files.filter((f) => f.approved === true).length
  const rejectedCount = files.filter((f) => f.approved === false).length
  const pendingCount = files.filter((f) => f.approved === null).length

  return (
    <div className="file-review">
      <div className="review-header">
        <h2>File Review</h2>
        <div className="review-stats">
          <span className="stat approved">✓ {approvedCount}</span>
          <span className="stat rejected">✗ {rejectedCount}</span>
          <span className="stat pending">⏳ {pendingCount}</span>
          <span className="stat total">Total: {files.length}</span>
        </div>
      </div>

      <div className="review-container">
        <div className="file-list">
          {files.map((file, idx) => (
            <div
              key={file.file_path}
              className={`file-item ${idx === selectedFile ? 'selected' : ''} ${
                file.approved === true
                  ? 'approved'
                  : file.approved === false
                  ? 'rejected'
                  : ''
              }`}
              onClick={() => setSelectedFile(idx)}
            >
              <div className="file-status">
                <span
                  className="status-icon"
                  style={{ backgroundColor: getStatusColor(file.status) }}
                >
                  {getStatusIcon(file.status)}
                </span>
              </div>
              <div className="file-info">
                <div className="file-name">{file.file_path}</div>
                <div className="file-changes">
                  <span className="additions">+{file.additions}</span>
                  <span className="deletions">-{file.deletions}</span>
                </div>
              </div>
              {file.approved !== null && (
                <div className="review-badge">
                  {file.approved ? '✓' : '✗'}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="file-detail">
          <div className="detail-header">
            <div className="file-path-header">
              <span
                className="status-badge"
                style={{ backgroundColor: getStatusColor(currentFile.status) }}
              >
                {currentFile.status}
              </span>
              <span className="file-path">{currentFile.file_path}</span>
            </div>
            <div className="change-stats">
              <span className="additions">+{currentFile.additions}</span>
              <span className="deletions">-{currentFile.deletions}</span>
            </div>
          </div>

          <div className="diff-container">
            {renderDiff(currentFile.diff)}
          </div>

          <div className="review-actions">
            <textarea
              className="comment-input"
              placeholder="Add optional comment..."
              value={comments[currentFile.file_path] || ''}
              onChange={(e) =>
                handleCommentChange(currentFile.file_path, e.target.value)
              }
              disabled={submitting === currentFile.file_path}
              rows={2}
            />
            <div className="action-buttons">
              <button
                className="reject-btn"
                onClick={() => handleReview(currentFile.file_path, false)}
                disabled={submitting === currentFile.file_path}
              >
                {submitting === currentFile.file_path ? 'Processing...' : '✗ Reject'}
              </button>
              <button
                className="approve-btn"
                onClick={() => handleReview(currentFile.file_path, true)}
                disabled={submitting === currentFile.file_path}
              >
                {submitting === currentFile.file_path ? 'Processing...' : '✓ Approve'}
              </button>
            </div>
          </div>

          <div className="navigation-buttons">
            <button
              onClick={() => setSelectedFile(Math.max(0, selectedFile - 1))}
              disabled={selectedFile === 0}
            >
              ← Previous
            </button>
            <span className="file-counter">
              File {selectedFile + 1} of {files.length}
            </span>
            <button
              onClick={() =>
                setSelectedFile(Math.min(files.length - 1, selectedFile + 1))
              }
              disabled={selectedFile === files.length - 1}
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FileReview
