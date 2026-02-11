import { useState, useEffect } from 'react'
import {
  Box,
  Button,
  Chip,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { useToast } from '../contexts/ToastContext'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'

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
  const toast = useToast()

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
      if (token) headers.Authorization = `Bearer ${token}`

      const response = await fetch(`/api/file-changes?${params}`, { headers })
      if (!response.ok) throw new Error(`HTTP error ${response.status}`)

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

      const headers: HeadersInit = { 'Content-Type': 'application/json' }
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) headers.Authorization = `Bearer ${token}`

      const response = await fetch(`/api/file-review?${params}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          file_path: filePath,
          approved,
          comment: comments[filePath] || null,
        }),
      })

      if (!response.ok) throw new Error(`HTTP error ${response.status}`)

      setFiles(files.map((f) => (f.file_path === filePath ? { ...f, approved } : f)))
      setComments((prev) => {
        const updated = { ...prev }
        delete updated[filePath]
        return updated
      })

      toast.success(approved ? 'File approved' : 'File rejected')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to submit review')
    } finally {
      setSubmitting(null)
    }
  }

  const handleCommentChange = (filePath: string, value: string) => {
    setComments((prev) => ({ ...prev, [filePath]: value }))
  }

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'added': return '#4caf50'
      case 'modified': return '#2196f3'
      case 'deleted': return '#f44336'
      default: return '#9e9e9e'
    }
  }

  const getStatusIcon = (status: string): string => {
    switch (status) {
      case 'added': return '+'
      case 'modified': return '~'
      case 'deleted': return '-'
      default: return '?'
    }
  }

  const renderDiff = (diff: string) => {
    if (!diff) return <Box className="no-diff" sx={{ p: 6, textAlign: 'center', color: 'text.secondary' }}>No diff available</Box>

    return (
      <Box className="diff-view" sx={{ fontFamily: 'monospace', fontSize: '0.75rem', lineHeight: 1.4 }}>
        {diff.split('\n').map((line, idx) => {
          let className = 'diff-line'
          if (line.startsWith('+') && !line.startsWith('+++')) className += ' addition'
          else if (line.startsWith('-') && !line.startsWith('---')) className += ' deletion'
          else if (line.startsWith('@@')) className += ' hunk'

          return (
            <Box
              key={idx}
              className={className}
              sx={{
                px: 1.5,
                whiteSpace: 'pre',
                overflowX: 'auto',
                bgcolor: className.includes('addition')
                  ? 'success.light'
                  : className.includes('deletion')
                    ? 'error.light'
                    : className.includes('hunk')
                      ? 'info.light'
                      : 'transparent',
                color: className.includes('hunk') ? 'text.secondary' : 'text.primary',
                fontWeight: className.includes('hunk') ? 600 : 400,
              }}
            >
              {line || ' '}
            </Box>
          )
        })}
      </Box>
    )
  }

  if (loading) {
    return (
      <Box className="file-review" sx={{ p: 3 }}>
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>File Review</Typography>
        <LoadingSpinner label="Loading file changes..." />
      </Box>
    )
  }

  if (error) {
    return (
      <Box className="file-review" sx={{ p: 3 }}>
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>File Review</Typography>
        <EmptyState
          icon={<span>⚠️</span>}
          title="Error loading files"
          description={error}
          size="sm"
        />
      </Box>
    )
  }

  if (files.length === 0) {
    return (
      <Box className="file-review" sx={{ p: 3 }}>
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>File Review</Typography>
        <EmptyState
          icon={<span>📁</span>}
          title="No file changes to review"
          description="File changes will appear here when they are ready for review."
          size="sm"
        />
      </Box>
    )
  }

  const currentFile = files[selectedFile]
  const approvedCount = files.filter((f) => f.approved === true).length
  const rejectedCount = files.filter((f) => f.approved === false).length
  const pendingCount = files.filter((f) => f.approved === null).length

  return (
    <Box
      className="file-review"
      sx={{
        background: 'background.paper',
        borderRadius: 2,
        p: 3,
        boxShadow: 1,
        mb: 3,
      }}
    >
      <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ sm: 'center' }} className="review-header" spacing={1} sx={{ mb: 1.5 }}>
        <Typography variant="h2" sx={{ fontSize: '1.125rem' }}>File Review</Typography>
        <Stack direction="row" spacing={0.75} className="review-stats" useFlexGap flexWrap="wrap">
          <Chip size="small" color="success" variant="outlined" label={`✓ ${approvedCount}`} />
          <Chip size="small" color="error" variant="outlined" label={`✗ ${rejectedCount}`} />
          <Chip size="small" color="warning" variant="outlined" label={`⏳ ${pendingCount}`} />
          <Chip size="small" variant="outlined" label={`Total: ${files.length}`} />
        </Stack>
      </Stack>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: '300px 1fr' },
          gap: 3,
          height: { xs: 'auto', lg: 600 },
        }}
      >
        <Box className="file-list" sx={{ border: 1, borderColor: 'divider', borderRadius: 2, overflowY: 'auto', bgcolor: 'background.default', maxHeight: { xs: 300, lg: 'none' } }}>
          {files.map((file, idx) => (
            <Box
              key={file.file_path}
              className={`file-item ${idx === selectedFile ? 'selected' : ''} ${file.approved === true ? 'approved' : file.approved === false ? 'rejected' : ''}`}
              onClick={() => setSelectedFile(idx)}
              sx={{
                borderBottom: 1,
                borderColor: 'divider',
                p: 1.25,
                cursor: 'pointer',
                position: 'relative',
                bgcolor: idx === selectedFile ? 'action.selected' : 'transparent',
                borderLeft: `4px solid ${
                  file.approved === true
                    ? 'var(--mui-palette-success-main)'
                    : file.approved === false
                      ? 'var(--mui-palette-error-main)'
                      : idx === selectedFile
                        ? 'var(--mui-palette-primary-main)'
                        : 'transparent'
                }`,
                '&:hover': {
                  bgcolor: 'action.hover',
                },
              }}
            >
              <Stack direction="row" spacing={1} alignItems="center">
                <div className="file-status">
                  <Box
                    component="span"
                    className="status-icon"
                    sx={{
                      display: 'inline-flex',
                      width: 24,
                      height: 24,
                      borderRadius: 1,
                      color: 'common.white',
                      fontWeight: 700,
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                    }}
                    style={{ backgroundColor: getStatusColor(file.status) }}
                  >
                    {getStatusIcon(file.status)}
                  </Box>
                </div>
                <Box className="file-info" sx={{ flex: 1, minWidth: 0 }}>
                  <Typography className="file-name" variant="body2" sx={{ fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {file.file_path}
                  </Typography>
                  <Box className="file-changes" sx={{ display: 'flex', gap: 1, fontSize: '0.75rem', mt: 0.25 }}>
                    <Box component="span" className="additions" sx={{ color: 'success.dark', fontWeight: 600 }}>+{file.additions}</Box>
                    <Box component="span" className="deletions" sx={{ color: 'error.dark', fontWeight: 600 }}>-{file.deletions}</Box>
                  </Box>
                </Box>
                {file.approved !== null && (
                  <Box
                    className="review-badge"
                    sx={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      bgcolor: file.approved ? 'success.main' : 'error.main',
                      color: 'common.white',
                    }}
                  >
                    {file.approved ? '✓' : '✗'}
                  </Box>
                )}
              </Stack>
            </Box>
          ))}
        </Box>

        <Box className="file-detail" sx={{ border: 1, borderColor: 'divider', borderRadius: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" className="detail-header" sx={{ p: 2, bgcolor: 'action.hover', borderBottom: 1, borderColor: 'divider', gap: 1, flexWrap: 'wrap' }}>
            <Box className="file-path-header" sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <Box component="span" className="status-badge" sx={{ px: 1, py: 0.25, borderRadius: 1, color: 'common.white', fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase' }} style={{ backgroundColor: getStatusColor(currentFile.status) }}>
                {currentFile.status}
              </Box>
              <Typography component="span" className="file-path" sx={{ fontFamily: 'monospace', fontSize: '0.875rem', fontWeight: 600 }}>{currentFile.file_path}</Typography>
            </Box>
            <Box className="change-stats" sx={{ display: 'flex', gap: 2 }}>
              <Box component="span" className="additions" sx={{ color: 'success.dark', fontWeight: 600 }}>+{currentFile.additions}</Box>
              <Box component="span" className="deletions" sx={{ color: 'error.dark', fontWeight: 600 }}>-{currentFile.deletions}</Box>
            </Box>
          </Stack>

          <Box className="diff-container" sx={{ flex: 1, overflowY: 'auto', bgcolor: 'background.default', maxHeight: { xs: 400, lg: 'none' } }}>{renderDiff(currentFile.diff)}</Box>

          <Box className="review-actions" sx={{ p: 2, borderTop: 1, borderColor: 'divider', bgcolor: 'background.paper', display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            <TextField
              className="comment-input"
              placeholder="Add optional comment..."
              value={comments[currentFile.file_path] || ''}
              onChange={(e) => handleCommentChange(currentFile.file_path, e.target.value)}
              disabled={submitting === currentFile.file_path}
              multiline
              minRows={2}
              fullWidth
            />
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} className="action-buttons" sx={{ mt: 1, justifyContent: 'flex-end' }}>
              <Button
                className="reject-btn"
                variant="outlined"
                color="error"
                onClick={() => handleReview(currentFile.file_path, false)}
                disabled={submitting === currentFile.file_path}
              >
                {submitting === currentFile.file_path ? 'Processing...' : '✗ Reject'}
              </Button>
              <Button
                className="approve-btn"
                variant="contained"
                color="success"
                onClick={() => handleReview(currentFile.file_path, true)}
                disabled={submitting === currentFile.file_path}
              >
                {submitting === currentFile.file_path ? 'Processing...' : '✓ Approve'}
              </Button>
            </Stack>
          </Box>

          <Box className="navigation-buttons" sx={{ p: 2, borderTop: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'action.hover' }}>
            <Button
              onClick={() => setSelectedFile(Math.max(0, selectedFile - 1))}
              disabled={selectedFile === 0}
              variant="outlined"
              size="small"
            >
              ← Previous
            </Button>
            <span className="file-counter">File {selectedFile + 1} of {files.length}</span>
            <Button
              onClick={() => setSelectedFile(Math.min(files.length - 1, selectedFile + 1))}
              disabled={selectedFile === files.length - 1}
              variant="outlined"
              size="small"
            >
              Next →
            </Button>
          </Box>
        </Box>
      </Box>
    </Box>
  )
}

export default FileReview
