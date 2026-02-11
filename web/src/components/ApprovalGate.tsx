import { useState, useEffect, useCallback } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Collapse,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import { useToast } from '../contexts/ToastContext'
import EmptyState from './EmptyState'
import LoadingSpinner from './LoadingSpinner'

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
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const toast = useToast()

  useEffect(() => {
    fetchApprovals()
  }, [projectDir])

  useChannel('approvals', useCallback(() => {
    fetchApprovals()
  }, [projectDir]))

  const fetchApprovals = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/approvals', projectDir), {
        headers: buildAuthHeaders(),
      })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setApprovals(Array.isArray(data) ? data : [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch approvals')
    } finally {
      setLoading(false)
    }
  }

  const handleRespond = async (requestId: string, approved: boolean) => {
    setSubmitting(requestId)

    try {
      const response = await fetch(buildApiUrl('/api/approvals/respond', projectDir), {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          request_id: requestId,
          approved,
          feedback: feedback[requestId] || null,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }

      await response.json()
      toast.success(approved ? 'Approved successfully' : 'Rejected successfully')

      setFeedback((prev) => {
        const updated = { ...prev }
        delete updated[requestId]
        return updated
      })

      await fetchApprovals()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to respond')
    } finally {
      setSubmitting(null)
    }
  }

  const formatTimestamp = (timestamp: string): string => {
    try {
      return new Date(timestamp).toLocaleString()
    } catch {
      return timestamp
    }
  }

  const renderContext = (approval: ApprovalGateInfo) => {
    const sections: Array<{ title: string; content: string }> = []

    if (approval.show_diff && approval.context.diff) {
      sections.push({ title: 'Diff', content: String(approval.context.diff) })
    }
    if (approval.show_plan && approval.context.plan) {
      sections.push({ title: 'Plan', content: String(approval.context.plan) })
    }
    if (approval.show_tests && approval.context.tests) {
      sections.push({ title: 'Tests', content: String(approval.context.tests) })
    }
    if (approval.show_review && approval.context.review) {
      sections.push({ title: 'Review', content: String(approval.context.review) })
    }

    const otherContext = Object.entries(approval.context).filter(
      ([key]) => !['diff', 'plan', 'tests', 'review'].includes(key)
    )
    if (otherContext.length > 0) {
      sections.push({
        title: 'Additional Context',
        content: JSON.stringify(Object.fromEntries(otherContext), null, 2),
      })
    }

    if (sections.length === 0) return null

    return (
      <Stack spacing={1.5} sx={{ mt: 1.5 }}>
        {sections.map((section) => (
          <Box key={section.title}>
            <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', fontWeight: 600 }}>
              {section.title}
            </Typography>
            <Box
              component="pre"
              sx={{
                fontSize: 12,
                p: 1,
                mt: 0.5,
                borderRadius: 1,
                bgcolor: 'background.default',
                border: 1,
                borderColor: 'divider',
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
              }}
            >
              {section.content}
            </Box>
          </Box>
        ))}
      </Stack>
    )
  }

  if (loading) {
    return (
      <Box className="approval-gate">
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1 }}>Pending Approvals</Typography>
        <LoadingSpinner label="Loading approvals..." />
      </Box>
    )
  }

  if (error) {
    return (
      <Box className="approval-gate">
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1 }}>Pending Approvals</Typography>
        <EmptyState
          icon={<span>⚠️</span>}
          title="Error loading approvals"
          description={error}
          size="sm"
        />
      </Box>
    )
  }

  if (approvals.length === 0) {
    return (
      <Box className="approval-gate">
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1 }}>Pending Approvals</Typography>
        <EmptyState
          icon={<span>✓</span>}
          title="No pending approvals"
          description="All approval gates have been resolved."
          size="sm"
        />
      </Box>
    )
  }

  return (
    <Box className="approval-gate">
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
        <Typography variant="h2" sx={{ fontSize: '1.125rem' }}>Pending Approvals</Typography>
        <Chip size="small" label={approvals.length} className="approval-count" />
      </Stack>

      <Stack spacing={1.25} className="approvals-list">
        {approvals.map((approval) => {
          const isOpen = expandedId === approval.request_id
          const busy = submitting === approval.request_id

          return (
            <Card key={approval.request_id} className="approval-card" variant="outlined">
              <CardContent sx={{ pb: '16px !important' }}>
                <Stack direction="row" justifyContent="space-between" spacing={1} sx={{ mb: 1 }}>
                  <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
                    <Chip size="small" label={approval.gate_type} className="gate-type" />
                    {approval.task_id && <Chip size="small" label={`Task: ${approval.task_id}`} className="task-id" />}
                    {approval.phase_id && <Chip size="small" label={`Phase: ${approval.phase_id}`} className="phase-id" />}
                  </Stack>
                  <Typography variant="caption" color="text.secondary" className="approval-timestamp">
                    {formatTimestamp(approval.created_at)}
                  </Typography>
                </Stack>

                <Typography className="approval-message" sx={{ mb: 1 }}>{approval.message}</Typography>

                {approval.timeout && (
                  <Alert severity="warning" sx={{ py: 0.25, mb: 1 }} className="approval-timeout">
                    Timeout: {approval.timeout} seconds
                  </Alert>
                )}

                <Button size="small" variant="text" onClick={() => setExpandedId(isOpen ? null : approval.request_id)}>
                  {isOpen ? 'Hide Context' : 'Show Context'}
                </Button>

                <Collapse in={isOpen}>
                  {renderContext(approval)}
                </Collapse>

                <Stack spacing={1} sx={{ mt: 1.5 }} className="approval-actions">
                  <TextField
                    className="feedback-input"
                    placeholder="Optional feedback..."
                    value={feedback[approval.request_id] || ''}
                    onChange={(e) => setFeedback((prev) => ({ ...prev, [approval.request_id]: e.target.value }))}
                    disabled={busy}
                    multiline
                    minRows={2}
                    fullWidth
                  />
                  <Stack direction="row" spacing={1} className="action-buttons">
                    <Button
                      className="approve-btn"
                      variant="contained"
                      color="success"
                      onClick={() => handleRespond(approval.request_id, true)}
                      disabled={busy}
                    >
                      {busy ? 'Processing...' : 'Approve'}
                    </Button>
                    <Button
                      className="reject-btn"
                      variant="outlined"
                      color="error"
                      onClick={() => handleRespond(approval.request_id, false)}
                      disabled={busy}
                    >
                      {busy ? 'Processing...' : 'Reject'}
                    </Button>
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          )
        })}
      </Stack>
    </Box>
  )
}

export default ApprovalGate
