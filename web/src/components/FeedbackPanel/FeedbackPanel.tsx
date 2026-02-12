/**
 * Structured feedback panel — allows humans to give actionable guidance to agents.
 */

import { useState, useEffect, useCallback } from 'react'
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { humanizeLabel } from '../../ui/labels'

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

const priorityColor = (priority: string) => {
  if (priority === 'must') return 'error.main'
  if (priority === 'should') return 'warning.main'
  return 'info.main'
}

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
        buildApiUrl(`/api/v3/collaboration/feedback/${taskId}`, projectDir),
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
        buildApiUrl('/api/v3/collaboration/feedback', projectDir),
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
      buildApiUrl(`/api/v3/collaboration/feedback/${feedbackId}/dismiss`, projectDir),
      { method: 'POST', headers: buildAuthHeaders() }
    )
    fetchFeedback()
  }

  const activeFeedback = feedback.filter(f => f.status === 'active')
  const addressedFeedback = feedback.filter(f => f.status === 'addressed')

  return (
    <Box sx={{ p: 1.5 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
        <Typography variant="h6" sx={{ fontSize: '1rem' }}>Feedback</Typography>
        <Chip size="small" label={`${activeFeedback.length} active`} variant="outlined" />
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="contained" onClick={() => setShowForm(!showForm)}>
          + Add Feedback
        </Button>
      </Stack>

      {showForm && (
        <Card variant="outlined" sx={{ mb: 1.5 }}>
          <CardContent sx={{ '&:last-child': { pb: 2 } }}>
            <Stack spacing={1}>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                <TextField size="small" select value={formType} onChange={(e) => setFormType(e.target.value)} fullWidth>
                  {FEEDBACK_TYPES.map(t => (
                    <MenuItem key={t.value} value={t.value}>{t.label}</MenuItem>
                  ))}
                </TextField>
                <TextField size="small" select value={formPriority} onChange={(e) => setFormPriority(e.target.value)} fullWidth>
                  {PRIORITIES.map(p => (
                    <MenuItem key={p.value} value={p.value}>{p.label}</MenuItem>
                  ))}
                </TextField>
              </Stack>

              <TextField
                placeholder="Summary (one line)"
                value={formSummary}
                onChange={(e) => setFormSummary(e.target.value)}
                size="small"
                fullWidth
              />
              <TextField
                placeholder="Details (optional)"
                value={formDetails}
                onChange={(e) => setFormDetails(e.target.value)}
                size="small"
                multiline
                minRows={3}
                fullWidth
              />
              <TextField
                placeholder="Target file (optional)"
                value={formFile}
                onChange={(e) => setFormFile(e.target.value)}
                size="small"
                fullWidth
              />

              <Stack direction="row" spacing={1}>
                <Button variant="contained" color="success" onClick={handleSubmit} disabled={submitting}>
                  {submitting ? 'Submitting...' : 'Submit Feedback'}
                </Button>
                <Button variant="outlined" onClick={() => setShowForm(false)}>Cancel</Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>
      )}

      <Stack spacing={1}>
        {activeFeedback.map(fb => (
          <Card key={fb.id} variant="outlined" sx={{ borderLeft: '3px solid', borderLeftColor: priorityColor(fb.priority) }}>
            <CardContent sx={{ '&:last-child': { pb: 1.5 } }}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
                <Chip
                  size="small"
                  label={humanizeLabel(fb.feedback_type)}
                  variant="outlined"
                  sx={{ textTransform: 'capitalize' }}
                />
                <Typography
                  variant="caption"
                  sx={{ textTransform: 'uppercase', fontWeight: 700, color: priorityColor(fb.priority) }}
                >
                  {fb.priority}
                </Typography>
                <Box sx={{ flex: 1 }} />
                <Button size="small" onClick={() => handleDismiss(fb.id)}>Dismiss</Button>
              </Stack>

              <Typography variant="body2" sx={{ fontWeight: 600 }}>{fb.summary}</Typography>
              {fb.details && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  {fb.details}
                </Typography>
              )}
              {fb.target_file && (
                <Typography component="code" variant="caption" sx={{ mt: 0.75, display: 'inline-block' }}>
                  {fb.target_file}
                </Typography>
              )}
            </CardContent>
          </Card>
        ))}

        {activeFeedback.length === 0 && !showForm && (
          <Typography color="text.secondary" sx={{ textAlign: 'center', py: 3 }}>
            No active feedback for this task.
          </Typography>
        )}
      </Stack>

      {addressedFeedback.length > 0 && (
        <Accordion sx={{ mt: 1.5 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>{addressedFeedback.length} addressed</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1}>
              {addressedFeedback.map(fb => (
                <Card key={fb.id} variant="outlined" sx={{ opacity: 0.75 }}>
                  <CardContent sx={{ '&:last-child': { pb: 1.5 } }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{fb.summary}</Typography>
                    {fb.agent_response && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                        Agent: {fb.agent_response}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              ))}
            </Stack>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  )
}
