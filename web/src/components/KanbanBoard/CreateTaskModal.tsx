/**
 * Modal for creating a new task.
 */

import { useState } from 'react'
import {
  Alert,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  MenuItem,
  Stack,
  TextField,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../../api'

interface Props {
  projectDir?: string
  onCreated: () => void
  onClose: () => void
}

export function CreateTaskModal({ projectDir, onCreated, onClose }: Props) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [taskType, setTaskType] = useState('feature')
  const [priority, setPriority] = useState('P2')
  const [labels, setLabels] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim()) return

    setSaving(true)
    setError(null)

    try {
      const resp = await fetch(
        buildApiUrl('/api/v3/tasks', projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            title: title.trim(),
            description: description.trim(),
            task_type: taskType,
            priority,
            labels: labels.split(',').map((l) => l.trim()).filter(Boolean),
          }),
        }
      )
      if (resp.ok) {
        onCreated()
      } else {
        const data = await resp.json()
        setError(data.detail || 'Failed to create task')
      }
    } catch {
      setError('Network error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Create Task</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={1.5} component="form" onSubmit={handleSubmit} id="create-task-form">
          <TextField
            label="Title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Task title"
            autoFocus
            required
            fullWidth
          />

          <TextField
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Task description..."
            multiline
            minRows={4}
            fullWidth
          />

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
            <TextField
              label="Type"
              select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              fullWidth
            >
              <MenuItem value="feature">Feature</MenuItem>
              <MenuItem value="bug">Bug</MenuItem>
              <MenuItem value="refactor">Refactor</MenuItem>
              <MenuItem value="research">Research</MenuItem>
              <MenuItem value="test">Test</MenuItem>
              <MenuItem value="docs">Docs</MenuItem>
              <MenuItem value="security">Security</MenuItem>
              <MenuItem value="performance">Performance</MenuItem>
            </TextField>

            <TextField
              label="Priority"
              select
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
              fullWidth
            >
              <MenuItem value="P0">P0 - Critical</MenuItem>
              <MenuItem value="P1">P1 - High</MenuItem>
              <MenuItem value="P2">P2 - Medium</MenuItem>
              <MenuItem value="P3">P3 - Low</MenuItem>
            </TextField>
          </Stack>

          <TextField
            label="Labels"
            value={labels}
            onChange={(e) => setLabels(e.target.value)}
            placeholder="auth, urgent, frontend"
            fullWidth
          />

          {error && <Alert severity="error">{error}</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving}>Cancel</Button>
        <Button type="submit" form="create-task-form" variant="contained" disabled={saving || !title.trim()}>
          {saving ? 'Creating...' : 'Create Task'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
