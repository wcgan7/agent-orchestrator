import { useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useToast } from '../contexts/ToastContext'

interface Props {
  currentTaskId?: string
  currentPhaseId?: string
  status?: string
  projectDir?: string
}

type ControlAction = 'retry' | 'skip' | 'resume' | 'stop'

export default function ControlPanel({ currentTaskId, currentPhaseId, status, projectDir }: Props) {
  const [loading, setLoading] = useState<ControlAction | null>(null)
  const [selectedStep, setSelectedStep] = useState('plan_impl')
  const toast = useToast()

  const steps = [
    { value: 'plan_impl', label: 'Plan Implementation' },
    { value: 'implement', label: 'Implement' },
    { value: 'verify', label: 'Verify' },
    { value: 'review', label: 'Review' },
    { value: 'commit', label: 'Commit' },
  ]

  const executeAction = async (action: ControlAction) => {
    if (!currentTaskId && action !== 'stop') {
      toast.error('No active task to control')
      return
    }

    setLoading(action)

    try {
      const body: any = {
        action,
        task_id: currentTaskId || null,
        params: action === 'retry' ? { step: selectedStep } : null,
      }

      const response = await fetch(buildApiUrl('/api/control', projectDir), {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(body),
      })

      const data = await response.json()
      if (data.success) {
        toast.success(data.message || `${action} completed successfully`)
      } else {
        toast.error(data.message || `${action} failed`)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to execute action')
    } finally {
      setLoading(null)
    }
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>Run Control</Typography>

      {currentTaskId ? (
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} sx={{ mb: 2 }}>
          <Chip size="small" variant="outlined" label={`Task: ${currentTaskId}`} />
          {currentPhaseId && <Chip size="small" variant="outlined" label={`Phase: ${currentPhaseId}`} />}
          {status && <Chip size="small" color="info" variant="outlined" label={`Status: ${status}`} />}
        </Stack>
      ) : (
        <Alert severity="info" sx={{ mb: 2 }}>No active task found. You can still stop the run.</Alert>
      )}

      <FormControl size="small" fullWidth sx={{ mb: 2 }}>
        <InputLabel id="control-panel-step-label">Retry from step</InputLabel>
        <Select
          labelId="control-panel-step-label"
          value={selectedStep}
          label="Retry from step"
          onChange={(e) => setSelectedStep(e.target.value)}
          className="control-panel-select"
        >
          {steps.map((step) => (
            <MenuItem key={step.value} value={step.value}>
              {step.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} useFlexGap flexWrap="wrap" sx={{ mb: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={() => executeAction('retry')}
          disabled={loading !== null || !currentTaskId}
          className={`control-btn control-btn-retry ${loading === 'retry' ? 'loading' : ''}`}
        >
          {loading === 'retry' ? 'Retrying...' : 'Retry Task'}
        </Button>

        <Button
          variant="outlined"
          color="warning"
          onClick={() => executeAction('skip')}
          disabled={loading !== null || !currentTaskId}
          className={`control-btn control-btn-skip ${loading === 'skip' ? 'loading' : ''}`}
        >
          {loading === 'skip' ? 'Skipping...' : 'Skip Step'}
        </Button>

        <Button
          variant="outlined"
          color="success"
          onClick={() => executeAction('resume')}
          disabled={loading !== null || !currentTaskId}
          className={`control-btn control-btn-resume ${loading === 'resume' ? 'loading' : ''}`}
        >
          {loading === 'resume' ? 'Resuming...' : 'Resume Task'}
        </Button>

        <Button
          variant="outlined"
          color="error"
          onClick={() => executeAction('stop')}
          disabled={loading !== null}
          className={`control-btn control-btn-stop ${loading === 'stop' ? 'loading' : ''}`}
        >
          {loading === 'stop' ? 'Stopping...' : 'Stop Run'}
        </Button>
      </Stack>

      <Typography variant="body2" color="text.secondary">
        Retry restarts at a selected step, Skip advances the task, Resume unblocks stalled tasks, and Stop sends a stop signal.
      </Typography>
    </Box>
  )
}
