import { useState } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import './TaskLauncher.css'

interface TaskLauncherProps {
  projectDir: string | null
  onRunStarted?: (runId: string) => void
}

interface StartRunResponse {
  success: boolean
  message: string
  run_id: string | null
  prd_path: string | null
}

interface ExecTaskResponse {
  success: boolean
  message: string
  run_id: string | null
  error: string | null
}

type LauncherMode = 'quick_task' | 'quick_prompt' | 'full_prd'

export default function TaskLauncher({ projectDir, onRunStarted }: TaskLauncherProps) {
  const [mode, setMode] = useState<LauncherMode>('quick_task')
  const [content, setContent] = useState('')

  // Config for full workflow (quick_prompt and full_prd)
  const [testCommand, setTestCommand] = useState('')
  const [buildCommand, setBuildCommand] = useState('')
  const [verificationProfile, setVerificationProfile] = useState<'none' | 'python'>('none')
  const [autoApprovePlans, setAutoApprovePlans] = useState(false)
  const [autoApproveChanges, setAutoApproveChanges] = useState(false)
  const [autoApproveCommits, setAutoApproveCommits] = useState(false)

  // Config for quick_task (exec)
  const [overrideAgents, setOverrideAgents] = useState(false)
  const [contextFiles, setContextFiles] = useState('')

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)

    if (!content.trim()) {
      setError('Please enter task description')
      return
    }

    if (!projectDir) {
      setError('No project selected')
      return
    }

    setIsSubmitting(true)

    try {
      if (mode === 'quick_task') {
        // Use the exec endpoint for one-off tasks
        const response = await fetch(buildApiUrl('/api/runs/exec', projectDir), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...buildAuthHeaders(),
          },
          body: JSON.stringify({
            prompt: content,
            override_agents: overrideAgents,
            context_files: contextFiles || null,
            shift_minutes: 45,
            heartbeat_seconds: 120,
          }),
        })

        const data: ExecTaskResponse = await response.json()

        if (data.success) {
          setSuccess('Task executed successfully!')
          setContent('') // Clear the input
        } else {
          setError(data.error || data.message || 'Failed to execute task')
        }
      } else {
        // Use the run endpoint for full workflow
        const response = await fetch(buildApiUrl('/api/runs/start', projectDir), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...buildAuthHeaders(),
          },
          body: JSON.stringify({
            mode,
            content,
            test_command: testCommand || null,
            build_command: buildCommand || null,
            verification_profile: verificationProfile,
            auto_approve_plans: autoApprovePlans,
            auto_approve_changes: autoApproveChanges,
            auto_approve_commits: autoApproveCommits,
          }),
        })

        const data: StartRunResponse = await response.json()

        if (data.success && data.run_id) {
          setSuccess(`Run started successfully! Run ID: ${data.run_id}`)
          setContent('') // Clear the input
          if (onRunStarted) {
            onRunStarted(data.run_id)
          }
        } else {
          setError(data.message || 'Failed to start run')
        }
      }
    } catch (err) {
      setError(`Error: ${err}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  const getModeDescription = () => {
    switch (mode) {
      case 'quick_task':
        return 'Execute a simple task immediately and terminate (no full workflow). Perfect for quick edits like "add folder to .gitignore".'
      case 'quick_prompt':
        return 'Enter a brief description. An AI will generate a full PRD and run the complete workflow.'
      case 'full_prd':
        return 'Paste a complete PRD document in markdown format for the full workflow.'
    }
  }

  const getPlaceholder = () => {
    switch (mode) {
      case 'quick_task':
        return 'e.g., "add node_modules to .gitignore" or "fix typo in README"'
      case 'quick_prompt':
        return 'e.g., "Add a user profile page with avatar upload and bio editing"'
      case 'full_prd':
        return '# Feature: [Your Feature Name]\n\n## Overview\n[Feature description...]\n\n## Requirements\n1. ...\n2. ...'
    }
  }

  const getTextareaRows = () => {
    switch (mode) {
      case 'quick_task':
        return 3
      case 'quick_prompt':
        return 5
      case 'full_prd':
        return 15
    }
  }

  return (
    <div className="task-launcher">
      <div className="task-launcher-header">
        <h2>Launch New Task</h2>
      </div>

      <form onSubmit={handleSubmit} className="task-launcher-form">
        {/* Mode Selection */}
        <div className="form-section">
          <label className="form-label">Mode</label>
          <div className="mode-toggle mode-toggle-three">
            <button
              type="button"
              className={`mode-button ${mode === 'quick_task' ? 'active' : ''}`}
              onClick={() => setMode('quick_task')}
            >
              Quick Task
            </button>
            <button
              type="button"
              className={`mode-button ${mode === 'quick_prompt' ? 'active' : ''}`}
              onClick={() => setMode('quick_prompt')}
            >
              Quick Prompt
            </button>
            <button
              type="button"
              className={`mode-button ${mode === 'full_prd' ? 'active' : ''}`}
              onClick={() => setMode('full_prd')}
            >
              Full PRD
            </button>
          </div>
          <p className="mode-description">{getModeDescription()}</p>
        </div>

        {/* Content Input */}
        <div className="form-section">
          <label className="form-label" htmlFor="content">
            {mode === 'full_prd' ? 'PRD Content' : mode === 'quick_task' ? 'Task Description' : 'Feature Prompt'}
          </label>
          <textarea
            id="content"
            className="content-textarea"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder={getPlaceholder()}
            rows={getTextareaRows()}
            disabled={isSubmitting}
          />
        </div>

        {/* Configuration Section for Quick Task */}
        {mode === 'quick_task' && (
          <div className="form-section config-section">
            <h3 className="section-title">Quick Task Options</h3>

            <div className="form-field">
              <label className="form-label" htmlFor="contextFiles">
                Focus Files (optional)
              </label>
              <input
                id="contextFiles"
                type="text"
                className="form-input"
                value={contextFiles}
                onChange={(e) => setContextFiles(e.target.value)}
                placeholder="e.g., src/auth.py,src/models/user.py"
                disabled={isSubmitting}
              />
              <p className="field-hint">Comma-separated file paths to limit scope</p>
            </div>

            <div className="form-field">
              <label className="checkbox-label checkbox-label-standalone">
                <input
                  type="checkbox"
                  checked={overrideAgents}
                  onChange={(e) => setOverrideAgents(e.target.checked)}
                  disabled={isSubmitting}
                />
                <span>
                  <strong>Override AGENTS.md rules</strong> (superadmin mode)
                  <br />
                  <small>Bypass file restrictions and normal workflow rules. Use for emergency fixes or administrative tasks.</small>
                </span>
              </label>
            </div>
          </div>
        )}

        {/* Configuration Section for Full Workflow */}
        {mode !== 'quick_task' && (
          <div className="form-section config-section">
            <h3 className="section-title">Workflow Configuration</h3>

            <div className="form-row">
              <div className="form-field">
                <label className="form-label" htmlFor="testCommand">
                  Test Command
                </label>
                <input
                  id="testCommand"
                  type="text"
                  className="form-input"
                  value={testCommand}
                  onChange={(e) => setTestCommand(e.target.value)}
                  placeholder="e.g., npm test"
                  disabled={isSubmitting}
                />
              </div>

              <div className="form-field">
                <label className="form-label" htmlFor="buildCommand">
                  Build Command
                </label>
                <input
                  id="buildCommand"
                  type="text"
                  className="form-input"
                  value={buildCommand}
                  onChange={(e) => setBuildCommand(e.target.value)}
                  placeholder="e.g., npm run build"
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <div className="form-field">
              <label className="form-label" htmlFor="verificationProfile">
                Verification Profile
              </label>
              <select
                id="verificationProfile"
                className="form-select"
                value={verificationProfile}
                onChange={(e) => setVerificationProfile(e.target.value as 'none' | 'python')}
                disabled={isSubmitting}
              >
                <option value="none">None</option>
                <option value="python">Python</option>
              </select>
            </div>

            <div className="form-field">
              <label className="form-label">Auto-approve Settings</label>
              <div className="checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={autoApprovePlans}
                    onChange={(e) => setAutoApprovePlans(e.target.checked)}
                    disabled={isSubmitting}
                  />
                  <span>Auto-approve plans</span>
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={autoApproveChanges}
                    onChange={(e) => setAutoApproveChanges(e.target.checked)}
                    disabled={isSubmitting}
                  />
                  <span>Auto-approve changes</span>
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={autoApproveCommits}
                    onChange={(e) => setAutoApproveCommits(e.target.checked)}
                    disabled={isSubmitting}
                  />
                  <span>Auto-approve commits</span>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Status Messages */}
        {error && <div className="message error-message">{error}</div>}
        {success && <div className="message success-message">{success}</div>}

        {/* Submit Button */}
        <div className="form-actions">
          <button type="submit" className="submit-button" disabled={isSubmitting || !content.trim()}>
            {isSubmitting
              ? mode === 'quick_task' ? 'Executing...' : 'Starting...'
              : mode === 'quick_task' ? 'Execute Task' : 'Start Run'}
          </button>
        </div>
      </form>
    </div>
  )
}
