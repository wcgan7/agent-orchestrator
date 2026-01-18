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

export default function TaskLauncher({ projectDir, onRunStarted }: TaskLauncherProps) {
  const [mode, setMode] = useState<'full_prd' | 'quick_prompt'>('quick_prompt')
  const [content, setContent] = useState('')
  const [testCommand, setTestCommand] = useState('')
  const [buildCommand, setBuildCommand] = useState('')
  const [verificationProfile, setVerificationProfile] = useState<'none' | 'python'>('none')
  const [autoApprovePlans, setAutoApprovePlans] = useState(false)
  const [autoApproveChanges, setAutoApproveChanges] = useState(false)
  const [autoApproveCommits, setAutoApproveCommits] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)

    if (!content.trim()) {
      setError(mode === 'full_prd' ? 'Please enter PRD content' : 'Please enter a prompt')
      return
    }

    if (!projectDir) {
      setError('No project selected')
      return
    }

    setIsSubmitting(true)

    try {
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
    } catch (err) {
      setError(`Error starting run: ${err}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="task-launcher">
      <div className="task-launcher-header">
        <h2>Launch New Run</h2>
      </div>

      <form onSubmit={handleSubmit} className="task-launcher-form">
        {/* Mode Selection */}
        <div className="form-section">
          <label className="form-label">Mode</label>
          <div className="mode-toggle">
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
          <p className="mode-description">
            {mode === 'quick_prompt'
              ? 'Enter a brief description. An AI will generate a full PRD from your prompt.'
              : 'Paste a complete PRD document in markdown format.'}
          </p>
        </div>

        {/* Content Input */}
        <div className="form-section">
          <label className="form-label" htmlFor="content">
            {mode === 'quick_prompt' ? 'Feature Prompt' : 'PRD Content'}
          </label>
          <textarea
            id="content"
            className="content-textarea"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder={
              mode === 'quick_prompt'
                ? 'e.g., "Add a user profile page with avatar upload and bio editing"'
                : '# Feature: [Your Feature Name]\n\n## Overview\n[Feature description...]\n\n## Requirements\n1. ...\n2. ...'
            }
            rows={mode === 'full_prd' ? 15 : 5}
            disabled={isSubmitting}
          />
        </div>

        {/* Configuration Section */}
        <div className="form-section config-section">
          <h3 className="section-title">Configuration</h3>

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

        {/* Status Messages */}
        {error && <div className="message error-message">{error}</div>}
        {success && <div className="message success-message">{success}</div>}

        {/* Submit Button */}
        <div className="form-actions">
          <button type="submit" className="submit-button" disabled={isSubmitting || !content.trim()}>
            {isSubmitting ? 'Starting Run...' : 'Start Run'}
          </button>
        </div>
      </form>
    </div>
  )
}
