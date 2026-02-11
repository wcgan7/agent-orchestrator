import { useState } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useToast } from '../contexts/ToastContext'

const TASK_LAUNCHER_STYLES = `
.task-launcher { background: var(--color-bg-primary); border-radius: var(--radius-lg); padding: var(--spacing-6); margin-bottom: var(--spacing-6); box-shadow: var(--shadow-sm); }
.task-launcher-header { margin-bottom: var(--spacing-6); }
.task-launcher-header h2 { margin: 0; font-size: var(--text-2xl); font-weight: var(--font-semibold); color: var(--color-text-primary); }
.task-launcher-form { display: flex; flex-direction: column; gap: var(--spacing-6); }
.form-section { display: flex; flex-direction: column; gap: var(--spacing-3); }
.section-title { margin: 0; font-size: var(--text-lg); font-weight: var(--font-semibold); color: var(--color-text-primary); }
.form-label { font-weight: var(--font-medium); color: var(--color-text-primary); font-size: var(--text-sm); }
.mode-toggle { display: flex; gap: var(--spacing-2); }
.mode-toggle-three { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--spacing-2); }
.mode-button { flex: 1; padding: var(--spacing-3) var(--spacing-4); border: 2px solid var(--color-border-default); background: var(--color-bg-primary); color: var(--color-text-primary); border-radius: var(--radius-md); font-size: var(--text-sm); font-weight: var(--font-medium); cursor: pointer; transition: all var(--transition-base); }
.mode-button:hover { background: var(--color-bg-secondary); }
.mode-button.active { background: var(--color-primary-500); color: var(--color-text-inverse); border-color: var(--color-primary-500); }
.mode-description { margin: 0; font-size: var(--text-sm); color: var(--color-text-secondary); line-height: var(--leading-normal); }
.content-textarea { width: 100%; padding: var(--spacing-3); border: 2px solid var(--color-border-default); border-radius: var(--radius-md); font-family: var(--font-mono); font-size: var(--text-sm); line-height: var(--leading-relaxed); background: var(--color-bg-primary); color: var(--color-text-primary); resize: vertical; min-height: 120px; }
.content-textarea:focus { outline: none; border-color: var(--color-primary-500); box-shadow: 0 0 0 3px var(--color-primary-100); }
.content-textarea::placeholder { color: var(--color-text-muted); }
.config-section { background: var(--color-bg-tertiary); padding: var(--spacing-5); border-radius: var(--radius-md); border: 1px solid var(--color-border-default); }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: var(--spacing-4); }
.form-field { display: flex; flex-direction: column; gap: var(--spacing-2); }
.form-input, .form-select { padding: var(--spacing-2-5) var(--spacing-3); border: 2px solid var(--color-border-default); border-radius: var(--radius-md); font-size: var(--text-sm); background: var(--color-bg-primary); color: var(--color-text-primary); }
.form-input:focus, .form-select:focus { outline: none; border-color: var(--color-primary-500); box-shadow: 0 0 0 3px var(--color-primary-100); }
.form-input::placeholder { color: var(--color-text-muted); }
.checkbox-group { display: flex; flex-direction: column; gap: var(--spacing-2-5); }
.checkbox-label { display: flex; align-items: center; gap: var(--spacing-2); cursor: pointer; font-size: var(--text-sm); color: var(--color-text-primary); }
.checkbox-label input[type='checkbox'] { width: 18px; height: 18px; cursor: pointer; flex-shrink: 0; }
.checkbox-label-standalone { align-items: flex-start; padding: var(--spacing-3); background: var(--color-bg-primary); border: 1px solid var(--color-border-default); border-radius: var(--radius-md); }
.checkbox-label-standalone input[type='checkbox'] { margin-top: 2px; }
.checkbox-label-standalone small { display: block; margin-top: var(--spacing-1); color: var(--color-text-secondary); font-weight: var(--font-normal); }
.field-hint { margin: var(--spacing-1) 0 0 0; font-size: var(--text-xs); color: var(--color-text-secondary); }
.form-actions { display: flex; justify-content: flex-end; }
.submit-button { padding: var(--spacing-3) var(--spacing-8); background: var(--color-primary-500); color: var(--color-text-inverse); border: none; border-radius: var(--radius-md); font-size: var(--text-base); font-weight: var(--font-semibold); cursor: pointer; transition: all var(--transition-base); }
.submit-button:hover:not(:disabled) { background: var(--color-primary-600); transform: translateY(-1px); box-shadow: var(--shadow-md); }
.submit-button:active:not(:disabled) { transform: translateY(0); }
.submit-button:disabled { background: var(--color-gray-300); cursor: not-allowed; opacity: 0.6; }
@media (max-width: 768px) { .form-row { grid-template-columns: 1fr; } .mode-toggle, .mode-toggle-three { display: flex; flex-direction: column; } .mode-button { font-size: var(--text-sm); padding: var(--spacing-3) var(--spacing-6); } .task-launcher { padding: var(--spacing-4); } }
`

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

interface QuickRunExecuteResponse {
  success: boolean
  message: string
  quick_run: {
    id: string
    status: string
  }
  error: string | null
}

type LauncherMode = 'quick_task' | 'quick_prompt' | 'full_prd'

export default function TaskLauncher({ projectDir, onRunStarted }: TaskLauncherProps) {
  const [mode, setMode] = useState<LauncherMode>('quick_task')
  const [content, setContent] = useState('')
  const toast = useToast()

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
  const [promoteToTask, setPromoteToTask] = useState(false)
  const [promoteTaskType, setPromoteTaskType] = useState('feature')
  const [promoteTaskPriority, setPromoteTaskPriority] = useState('P2')

  // Advanced options (Batch 6)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [language, setLanguage] = useState('')
  const [resetState, setResetState] = useState(false)
  const [requireClean, setRequireClean] = useState(true)
  const [commitEnabled, setCommitEnabled] = useState(true)
  const [pushEnabled, setPushEnabled] = useState(true)
  const [parallel, setParallel] = useState(false)
  const [maxWorkers, setMaxWorkers] = useState(3)
  const [ensureRuff, setEnsureRuff] = useState('off')
  const [ensureDeps, setEnsureDeps] = useState('off')
  const [ensureDepsCommand, setEnsureDepsCommand] = useState('')
  const [shiftMinutes, setShiftMinutes] = useState(45)
  const [maxTaskAttempts, setMaxTaskAttempts] = useState(5)
  const [maxReviewAttempts, setMaxReviewAttempts] = useState(10)
  const [worker, setWorker] = useState('')
  const [codexCommand, setCodexCommand] = useState('')

  const [isSubmitting, setIsSubmitting] = useState(false)

  const buildPromotionTitle = (prompt: string): string => {
    const firstLine = prompt
      .split('\n')
      .map((line) => line.trim())
      .find((line) => line.length > 0) || 'Quick action follow-up'
    const compact = firstLine.replace(/\s+/g, ' ')
    return compact.length > 80 ? `${compact.slice(0, 77).trimEnd()}...` : compact
  }

  const promoteQuickActionToTask = async (quickRunId: string, prompt: string): Promise<string | null> => {
    if (!projectDir) return null
    const response = await fetch(buildApiUrl(`/api/v3/quick-runs/${quickRunId}/promote`, projectDir), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...buildAuthHeaders(),
      },
      body: JSON.stringify({
        title: buildPromotionTitle(prompt),
        task_type: promoteTaskType,
        priority: promoteTaskPriority,
      }),
    })

    if (!response.ok) {
      let detail = 'Failed to promote quick action to task'
      try {
        const data = await response.json()
        if (data?.detail && typeof data.detail === 'string') detail = data.detail
      } catch {
        // ignore parse errors and keep default detail
      }
      throw new Error(detail)
    }

    const data = await response.json()
    return typeof data?.task_id === 'string' ? data.task_id : null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!content.trim()) {
      toast.error(mode === 'quick_task' ? 'Please enter quick action prompt' : 'Please enter task description')
      return
    }

    if (!projectDir) {
      toast.error('No project selected')
      return
    }

    setIsSubmitting(true)

    try {
      if (mode === 'quick_task') {
        // Use the exec endpoint for one-off actions
        const response = await fetch(buildApiUrl('/api/v3/quick-runs', projectDir), {
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
        const data = await response.json() as QuickRunExecuteResponse & { detail?: string }
        if (!response.ok) {
          toast.error(data.detail || data.error || data.message || 'Failed to execute action')
          return
        }

        if (data.success) {
          let promotedTaskId: string | null = null
          const quickRunId = data.quick_run?.id
          if (promoteToTask && quickRunId) {
            try {
              promotedTaskId = await promoteQuickActionToTask(quickRunId, content)
            } catch (promotionErr) {
              toast.error(`Quick action succeeded, but promotion failed: ${promotionErr}`)
            }
          } else if (promoteToTask && !quickRunId) {
            toast.error('Quick action succeeded, but promotion failed: missing quick run ID')
          }
          if (promoteToTask && promotedTaskId) {
            toast.success(`Quick action executed and promoted to task ${promotedTaskId}`)
          } else {
            toast.success('Quick action executed successfully!')
          }
          setContent('') // Clear the input
        } else {
          toast.error(data.error || data.message || 'Failed to execute action')
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
            // Advanced options
            language: language || null,
            reset_state: resetState,
            require_clean: requireClean,
            commit_enabled: commitEnabled,
            push_enabled: pushEnabled,
            parallel,
            max_workers: maxWorkers,
            ensure_ruff: ensureRuff,
            ensure_deps: ensureDeps,
            ensure_deps_command: ensureDepsCommand || null,
            shift_minutes: shiftMinutes,
            max_task_attempts: maxTaskAttempts,
            max_review_attempts: maxReviewAttempts,
            worker: worker || null,
            codex_command: codexCommand || null,
          }),
        })

        const data: StartRunResponse = await response.json()

        if (data.success && data.run_id) {
          toast.success(`Run started! ID: ${data.run_id}`)
          setContent('') // Clear the input
          if (onRunStarted) {
            onRunStarted(data.run_id)
          }
        } else {
          toast.error(data.message || 'Failed to start run')
        }
      }
    } catch (err) {
      toast.error(`Error: ${err}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  const getModeDescription = () => {
    switch (mode) {
      case 'quick_task':
        return 'Run a one-off action immediately. This does not create a board task unless you later promote it.'
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
      <style>{TASK_LAUNCHER_STYLES}</style>
      <div className="task-launcher-header">
        <h2>Launch New Run</h2>
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
              Quick Action
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
            {mode === 'full_prd' ? 'PRD Content' : mode === 'quick_task' ? 'Action Prompt' : 'Feature Prompt'}
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

        {/* Configuration Section for Quick Action */}
        {mode === 'quick_task' && (
          <div className="form-section config-section">
            <h3 className="section-title">Quick Action Options</h3>

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

            <div className="form-field">
              <label className="checkbox-label checkbox-label-standalone">
                <input
                  type="checkbox"
                  checked={promoteToTask}
                  onChange={(e) => setPromoteToTask(e.target.checked)}
                  disabled={isSubmitting}
                />
                <span>
                  <strong>Save result as task</strong>
                  <br />
                  <small>Create a board task after this quick action succeeds.</small>
                </span>
              </label>
            </div>

            {promoteToTask && (
              <div className="form-row">
                <div className="form-field">
                  <label className="form-label" htmlFor="promoteTaskType">Task Type</label>
                  <select
                    id="promoteTaskType"
                    className="form-select"
                    value={promoteTaskType}
                    onChange={(e) => setPromoteTaskType(e.target.value)}
                    disabled={isSubmitting}
                  >
                    <option value="feature">Feature</option>
                    <option value="bug">Bug</option>
                    <option value="refactor">Refactor</option>
                    <option value="research">Research</option>
                    <option value="test">Test</option>
                    <option value="docs">Docs</option>
                    <option value="security">Security</option>
                    <option value="performance">Performance</option>
                  </select>
                </div>

                <div className="form-field">
                  <label className="form-label" htmlFor="promoteTaskPriority">Priority</label>
                  <select
                    id="promoteTaskPriority"
                    className="form-select"
                    value={promoteTaskPriority}
                    onChange={(e) => setPromoteTaskPriority(e.target.value)}
                    disabled={isSubmitting}
                  >
                    <option value="P0">P0 - Critical</option>
                    <option value="P1">P1 - High</option>
                    <option value="P2">P2 - Medium</option>
                    <option value="P3">P3 - Low</option>
                  </select>
                </div>
              </div>
            )}
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

        {/* Advanced Options (Batch 6) */}
        {mode !== 'quick_task' && (
          <div className="form-section config-section">
            <button
              type="button"
              className="section-title"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%', textAlign: 'left' }}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span style={{ transform: showAdvanced ? 'rotate(90deg)' : 'none', display: 'inline-block', transition: 'transform 0.2s' }}>{'\u25B6'}</span>
              <h3 className="section-title" style={{ margin: 0 }}>Advanced Options</h3>
            </button>

            {showAdvanced && (
              <>
                {/* Language */}
                <div className="form-field" style={{ marginTop: '0.75rem' }}>
                  <label className="form-label" htmlFor="language">Language Profile</label>
                  <select id="language" className="form-select" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={isSubmitting}>
                    <option value="">Auto-detect</option>
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="typescript">TypeScript</option>
                    <option value="nextjs">Next.js</option>
                    <option value="go">Go</option>
                    <option value="rust">Rust</option>
                    <option value="java">Java</option>
                  </select>
                </div>

                {/* Execution */}
                <div className="form-field">
                  <label className="form-label">Execution</label>
                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input type="checkbox" checked={parallel} onChange={(e) => setParallel(e.target.checked)} disabled={isSubmitting} />
                      <span>Parallel execution</span>
                    </label>
                    <label className="checkbox-label">
                      <input type="checkbox" checked={resetState} onChange={(e) => setResetState(e.target.checked)} disabled={isSubmitting} />
                      <span>Reset state (fresh start)</span>
                    </label>
                  </div>
                </div>

                {parallel && (
                  <div className="form-field">
                    <label className="form-label" htmlFor="maxWorkers">Max Workers</label>
                    <input id="maxWorkers" type="number" className="form-input" value={maxWorkers} onChange={(e) => setMaxWorkers(Number(e.target.value))} min={1} max={10} disabled={isSubmitting} />
                  </div>
                )}

                {/* Git */}
                <div className="form-field">
                  <label className="form-label">Git</label>
                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input type="checkbox" checked={requireClean} onChange={(e) => setRequireClean(e.target.checked)} disabled={isSubmitting} />
                      <span>Require clean working tree</span>
                    </label>
                    <label className="checkbox-label">
                      <input type="checkbox" checked={commitEnabled} onChange={(e) => setCommitEnabled(e.target.checked)} disabled={isSubmitting} />
                      <span>Auto-commit changes</span>
                    </label>
                    <label className="checkbox-label">
                      <input type="checkbox" checked={pushEnabled} onChange={(e) => setPushEnabled(e.target.checked)} disabled={isSubmitting} />
                      <span>Auto-push to remote</span>
                    </label>
                  </div>
                </div>

                {/* Limits */}
                <div className="form-row">
                  <div className="form-field">
                    <label className="form-label" htmlFor="shiftMinutes">Timeout (min)</label>
                    <input id="shiftMinutes" type="number" className="form-input" value={shiftMinutes} onChange={(e) => setShiftMinutes(Number(e.target.value))} min={5} disabled={isSubmitting} />
                  </div>
                  <div className="form-field">
                    <label className="form-label" htmlFor="maxTaskAttempts">Max Task Attempts</label>
                    <input id="maxTaskAttempts" type="number" className="form-input" value={maxTaskAttempts} onChange={(e) => setMaxTaskAttempts(Number(e.target.value))} min={1} disabled={isSubmitting} />
                  </div>
                  <div className="form-field">
                    <label className="form-label" htmlFor="maxReviewAttempts">Max Review Attempts</label>
                    <input id="maxReviewAttempts" type="number" className="form-input" value={maxReviewAttempts} onChange={(e) => setMaxReviewAttempts(Number(e.target.value))} min={1} disabled={isSubmitting} />
                  </div>
                </div>

                {/* Quality */}
                <div className="form-row">
                  <div className="form-field">
                    <label className="form-label" htmlFor="ensureRuff">Ruff Linting</label>
                    <select id="ensureRuff" className="form-select" value={ensureRuff} onChange={(e) => setEnsureRuff(e.target.value)} disabled={isSubmitting}>
                      <option value="off">Off</option>
                      <option value="check">Check</option>
                      <option value="fix">Fix</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="form-label" htmlFor="ensureDeps">Dependency Check</label>
                    <select id="ensureDeps" className="form-select" value={ensureDeps} onChange={(e) => setEnsureDeps(e.target.value)} disabled={isSubmitting}>
                      <option value="off">Off</option>
                      <option value="check">Check</option>
                      <option value="install">Install</option>
                    </select>
                  </div>
                </div>

                {ensureDeps !== 'off' && (
                  <div className="form-field">
                    <label className="form-label" htmlFor="ensureDepsCommand">Deps Command</label>
                    <input id="ensureDepsCommand" type="text" className="form-input" value={ensureDepsCommand} onChange={(e) => setEnsureDepsCommand(e.target.value)} placeholder="e.g., pip install -r requirements.txt" disabled={isSubmitting} />
                  </div>
                )}

                {/* Worker */}
                <div className="form-row">
                  <div className="form-field">
                    <label className="form-label" htmlFor="worker">Worker Provider</label>
                    <input id="worker" type="text" className="form-input" value={worker} onChange={(e) => setWorker(e.target.value)} placeholder="Default" disabled={isSubmitting} />
                  </div>
                  <div className="form-field">
                    <label className="form-label" htmlFor="codexCommand">Codex Command</label>
                    <input id="codexCommand" type="text" className="form-input" value={codexCommand} onChange={(e) => setCodexCommand(e.target.value)} placeholder="Auto-detect" disabled={isSubmitting} />
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Submit Button */}
        <div className="form-actions">
          <button type="submit" className="submit-button" disabled={isSubmitting || !content.trim()}>
            {isSubmitting
              ? mode === 'quick_task' ? 'Executing...' : 'Starting...'
              : mode === 'quick_task' ? 'Run Quick Action' : 'Start Run'}
          </button>
        </div>
      </form>
    </div>
  )
}
