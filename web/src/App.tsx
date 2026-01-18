import { useState, useEffect } from 'react'
import './App.css'
import RunDashboard from './components/RunDashboard'
import PhaseTimeline from './components/PhaseTimeline'
import LiveLog from './components/LiveLog'
import MetricsPanel from './components/MetricsPanel'
import MetricsChart from './components/MetricsChart'
import ControlPanel from './components/ControlPanel'
import DependencyGraph from './components/DependencyGraph'
import ProjectSelector from './components/ProjectSelector'
import Login from './components/Login'
import ApprovalGate from './components/ApprovalGate'
import Chat from './components/Chat'
import FileReview from './components/FileReview'
import TasksPanel from './components/TasksPanel'
import RunsPanel from './components/RunsPanel'
import BreakpointsPanel from './components/BreakpointsPanel'
import TaskLauncher from './components/TaskLauncher'

interface ProjectStatus {
  project_dir: string
  status: string
  current_task_id?: string
  current_phase_id?: string
  run_id?: string
  last_error?: string
  phases_completed: number
  phases_total: number
  tasks_ready: number
  tasks_running: number
  tasks_done: number
  tasks_blocked: number
}

interface AuthStatus {
  enabled: boolean
  authenticated: boolean
  username: string | null
}

const STORAGE_KEY_PROJECT = 'feature-prd-runner-selected-project'
const STORAGE_KEY_TOKEN = 'feature-prd-runner-auth-token'
const STORAGE_KEY_USERNAME = 'feature-prd-runner-username'

function App() {
  const [status, setStatus] = useState<ProjectStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [currentProject, setCurrentProject] = useState<string | null>(() => {
    return localStorage.getItem(STORAGE_KEY_PROJECT)
  })
  const [showLauncher, setShowLauncher] = useState(false)

  // Check auth status on mount
  useEffect(() => {
    checkAuthStatus()
  }, [])

  // Fetch project status after auth is checked
  useEffect(() => {
    if (authChecked && isAuthenticated()) {
      fetchStatus()
      const interval = setInterval(fetchStatus, 5000)
      return () => clearInterval(interval)
    }
  }, [currentProject, authChecked])

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('/api/auth/status')
      if (response.ok) {
        const data = await response.json()
        setAuthStatus(data)

        // If auth is disabled, we're automatically authenticated
        if (!data.enabled) {
          setAuthChecked(true)
          return
        }

        // Check if we have a stored token
        const token = localStorage.getItem(STORAGE_KEY_TOKEN)
        if (token) {
          // Token exists, assume authenticated for now
          // (could add token validation here)
          setAuthChecked(true)
        } else {
          setAuthChecked(true)
        }
      }
    } catch (err) {
      console.error('Failed to check auth status:', err)
      // On error, assume no auth required
      setAuthStatus({ enabled: false, authenticated: true, username: null })
      setAuthChecked(true)
    }
  }

  const isAuthenticated = (): boolean => {
    if (!authStatus) return false
    if (!authStatus.enabled) return true
    return !!localStorage.getItem(STORAGE_KEY_TOKEN)
  }

  const handleLoginSuccess = (token: string, username: string) => {
    localStorage.setItem(STORAGE_KEY_TOKEN, token)
    localStorage.setItem(STORAGE_KEY_USERNAME, username)
    setAuthStatus({
      enabled: true,
      authenticated: true,
      username,
    })
    setLoading(true)
  }

  const handleLogout = () => {
    localStorage.removeItem(STORAGE_KEY_TOKEN)
    localStorage.removeItem(STORAGE_KEY_USERNAME)
    setAuthStatus({
      enabled: true,
      authenticated: false,
      username: null,
    })
  }

  const fetchStatus = async () => {
    try {
      const url = currentProject
        ? `/api/status?project_dir=${encodeURIComponent(currentProject)}`
        : '/api/status'

      const headers: HeadersInit = {}
      const token = localStorage.getItem(STORAGE_KEY_TOKEN)
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(url, { headers })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setStatus(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status')
    } finally {
      setLoading(false)
    }
  }

  const handleProjectChange = (projectPath: string) => {
    setCurrentProject(projectPath)
    localStorage.setItem(STORAGE_KEY_PROJECT, projectPath)
    setLoading(true)
  }

  const handleRunStarted = (_runId: string) => {
    setShowLauncher(false)
    // Refresh status to show the new run
    fetchStatus()
  }

  // Show login page if auth is enabled and not authenticated
  if (!authChecked) {
    return (
      <div className="app">
        <div className="loading">Checking authentication...</div>
      </div>
    )
  }

  if (authStatus?.enabled && !isAuthenticated()) {
    return <Login onLoginSuccess={handleLoginSuccess} />
  }

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <header className="header">
          <h1>Feature PRD Runner Dashboard</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <ProjectSelector
              currentProject={currentProject}
              onProjectChange={handleProjectChange}
            />
            {authStatus?.enabled && (
              <button
                onClick={handleLogout}
                style={{
                  padding: '0.5rem 1rem',
                  background: '#f44336',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '0.875rem',
                  cursor: 'pointer',
                }}
              >
                Logout
              </button>
            )}
          </div>
        </header>
        <div className="error">
          <h2>Error</h2>
          <p>{error}</p>
          <p className="hint">
            {currentProject
              ? 'Make sure the selected project has a valid .prd_runner directory'
              : 'Select a project or make sure the backend server is running on port 8080'}
          </p>
          <button onClick={fetchStatus}>Retry</button>
        </div>
      </div>
    )
  }

  const username = localStorage.getItem(STORAGE_KEY_USERNAME) || authStatus?.username

  return (
    <div className="app">
      <header className="header">
        <h1>Feature PRD Runner Dashboard</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {username && (
            <div style={{ fontSize: '0.875rem', color: '#666' }}>
              {username}
            </div>
          )}
          <ProjectSelector
            currentProject={currentProject}
            onProjectChange={handleProjectChange}
          />
          <div className="status-badge" data-status={status?.status}>
            {status?.status || 'unknown'}
          </div>
          {authStatus?.enabled && (
            <button
              onClick={handleLogout}
              style={{
                padding: '0.5rem 1rem',
                background: '#f44336',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                fontSize: '0.875rem',
                cursor: 'pointer',
              }}
            >
              Logout
            </button>
          )}
        </div>
      </header>

      <div className="container">
        <RunDashboard status={status} />

        {/* Task Launcher Section */}
        <div style={{ marginBottom: '1.5rem' }}>
          <button
            onClick={() => setShowLauncher(!showLauncher)}
            style={{
              padding: '0.75rem 1.5rem',
              background: showLauncher ? '#666' : '#007bff',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              fontSize: '0.95rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            {showLauncher ? 'Hide Task Launcher' : 'Launch New Run'}
          </button>
        </div>

        {showLauncher && (
          <TaskLauncher
            projectDir={currentProject}
            onRunStarted={handleRunStarted}
          />
        )}

        <ControlPanel
          currentTaskId={status?.current_task_id}
          currentPhaseId={status?.current_phase_id}
          status={status?.status}
          projectDir={currentProject || undefined}
        />

        <ApprovalGate projectDir={currentProject || undefined} />

        <FileReview
          taskId={status?.current_task_id}
          projectDir={currentProject || undefined}
        />

        <div className="grid">
          <div className="col-2">
            <PhaseTimeline projectDir={currentProject || undefined} />
          </div>
          <div className="col-2">
            <MetricsPanel status={status} projectDir={currentProject || undefined} />
          </div>
        </div>

        <MetricsChart projectDir={currentProject || undefined} />

        <DependencyGraph projectDir={currentProject || undefined} />

        <div className="grid">
          <div className="col-2">
            <TasksPanel
              projectDir={currentProject || undefined}
              currentTaskId={status?.current_task_id}
            />
          </div>
          <div className="col-2">
            <RunsPanel
              projectDir={currentProject || undefined}
              currentRunId={status?.run_id}
            />
          </div>
        </div>

        <BreakpointsPanel projectDir={currentProject || undefined} />

        {status?.run_id && (
          <LiveLog runId={status.run_id} projectDir={currentProject || undefined} />
        )}
      </div>

      <Chat runId={status?.run_id} projectDir={currentProject || undefined} />
    </div>
  )
}

export default App
