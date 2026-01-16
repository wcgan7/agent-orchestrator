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

const STORAGE_KEY = 'feature-prd-runner-selected-project'

function App() {
  const [status, setStatus] = useState<ProjectStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [currentProject, setCurrentProject] = useState<string | null>(() => {
    // Load from local storage on mount
    return localStorage.getItem(STORAGE_KEY)
  })

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000) // Poll every 5 seconds
    return () => clearInterval(interval)
  }, [currentProject])

  const fetchStatus = async () => {
    try {
      const url = currentProject
        ? `/api/status?project_dir=${encodeURIComponent(currentProject)}`
        : '/api/status'

      const response = await fetch(url)
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
    localStorage.setItem(STORAGE_KEY, projectPath)
    setLoading(true)
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
          <ProjectSelector
            currentProject={currentProject}
            onProjectChange={handleProjectChange}
          />
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

  return (
    <div className="app">
      <header className="header">
        <h1>Feature PRD Runner Dashboard</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <ProjectSelector
            currentProject={currentProject}
            onProjectChange={handleProjectChange}
          />
          <div className="status-badge" data-status={status?.status}>
            {status?.status || 'unknown'}
          </div>
        </div>
      </header>

      <div className="container">
        <RunDashboard status={status} />

        <ControlPanel
          currentTaskId={status?.current_task_id}
          currentPhaseId={status?.current_phase_id}
          status={status?.status}
        />

        <div className="grid">
          <div className="col-2">
            <PhaseTimeline />
          </div>
          <div className="col-2">
            <MetricsPanel status={status} />
          </div>
        </div>

        <MetricsChart />

        <DependencyGraph />

        {status?.run_id && (
          <LiveLog runId={status.run_id} />
        )}
      </div>
    </div>
  )
}

export default App
