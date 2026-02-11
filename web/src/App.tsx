import { useState, useEffect, useMemo, useCallback } from 'react'
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
import CostBreakdown from './components/CostBreakdown'
import BreakpointsPanel from './components/BreakpointsPanel'
import TaskLauncher from './components/TaskLauncher'
import LoadingSpinner from './components/LoadingSpinner'
import SplitPane from './components/SplitPane/SplitPane'
import KanbanBoard from './components/KanbanBoard/KanbanBoard'
import AgentPanel from './components/AgentCard/AgentCard'
import CommandPalette, { useCommandPalette, Command } from './components/CommandPalette/CommandPalette'
import HITLModeSelector from './components/HITLModeSelector/HITLModeSelector'
import NotificationCenter from './components/NotificationCenter/NotificationCenter'
import OnlineUsers from './components/OnlineUsers'
import { ToastProvider } from './contexts/ToastContext'
import { WebSocketProvider, useChannel } from './contexts/WebSocketContext'
import { ThemeProvider, useTheme } from './contexts/ThemeContext'

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

type ViewTab = 'dashboard' | 'board' | 'agents'

const STORAGE_KEY_PROJECT = 'feature-prd-runner-selected-project'
const STORAGE_KEY_TOKEN = 'feature-prd-runner-auth-token'
const STORAGE_KEY_USERNAME = 'feature-prd-runner-username'
const STORAGE_KEY_VIEW = 'feature-prd-runner-view'

function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="theme-toggle-group">
      <button
        className={`theme-btn ${theme === 'light' ? 'active' : ''}`}
        onClick={() => setTheme('light')}
        title="Light mode"
        aria-label="Light mode"
      >
        &#x2600;
      </button>
      <button
        className={`theme-btn ${theme === 'dark' ? 'active' : ''}`}
        onClick={() => setTheme('dark')}
        title="Dark mode"
        aria-label="Dark mode"
      >
        &#x263E;
      </button>
      <button
        className={`theme-btn ${theme === 'system' ? 'active' : ''}`}
        onClick={() => setTheme('system')}
        title="System preference"
        aria-label="System preference"
      >
        &#x2699;
      </button>
    </div>
  )
}

function AppContent() {
  const [status, setStatus] = useState<ProjectStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [currentProject, setCurrentProject] = useState<string | null>(() => {
    return localStorage.getItem(STORAGE_KEY_PROJECT)
  })
  const [showLauncher, setShowLauncher] = useState(false)
  const [activeView, setActiveView] = useState<ViewTab>(() => {
    return (localStorage.getItem(STORAGE_KEY_VIEW) as ViewTab) || 'dashboard'
  })
  const [hitlMode, setHitlMode] = useState('autopilot')

  const { effectiveTheme, toggleTheme } = useTheme()
  const { isOpen: paletteOpen, open: openPalette, close: closePalette } = useCommandPalette()

  // Persist active view
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_VIEW, activeView)
  }, [activeView])

  // Check auth status on mount
  useEffect(() => {
    checkAuthStatus()
  }, [])

  // Fetch project status after auth is checked
  useEffect(() => {
    if (authChecked && isAuthenticated()) {
      fetchStatus()
    }
  }, [currentProject, authChecked])

  useChannel('status', useCallback(() => {
    if (authChecked && isAuthenticated()) {
      fetchStatus()
    }
  }, [currentProject, authChecked]))

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('/api/auth/status')
      if (response.ok) {
        const data = await response.json()
        setAuthStatus(data)

        if (!data.enabled) {
          setAuthChecked(true)
          return
        }

        setAuthChecked(true)
      }
    } catch (err) {
      console.error('Failed to check auth status:', err)
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
    fetchStatus()
  }

  // Command palette commands
  const commands: Command[] = useMemo(() => [
    {
      id: 'nav-board',
      label: 'Go to Task Board',
      category: 'Navigation',
      icon: '&#x25A6;',
      shortcut: 'B',
      action: () => setActiveView('board'),
    },
    {
      id: 'nav-dashboard',
      label: 'Go to Dashboard',
      category: 'Navigation',
      icon: '&#x25A3;',
      shortcut: 'D',
      action: () => setActiveView('dashboard'),
    },
    {
      id: 'nav-agents',
      label: 'Go to Agents',
      category: 'Navigation',
      icon: '&#x2699;',
      shortcut: 'A',
      action: () => setActiveView('agents'),
    },
    {
      id: 'toggle-theme',
      label: `Switch to ${effectiveTheme === 'light' ? 'Dark' : 'Light'} Mode`,
      category: 'Settings',
      icon: effectiveTheme === 'light' ? '&#x263E;' : '&#x2600;',
      shortcut: 'T',
      action: toggleTheme,
    },
    {
      id: 'toggle-launcher',
      label: showLauncher ? 'Hide Task Launcher' : 'Launch New Run',
      category: 'Actions',
      icon: '&#x25B6;',
      shortcut: 'L',
      action: () => { setActiveView('dashboard'); setShowLauncher(!showLauncher) },
    },
    {
      id: 'refresh',
      label: 'Refresh Status',
      category: 'Actions',
      icon: '&#x21BB;',
      shortcut: 'R',
      action: fetchStatus,
    },
  ], [effectiveTheme, toggleTheme, showLauncher])

  // Show login page if auth is enabled and not authenticated
  if (!authChecked) {
    return (
      <div className="app">
        <div className="loading-screen">
          <LoadingSpinner size="lg" label="Checking authentication..." />
        </div>
      </div>
    )
  }

  if (authStatus?.enabled && !isAuthenticated()) {
    return <Login onLoginSuccess={handleLoginSuccess} />
  }

  if (loading) {
    return (
      <div className="app">
        <div className="loading-screen">
          <LoadingSpinner size="lg" label="Loading dashboard..." />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <header className="header">
          <div className="header-left">
            <h1 className="header-title">Feature PRD Runner</h1>
          </div>
          <div className="header-right">
            <ProjectSelector
              currentProject={currentProject}
              onProjectChange={handleProjectChange}
            />
            <ThemeToggle />
            {authStatus?.enabled && (
              <button onClick={handleLogout} className="btn-logout">
                Logout
              </button>
            )}
          </div>
        </header>
        <div className="error">
          <h2>Connection Error</h2>
          <p>{error}</p>
          <p className="hint">
            {currentProject
              ? 'Make sure the selected project has a valid .prd_runner directory'
              : 'Select a project or make sure the backend server is running on port 8080'}
          </p>
          <button onClick={fetchStatus} className="btn btn-primary">
            Retry
          </button>
        </div>
      </div>
    )
  }

  const username = localStorage.getItem(STORAGE_KEY_USERNAME) || authStatus?.username

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="header-title">Feature PRD Runner</h1>
          <nav className="header-nav">
            <button
              className={`nav-tab ${activeView === 'board' ? 'active' : ''}`}
              onClick={() => setActiveView('board')}
            >
              Task Board
            </button>
            <button
              className={`nav-tab ${activeView === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveView('dashboard')}
            >
              Dashboard
            </button>
            <button
              className={`nav-tab ${activeView === 'agents' ? 'active' : ''}`}
              onClick={() => setActiveView('agents')}
            >
              Agents
            </button>
          </nav>
        </div>
        <div className="header-center">
          <ProjectSelector
            currentProject={currentProject}
            onProjectChange={handleProjectChange}
          />
        </div>
        <div className="header-right">
          <button
            className="cmd-k-btn"
            onClick={openPalette}
            title="Command Palette (Cmd+K)"
          >
            <kbd>&#x2318;K</kbd>
          </button>
          <OnlineUsers projectDir={currentProject || undefined} />
          <NotificationCenter />
          <ThemeToggle />
          {username && <span className="header-username">{username}</span>}
          <div className="status-badge" data-status={status?.status}>
            {status?.status || 'unknown'}
          </div>
          {authStatus?.enabled && (
            <button onClick={handleLogout} className="btn-logout">
              Logout
            </button>
          )}
        </div>
      </header>

      {/* Main content area */}
      {activeView === 'board' ? (
        <div className="main-content">
          <KanbanBoard projectDir={currentProject || undefined} />
        </div>
      ) : activeView === 'agents' ? (
        <div className="main-content main-content-scroll">
          <div className="agents-header-bar">
            <HITLModeSelector
              currentMode={hitlMode}
              onModeChange={setHitlMode}
              projectDir={currentProject || undefined}
            />
          </div>
          <AgentPanel projectDir={currentProject || undefined} />
        </div>
      ) : status?.run_id ? (
        <SplitPane
          defaultLeftWidth={65}
          minLeftWidth={40}
          maxLeftWidth={80}
          className="dashboard-split"
          left={
            <div className="container">
              <RunDashboard status={status} />

              {/* Task Launcher Section */}
              <div className="launcher-toggle-section">
                <button
                  onClick={() => setShowLauncher(!showLauncher)}
                  className={`btn-launcher-toggle ${showLauncher ? 'active' : ''}`}
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
                  <MetricsPanel projectDir={currentProject || undefined} />
                </div>
              </div>

              <MetricsChart projectDir={currentProject || undefined} />

              <CostBreakdown projectDir={currentProject || undefined} />

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
            </div>
          }
          right={
            <div className="split-log-panel">
              <LiveLog runId={status.run_id} projectDir={currentProject || undefined} />
            </div>
          }
        />
      ) : (
        <div className="container">
          <RunDashboard status={status} />

          {/* Task Launcher Section */}
          <div className="launcher-toggle-section">
            <button
              onClick={() => setShowLauncher(!showLauncher)}
              className={`btn-launcher-toggle ${showLauncher ? 'active' : ''}`}
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
              <MetricsPanel projectDir={currentProject || undefined} />
            </div>
          </div>

          <MetricsChart projectDir={currentProject || undefined} />

          <CostBreakdown projectDir={currentProject || undefined} />

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
        </div>
      )}

      <Chat runId={status?.run_id} projectDir={currentProject || undefined} />

      {/* Command Palette */}
      <CommandPalette
        commands={commands}
        isOpen={paletteOpen}
        onClose={closePalette}
      />
    </div>
  )
}

function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <WebSocketProvider>
          <AppContent />
        </WebSocketProvider>
      </ToastProvider>
    </ThemeProvider>
  )
}

export default App
