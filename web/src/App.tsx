import { useState, useEffect, useMemo, useCallback, lazy, Suspense } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  IconButton,
  Stack,
  Tab,
  Tabs,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import LightModeIcon from '@mui/icons-material/LightMode'
import DarkModeIcon from '@mui/icons-material/DarkMode'
import SettingsSuggestIcon from '@mui/icons-material/SettingsSuggest'
import LogoutIcon from '@mui/icons-material/Logout'
import RefreshIcon from '@mui/icons-material/Refresh'
import ProjectSelector from './components/ProjectSelector'
import Login from './components/Login'
import RunDashboard from './components/RunDashboard'
import PhaseTimeline from './components/PhaseTimeline'
import LiveLog from './components/LiveLog'
import MetricsPanel from './components/MetricsPanel'
import ControlPanel from './components/ControlPanel'
import ApprovalGate from './components/ApprovalGate'
import Chat from './components/Chat'
import TasksPanel from './components/TasksPanel'
import RunsPanel from './components/RunsPanel'
import BreakpointsPanel from './components/BreakpointsPanel'
import TaskLauncher from './components/TaskLauncher'
import LoadingSpinner from './components/LoadingSpinner'
import CommandPalette, { useCommandPalette, Command } from './components/CommandPalette/CommandPalette'
import NotificationCenter from './components/NotificationCenter/NotificationCenter'
import OnlineUsers from './components/OnlineUsers'
import { ToastProvider } from './contexts/ToastContext'
import { WebSocketProvider, useChannel } from './contexts/WebSocketContext'
import { ThemeProvider as AppThemeProvider, useTheme } from './contexts/ThemeContext'
import type { ThemeMode } from './contexts/ThemeContext'
import { ThemeProvider as MuiThemeProvider, CssBaseline } from '@mui/material'
import { createCockpitTheme } from './ui/theme'
import AppShell from './ui/layout/AppShell'
import { mapStatusSummary } from './ui/status'
import type {
  AppNavSection,
  CockpitPanelState,
  CockpitView,
  DashboardLayoutConfig,
  TaskDetailTab,
} from './types/ui'

const DryRunPanel = lazy(() => import('./components/DryRunPanel'))
const DoctorPanel = lazy(() => import('./components/DoctorPanel'))
const WorkersPanel = lazy(() => import('./components/WorkersPanel'))
const ParallelPlanView = lazy(() => import('./components/ParallelPlanView'))
const RequirementForm = lazy(() => import('./components/RequirementForm'))
const KanbanBoard = lazy(() => import('./components/KanbanBoard/KanbanBoard'))
const AgentPanel = lazy(() => import('./components/AgentCard/AgentCard'))
const HITLModeSelector = lazy(() => import('./components/HITLModeSelector/HITLModeSelector'))
const MetricsChart = lazy(() => import('./components/MetricsChart'))
const DependencyGraph = lazy(() => import('./components/DependencyGraph'))
const FileReview = lazy(() => import('./components/FileReview'))
const CostBreakdown = lazy(() => import('./components/CostBreakdown'))

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
const STORAGE_KEY_VIEW = 'feature-prd-runner-view'
const STORAGE_KEY_TASK_TAB = 'feature-prd-runner-task-tab'

const navSections: AppNavSection[] = [
  { id: 'overview', label: 'Overview', description: 'Current run health and priorities' },
  { id: 'execution', label: 'Execution', description: 'Run controls, approvals, and live activity' },
  { id: 'tasks', label: 'Tasks', description: 'Board, dependencies, and task interventions' },
  { id: 'agents', label: 'Agents', description: 'Agent orchestration and worker capacity' },
  { id: 'diagnostics', label: 'Diagnostics', description: 'Dry-run, doctor, cost, and metrics' },
]

const dashboardLayoutConfig: DashboardLayoutConfig = {
  now: ['run_dashboard', 'approval_gate', 'file_review'],
  flow: ['control_panel', 'phase_timeline', 'tasks_runs'],
  insights: ['metrics', 'cost', 'dependency_parallel'],
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <ToggleButtonGroup
      size="small"
      value={theme}
      exclusive
      onChange={(_, next: ThemeMode | null) => {
        if (next) setTheme(next)
      }}
      aria-label="Theme selection"
    >
      <ToggleButton value="light" aria-label="Light mode">
        <LightModeIcon fontSize="small" />
      </ToggleButton>
      <ToggleButton value="dark" aria-label="Dark mode">
        <DarkModeIcon fontSize="small" />
      </ToggleButton>
      <ToggleButton value="system" aria-label="System mode">
        <SettingsSuggestIcon fontSize="small" />
      </ToggleButton>
    </ToggleButtonGroup>
  )
}

function ActivityRail({
  status,
  currentProject,
}: {
  status: ProjectStatus | null
  currentProject: string | null
}) {
  const statusSummary = mapStatusSummary(status?.status)

  return (
    <Stack spacing={2}>
      <Card>
        <CardContent>
          <Typography variant="overline" color="text.secondary">Run Health</Typography>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ mt: 0.5 }}>
            <Chip size="small" label={statusSummary.label} color={statusSummary.color} />
            <Typography variant="body2" color="text.secondary">
              Phase {status?.phases_completed ?? 0}/{status?.phases_total ?? 0}
            </Typography>
          </Stack>
          <Stack direction="row" spacing={1} sx={{ mt: 1.5, flexWrap: 'wrap' }}>
            <Chip size="small" color="success" variant="outlined" label={`Done ${status?.tasks_done ?? 0}`} />
            <Chip size="small" color="info" variant="outlined" label={`Running ${status?.tasks_running ?? 0}`} />
            <Chip size="small" color="warning" variant="outlined" label={`Ready ${status?.tasks_ready ?? 0}`} />
            <Chip size="small" color="error" variant="outlined" label={`Blocked ${status?.tasks_blocked ?? 0}`} />
          </Stack>
          {status?.current_phase_id && (
            <Typography variant="body2" sx={{ mt: 1.5 }}>
              Current: <strong>{status.current_phase_id}</strong>
            </Typography>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="overline" color="text.secondary">Approvals and Risk</Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Blocked tasks: <strong>{status?.tasks_blocked ?? 0}</strong>
          </Typography>
          <Typography variant="body2">
            Review task: <strong>{status?.current_task_id || 'none'}</strong>
          </Typography>
          {status?.last_error && (
            <Alert severity="error" sx={{ mt: 1.5 }}>
              {status.last_error}
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="overline" color="text.secondary">Context</Typography>
          <Typography variant="body2" sx={{ mt: 1, wordBreak: 'break-word' }}>
            {currentProject || 'No project selected'}
          </Typography>
          <Box sx={{ mt: 1.5 }}>
            <OnlineUsers projectDir={currentProject || undefined} />
          </Box>
        </CardContent>
      </Card>
    </Stack>
  )
}

function CockpitOverview({
  status,
  currentProject,
}: {
  status: ProjectStatus | null
  currentProject: string | null
}) {
  return (
    <Stack spacing={2.5}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="h5">Overview</Typography>
        <Chip size="small" variant="outlined" label="Now / Flow / Insights" />
      </Stack>

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, xl: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1.5 }}>Now</Typography>
              <RunDashboard status={status} />
              <Divider sx={{ my: 2 }} />
              <ApprovalGate projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <FileReview
                taskId={status?.current_task_id}
                projectDir={currentProject || undefined}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, xl: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1.5 }}>Flow</Typography>
              <ControlPanel
                currentTaskId={status?.current_task_id}
                currentPhaseId={status?.current_phase_id}
                status={status?.status}
                projectDir={currentProject || undefined}
              />
              <Divider sx={{ my: 2 }} />
              <PhaseTimeline projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <Grid container spacing={2}>
                <Grid size={{ xs: 12 }}>
                  <TasksPanel
                    projectDir={currentProject || undefined}
                    currentTaskId={status?.current_task_id}
                  />
                </Grid>
                <Grid size={{ xs: 12 }}>
                  <RunsPanel
                    projectDir={currentProject || undefined}
                    currentRunId={status?.run_id}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, xl: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1.5 }}>Insights</Typography>
              <MetricsPanel projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <MetricsChart projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <CostBreakdown projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <DependencyGraph projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <ParallelPlanView projectDir={currentProject || undefined} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Stack>
  )
}

function AppContent() {
  const [status, setStatus] = useState<ProjectStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null)
  const [authChecked, setAuthChecked] = useState(false)
  const [currentProject, setCurrentProject] = useState<string | null>(() => localStorage.getItem(STORAGE_KEY_PROJECT))
  const [activeView, setActiveView] = useState<CockpitView>(() => {
    const stored = localStorage.getItem(STORAGE_KEY_VIEW) as CockpitView | null
    return stored || 'overview'
  })
  const [taskTab, setTaskTab] = useState<TaskDetailTab>(() => {
    const stored = localStorage.getItem(STORAGE_KEY_TASK_TAB) as TaskDetailTab | null
    return stored || 'summary'
  })
  const [hitlMode, setHitlMode] = useState('autopilot')
  const [panelState, setPanelState] = useState<CockpitPanelState>({
    showLauncher: false,
    showLiveLog: true,
  })

  const { effectiveTheme, toggleTheme } = useTheme()
  const { isOpen: paletteOpen, open: openPalette, close: closePalette } = useCommandPalette()

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_VIEW, activeView)
  }, [activeView])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_TASK_TAB, taskTab)
  }, [taskTab])

  useEffect(() => {
    checkAuthStatus()
  }, [])

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
        setAuthChecked(true)
      }
    } catch {
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
    setAuthStatus({ enabled: true, authenticated: false, username: null })
  }

  const fetchStatus = useCallback(async () => {
    try {
      const url = currentProject
        ? `/api/status?project_dir=${encodeURIComponent(currentProject)}`
        : '/api/status'

      const headers: HeadersInit = {}
      const token = localStorage.getItem(STORAGE_KEY_TOKEN)
      if (token) headers.Authorization = `Bearer ${token}`

      const response = await fetch(url, { headers })
      if (!response.ok) throw new Error(`HTTP error ${response.status}`)

      const data = await response.json()
      setStatus(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status')
    } finally {
      setLoading(false)
    }
  }, [currentProject])

  const handleProjectChange = (projectPath: string) => {
    setCurrentProject(projectPath)
    localStorage.setItem(STORAGE_KEY_PROJECT, projectPath)
    setLoading(true)
  }

  const handleRunStarted = () => {
    setPanelState((prev) => ({ ...prev, showLauncher: false }))
    fetchStatus()
  }

  const commands: Command[] = useMemo(() => [
    {
      id: 'nav-overview',
      label: 'Go to Overview',
      category: 'Navigation',
      icon: 'O',
      shortcut: 'O',
      action: () => setActiveView('overview'),
    },
    {
      id: 'nav-execution',
      label: 'Go to Execution',
      category: 'Navigation',
      icon: 'E',
      shortcut: 'E',
      action: () => setActiveView('execution'),
    },
    {
      id: 'nav-tasks',
      label: 'Go to Tasks',
      category: 'Navigation',
      icon: 'T',
      shortcut: 'T',
      action: () => setActiveView('tasks'),
    },
    {
      id: 'nav-agents',
      label: 'Go to Agents',
      category: 'Navigation',
      icon: 'A',
      shortcut: 'A',
      action: () => setActiveView('agents'),
    },
    {
      id: 'nav-diagnostics',
      label: 'Go to Diagnostics',
      category: 'Navigation',
      icon: 'D',
      shortcut: 'G',
      action: () => setActiveView('diagnostics'),
    },
    {
      id: 'toggle-theme',
      label: `Switch to ${effectiveTheme === 'light' ? 'Dark' : 'Light'} Mode`,
      category: 'Settings',
      icon: effectiveTheme === 'light' ? 'M' : 'S',
      shortcut: 'Shift+T',
      action: toggleTheme,
    },
    {
      id: 'toggle-launcher',
      label: panelState.showLauncher ? 'Hide Task Launcher' : 'Launch New Run',
      category: 'Actions',
      icon: 'L',
      shortcut: 'L',
      action: () => {
        setActiveView('execution')
        setPanelState((prev) => ({ ...prev, showLauncher: !prev.showLauncher }))
      },
    },
    {
      id: 'toggle-log',
      label: panelState.showLiveLog ? 'Hide Live Log' : 'Show Live Log',
      category: 'Actions',
      icon: 'R',
      shortcut: 'V',
      action: () => setPanelState((prev) => ({ ...prev, showLiveLog: !prev.showLiveLog })),
    },
    {
      id: 'refresh',
      label: 'Refresh Status',
      category: 'Actions',
      icon: 'R',
      shortcut: 'R',
      action: fetchStatus,
    },
  ], [effectiveTheme, toggleTheme, panelState, fetchStatus])

  if (!authChecked) {
    return (
      <Box component="main" sx={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
        <LoadingSpinner size="lg" label="Checking authentication..." />
      </Box>
    )
  }

  if (authStatus?.enabled && !isAuthenticated()) {
    return (
      <Box component="main">
        <Login onLoginSuccess={handleLoginSuccess} />
      </Box>
    )
  }

  if (loading) {
    return (
      <Box component="main" sx={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
        <Stack spacing={2} alignItems="center">
          <CircularProgress size={28} />
          <Typography color="text.secondary">Loading cockpit...</Typography>
        </Stack>
      </Box>
    )
  }

  if (error) {
    return (
      <Box component="main" sx={{ p: 3, maxWidth: 760, mx: 'auto', mt: 10 }}>
        <Typography variant="h5" sx={{ mb: 1 }}>
          Connection Error
        </Typography>
        <Alert severity="error" sx={{ mb: 2 }}>
          Connection error: {error}
        </Alert>
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          {currentProject
            ? 'Ensure the selected project has a valid .prd_runner directory.'
            : 'Select a project or ensure the backend server is running on port 8080.'}
        </Typography>
        <Button variant="contained" onClick={fetchStatus}>Retry</Button>
      </Box>
    )
  }

  const username = localStorage.getItem(STORAGE_KEY_USERNAME) || authStatus?.username
  const statusSummary = mapStatusSummary(status?.status)

  const executionContent = (
    <Stack spacing={2.5}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={1.5}>
        <Typography variant="h5">Execution</Typography>
        <Stack direction="row" spacing={1}>
          <Button
            variant={panelState.showLauncher ? 'outlined' : 'contained'}
            onClick={() => setPanelState((prev) => ({ ...prev, showLauncher: !prev.showLauncher }))}
          >
            {panelState.showLauncher ? 'Hide Launcher' : 'Launch New Run'}
          </Button>
          <Button
            variant="outlined"
            onClick={() => setPanelState((prev) => ({ ...prev, showLiveLog: !prev.showLiveLog }))}
          >
            {panelState.showLiveLog ? 'Hide Live Log' : 'Show Live Log'}
          </Button>
        </Stack>
      </Stack>

      {panelState.showLauncher && (
        <Card>
          <CardContent>
            <TaskLauncher projectDir={currentProject} onRunStarted={handleRunStarted} />
          </CardContent>
        </Card>
      )}

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, lg: panelState.showLiveLog ? 8 : 12 }}>
          <Stack spacing={2}>
            <Card>
              <CardContent>
                <ControlPanel
                  currentTaskId={status?.current_task_id}
                  currentPhaseId={status?.current_phase_id}
                  status={status?.status}
                  projectDir={currentProject || undefined}
                />
                <Divider sx={{ my: 2 }} />
                <ApprovalGate projectDir={currentProject || undefined} />
                <Divider sx={{ my: 2 }} />
                <FileReview taskId={status?.current_task_id} projectDir={currentProject || undefined} />
                <Divider sx={{ my: 2 }} />
                <BreakpointsPanel projectDir={currentProject || undefined} />
              </CardContent>
            </Card>
          </Stack>
        </Grid>
        {panelState.showLiveLog && (
          <Grid size={{ xs: 12, lg: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 1.5 }}>Live Log</Typography>
                {status?.run_id ? (
                  <LiveLog runId={status.run_id} projectDir={currentProject || undefined} />
                ) : (
                  <Alert severity="info">Start a run to stream logs.</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Stack>
  )

  const taskContent = (
    <Stack spacing={2.5}>
      <Typography variant="h5">Tasks</Typography>
      <Card>
        <CardContent>
          <KanbanBoard projectDir={currentProject || undefined} />
        </CardContent>
      </Card>
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 1.5 }}>Task Detail Workbench</Typography>
          <Tabs
            value={taskTab}
            onChange={(_, value: TaskDetailTab) => setTaskTab(value)}
            variant="scrollable"
            allowScrollButtonsMobile
          >
            <Tab value="summary" label="Summary" />
            <Tab value="dependencies" label="Dependencies" />
            <Tab value="logs" label="Logs" />
            <Tab value="interventions" label="Interventions" />
          </Tabs>
          <Divider sx={{ my: 2 }} />

          {taskTab === 'summary' && (
            <Grid container spacing={2}>
              <Grid size={{ xs: 12, lg: 6 }}>
                <TasksPanel
                  projectDir={currentProject || undefined}
                  currentTaskId={status?.current_task_id}
                />
              </Grid>
              <Grid size={{ xs: 12, lg: 6 }}>
                <RunsPanel
                  projectDir={currentProject || undefined}
                  currentRunId={status?.run_id}
                />
              </Grid>
            </Grid>
          )}

          {taskTab === 'dependencies' && (
            <Stack spacing={2}>
              <DependencyGraph projectDir={currentProject || undefined} />
              <ParallelPlanView projectDir={currentProject || undefined} />
            </Stack>
          )}

          {taskTab === 'logs' && (
            status?.run_id
              ? <LiveLog runId={status.run_id} projectDir={currentProject || undefined} />
              : <Alert severity="info">No active run logs available.</Alert>
          )}

          {taskTab === 'interventions' && (
            <Stack spacing={2}>
              <RequirementForm projectDir={currentProject || undefined} />
              <Typography variant="body2" color="text.secondary">
                Use the floating chat entry point to collaborate with the worker.
              </Typography>
            </Stack>
          )}
        </CardContent>
      </Card>
    </Stack>
  )

  const agentsContent = (
    <Stack spacing={2.5}>
      <Typography variant="h5">Agents</Typography>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, lg: 4 }}>
          <Stack spacing={2}>
            <Card>
              <CardContent>
                <HITLModeSelector
                  currentMode={hitlMode}
                  onModeChange={setHitlMode}
                  projectDir={currentProject || undefined}
                />
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <WorkersPanel projectDir={currentProject || undefined} />
              </CardContent>
            </Card>
          </Stack>
        </Grid>
        <Grid size={{ xs: 12, lg: 8 }}>
          <Card>
            <CardContent>
              <AgentPanel projectDir={currentProject || undefined} />
              <Divider sx={{ my: 2 }} />
              <RequirementForm projectDir={currentProject || undefined} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Stack>
  )

  const diagnosticsContent = (
    <Stack spacing={2.5}>
      <Typography variant="h5">Diagnostics</Typography>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, lg: 6 }}>
          <Card><CardContent><DryRunPanel projectDir={currentProject || undefined} /></CardContent></Card>
        </Grid>
        <Grid size={{ xs: 12, lg: 6 }}>
          <Card><CardContent><DoctorPanel projectDir={currentProject || undefined} /></CardContent></Card>
        </Grid>
      </Grid>
      <Card><CardContent><MetricsPanel projectDir={currentProject || undefined} /></CardContent></Card>
      <Card><CardContent><MetricsChart projectDir={currentProject || undefined} /></CardContent></Card>
      <Card><CardContent><CostBreakdown projectDir={currentProject || undefined} /></CardContent></Card>
    </Stack>
  )

  return (
    <>
      <Box className="sr-only" aria-live="polite" aria-atomic="true">
        {`Run status ${statusSummary.label}`}
      </Box>

      <AppShell
        title="Feature PRD Runner"
        sections={navSections}
        activeSection={activeView}
        onSectionChange={setActiveView}
        statusSummary={statusSummary}
        commandHint="Cmd/Ctrl + K"
        onOpenCommandPalette={openPalette}
        commandBarCenter={(
          <Box sx={{ width: { xs: '100%', md: 420 }, maxWidth: '100%' }}>
            <ProjectSelector
              currentProject={currentProject}
              onProjectChange={handleProjectChange}
            />
          </Box>
        )}
        commandBarRight={(
          <Stack direction="row" spacing={1.2} alignItems="center">
            <IconButton aria-label="Refresh status" onClick={fetchStatus}>
              <RefreshIcon />
            </IconButton>
            <NotificationCenter />
            <ThemeToggle />
            {username && (
              <Chip size="small" variant="outlined" label={username} />
            )}
            {authStatus?.enabled && (
              <Button
                size="small"
                color="error"
                variant="outlined"
                startIcon={<LogoutIcon />}
                onClick={handleLogout}
              >
                Logout
              </Button>
            )}
          </Stack>
        )}
        rightRail={<ActivityRail status={status} currentProject={currentProject} />}
      >
        <Suspense
          fallback={(
            <Box sx={{ minHeight: 220, display: 'grid', placeItems: 'center' }}>
              <LoadingSpinner label="Loading section..." />
            </Box>
          )}
        >
          {activeView === 'overview' && <CockpitOverview status={status} currentProject={currentProject} />}
          {activeView === 'execution' && executionContent}
          {activeView === 'tasks' && taskContent}
          {activeView === 'agents' && agentsContent}
          {activeView === 'diagnostics' && diagnosticsContent}
        </Suspense>
      </AppShell>

      <CommandPalette
        commands={commands}
        isOpen={paletteOpen}
        onClose={closePalette}
      />

      <Chat runId={status?.run_id} projectDir={currentProject || undefined} />

      <Box sx={{ display: 'none' }}>
        {dashboardLayoutConfig.now.join(',')}
        {dashboardLayoutConfig.flow.join(',')}
        {dashboardLayoutConfig.insights.join(',')}
      </Box>
    </>
  )
}

function CockpitThemeRoot() {
  const { effectiveTheme } = useTheme()
  const muiTheme = useMemo(() => createCockpitTheme(effectiveTheme), [effectiveTheme])

  return (
    <MuiThemeProvider theme={muiTheme}>
      <CssBaseline />
      <ToastProvider>
        <WebSocketProvider>
          <AppContent />
        </WebSocketProvider>
      </ToastProvider>
    </MuiThemeProvider>
  )
}

function App() {
  return (
    <AppThemeProvider>
      <CockpitThemeRoot />
    </AppThemeProvider>
  )
}

export default App
