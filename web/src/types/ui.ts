export type CockpitView = 'overview' | 'execution' | 'tasks' | 'agents' | 'diagnostics'

export interface AppNavSection {
  id: CockpitView
  label: string
  description: string
}

export interface GlobalStatusSummary {
  rawStatus: string
  label: string
  severity: 'success' | 'warning' | 'error' | 'info' | 'default'
  color: 'success' | 'warning' | 'error' | 'info' | 'default' | 'primary'
  icon: string
}

export interface CockpitPanelState {
  showLauncher: boolean
  showLiveLog: boolean
}

export type TaskDetailTab = 'summary' | 'dependencies' | 'logs' | 'interventions'

export interface DashboardLayoutConfig {
  now: string[]
  flow: string[]
  insights: string[]
}
