import { FormEvent, useEffect, useState } from 'react'
import { buildApiUrl } from './api'
import { ImportJobPanel } from './components/AppPanels/ImportJobPanel'
import { QuickActionDetailPanel } from './components/AppPanels/QuickActionDetailPanel'
import { TaskExplorerPanel } from './components/AppPanels/TaskExplorerPanel'
import { humanizeLabel } from './ui/labels'
import './styles/orchestrator.css'

type RouteKey = 'board' | 'execution' | 'review' | 'agents' | 'settings'
type CreateTab = 'task' | 'import' | 'quick'

type TaskRecord = {
  id: string
  title: string
  description?: string
  task_type?: string
  priority: string
  status: string
  labels?: string[]
  approval_mode?: 'human_review' | 'auto_approve'
  blocked_by?: string[]
  blocks?: string[]
  parent_id?: string | null
  pipeline_template?: string[]
  retry_count?: number
  metadata?: Record<string, unknown>
}

type BoardResponse = {
  columns: Record<string, TaskRecord[]>
}

type PreviewNode = {
  id: string
  title: string
  priority: string
}

type PreviewEdge = {
  from: string
  to: string
}

type PrdPreview = {
  nodes: PreviewNode[]
  edges: PreviewEdge[]
}

type OrchestratorStatus = {
  status: string
  queue_depth: number
  in_progress: number
  draining: boolean
  run_branch?: string | null
}

type AgentRecord = {
  id: string
  role: string
  status: string
  capacity: number
  override_provider?: string | null
}

type ProjectRef = {
  id: string
  path: string
  source: string
  is_git: boolean
}

type PinnedProjectRef = {
  id: string
  path: string
  pinned_at?: string
}

type QuickActionRecord = {
  id: string
  prompt: string
  status: string
  started_at?: string | null
  finished_at?: string | null
  result_summary?: string | null
  promoted_task_id?: string | null
  kind?: string | null
  command?: string | null
  exit_code?: number | null
}

type ImportJobRecord = {
  id: string
  project_id?: string
  title?: string
  status?: string
  created_at?: string
  created_task_ids?: string[]
  tasks?: Array<{ title?: string; priority?: string }>
}

type BrowseDirectoryEntry = {
  name: string
  path: string
  is_git: boolean
}

type BrowseProjectsResponse = {
  path: string
  parent: string | null
  current_is_git: boolean
  directories: BrowseDirectoryEntry[]
  truncated: boolean
}

const STORAGE_PROJECT = 'feature-prd-runner-v3-project'
const STORAGE_ROUTE = 'feature-prd-runner-v3-route'
const ADD_REPO_VALUE = '__add_new_repo__'

const ROUTES: Array<{ key: RouteKey; label: string }> = [
  { key: 'board', label: 'Board' },
  { key: 'execution', label: 'Execution' },
  { key: 'review', label: 'Review Queue' },
  { key: 'agents', label: 'Agents' },
  { key: 'settings', label: 'Settings' },
]

const TASK_TYPE_OPTIONS = [
  'feature',
  'bug',
  'refactor',
  'research',
  'test',
  'docs',
  'security',
  'performance',
]

const TASK_STATUS_OPTIONS = ['backlog', 'ready', 'in_progress', 'in_review', 'blocked', 'done', 'cancelled']
const AGENT_ROLE_OPTIONS = ['general', 'implementer', 'reviewer', 'researcher', 'tester', 'planner', 'debugger']

function routeFromHash(hash: string): RouteKey {
  const cleaned = hash.replace(/^#\/?/, '').trim().toLowerCase()
  const found = ROUTES.find((route) => route.key === cleaned)
  return found?.key ?? 'board'
}

function toHash(route: RouteKey): string {
  return `#/${route}`
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let detail = ''
    try {
      const payload = await response.json() as { detail?: string }
      detail = payload?.detail ? `: ${payload.detail}` : ''
    } catch {
      // ignore parse failures for non-json bodies
    }
    throw new Error(`${response.status} ${response.statusText} [${url}]${detail}`)
  }
  return response.json() as Promise<T>
}

export default function App() {
  const [route, setRoute] = useState<RouteKey>(() => routeFromHash(window.location.hash || localStorage.getItem(STORAGE_ROUTE) || '#/board'))
  const [projectDir, setProjectDir] = useState<string>(() => localStorage.getItem(STORAGE_PROJECT) || '')
  const [board, setBoard] = useState<BoardResponse>({ columns: {} })
  const [orchestrator, setOrchestrator] = useState<OrchestratorStatus | null>(null)
  const [reviewQueue, setReviewQueue] = useState<TaskRecord[]>([])
  const [agents, setAgents] = useState<AgentRecord[]>([])
  const [projects, setProjects] = useState<ProjectRef[]>([])
  const [pinnedProjects, setPinnedProjects] = useState<PinnedProjectRef[]>([])
  const [quickActions, setQuickActions] = useState<QuickActionRecord[]>([])
  const [taskExplorerItems, setTaskExplorerItems] = useState<TaskRecord[]>([])
  const [executionBatches, setExecutionBatches] = useState<string[][]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  const [workOpen, setWorkOpen] = useState(false)
  const [createTab, setCreateTab] = useState<CreateTab>('task')
  const [selectedTaskId, setSelectedTaskId] = useState<string>('')
  const [selectedTaskDetail, setSelectedTaskDetail] = useState<TaskRecord | null>(null)
  const [selectedTaskDetailLoading, setSelectedTaskDetailLoading] = useState(false)
  const [editTaskTitle, setEditTaskTitle] = useState('')
  const [editTaskDescription, setEditTaskDescription] = useState('')
  const [editTaskType, setEditTaskType] = useState('feature')
  const [editTaskPriority, setEditTaskPriority] = useState('P2')
  const [editTaskLabels, setEditTaskLabels] = useState('')
  const [editTaskApprovalMode, setEditTaskApprovalMode] = useState<'human_review' | 'auto_approve'>('human_review')

  const [newTaskTitle, setNewTaskTitle] = useState('')
  const [newTaskDescription, setNewTaskDescription] = useState('')
  const [newTaskType, setNewTaskType] = useState('feature')
  const [newTaskPriority, setNewTaskPriority] = useState('P2')
  const [newTaskLabels, setNewTaskLabels] = useState('')
  const [newTaskBlockedBy, setNewTaskBlockedBy] = useState('')
  const [newTaskApprovalMode, setNewTaskApprovalMode] = useState<'human_review' | 'auto_approve'>('human_review')
  const [newTaskParentId, setNewTaskParentId] = useState('')
  const [newTaskPipelineTemplate, setNewTaskPipelineTemplate] = useState('')
  const [newTaskMetadata, setNewTaskMetadata] = useState('')
  const [selectedTaskTransition, setSelectedTaskTransition] = useState('ready')
  const [newDependencyId, setNewDependencyId] = useState('')
  const [taskExplorerQuery, setTaskExplorerQuery] = useState('')
  const [taskExplorerStatus, setTaskExplorerStatus] = useState('')
  const [taskExplorerType, setTaskExplorerType] = useState('')
  const [taskExplorerPriority, setTaskExplorerPriority] = useState('')
  const [taskExplorerOnlyBlocked, setTaskExplorerOnlyBlocked] = useState(false)
  const [taskExplorerLoading, setTaskExplorerLoading] = useState(false)
  const [taskExplorerError, setTaskExplorerError] = useState('')
  const [taskExplorerPage, setTaskExplorerPage] = useState(1)
  const [taskExplorerPageSize, setTaskExplorerPageSize] = useState(6)

  const [importText, setImportText] = useState('')
  const [importJobId, setImportJobId] = useState('')
  const [importPreview, setImportPreview] = useState<PrdPreview | null>(null)
  const [recentImportJobIds, setRecentImportJobIds] = useState<string[]>([])
  const [recentImportCommitMap, setRecentImportCommitMap] = useState<Record<string, string[]>>({})
  const [selectedImportJobId, setSelectedImportJobId] = useState('')
  const [selectedImportJob, setSelectedImportJob] = useState<ImportJobRecord | null>(null)
  const [selectedImportJobLoading, setSelectedImportJobLoading] = useState(false)
  const [selectedImportJobError, setSelectedImportJobError] = useState('')
  const [selectedImportJobErrorAt, setSelectedImportJobErrorAt] = useState('')

  const [quickPrompt, setQuickPrompt] = useState('')
  const [selectedQuickActionId, setSelectedQuickActionId] = useState('')
  const [selectedQuickActionDetail, setSelectedQuickActionDetail] = useState<QuickActionRecord | null>(null)
  const [selectedQuickActionLoading, setSelectedQuickActionLoading] = useState(false)
  const [selectedQuickActionError, setSelectedQuickActionError] = useState('')
  const [selectedQuickActionErrorAt, setSelectedQuickActionErrorAt] = useState('')
  const [reviewGuidance, setReviewGuidance] = useState('')
  const [spawnRole, setSpawnRole] = useState('general')
  const [spawnCapacity, setSpawnCapacity] = useState('1')
  const [spawnProviderOverride, setSpawnProviderOverride] = useState('')

  const [manualPinPath, setManualPinPath] = useState('')
  const [allowNonGit, setAllowNonGit] = useState(false)
  const [projectSearch, setProjectSearch] = useState('')
  const [browseOpen, setBrowseOpen] = useState(false)
  const [browsePath, setBrowsePath] = useState('')
  const [browseParentPath, setBrowseParentPath] = useState<string | null>(null)
  const [browseDirectories, setBrowseDirectories] = useState<BrowseDirectoryEntry[]>([])
  const [browseCurrentIsGit, setBrowseCurrentIsGit] = useState(false)
  const [browseLoading, setBrowseLoading] = useState(false)
  const [browseError, setBrowseError] = useState('')
  const [browseAllowNonGit, setBrowseAllowNonGit] = useState(false)

  useEffect(() => {
    const syncFromHash = () => {
      const next = routeFromHash(window.location.hash)
      setRoute(next)
      localStorage.setItem(STORAGE_ROUTE, toHash(next))
    }
    window.addEventListener('hashchange', syncFromHash)
    if (!window.location.hash) {
      window.location.hash = toHash(route)
    }
    return () => window.removeEventListener('hashchange', syncFromHash)
  }, [route])

  useEffect(() => {
    if (projectDir) {
      localStorage.setItem(STORAGE_PROJECT, projectDir)
    } else {
      localStorage.removeItem(STORAGE_PROJECT)
    }
  }, [projectDir])

  useEffect(() => {
    const hasModalOpen = workOpen || browseOpen
    document.documentElement.classList.toggle('modal-open', hasModalOpen)
    document.body.classList.toggle('modal-open', hasModalOpen)
    return () => {
      document.documentElement.classList.remove('modal-open')
      document.body.classList.remove('modal-open')
    }
  }, [workOpen, browseOpen])

  useEffect(() => {
    const columns = ['backlog', 'ready', 'in_progress', 'in_review', 'blocked', 'done'] as const
    const allTasks = columns.flatMap((column) => board.columns[column] || [])
    if (!selectedTaskId && allTasks.length > 0) {
      setSelectedTaskId(allTasks[0].id)
    }
    if (selectedTaskId && allTasks.every((task) => task.id !== selectedTaskId)) {
      setSelectedTaskId(allTasks[0]?.id || '')
      setSelectedTaskDetail(null)
    }
  }, [board, selectedTaskId])

  async function loadTaskDetail(taskId: string): Promise<void> {
    if (!taskId) {
      setSelectedTaskDetail(null)
      return
    }
    setSelectedTaskDetailLoading(true)
    try {
      const detail = await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/v3/tasks/${taskId}`, projectDir))
      const task = detail.task
      setSelectedTaskDetail(task)
      setEditTaskTitle(task.title || '')
      setEditTaskDescription(task.description || '')
      setEditTaskType(task.task_type || 'feature')
      setEditTaskPriority(task.priority || 'P2')
      setEditTaskLabels((task.labels || []).join(', '))
      setEditTaskApprovalMode(task.approval_mode || 'human_review')
    } catch {
      setSelectedTaskDetail(null)
    } finally {
      setSelectedTaskDetailLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedTaskId) return
    void loadTaskDetail(selectedTaskId)
  }, [selectedTaskId, projectDir])

  async function loadTaskExplorer(): Promise<void> {
    setTaskExplorerLoading(true)
    setTaskExplorerError('')
    try {
      const params: Record<string, string> = {}
      const effectiveStatus = taskExplorerOnlyBlocked ? 'blocked' : taskExplorerStatus
      if (effectiveStatus) params.status = effectiveStatus
      if (taskExplorerType) params.task_type = taskExplorerType
      if (taskExplorerPriority) params.priority = taskExplorerPriority
      const response = await requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/v3/tasks', projectDir, params))
      const tasks = response.tasks || []
      const query = taskExplorerQuery.trim().toLowerCase()
      const filtered = query
        ? tasks.filter((task) => {
            const haystack = `${task.title} ${task.description || ''} ${task.id}`.toLowerCase()
            return haystack.includes(query)
          })
        : tasks
      setTaskExplorerItems(filtered)
    } catch (err) {
      setTaskExplorerItems([])
      const detail = err instanceof Error ? err.message : 'unknown error'
      setTaskExplorerError(`Failed to load task explorer (${detail})`)
    } finally {
      setTaskExplorerLoading(false)
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadTaskExplorer()
    }, 250)
    return () => window.clearTimeout(timer)
  }, [projectDir, taskExplorerQuery, taskExplorerStatus, taskExplorerType, taskExplorerPriority, taskExplorerOnlyBlocked])

  useEffect(() => {
    setTaskExplorerPage(1)
  }, [taskExplorerQuery, taskExplorerStatus, taskExplorerType, taskExplorerPriority, taskExplorerOnlyBlocked, taskExplorerPageSize])

  useEffect(() => {
    const maxPage = Math.max(1, Math.ceil(taskExplorerItems.length / taskExplorerPageSize))
    if (taskExplorerPage > maxPage) {
      setTaskExplorerPage(maxPage)
    }
  }, [taskExplorerItems.length, taskExplorerPageSize, taskExplorerPage])

  async function loadImportJobDetail(jobId: string): Promise<void> {
    if (!jobId) {
      setSelectedImportJob(null)
      return
    }
    setSelectedImportJobLoading(true)
    setSelectedImportJobError('')
    setSelectedImportJobErrorAt('')
    try {
      const payload = await requestJson<{ job: ImportJobRecord }>(buildApiUrl(`/api/v3/import/${jobId}`, projectDir))
      setSelectedImportJob(payload.job)
    } catch (err) {
      setSelectedImportJob(null)
      const detail = err instanceof Error ? err.message : 'unknown error'
      setSelectedImportJobError(`Failed to load import job detail (${detail})`)
      setSelectedImportJobErrorAt(new Date().toLocaleTimeString())
    } finally {
      setSelectedImportJobLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedImportJobId) return
    void loadImportJobDetail(selectedImportJobId)
  }, [selectedImportJobId, projectDir])

  useEffect(() => {
    if (!workOpen || createTab !== 'import') return
    if (!selectedImportJobId) return
    if (!selectedImportJob) return
    const status = String(selectedImportJob.status || '').toLowerCase()
    if (!['preview_ready', 'committing'].includes(status)) return
    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      if (Date.now() - startedAt > 60_000) {
        window.clearInterval(timer)
        return
      }
      void loadImportJobDetail(selectedImportJobId)
    }, 2_000)
    return () => window.clearInterval(timer)
  }, [workOpen, createTab, selectedImportJobId, selectedImportJob, projectDir])

  async function loadQuickActionDetail(quickActionId: string): Promise<void> {
    if (!quickActionId) {
      setSelectedQuickActionDetail(null)
      return
    }
    setSelectedQuickActionLoading(true)
    setSelectedQuickActionError('')
    setSelectedQuickActionErrorAt('')
    try {
      const payload = await requestJson<{ quick_action: QuickActionRecord }>(buildApiUrl(`/api/v3/quick-actions/${quickActionId}`, projectDir))
      setSelectedQuickActionDetail(payload.quick_action)
    } catch (err) {
      setSelectedQuickActionDetail(null)
      const detail = err instanceof Error ? err.message : 'unknown error'
      setSelectedQuickActionError(`Failed to load quick action detail (${detail})`)
      setSelectedQuickActionErrorAt(new Date().toLocaleTimeString())
    } finally {
      setSelectedQuickActionLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedQuickActionId) return
    void loadQuickActionDetail(selectedQuickActionId)
  }, [selectedQuickActionId, projectDir])

  async function reloadAll(): Promise<void> {
    setLoading(true)
    setError('')
    try {
      const [boardData, orchestratorData, reviewData, agentData, projectData, pinnedData, quickActionData, executionOrderData] = await Promise.all([
        requestJson<BoardResponse>(buildApiUrl('/api/v3/tasks/board', projectDir)),
        requestJson<OrchestratorStatus>(buildApiUrl('/api/v3/orchestrator/status', projectDir)),
        requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/v3/review-queue', projectDir)),
        requestJson<{ agents: AgentRecord[] }>(buildApiUrl('/api/v3/agents', projectDir)),
        requestJson<{ projects: ProjectRef[] }>(buildApiUrl('/api/v3/projects', projectDir)),
        requestJson<{ items: PinnedProjectRef[] }>(buildApiUrl('/api/v3/projects/pinned', projectDir)),
        requestJson<{ quick_actions: QuickActionRecord[] }>(buildApiUrl('/api/v3/quick-actions', projectDir)),
        requestJson<{ batches: string[][] }>(buildApiUrl('/api/v3/tasks/execution-order', projectDir)),
      ])
      setBoard(boardData)
      setOrchestrator(orchestratorData)
      setReviewQueue(reviewData.tasks)
      setAgents(agentData.agents)
      setProjects(projectData.projects)
      setPinnedProjects(pinnedData.items || [])
      setQuickActions(quickActionData.quick_actions || [])
      setExecutionBatches(executionOrderData.batches || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reloadAll()
  }, [projectDir])

  useEffect(() => {
    const socket = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`)
    socket.addEventListener('open', () => {
      socket.send(JSON.stringify({ action: 'subscribe', channels: ['tasks', 'queue', 'agents', 'review', 'quick_actions', 'notifications', 'system'] }))
    })
    socket.addEventListener('message', (event) => {
      void reloadAll()
      try {
        const data = JSON.parse(event.data)
        if (data.channel === 'quick_actions' && selectedQuickActionId) {
          void loadQuickActionDetail(selectedQuickActionId)
        }
      } catch {
        // ignore non-JSON messages
      }
    })
    socket.addEventListener('error', () => {
      socket.close()
    })
    return () => socket.close()
  }, [projectDir])

  async function submitTask(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!newTaskTitle.trim()) return
    let parsedMetadata: Record<string, unknown> | undefined
    if (newTaskMetadata.trim()) {
      try {
        const metadataJson = JSON.parse(newTaskMetadata)
        if (metadataJson && typeof metadataJson === 'object' && !Array.isArray(metadataJson)) {
          parsedMetadata = metadataJson as Record<string, unknown>
        } else {
          setError('Task metadata must be a JSON object')
          return
        }
      } catch {
        setError('Task metadata must be valid JSON')
        return
      }
    }
    const parsedPipelineTemplate = newTaskPipelineTemplate
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    await requestJson<{ task: TaskRecord }>(buildApiUrl('/api/v3/tasks', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: newTaskTitle.trim(),
        description: newTaskDescription,
        task_type: newTaskType,
        priority: newTaskPriority,
        labels: newTaskLabels.split(',').map((item) => item.trim()).filter(Boolean),
        blocked_by: newTaskBlockedBy.split(',').map((item) => item.trim()).filter(Boolean),
        approval_mode: newTaskApprovalMode,
        parent_id: newTaskParentId.trim() || undefined,
        pipeline_template: parsedPipelineTemplate.length > 0 ? parsedPipelineTemplate : undefined,
        metadata: parsedMetadata,
        status: 'backlog',
      }),
    })
    setError('')
    setNewTaskTitle('')
    setNewTaskDescription('')
    setNewTaskType('feature')
    setNewTaskPriority('P2')
    setNewTaskLabels('')
    setNewTaskBlockedBy('')
    setNewTaskApprovalMode('human_review')
    setNewTaskParentId('')
    setNewTaskPipelineTemplate('')
    setNewTaskMetadata('')
    setWorkOpen(false)
    await reloadAll()
  }

  async function previewImport(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!importText.trim()) return
    const preview = await requestJson<{ job_id: string; preview: PrdPreview }>(buildApiUrl('/api/v3/import/prd/preview', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: importText, default_priority: 'P2' }),
    })
    setImportJobId(preview.job_id)
    setImportPreview(preview.preview)
    setSelectedImportJobId(preview.job_id)
    setRecentImportJobIds((prev) => [preview.job_id, ...prev.filter((item) => item !== preview.job_id)].slice(0, 8))
  }

  async function commitImport(): Promise<void> {
    if (!importJobId) return
    const commitResponse = await requestJson<{ created_task_ids: string[] }>(buildApiUrl('/api/v3/import/prd/commit', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: importJobId }),
    })
    setRecentImportCommitMap((prev) => ({ ...prev, [importJobId]: commitResponse.created_task_ids || [] }))
    if (importJobId) {
      setSelectedImportJobId(importJobId)
      setRecentImportJobIds((prev) => [importJobId, ...prev.filter((item) => item !== importJobId)].slice(0, 8))
      await loadImportJobDetail(importJobId)
    }
    setImportJobId('')
    setImportPreview(null)
    setImportText('')
    setWorkOpen(false)
    await reloadAll()
  }

  async function submitQuickAction(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!quickPrompt.trim()) return
    const resp = await requestJson<{ quick_action: QuickActionRecord }>(buildApiUrl('/api/v3/quick-actions', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: quickPrompt.trim() }),
    })
    if (resp.quick_action) {
      setSelectedQuickActionId(resp.quick_action.id)
      setSelectedQuickActionDetail(resp.quick_action)
    }
    setQuickPrompt('')
    setWorkOpen(false)
    await reloadAll()
  }

  async function taskAction(taskId: string, action: 'run' | 'retry' | 'cancel'): Promise<void> {
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/v3/tasks/${taskId}/${action}`, projectDir), {
      method: 'POST',
    })
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
    }
  }

  async function transitionTask(taskId: string): Promise<void> {
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/v3/tasks/${taskId}/transition`, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: selectedTaskTransition }),
    })
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
    }
  }

  async function addDependency(taskId: string): Promise<void> {
    if (!newDependencyId.trim()) return
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/v3/tasks/${taskId}/dependencies`, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ depends_on: newDependencyId.trim() }),
    })
    setNewDependencyId('')
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
    }
  }

  async function removeDependency(taskId: string, depId: string): Promise<void> {
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/v3/tasks/${taskId}/dependencies/${depId}`, projectDir), {
      method: 'DELETE',
    })
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
    }
  }

  async function saveTaskEdits(taskId: string): Promise<void> {
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/v3/tasks/${taskId}`, projectDir), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: editTaskTitle.trim(),
        description: editTaskDescription,
        task_type: editTaskType,
        priority: editTaskPriority,
        labels: editTaskLabels.split(',').map((item) => item.trim()).filter(Boolean),
        approval_mode: editTaskApprovalMode,
      }),
    })
    await reloadAll()
    await loadTaskDetail(taskId)
  }

  async function promoteQuickAction(quickActionId: string): Promise<void> {
    await requestJson<{ task: TaskRecord; already_promoted: boolean }>(buildApiUrl(`/api/v3/quick-actions/${quickActionId}/promote`, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ priority: 'P2' }),
    })
    await reloadAll()
    if (selectedQuickActionId === quickActionId) {
      await loadQuickActionDetail(quickActionId)
    }
  }

  async function controlOrchestrator(action: 'pause' | 'resume' | 'drain' | 'stop'): Promise<void> {
    await requestJson<OrchestratorStatus>(buildApiUrl('/api/v3/orchestrator/control', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    })
    await reloadAll()
  }

  async function reviewAction(taskId: string, action: 'approve' | 'request-changes'): Promise<void> {
    const endpoint = action === 'approve' ? `/api/v3/review/${taskId}/approve` : `/api/v3/review/${taskId}/request-changes`
    await requestJson<{ task: TaskRecord }>(buildApiUrl(endpoint, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ guidance: reviewGuidance.trim() || undefined }),
    })
    setReviewGuidance('')
    await reloadAll()
  }

  async function spawnAgent(): Promise<void> {
    await requestJson<{ agent: AgentRecord }>(buildApiUrl('/api/v3/agents/spawn', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role: spawnRole,
        capacity: Math.max(1, Number(spawnCapacity) || 1),
        override_provider: spawnProviderOverride.trim() || undefined,
      }),
    })
    setSpawnProviderOverride('')
    await reloadAll()
  }

  async function agentAction(agentId: string, action: 'pause' | 'resume' | 'terminate'): Promise<void> {
    await requestJson<{ agent: AgentRecord }>(buildApiUrl(`/api/v3/agents/${agentId}/${action}`, projectDir), { method: 'POST' })
    await reloadAll()
  }

  async function pinProjectPath(path: string, allowNonGitValue: boolean): Promise<void> {
    const pinned = await requestJson<{ project: ProjectRef }>(buildApiUrl('/api/v3/projects/pinned', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, allow_non_git: allowNonGitValue }),
    })
    setProjectDir(pinned.project.path)
    await reloadAll()
  }

  async function unpinProject(projectId: string): Promise<void> {
    await requestJson<{ removed: boolean }>(buildApiUrl(`/api/v3/projects/pinned/${projectId}`, projectDir), {
      method: 'DELETE',
    })
    await reloadAll()
  }

  async function pinManualProject(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!manualPinPath.trim()) return
    await pinProjectPath(manualPinPath.trim(), allowNonGit)
    setManualPinPath('')
  }

  async function handleTopbarProjectChange(nextValue: string): Promise<void> {
    if (nextValue === ADD_REPO_VALUE) {
      setBrowseOpen(true)
      void loadBrowseDirectories()
      return
    }
    setProjectDir(nextValue)
  }

  async function loadBrowseDirectories(nextPath?: string): Promise<void> {
    setBrowseLoading(true)
    setBrowseError('')
    try {
      const data = await requestJson<BrowseProjectsResponse>(
        buildApiUrl('/api/v3/projects/browse', projectDir, nextPath ? { path: nextPath } : {})
      )
      setBrowsePath(data.path)
      setBrowseParentPath(data.parent)
      setBrowseCurrentIsGit(data.current_is_git)
      setBrowseDirectories(data.directories)
    } catch (err) {
      setBrowseError(err instanceof Error ? err.message : 'Failed to browse directories')
    } finally {
      setBrowseLoading(false)
    }
  }

  async function pinFromBrowse(): Promise<void> {
    if (!browsePath) return
    await pinProjectPath(browsePath, browseAllowNonGit)
    setBrowseOpen(false)
  }

  function renderBoard(): JSX.Element {
    const columns = ['backlog', 'ready', 'in_progress', 'in_review', 'blocked', 'done']
    const allTasks = columns.flatMap((column) => board.columns[column] || [])
    const selectedTask = allTasks.find((task) => task.id === selectedTaskId) || allTasks[0]
    const selectedTaskView = selectedTaskDetail && selectedTask && selectedTaskDetail.id === selectedTask.id ? selectedTaskDetail : selectedTask
    const queueTasks = [...(board.columns.ready || []), ...(board.columns.in_progress || [])]
    const totalExplorerItems = taskExplorerItems.length
    const explorerStart = (taskExplorerPage - 1) * taskExplorerPageSize
    const explorerEnd = explorerStart + taskExplorerPageSize
    const pagedExplorerItems = taskExplorerItems.slice(explorerStart, explorerEnd)
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Board</h2>
        </header>
        <div className="workbench-grid">
          <article className="workbench-pane">
            <h3>Kanban</h3>
            <div className="board-grid">
              {columns.map((column) => (
                <article className="board-col" key={column}>
                  <h3>{humanizeLabel(column)}</h3>
                  <div className="card-list">
                    {(board.columns[column] || []).map((task) => (
                      <button className="task-card task-card-button" key={task.id} onClick={() => setSelectedTaskId(task.id)}>
                        <p className="task-title">{task.title}</p>
                        <p className="task-meta">{task.priority} · {task.id}</p>
                        {task.description ? <p className="task-desc">{task.description}</p> : null}
                      </button>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </article>
          <article className="workbench-pane">
            <h3>Task Detail</h3>
            {selectedTaskView ? (
              <div className="detail-card">
                {selectedTaskDetailLoading ? <p className="field-label">Loading full task detail...</p> : null}
                <p className="task-title">{selectedTaskView.title}</p>
                <p className="task-meta">{selectedTaskView.id} · {selectedTaskView.priority} · {humanizeLabel(selectedTaskView.status)} · {humanizeLabel(selectedTaskView.task_type || 'feature')}</p>
                {selectedTaskView.description ? <p className="task-desc">{selectedTaskView.description}</p> : <p className="task-desc">No description.</p>}
                <p className="field-label">Blockers: {(selectedTaskView.blocked_by || []).join(', ') || 'None'}</p>
                <div className="form-stack">
                  <label className="field-label" htmlFor="edit-task-title">Edit title</label>
                  <input id="edit-task-title" value={editTaskTitle} onChange={(event) => setEditTaskTitle(event.target.value)} />
                  <label className="field-label" htmlFor="edit-task-description">Edit description</label>
                  <textarea id="edit-task-description" rows={3} value={editTaskDescription} onChange={(event) => setEditTaskDescription(event.target.value)} />
                  <div className="inline-actions">
                    <select value={editTaskType} onChange={(event) => setEditTaskType(event.target.value)}>
                      {TASK_TYPE_OPTIONS.map((taskType) => (
                        <option key={taskType} value={taskType}>{humanizeLabel(taskType)}</option>
                      ))}
                    </select>
                    <select value={editTaskPriority} onChange={(event) => setEditTaskPriority(event.target.value)}>
                      <option value="P0">P0</option>
                      <option value="P1">P1</option>
                      <option value="P2">P2</option>
                      <option value="P3">P3</option>
                    </select>
                    <select value={editTaskApprovalMode} onChange={(event) => setEditTaskApprovalMode(event.target.value as 'human_review' | 'auto_approve')}>
                      <option value="human_review">{humanizeLabel('human_review')}</option>
                      <option value="auto_approve">{humanizeLabel('auto_approve')}</option>
                    </select>
                  </div>
                  <label className="field-label" htmlFor="edit-task-labels">Labels (comma-separated)</label>
                  <input id="edit-task-labels" value={editTaskLabels} onChange={(event) => setEditTaskLabels(event.target.value)} />
                  <button className="button" onClick={() => void saveTaskEdits(selectedTaskView.id)}>Save edits</button>
                </div>
                <div className="inline-actions">
                  <button className="button" onClick={() => void taskAction(selectedTaskView.id, 'run')}>Run</button>
                  <button className="button" onClick={() => void taskAction(selectedTaskView.id, 'retry')}>Retry</button>
                  <button className="button button-danger" onClick={() => void taskAction(selectedTaskView.id, 'cancel')}>Cancel</button>
                </div>
                <div className="inline-actions">
                  <select value={selectedTaskTransition} onChange={(event) => setSelectedTaskTransition(event.target.value)}>
                    {TASK_STATUS_OPTIONS.map((status) => (
                      <option key={status} value={status}>{humanizeLabel(status)}</option>
                    ))}
                  </select>
                  <button className="button" onClick={() => void transitionTask(selectedTaskView.id)}>Transition</button>
                </div>
                <div className="form-stack">
                  <label className="field-label" htmlFor="task-blocker-input">Add blocker task ID</label>
                  <div className="inline-actions">
                    <input
                      id="task-blocker-input"
                      value={newDependencyId}
                      onChange={(event) => setNewDependencyId(event.target.value)}
                      placeholder="task-xxxxxxxxxx"
                    />
                    <button className="button" onClick={() => void addDependency(selectedTaskView.id)}>Add dependency</button>
                  </div>
                  {(selectedTaskView.blocked_by || []).map((depId) => (
                    <div className="row-card" key={depId}>
                      <p className="task-meta">{depId}</p>
                      <button className="button button-danger" onClick={() => void removeDependency(selectedTaskView.id, depId)}>
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
                <p className="field-label">Activity: Review queue size is {reviewQueue.length}. Trigger actions from Review Queue.</p>
              </div>
            ) : (
              <p className="empty">No tasks on board yet.</p>
            )}
          </article>
          <article className="workbench-pane">
            <h3>Queue & Agents</h3>
            <div className="list-stack">
              <p className="field-label">Queue depth: {orchestrator?.queue_depth ?? 0}</p>
              <p className="field-label">In progress: {orchestrator?.in_progress ?? 0}</p>
              {queueTasks.slice(0, 5).map((task) => (
                <div key={task.id} className="row-card">
                  <p className="task-title">{task.title}</p>
                  <p className="task-meta">{humanizeLabel(task.status)}</p>
                </div>
              ))}
              {queueTasks.length === 0 ? <p className="empty">No queued or running tasks.</p> : null}
              <p className="field-label">Agents ({agents.length})</p>
              {agents.slice(0, 4).map((agent) => (
                <div key={agent.id} className="row-card">
                  <p className="task-title">{humanizeLabel(agent.role)}</p>
                  <p className="task-meta">{humanizeLabel(agent.status)}</p>
                </div>
              ))}
              <TaskExplorerPanel
                query={taskExplorerQuery}
                status={taskExplorerStatus}
                taskType={taskExplorerType}
                priority={taskExplorerPriority}
                onlyBlocked={taskExplorerOnlyBlocked}
                loading={taskExplorerLoading}
                error={taskExplorerError}
                items={pagedExplorerItems}
                page={taskExplorerPage}
                pageSize={taskExplorerPageSize}
                totalItems={totalExplorerItems}
                statusOptions={TASK_STATUS_OPTIONS}
                typeOptions={TASK_TYPE_OPTIONS}
                onQueryChange={setTaskExplorerQuery}
                onStatusChange={setTaskExplorerStatus}
                onTypeChange={setTaskExplorerType}
                onPriorityChange={setTaskExplorerPriority}
                onOnlyBlockedChange={setTaskExplorerOnlyBlocked}
                onPageChange={setTaskExplorerPage}
                onPageSizeChange={setTaskExplorerPageSize}
                onSelectTask={setSelectedTaskId}
                onRetry={() => void loadTaskExplorer()}
              />
            </div>
          </article>
        </div>
      </section>
    )
  }

  function renderExecution(): JSX.Element {
    const taskById = new Map<string, TaskRecord>()
    for (const tasks of Object.values(board.columns)) {
      for (const task of tasks) {
        taskById.set(task.id, task)
      }
    }
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Execution</h2>
          <div className="inline-actions">
            <button className="button" onClick={() => void controlOrchestrator('pause')}>Pause</button>
            <button className="button" onClick={() => void controlOrchestrator('resume')}>Resume</button>
            <button className="button" onClick={() => void controlOrchestrator('drain')}>Drain</button>
            <button className="button button-danger" onClick={() => void controlOrchestrator('stop')}>Stop</button>
          </div>
        </header>
        <div className="status-grid">
          <div className="status-card">
            <span>State</span>
            <strong>{humanizeLabel(orchestrator?.status ?? 'unknown')}</strong>
          </div>
          <div className="status-card">
            <span>Queue</span>
            <strong>{orchestrator?.queue_depth ?? 0}</strong>
          </div>
          <div className="status-card">
            <span>In Progress</span>
            <strong>{orchestrator?.in_progress ?? 0}</strong>
          </div>
          <div className="status-card">
            <span>Run Branch</span>
            <strong>{orchestrator?.run_branch || '-'}</strong>
          </div>
        </div>
        <div className="list-stack">
          <p className="field-label">Execution order</p>
          {executionBatches.map((batch, index) => (
            <div className="row-card" key={`batch-${index}`}>
              <p className="task-title">Batch {index + 1}</p>
              <p className="task-meta">{batch.map((taskId) => taskById.get(taskId)?.title || taskId).join(' | ')}</p>
            </div>
          ))}
          {executionBatches.length === 0 ? <p className="empty">No execution batches available.</p> : null}
        </div>
      </section>
    )
  }

  function renderReviewQueue(): JSX.Element {
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Review Queue</h2>
        </header>
        <div className="list-stack">
          <label className="field-label" htmlFor="review-guidance">Optional review guidance</label>
          <input
            id="review-guidance"
            value={reviewGuidance}
            onChange={(event) => setReviewGuidance(event.target.value)}
            placeholder="What should be fixed or accepted?"
          />
          {reviewQueue.map((task) => (
            <div className="row-card" key={task.id}>
              <div>
                <p className="task-title">{task.title}</p>
                <p className="task-meta">{task.id}</p>
              </div>
              <div className="inline-actions">
                <button className="button" onClick={() => void reviewAction(task.id, 'request-changes')}>Request changes</button>
                <button className="button button-primary" onClick={() => void reviewAction(task.id, 'approve')}>Approve</button>
              </div>
            </div>
          ))}
          {reviewQueue.length === 0 ? <p className="empty">No tasks waiting for review.</p> : null}
        </div>
      </section>
    )
  }

  function renderAgents(): JSX.Element {
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Agents</h2>
          <div className="inline-actions">
            <select value={spawnRole} onChange={(event) => setSpawnRole(event.target.value)}>
              {AGENT_ROLE_OPTIONS.map((role) => (
                <option key={role} value={role}>{humanizeLabel(role)}</option>
              ))}
            </select>
            <input
              value={spawnCapacity}
              onChange={(event) => setSpawnCapacity(event.target.value)}
              placeholder="capacity"
              aria-label="Spawn capacity"
            />
            <input
              value={spawnProviderOverride}
              onChange={(event) => setSpawnProviderOverride(event.target.value)}
              placeholder="provider override (optional)"
              aria-label="Provider override"
            />
            <button className="button button-primary" onClick={() => void spawnAgent()}>Spawn agent</button>
          </div>
        </header>
        <div className="list-stack">
          {agents.map((agent) => (
            <div className="row-card" key={agent.id}>
              <div>
                <p className="task-title">{humanizeLabel(agent.role)}</p>
                <p className="task-meta">{agent.id} · {humanizeLabel(agent.status)}</p>
              </div>
              <div className="inline-actions">
                <button className="button" onClick={() => void agentAction(agent.id, 'pause')}>Pause</button>
                <button className="button" onClick={() => void agentAction(agent.id, 'resume')}>Resume</button>
                <button className="button button-danger" onClick={() => void agentAction(agent.id, 'terminate')}>Terminate</button>
              </div>
            </div>
          ))}
          {agents.length === 0 ? <p className="empty">No agents active.</p> : null}
        </div>
      </section>
    )
  }

  function renderSettings(): JSX.Element {
    const filteredProjects = projects.filter((project) => project.path.toLowerCase().includes(projectSearch.toLowerCase()))
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Settings</h2>
        </header>

        <div className="settings-grid">
          <article className="settings-card">
            <h3>Projects</h3>
            <label className="field-label" htmlFor="project-selector">Active project</label>
            <input
              id="project-search"
              value={projectSearch}
              onChange={(event) => setProjectSearch(event.target.value)}
              placeholder="Search discovered/pinned projects"
            />
            <select
              id="project-selector"
              value={projectDir}
              onChange={(event) => setProjectDir(event.target.value)}
            >
              <option value="">Current workspace</option>
              {filteredProjects.map((project) => (
                <option key={`${project.id}-${project.path}`} value={project.path}>
                  {project.path} ({humanizeLabel(project.source)})
                </option>
              ))}
            </select>

            <form className="form-stack" onSubmit={(event) => void pinManualProject(event)}>
              <label className="field-label" htmlFor="manual-project-path">Pin project by absolute path</label>
              <input
                id="manual-project-path"
                value={manualPinPath}
                onChange={(event) => setManualPinPath(event.target.value)}
                placeholder="/absolute/path/to/repo"
                required
              />
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={allowNonGit}
                  onChange={(event) => setAllowNonGit(event.target.checked)}
                />
                Allow non-git directory
              </label>
              <button className="button button-primary" type="submit">Pin project</button>
            </form>
            <div className="list-stack">
              <p className="field-label">Pinned projects</p>
              {pinnedProjects.map((project) => (
                <div className="row-card" key={project.id}>
                  <div>
                    <p className="task-title">{project.id}</p>
                    <p className="task-meta">{project.path}</p>
                  </div>
                  <button className="button button-danger" onClick={() => void unpinProject(project.id)}>Unpin</button>
                </div>
              ))}
              {pinnedProjects.length === 0 ? <p className="empty">No pinned projects.</p> : null}
            </div>
          </article>

          <article className="settings-card">
            <h3>Diagnostics</h3>
            <p>Schema version: 3</p>
            <p>Selected route: {humanizeLabel(route)}</p>
            <p>Project dir: {projectDir || 'current workspace'}</p>
          </article>
        </div>
      </section>
    )
  }

  function renderRoute(): JSX.Element {
    if (route === 'execution') return renderExecution()
    if (route === 'review') return renderReviewQueue()
    if (route === 'agents') return renderAgents()
    if (route === 'settings') return renderSettings()
    return renderBoard()
  }

  return (
    <div className="orchestrator-app">
      <div className="bg-layer" aria-hidden="true" />
      <header className="topbar">
        <div>
          <p className="kicker">orchestrator-first</p>
          <h1>Feature PRD Runner</h1>
        </div>
        <div className="topbar-actions">
          <select
            className="topbar-project-select"
            value={projectDir}
            onChange={(event) => {
              void handleTopbarProjectChange(event.target.value)
            }}
            aria-label="Active repo"
          >
            <option value="">Current workspace</option>
            {projects.map((project) => (
              <option key={`${project.id}-${project.path}`} value={project.path}>
                {project.path}
              </option>
            ))}
            <option value={ADD_REPO_VALUE}>Add repo...</option>
          </select>
          <button className="button" onClick={() => void reloadAll()} disabled={loading}>Refresh</button>
          <button className="button button-primary" onClick={() => setWorkOpen(true)}>Create Work</button>
        </div>
      </header>

      <nav className="nav-strip" aria-label="Main navigation">
        {ROUTES.map((item) => (
          <button
            key={item.key}
            className={`nav-pill ${route === item.key ? 'is-active' : ''}`}
            onClick={() => {
              window.location.hash = toHash(item.key)
              setRoute(item.key)
            }}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main>{renderRoute()}</main>

      {error ? <p className="error-banner">{error}</p> : null}

      {workOpen ? (
        <div className="modal-scrim" role="dialog" aria-modal="true" aria-label="Create Work modal">
          <div className="modal-card create-work-modal">
            <div className="modal-sticky-top">
              <header className="panel-head">
                <h2>Create Work</h2>
                <button className="button" onClick={() => setWorkOpen(false)}>Close</button>
              </header>

              <div className="tab-row">
                <button className={`tab ${createTab === 'task' ? 'is-active' : ''}`} onClick={() => setCreateTab('task')}>Create Task</button>
                <button className={`tab ${createTab === 'import' ? 'is-active' : ''}`} onClick={() => setCreateTab('import')}>Import PRD</button>
                <button className={`tab ${createTab === 'quick' ? 'is-active' : ''}`} onClick={() => setCreateTab('quick')}>Quick Action</button>
              </div>
            </div>

            <div className="modal-body">
              {createTab === 'task' ? (
                <form id="create-task-form" className="form-stack create-task-form" onSubmit={(event) => void submitTask(event)}>
                  <label className="field-label" htmlFor="task-title">Title</label>
                  <input id="task-title" value={newTaskTitle} onChange={(event) => setNewTaskTitle(event.target.value)} required />
                  <label className="field-label" htmlFor="task-description">Description</label>
                  <textarea id="task-description" rows={4} value={newTaskDescription} onChange={(event) => setNewTaskDescription(event.target.value)} />
                  <label className="field-label" htmlFor="task-type">Task Type</label>
                  <select id="task-type" value={newTaskType} onChange={(event) => setNewTaskType(event.target.value)}>
                    {TASK_TYPE_OPTIONS.map((taskType) => (
                      <option key={taskType} value={taskType}>{humanizeLabel(taskType)}</option>
                    ))}
                  </select>
                  <label className="field-label">Priority</label>
                  <div className="toggle-group" role="group" aria-label="Task priority">
                    {['P0', 'P1', 'P2', 'P3'].map((priority) => (
                      <button
                        key={priority}
                        type="button"
                        className={`toggle-button ${newTaskPriority === priority ? 'is-active' : ''}`}
                        aria-pressed={newTaskPriority === priority}
                        onClick={() => setNewTaskPriority(priority)}
                      >
                        {priority}
                      </button>
                    ))}
                  </div>
                  <details className="advanced-fields">
                    <summary>Advanced</summary>
                    <div className="form-stack advanced-fields-body">
                      <label className="field-label">Approval mode</label>
                      <div className="toggle-group" role="group" aria-label="Task approval mode">
                        <button
                          type="button"
                          className={`toggle-button ${newTaskApprovalMode === 'human_review' ? 'is-active' : ''}`}
                          aria-pressed={newTaskApprovalMode === 'human_review'}
                          onClick={() => setNewTaskApprovalMode('human_review')}
                        >
                          {humanizeLabel('human_review')}
                        </button>
                        <button
                          type="button"
                          className={`toggle-button ${newTaskApprovalMode === 'auto_approve' ? 'is-active' : ''}`}
                          aria-pressed={newTaskApprovalMode === 'auto_approve'}
                          onClick={() => setNewTaskApprovalMode('auto_approve')}
                        >
                          {humanizeLabel('auto_approve')}
                        </button>
                      </div>
                      <label className="field-label" htmlFor="task-labels">Labels (comma-separated)</label>
                      <input
                        id="task-labels"
                        value={newTaskLabels}
                        onChange={(event) => setNewTaskLabels(event.target.value)}
                        placeholder="frontend, urgent"
                      />
                      <label className="field-label" htmlFor="task-blocked-by">Blocked by task IDs (comma-separated)</label>
                      <input
                        id="task-blocked-by"
                        value={newTaskBlockedBy}
                        onChange={(event) => setNewTaskBlockedBy(event.target.value)}
                        placeholder="task-abc123, task-def456"
                      />
                      <label className="field-label" htmlFor="task-parent-id">Parent task ID (optional)</label>
                      <input
                        id="task-parent-id"
                        value={newTaskParentId}
                        onChange={(event) => setNewTaskParentId(event.target.value)}
                        placeholder="task-parent-id"
                      />
                      <label className="field-label" htmlFor="task-pipeline-template">Pipeline template steps (comma-separated, optional)</label>
                      <input
                        id="task-pipeline-template"
                        value={newTaskPipelineTemplate}
                        onChange={(event) => setNewTaskPipelineTemplate(event.target.value)}
                        placeholder="plan, implement, verify, review"
                      />
                      <label className="field-label" htmlFor="task-metadata">Metadata JSON object (optional)</label>
                      <textarea
                        id="task-metadata"
                        rows={4}
                        value={newTaskMetadata}
                        onChange={(event) => setNewTaskMetadata(event.target.value)}
                        placeholder='{"epic":"checkout","owner":"web"}'
                      />
                    </div>
                  </details>
                </form>
              ) : null}

              {createTab === 'import' ? (
                <div className="form-stack">
                  <form className="form-stack" onSubmit={(event) => void previewImport(event)}>
                    <label className="field-label" htmlFor="prd-text">PRD text</label>
                    <textarea id="prd-text" rows={8} value={importText} onChange={(event) => setImportText(event.target.value)} placeholder="- Task 1\n- Task 2" required />
                    <button className="button" type="submit">Preview</button>
                  </form>
                  <ImportJobPanel
                    importJobId={importJobId}
                    importPreview={importPreview}
                    recentImportJobIds={recentImportJobIds}
                    selectedImportJobId={selectedImportJobId}
                    selectedImportJob={selectedImportJob}
                    selectedImportJobLoading={selectedImportJobLoading}
                    selectedImportJobError={`${selectedImportJobError}${selectedImportJobErrorAt ? ` at ${selectedImportJobErrorAt}` : ''}`}
                    selectedCreatedTaskIds={recentImportCommitMap[selectedImportJobId] || selectedImportJob?.created_task_ids || []}
                    onCommitImport={() => void commitImport()}
                    onSelectImportJob={setSelectedImportJobId}
                    onRefreshImportJob={() => void loadImportJobDetail(selectedImportJobId)}
                    onRetryLoadImportJob={() => void loadImportJobDetail(selectedImportJobId)}
                  />
                </div>
              ) : null}

              {createTab === 'quick' ? (
                <div className="form-stack">
                  <form className="form-stack" onSubmit={(event) => void submitQuickAction(event)}>
                    <p className="hint">Quick Action is ephemeral. Promote explicitly if you want it on the board.</p>
                    <label className="field-label" htmlFor="quick-prompt">Prompt</label>
                    <textarea id="quick-prompt" rows={6} value={quickPrompt} onChange={(event) => setQuickPrompt(event.target.value)} required />
                    <button className="button button-primary" type="submit">Run Quick Action</button>
                  </form>
                  <QuickActionDetailPanel
                    quickActions={quickActions}
                    selectedQuickActionId={selectedQuickActionId}
                    selectedQuickActionDetail={selectedQuickActionDetail}
                    selectedQuickActionLoading={selectedQuickActionLoading}
                    selectedQuickActionError={`${selectedQuickActionError}${selectedQuickActionErrorAt ? ` at ${selectedQuickActionErrorAt}` : ''}`}
                    onSelectQuickAction={setSelectedQuickActionId}
                    onPromoteQuickAction={(quickActionId) => void promoteQuickAction(quickActionId)}
                    onRefreshQuickActionDetail={() => void loadQuickActionDetail(selectedQuickActionId)}
                    onRetryLoadQuickActionDetail={() => void loadQuickActionDetail(selectedQuickActionId)}
                  />
                </div>
              ) : null}
            </div>
            {createTab === 'task' ? (
              <div className="modal-footer">
                <button className="button button-primary" type="submit" form="create-task-form">Create Task</button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {browseOpen ? (
        <div className="modal-scrim" role="dialog" aria-modal="true" aria-label="Browse repositories">
          <div className="modal-card">
            <header className="panel-head">
              <h2>Browse Repositories</h2>
              <button className="button" onClick={() => setBrowseOpen(false)}>Close</button>
            </header>
            <div className="browse-toolbar">
              <button className="button" onClick={() => void loadBrowseDirectories(browseParentPath || undefined)} disabled={!browseParentPath || browseLoading}>
                Up
              </button>
              <button className="button" onClick={() => void loadBrowseDirectories(browsePath || undefined)} disabled={browseLoading}>
                Refresh
              </button>
              <div className="browse-path-wrap">
                <input
                  className={`browse-path-input ${browseCurrentIsGit ? 'is-git' : ''}`}
                  value={browsePath}
                  onChange={(event) => setBrowsePath(event.target.value)}
                  aria-label="Browse path"
                />
                {browseCurrentIsGit ? <span className="git-chip">Git repo</span> : null}
              </div>
              <button className="button" onClick={() => void loadBrowseDirectories(browsePath || undefined)} disabled={!browsePath || browseLoading}>
                Go
              </button>
            </div>

            {browseError ? <p className="error-banner">{browseError}</p> : null}
            <div className="browse-list">
              {browseDirectories.map((entry) => (
                <button
                  key={entry.path}
                  className={`browse-item ${entry.is_git ? 'is-git' : ''}`}
                  onClick={() => void loadBrowseDirectories(entry.path)}
                >
                  <span className="browse-item-name">{entry.name}</span>
                  <span className="browse-item-kind">{entry.is_git ? 'git' : 'dir'}</span>
                </button>
              ))}
              {browseDirectories.length === 0 && !browseLoading ? <p className="empty">No directories found.</p> : null}
            </div>
            <div className="browse-actions">
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={browseAllowNonGit}
                  onChange={(event) => setBrowseAllowNonGit(event.target.checked)}
                />
                Allow non-git directory
              </label>
              <button className="button button-primary" onClick={() => void pinFromBrowse()} disabled={!browsePath || browseLoading}>
                Pin this folder
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
