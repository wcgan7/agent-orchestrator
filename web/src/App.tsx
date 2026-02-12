import { FormEvent, useEffect, useState } from 'react'
import { buildApiUrl } from './api'
import './styles/orchestrator.css'

type RouteKey = 'board' | 'execution' | 'review' | 'agents' | 'settings'
type CreateTab = 'task' | 'import' | 'quick'

type TaskRecord = {
  id: string
  title: string
  description?: string
  priority: string
  status: string
  blocked_by?: string[]
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

const STORAGE_PROJECT = 'feature-prd-runner-v3-project'
const STORAGE_ROUTE = 'feature-prd-runner-v3-route'

const ROUTES: Array<{ key: RouteKey; label: string }> = [
  { key: 'board', label: 'Board' },
  { key: 'execution', label: 'Execution' },
  { key: 'review', label: 'Review Queue' },
  { key: 'agents', label: 'Agents' },
  { key: 'settings', label: 'Settings' },
]

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
    throw new Error(`${response.status} ${response.statusText}`)
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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  const [workOpen, setWorkOpen] = useState(false)
  const [createTab, setCreateTab] = useState<CreateTab>('task')
  const [selectedTaskId, setSelectedTaskId] = useState<string>('')

  const [newTaskTitle, setNewTaskTitle] = useState('')
  const [newTaskDescription, setNewTaskDescription] = useState('')
  const [newTaskPriority, setNewTaskPriority] = useState('P2')

  const [importText, setImportText] = useState('')
  const [importJobId, setImportJobId] = useState('')
  const [importPreview, setImportPreview] = useState<PrdPreview | null>(null)

  const [quickPrompt, setQuickPrompt] = useState('')

  const [manualPinPath, setManualPinPath] = useState('')
  const [allowNonGit, setAllowNonGit] = useState(false)
  const [projectSearch, setProjectSearch] = useState('')

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

  async function reloadAll(): Promise<void> {
    setLoading(true)
    setError('')
    try {
      const [boardData, orchestratorData, reviewData, agentData, projectData] = await Promise.all([
        requestJson<BoardResponse>(buildApiUrl('/api/v3/tasks/board', projectDir)),
        requestJson<OrchestratorStatus>(buildApiUrl('/api/v3/orchestrator/status', projectDir)),
        requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/v3/review-queue', projectDir)),
        requestJson<{ agents: AgentRecord[] }>(buildApiUrl('/api/v3/agents', projectDir)),
        requestJson<{ projects: ProjectRef[] }>(buildApiUrl('/api/v3/projects', projectDir)),
      ])
      setBoard(boardData)
      setOrchestrator(orchestratorData)
      setReviewQueue(reviewData.tasks)
      setAgents(agentData.agents)
      setProjects(projectData.projects)
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
    socket.addEventListener('message', () => {
      void reloadAll()
    })
    socket.addEventListener('error', () => {
      socket.close()
    })
    return () => socket.close()
  }, [projectDir])

  async function submitTask(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!newTaskTitle.trim()) return
    await requestJson<{ task: TaskRecord }>(buildApiUrl('/api/v3/tasks', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: newTaskTitle.trim(),
        description: newTaskDescription,
        priority: newTaskPriority,
        status: 'backlog',
      }),
    })
    setNewTaskTitle('')
    setNewTaskDescription('')
    setNewTaskPriority('P2')
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
  }

  async function commitImport(): Promise<void> {
    if (!importJobId) return
    await requestJson<{ created_task_ids: string[] }>(buildApiUrl('/api/v3/import/prd/commit', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: importJobId }),
    })
    setImportJobId('')
    setImportPreview(null)
    setImportText('')
    setWorkOpen(false)
    await reloadAll()
  }

  async function submitQuickAction(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!quickPrompt.trim()) return
    await requestJson<{ quick_action: { id: string } }>(buildApiUrl('/api/v3/quick-actions', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: quickPrompt.trim() }),
    })
    setQuickPrompt('')
    setWorkOpen(false)
    await reloadAll()
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
      body: JSON.stringify({}),
    })
    await reloadAll()
  }

  async function spawnAgent(): Promise<void> {
    await requestJson<{ agent: AgentRecord }>(buildApiUrl('/api/v3/agents/spawn', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: 'general', capacity: 1 }),
    })
    await reloadAll()
  }

  async function agentAction(agentId: string, action: 'pause' | 'resume' | 'terminate'): Promise<void> {
    await requestJson<{ agent: AgentRecord }>(buildApiUrl(`/api/v3/agents/${agentId}/${action}`, projectDir), { method: 'POST' })
    await reloadAll()
  }

  async function pinManualProject(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!manualPinPath.trim()) return
    await requestJson<{ project: ProjectRef }>(buildApiUrl('/api/v3/projects/pinned', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: manualPinPath.trim(), allow_non_git: allowNonGit }),
    })
    setManualPinPath('')
    await reloadAll()
  }

  function renderBoard(): JSX.Element {
    const columns = ['backlog', 'ready', 'in_progress', 'in_review', 'blocked', 'done']
    const allTasks = columns.flatMap((column) => board.columns[column] || [])
    const selectedTask = allTasks.find((task) => task.id === selectedTaskId) || allTasks[0]
    const queueTasks = [...(board.columns.ready || []), ...(board.columns.in_progress || [])]
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Board</h2>
          <button className="button button-primary" onClick={() => setWorkOpen(true)}>Create Work</button>
        </header>
        <div className="workbench-grid">
          <article className="workbench-pane">
            <h3>Kanban</h3>
            <div className="board-grid">
              {columns.map((column) => (
                <article className="board-col" key={column}>
                  <h3>{column.replace('_', ' ')}</h3>
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
            {selectedTask ? (
              <div className="detail-card">
                <p className="task-title">{selectedTask.title}</p>
                <p className="task-meta">{selectedTask.id} · {selectedTask.priority} · {selectedTask.status}</p>
                {selectedTask.description ? <p className="task-desc">{selectedTask.description}</p> : <p className="task-desc">No description.</p>}
                <p className="field-label">Blockers: {(selectedTask.blocked_by || []).join(', ') || 'None'}</p>
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
                  <p className="task-meta">{task.status}</p>
                </div>
              ))}
              {queueTasks.length === 0 ? <p className="empty">No queued or running tasks.</p> : null}
              <p className="field-label">Agents ({agents.length})</p>
              {agents.slice(0, 4).map((agent) => (
                <div key={agent.id} className="row-card">
                  <p className="task-title">{agent.role}</p>
                  <p className="task-meta">{agent.status}</p>
                </div>
              ))}
            </div>
          </article>
        </div>
      </section>
    )
  }

  function renderExecution(): JSX.Element {
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
            <strong>{orchestrator?.status ?? 'unknown'}</strong>
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
          <button className="button button-primary" onClick={() => void spawnAgent()}>Spawn agent</button>
        </header>
        <div className="list-stack">
          {agents.map((agent) => (
            <div className="row-card" key={agent.id}>
              <div>
                <p className="task-title">{agent.role}</p>
                <p className="task-meta">{agent.id} · {agent.status}</p>
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
                  {project.path} ({project.source})
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
          </article>

          <article className="settings-card">
            <h3>Diagnostics</h3>
            <p>Schema version: 3</p>
            <p>Selected route: {route}</p>
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
          <div className="modal-card">
            <header className="panel-head">
              <h2>Create Work</h2>
              <button className="button" onClick={() => setWorkOpen(false)}>Close</button>
            </header>

            <div className="tab-row">
              <button className={`tab ${createTab === 'task' ? 'is-active' : ''}`} onClick={() => setCreateTab('task')}>Create Task</button>
              <button className={`tab ${createTab === 'import' ? 'is-active' : ''}`} onClick={() => setCreateTab('import')}>Import PRD</button>
              <button className={`tab ${createTab === 'quick' ? 'is-active' : ''}`} onClick={() => setCreateTab('quick')}>Quick Action</button>
            </div>

            {createTab === 'task' ? (
              <form className="form-stack" onSubmit={(event) => void submitTask(event)}>
                <label className="field-label" htmlFor="task-title">Title</label>
                <input id="task-title" value={newTaskTitle} onChange={(event) => setNewTaskTitle(event.target.value)} required />
                <label className="field-label" htmlFor="task-description">Description</label>
                <textarea id="task-description" rows={4} value={newTaskDescription} onChange={(event) => setNewTaskDescription(event.target.value)} />
                <label className="field-label" htmlFor="task-priority">Priority</label>
                <select id="task-priority" value={newTaskPriority} onChange={(event) => setNewTaskPriority(event.target.value)}>
                  <option value="P0">P0</option>
                  <option value="P1">P1</option>
                  <option value="P2">P2</option>
                  <option value="P3">P3</option>
                </select>
                <button className="button button-primary" type="submit">Create Task</button>
              </form>
            ) : null}

            {createTab === 'import' ? (
              <div className="form-stack">
                <form className="form-stack" onSubmit={(event) => void previewImport(event)}>
                  <label className="field-label" htmlFor="prd-text">PRD text</label>
                  <textarea id="prd-text" rows={8} value={importText} onChange={(event) => setImportText(event.target.value)} placeholder="- Task 1\n- Task 2" required />
                  <button className="button" type="submit">Preview</button>
                </form>
                {importJobId ? (
                  <div className="preview-box">
                    <p>Preview ready: {importJobId}</p>
                    {importPreview ? (
                      <div className="import-preview-graph">
                        <p className="field-label">Preview Graph</p>
                        {importPreview.nodes.map((node) => (
                          <div key={node.id} className="import-node">
                            <span>{node.id}</span>
                            <span>{node.title}</span>
                            <span>{node.priority}</span>
                          </div>
                        ))}
                        {importPreview.edges.length > 0 ? (
                          <p className="field-label">
                            Edges: {importPreview.edges.map((edge) => `${edge.from} -> ${edge.to}`).join(', ')}
                          </p>
                        ) : (
                          <p className="field-label">Edges: none</p>
                        )}
                      </div>
                    ) : null}
                    <button className="button button-primary" onClick={() => void commitImport()}>Commit to board</button>
                  </div>
                ) : null}
              </div>
            ) : null}

            {createTab === 'quick' ? (
              <form className="form-stack" onSubmit={(event) => void submitQuickAction(event)}>
                <p className="hint">Quick Action is ephemeral. Promote explicitly if you want it on the board.</p>
                <label className="field-label" htmlFor="quick-prompt">Prompt</label>
                <textarea id="quick-prompt" rows={6} value={quickPrompt} onChange={(event) => setQuickPrompt(event.target.value)} required />
                <button className="button button-primary" type="submit">Run Quick Action</button>
              </form>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
