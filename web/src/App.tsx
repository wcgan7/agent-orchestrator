import { FormEvent, useEffect, useRef, useState } from 'react'
import { buildApiUrl, buildAuthHeaders } from './api'
import { ImportJobPanel } from './components/AppPanels/ImportJobPanel'
import { QuickActionDetailPanel } from './components/AppPanels/QuickActionDetailPanel'
import HITLModeSelector from './components/HITLModeSelector/HITLModeSelector'
import Markdown from 'react-markdown'
import { humanizeLabel } from './ui/labels'
import './styles/orchestrator.css'

type RouteKey = 'board' | 'planning' | 'execution' | 'review' | 'agents' | 'settings'
type CreateTab = 'task' | 'import' | 'quick'
type TaskDetailTab = 'overview' | 'logs' | 'activity' | 'dependencies' | 'configuration'
type TaskActionKey = 'save' | 'run' | 'retry' | 'cancel' | 'transition'

type TaskRecord = {
  id: string
  title: string
  description?: string
  task_type?: string
  current_step?: string | null
  current_agent_id?: string | null
  worker_model?: string | null
  priority: string
  status: string
  labels?: string[]
  approval_mode?: 'human_review' | 'auto_approve'
  blocked_by?: string[]
  blocks?: string[]
  parent_id?: string | null
  pipeline_template?: string[]
  retry_count?: number
  hitl_mode?: string
  pending_gate?: string | null
  quality_gate?: Record<string, number>
  metadata?: Record<string, unknown>
  human_blocking_issues?: HumanBlockingIssue[]
  error?: string | null
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

type WorkerHealthRecord = {
  name: string
  type: string
  configured: boolean
  healthy: boolean
  status: 'connected' | 'unavailable' | 'not_configured'
  detail: string
  checked_at: string
  command?: string | null
  endpoint?: string | null
  model?: string | null
}

type WorkerRoutingRow = {
  step: string
  provider: string
  provider_type?: string | null
  source: 'default' | 'explicit'
  configured: boolean
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

type TaskLogsSnapshot = {
  mode: 'active' | 'last' | 'none'
  task_status?: string
  step?: string
  stdout: string
  stderr: string
  stdout_offset?: number
  stderr_offset?: number
  stdout_chunk_start?: number
  stderr_chunk_start?: number
  stdout_tail_start?: number
  stderr_tail_start?: number
  started_at?: string | null
  finished_at?: string | null
  log_id?: string
  progress?: Record<string, unknown>
}

type LogAccumPhase = 'init' | 'backfill' | 'forward'

type LogAccumState = {
  taskId: string
  logId: string
  stdoutOffset: number
  stderrOffset: number
  stdoutBackfillOffset: number
  stderrBackfillOffset: number
  stdout: string
  stderr: string
  phase: LogAccumPhase
  stdoutTailStart: number
  stderrTailStart: number
}

type LogPaneKey = 'stdout' | 'stderr'

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

type CollaborationMode = {
  mode: string
  display_name: string
  description: string
}

type WorkerProviderSettings = {
  type: 'codex' | 'ollama' | 'claude'
  command?: string
  reasoning_effort?: 'low' | 'medium' | 'high'
  endpoint?: string
  model?: string
  temperature?: number
  num_ctx?: number
}

type LanguageCommandSettings = Record<string, string>

type SystemSettings = {
  orchestrator: {
    concurrency: number
    auto_deps: boolean
    max_review_attempts: number
  }
  agent_routing: {
    default_role: string
    task_type_roles: Record<string, string>
    role_provider_overrides: Record<string, string>
  }
  defaults: {
    quality_gate: {
      critical: number
      high: number
      medium: number
      low: number
    }
  }
  workers: {
    default: string
    default_model: string
    routing: Record<string, string>
    providers: Record<string, WorkerProviderSettings>
  }
  project: {
    commands: Record<string, LanguageCommandSettings>
  }
}

type MetricsSnapshot = {
  tokens_used: number
  api_calls: number
  estimated_cost_usd: number
  wall_time_seconds: number
  phases_completed: number
  phases_total: number
  files_changed: number
  lines_added: number
  lines_removed: number
  queue_depth: number
  in_progress: number
}

type RootSnapshot = {
  project_id?: string
}


type CollaborationTimelineEvent = {
  id: string
  type: string
  timestamp: string
  actor: string
  actor_type: string
  summary: string
  details: string
  human_blocking_issues?: HumanBlockingIssue[]
}

type HumanBlockingIssue = {
  summary: string
  details?: string
  category?: string
  action?: string
  blocking_on?: string
  severity?: string
}

type CollaborationFeedbackItem = {
  id: string
  task_id: string
  feedback_type: string
  priority: string
  status: string
  summary: string
  details: string
  target_file?: string | null
  created_by?: string | null
  created_at?: string | null
  agent_response?: string | null
}

type CollaborationCommentItem = {
  id: string
  task_id: string
  file_path: string
  line_number: number
  line_type?: string | null
  body: string
  author?: string | null
  created_at?: string | null
  resolved: boolean
  parent_id?: string | null
}

type PlanRevisionRecord = {
  id: string
  task_id: string
  created_at: string
  source: 'worker_plan' | 'worker_refine' | 'human_edit' | 'import'
  parent_revision_id?: string | null
  step?: string | null
  feedback_note?: string | null
  provider?: string | null
  model?: string | null
  content: string
  content_hash: string
  status: 'draft' | 'committed'
}

type PlanRefineJobRecord = {
  id: string
  task_id: string
  base_revision_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  feedback: string
  instructions?: string | null
  priority?: 'normal' | 'high'
  result_revision_id?: string | null
  error?: string | null
}

type TaskPlanDocument = {
  task_id: string
  latest_revision_id?: string | null
  committed_revision_id?: string | null
  revisions: PlanRevisionRecord[]
  active_refine_job?: PlanRefineJobRecord | null
  plans: Array<{ step?: string | null; ts?: string | null; content?: string }>
  latest?: { step?: string | null; ts?: string | null; content?: string } | null
}

const STORAGE_PROJECT = 'agent-orchestrator-project'
const STORAGE_ROUTE = 'agent-orchestrator-route'
const ADD_REPO_VALUE = '__add_new_repo__'
const MOBILE_BOARD_BREAKPOINT = 640
const WS_RELOAD_CHANNELS = new Set(['tasks', 'queue', 'agents', 'review', 'quick_actions', 'notifications'])
const LOG_CHUNK_CHARS = 200_000
const LOG_HISTORY_MAX_CHARS = 5_000_000
const LOG_NEAR_BOTTOM_PX = 120

function createEmptyLogAccum(taskId = ''): LogAccumState {
  return {
    taskId,
    logId: '',
    stdoutOffset: 0,
    stderrOffset: 0,
    stdoutBackfillOffset: 0,
    stderrBackfillOffset: 0,
    stdout: '',
    stderr: '',
    phase: 'init',
    stdoutTailStart: 0,
    stderrTailStart: 0,
  }
}

const ROUTES: Array<{ key: RouteKey; label: string }> = [
  { key: 'board', label: 'Board' },
  { key: 'planning', label: 'Planning' },
  { key: 'execution', label: 'Execution' },
  { key: 'review', label: 'Review Queue' },
  { key: 'agents', label: 'Workers' },
  { key: 'settings', label: 'Settings' },
]

const TASK_TYPE_OPTIONS = [
  'feature',
  'plan',
  'bug',
  'refactor',
  'research',
  'test',
  'docs',
  'security',
  'performance',
]

const TASK_STATUS_OPTIONS = ['backlog', 'queued', 'in_progress', 'in_review', 'blocked', 'done', 'cancelled']
const DEFAULT_COLLABORATION_MODES: CollaborationMode[] = [
  { mode: 'autopilot', display_name: 'Autopilot', description: 'Agents run freely.' },
  { mode: 'supervised', display_name: 'Supervised', description: 'Approve each step.' },
  { mode: 'collaborative', display_name: 'Collaborative', description: 'Work together with agents.' },
  { mode: 'review_only', display_name: 'Review Only', description: 'Review changes before commit.' },
]
const DEFAULT_SETTINGS: SystemSettings = {
  orchestrator: {
    concurrency: 2,
    auto_deps: true,
    max_review_attempts: 10,
  },
  agent_routing: {
    default_role: 'general',
    task_type_roles: {},
    role_provider_overrides: {},
  },
  defaults: {
    quality_gate: {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
    },
  },
  workers: {
    default: 'codex',
    default_model: '',
    routing: {},
    providers: {
      codex: { type: 'codex', command: 'codex exec' },
    },
  },
  project: {
    commands: {},
  },
}

const TASK_TYPE_ROLE_MAP_EXAMPLE = `{
  "bug": "debugger",
  "docs": "researcher"
}`

const ROLE_PROVIDER_OVERRIDES_EXAMPLE = `{
  "reviewer": "codex"
}`

const WORKER_ROUTING_EXAMPLE = `{
  "plan": "codex",
  "implement": "ollama",
  "review": "codex"
}`

const WORKER_PROVIDERS_EXAMPLE = `{
  "review-fastlane": {
    "type": "claude",
    "command": "claude -p",
    "model": "sonnet",
    "reasoning_effort": "high"
  },
  "local-lab-8b": {
    "type": "ollama",
    "endpoint": "http://localhost:11434",
    "model": "llama3.1:8b"
  }
}`

const PROJECT_COMMANDS_EXAMPLE = `{
  "python": {
    "test": ".venv/bin/pytest -n auto",
    "lint": ".venv/bin/ruff check ."
  },
  "typescript": {
    "test": "npm test",
    "lint": "npm run lint"
  }
}`

function RenderedMarkdown({ content, className }: { content: string; className?: string }): JSX.Element {
  return (
    <div className={`rendered-markdown ${className || ''}`}>
      <Markdown>{content}</Markdown>
    </div>
  )
}

function statusPillClass(status: string): string {
  switch (status) {
    case 'in_progress': return 'status-running'
    case 'in_review': return 'status-review'
    case 'done': return 'status-done'
    case 'blocked': return 'status-blocked'
    case 'cancelled': return 'status-failed'
    default: return 'status-paused'
  }
}

function routeFromHash(hash: string): RouteKey {
  const cleaned = hash.replace(/^#\/?/, '').trim().toLowerCase()
  const found = ROUTES.find((route) => route.key === cleaned)
  return found?.key ?? 'board'
}

function toHash(route: RouteKey): string {
  return `#/${route}`
}

function isMobileBoardViewport(): boolean {
  return window.innerWidth <= MOBILE_BOARD_BREAKPOINT
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: buildAuthHeaders(init?.headers || {}),
  })
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

function parseStringMap(input: string, label: string): Record<string, string> {
  if (!input.trim()) return {}
  let parsed: unknown
  try {
    parsed = JSON.parse(input)
  } catch {
    throw new Error(`${label} must be valid JSON`)
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`)
  }
  const out: Record<string, string> = {}
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    const normalizedKey = String(key || '').trim()
    const normalizedValue = String(value || '').trim()
    if (normalizedKey && normalizedValue) {
      out[normalizedKey] = normalizedValue
    }
  }
  return out
}

function parseProjectCommands(input: string): Record<string, LanguageCommandSettings> {
  if (!input.trim()) return {}
  let parsed: unknown
  try {
    parsed = JSON.parse(input)
  } catch {
    throw new Error('Project commands must be valid JSON')
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Project commands must be a JSON object')
  }

  const out: Record<string, LanguageCommandSettings> = {}
  for (const [rawLanguage, rawCommands] of Object.entries(parsed as Record<string, unknown>)) {
    const language = String(rawLanguage || '').trim().toLowerCase()
    if (!language) continue
    if (!rawCommands || typeof rawCommands !== 'object' || Array.isArray(rawCommands)) {
      throw new Error(`Project commands for "${language}" must be a JSON object`)
    }
    const commands: LanguageCommandSettings = {}
    for (const [rawField, rawValue] of Object.entries(rawCommands as Record<string, unknown>)) {
      const field = String(rawField || '').trim()
      if (!field) continue
      if (typeof rawValue !== 'string') {
        throw new Error(`Project command "${language}.${field}" must be a string`)
      }
      commands[field] = rawValue
    }
    if (Object.keys(commands).length > 0) {
      out[language] = commands
    }
  }
  return out
}

function formatJsonObjectInput(input: string, label: string): string {
  const trimmed = input.trim()
  if (!trimmed) return ''
  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    throw new Error(`${label} must be valid JSON`)
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`)
  }
  return JSON.stringify(parsed, null, 2)
}

function parseWorkerProviders(input: string): Record<string, WorkerProviderSettings> {
  if (!input.trim()) {
    return {}
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(input)
  } catch {
    throw new Error('Worker providers must be valid JSON')
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Worker providers must be a JSON object')
  }
  const out: Record<string, WorkerProviderSettings> = {}
  for (const [rawName, rawValue] of Object.entries(parsed as Record<string, unknown>)) {
    const name = String(rawName || '').trim()
    if (!name) continue
    if (!rawValue || typeof rawValue !== 'object' || Array.isArray(rawValue)) {
      throw new Error(`Worker provider "${name}" must be a JSON object`)
    }
    const record = rawValue as Record<string, unknown>
    let type = String(record.type || (name === 'codex' ? 'codex' : name === 'claude' ? 'claude' : '')).trim().toLowerCase()
    if (type === 'local') type = 'ollama'
    if (type !== 'codex' && type !== 'ollama' && type !== 'claude') {
      throw new Error(`Worker provider "${name}" has invalid type "${type}" (allowed: codex, ollama, claude)`)
    }
    if (type === 'codex' || type === 'claude') {
      const defaultCommand = type === 'codex' ? 'codex exec' : 'claude -p'
      const command = String(record.command || defaultCommand).trim() || defaultCommand
      const provider: WorkerProviderSettings = { type, command }
      const model = String(record.model || '').trim()
      if (model) provider.model = model
      const reasoningEffort = String(record.reasoning_effort || '').trim().toLowerCase()
      if (reasoningEffort === 'low' || reasoningEffort === 'medium' || reasoningEffort === 'high') {
        provider.reasoning_effort = reasoningEffort
      }
      out[name] = provider
      continue
    }
    const provider: WorkerProviderSettings = { type: 'ollama' }
    const endpoint = String(record.endpoint || '').trim()
    const model = String(record.model || '').trim()
    if (endpoint) provider.endpoint = endpoint
    if (model) provider.model = model
    const maybeTemperature = Number(record.temperature)
    if (Number.isFinite(maybeTemperature)) {
      provider.temperature = maybeTemperature
    }
    const maybeNumCtx = Number(record.num_ctx)
    if (Number.isFinite(maybeNumCtx) && maybeNumCtx > 0) {
      provider.num_ctx = Math.floor(maybeNumCtx)
    }
    out[name] = provider
  }
  return out
}

function parseNonNegativeInt(input: string, fallback: number): number {
  const parsed = Number(input)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(0, Math.floor(parsed))
}

function normalizeWorkers(payload: Partial<SystemSettings['workers']> | null | undefined): SystemSettings['workers'] {
  const defaultWorker = String(payload?.default || DEFAULT_SETTINGS.workers.default).trim() || 'codex'
  const defaultModel = String(payload?.default_model || '').trim()
  const routingRaw = payload?.routing && typeof payload.routing === 'object' ? payload.routing : {}
  const providersRaw = payload?.providers && typeof payload.providers === 'object' ? payload.providers : {}
  const routing: Record<string, string> = {}
  for (const [rawKey, rawValue] of Object.entries(routingRaw)) {
    const key = String(rawKey || '').trim()
    const value = String(rawValue || '').trim()
    if (key && value) {
      routing[key] = value
    }
  }

  const providers: Record<string, WorkerProviderSettings> = {}
  for (const [rawName, rawValue] of Object.entries(providersRaw)) {
    const name = String(rawName || '').trim()
    if (!name || !rawValue || typeof rawValue !== 'object') continue
    const value = rawValue as Record<string, unknown>
    let type = String(value.type || (name === 'codex' ? 'codex' : name === 'claude' ? 'claude' : '')).trim().toLowerCase()
    if (type === 'local') type = 'ollama'
    if (type !== 'codex' && type !== 'ollama' && type !== 'claude') continue
    if (type === 'codex' || type === 'claude') {
      const defaultCommand = type === 'codex' ? 'codex exec' : 'claude -p'
      const provider: WorkerProviderSettings = {
        type,
        command: String(value.command || defaultCommand).trim() || defaultCommand,
      }
      const model = String(value.model || '').trim()
      if (model) provider.model = model
      const reasoningEffort = String(value.reasoning_effort || '').trim().toLowerCase()
      if (reasoningEffort === 'low' || reasoningEffort === 'medium' || reasoningEffort === 'high') {
        provider.reasoning_effort = reasoningEffort
      }
      providers[name] = provider
      continue
    }
    const provider: WorkerProviderSettings = { type: 'ollama' }
    const endpoint = String(value.endpoint || '').trim()
    const model = String(value.model || '').trim()
    if (endpoint) provider.endpoint = endpoint
    if (model) provider.model = model
    const maybeTemperature = Number(value.temperature)
    if (Number.isFinite(maybeTemperature)) provider.temperature = maybeTemperature
    const maybeNumCtx = Number(value.num_ctx)
    if (Number.isFinite(maybeNumCtx) && maybeNumCtx > 0) provider.num_ctx = Math.floor(maybeNumCtx)
    providers[name] = provider
  }
  if (!providers.codex || providers.codex.type !== 'codex') {
    providers.codex = { type: 'codex', command: 'codex exec' }
  }
  const effectiveDefault = providers[defaultWorker] ? defaultWorker : 'codex'
  return {
    default: effectiveDefault,
    default_model: defaultModel,
    routing,
    providers,
  }
}

function normalizeSettings(payload: Partial<SystemSettings> | null | undefined): SystemSettings {
  const orchestrator: Partial<SystemSettings['orchestrator']> = payload?.orchestrator || {}
  const routing: Partial<SystemSettings['agent_routing']> = payload?.agent_routing || {}
  const defaults: Partial<SystemSettings['defaults']> = payload?.defaults || {}
  const qualityGate: Partial<SystemSettings['defaults']['quality_gate']> = defaults.quality_gate || {}
  const workers = normalizeWorkers(payload?.workers)
  const projectCommandsRaw = payload?.project?.commands
  const projectCommands: Record<string, LanguageCommandSettings> = {}
  if (projectCommandsRaw && typeof projectCommandsRaw === 'object') {
    for (const [rawLanguage, rawCommands] of Object.entries(projectCommandsRaw)) {
      const language = String(rawLanguage || '').trim().toLowerCase()
      if (!language || !rawCommands || typeof rawCommands !== 'object' || Array.isArray(rawCommands)) continue
      const commands: LanguageCommandSettings = {}
      for (const [rawField, rawValue] of Object.entries(rawCommands as Record<string, unknown>)) {
        const field = String(rawField || '').trim()
        if (!field || typeof rawValue !== 'string') continue
        commands[field] = rawValue
      }
      if (Object.keys(commands).length > 0) {
        projectCommands[language] = commands
      }
    }
  }

  const maybeConcurrency = Number(orchestrator.concurrency)
  const maybeMaxReviewAttempts = Number(orchestrator.max_review_attempts)
  const maybeCritical = Number(qualityGate.critical)
  const maybeHigh = Number(qualityGate.high)
  const maybeMedium = Number(qualityGate.medium)
  const maybeLow = Number(qualityGate.low)

  return {
    orchestrator: {
      concurrency: Number.isFinite(maybeConcurrency) ? Math.max(1, Math.floor(maybeConcurrency)) : DEFAULT_SETTINGS.orchestrator.concurrency,
      auto_deps: typeof orchestrator.auto_deps === 'boolean' ? orchestrator.auto_deps : DEFAULT_SETTINGS.orchestrator.auto_deps,
      max_review_attempts: Number.isFinite(maybeMaxReviewAttempts) ? Math.max(1, Math.floor(maybeMaxReviewAttempts)) : DEFAULT_SETTINGS.orchestrator.max_review_attempts,
    },
    agent_routing: {
      default_role: String(routing.default_role || DEFAULT_SETTINGS.agent_routing.default_role),
      task_type_roles: routing.task_type_roles && typeof routing.task_type_roles === 'object' ? routing.task_type_roles : {},
      role_provider_overrides: routing.role_provider_overrides && typeof routing.role_provider_overrides === 'object' ? routing.role_provider_overrides : {},
    },
    defaults: {
      quality_gate: {
        critical: Number.isFinite(maybeCritical) ? Math.max(0, Math.floor(maybeCritical)) : DEFAULT_SETTINGS.defaults.quality_gate.critical,
        high: Number.isFinite(maybeHigh) ? Math.max(0, Math.floor(maybeHigh)) : DEFAULT_SETTINGS.defaults.quality_gate.high,
        medium: Number.isFinite(maybeMedium) ? Math.max(0, Math.floor(maybeMedium)) : DEFAULT_SETTINGS.defaults.quality_gate.medium,
        low: Number.isFinite(maybeLow) ? Math.max(0, Math.floor(maybeLow)) : DEFAULT_SETTINGS.defaults.quality_gate.low,
      },
    },
    workers,
    project: {
      commands: projectCommands,
    },
  }
}

function describeTask(taskId: string, taskIndex: Map<string, TaskRecord>): { label: string; status: string } {
  const task = taskIndex.get(taskId)
  if (!task) {
    return { label: taskId, status: 'unknown' }
  }
  const title = task.title?.trim() || taskId
  return {
    label: `${title} (${task.id})`,
    status: task.status || 'unknown',
  }
}

function normalizeHumanBlockingIssues(value: unknown): HumanBlockingIssue[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => {
      const summary = String(item.summary || item.issue || '').trim()
      const details = String(item.details || item.rationale || '').trim()
      const category = String(item.category || '').trim()
      const action = String(item.action || '').trim()
      const blockingOn = String(item.blocking_on || '').trim()
      const severity = String(item.severity || '').trim()
      return {
        summary: summary || (details ? details.split('\n')[0] : ''),
        details: details || undefined,
        category: category || undefined,
        action: action || undefined,
        blocking_on: blockingOn || undefined,
        severity: severity || undefined,
      }
    })
    .filter((item) => !!item.summary)
}

function normalizeMetrics(payload: unknown): MetricsSnapshot | null {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return null
  }
  const raw = payload as Record<string, unknown>
  const toNumber = (value: unknown, fallback = 0): number => {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return {
    tokens_used: toNumber(raw.tokens_used),
    api_calls: toNumber(raw.api_calls),
    estimated_cost_usd: toNumber(raw.estimated_cost_usd),
    wall_time_seconds: toNumber(raw.wall_time_seconds),
    phases_completed: toNumber(raw.phases_completed),
    phases_total: toNumber(raw.phases_total),
    files_changed: toNumber(raw.files_changed),
    lines_added: toNumber(raw.lines_added),
    lines_removed: toNumber(raw.lines_removed),
    queue_depth: toNumber(raw.queue_depth),
    in_progress: toNumber(raw.in_progress),
  }
}

function normalizeWorkerHealth(payload: unknown): WorkerHealthRecord[] {
  const itemsRaw = payload && typeof payload === 'object' && !Array.isArray(payload) && Array.isArray((payload as { providers?: unknown[] }).providers)
    ? (payload as { providers: unknown[] }).providers
    : []
  return itemsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => {
      const rawStatus = String(item.status || '').trim().toLowerCase()
      const status: WorkerHealthRecord['status'] =
        rawStatus === 'connected' || rawStatus === 'unavailable' || rawStatus === 'not_configured'
          ? rawStatus
          : 'unavailable'
      return {
        name: String(item.name || '').trim(),
        type: String(item.type || '').trim(),
        configured: Boolean(item.configured),
        healthy: Boolean(item.healthy),
        status,
        detail: String(item.detail || '').trim(),
        checked_at: String(item.checked_at || '').trim(),
        command: item.command ? String(item.command) : null,
        endpoint: item.endpoint ? String(item.endpoint) : null,
        model: item.model ? String(item.model) : null,
      }
    })
    .filter((item) => !!item.name)
}

function normalizeWorkerRouting(payload: unknown): { defaultProvider: string; rows: WorkerRoutingRow[] } {
  const root = payload && typeof payload === 'object' && !Array.isArray(payload)
    ? payload as { default?: unknown; rows?: unknown[] }
    : {}
  const rowsRaw = Array.isArray(root.rows) ? root.rows : []
  const rows = rowsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => {
      const source: WorkerRoutingRow['source'] =
        String(item.source || '').trim().toLowerCase() === 'explicit' ? 'explicit' : 'default'
      return {
        step: String(item.step || '').trim(),
        provider: String(item.provider || '').trim(),
        provider_type: item.provider_type ? String(item.provider_type) : null,
        source,
        configured: Boolean(item.configured),
      }
    })
    .filter((item) => !!item.step && !!item.provider)
  return {
    defaultProvider: String(root.default || 'codex').trim() || 'codex',
    rows,
  }
}

function normalizeTimelineEvents(payload: unknown): CollaborationTimelineEvent[] {
  const eventsRaw = payload && typeof payload === 'object' && !Array.isArray(payload) && Array.isArray((payload as { events?: unknown[] }).events)
    ? (payload as { events: unknown[] }).events
    : []
  return eventsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => ({
      id: String(item.id || '').trim(),
      type: String(item.type || 'event').trim(),
      timestamp: String(item.timestamp || '').trim(),
      actor: String(item.actor || 'system').trim(),
      actor_type: String(item.actor_type || 'system').trim(),
      summary: String(item.summary || '').trim(),
      details: String(item.details || '').trim(),
      human_blocking_issues: normalizeHumanBlockingIssues(item.human_blocking_issues),
    }))
    .filter((item) => !!item.id)
}

function normalizeFeedbackItems(payload: unknown): CollaborationFeedbackItem[] {
  const feedbackRaw = payload && typeof payload === 'object' && !Array.isArray(payload) && Array.isArray((payload as { feedback?: unknown[] }).feedback)
    ? (payload as { feedback: unknown[] }).feedback
    : []
  return feedbackRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => ({
      id: String(item.id || '').trim(),
      task_id: String(item.task_id || '').trim(),
      feedback_type: String(item.feedback_type || 'general').trim(),
      priority: String(item.priority || 'should').trim(),
      status: String(item.status || 'active').trim(),
      summary: String(item.summary || '').trim(),
      details: String(item.details || '').trim(),
      target_file: item.target_file ? String(item.target_file) : null,
      created_by: item.created_by ? String(item.created_by) : null,
      created_at: item.created_at ? String(item.created_at) : null,
      agent_response: item.agent_response ? String(item.agent_response) : null,
    }))
    .filter((item) => !!item.id)
}

function normalizeComments(payload: unknown): CollaborationCommentItem[] {
  const commentsRaw = payload && typeof payload === 'object' && !Array.isArray(payload) && Array.isArray((payload as { comments?: unknown[] }).comments)
    ? (payload as { comments: unknown[] }).comments
    : []
  return commentsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => ({
      id: String(item.id || '').trim(),
      task_id: String(item.task_id || '').trim(),
      file_path: String(item.file_path || '').trim(),
      line_number: Number.isFinite(Number(item.line_number)) ? Math.max(0, Math.floor(Number(item.line_number))) : 0,
      line_type: item.line_type ? String(item.line_type) : null,
      body: String(item.body || '').trim(),
      author: item.author ? String(item.author) : null,
      created_at: item.created_at ? String(item.created_at) : null,
      resolved: Boolean(item.resolved),
      parent_id: item.parent_id ? String(item.parent_id) : null,
    }))
    .filter((item) => !!item.id)
}

function normalizePlanRevision(item: unknown): PlanRevisionRecord | null {
  if (!item || typeof item !== 'object' || Array.isArray(item)) return null
  const raw = item as Record<string, unknown>
  const id = String(raw.id || '').trim()
  const taskId = String(raw.task_id || '').trim()
  const content = String(raw.content || '')
  if (!id || !taskId) return null
  const sourceRaw = String(raw.source || 'human_edit').trim()
  const source: PlanRevisionRecord['source'] = (
    sourceRaw === 'worker_plan' || sourceRaw === 'worker_refine' || sourceRaw === 'import'
  ) ? sourceRaw : 'human_edit'
  const statusRaw = String(raw.status || 'draft').trim()
  const status: PlanRevisionRecord['status'] = statusRaw === 'committed' ? 'committed' : 'draft'
  return {
    id,
    task_id: taskId,
    created_at: String(raw.created_at || ''),
    source,
    parent_revision_id: raw.parent_revision_id ? String(raw.parent_revision_id) : null,
    step: raw.step ? String(raw.step) : null,
    feedback_note: raw.feedback_note ? String(raw.feedback_note) : null,
    provider: raw.provider ? String(raw.provider) : null,
    model: raw.model ? String(raw.model) : null,
    content,
    content_hash: String(raw.content_hash || ''),
    status,
  }
}

function normalizePlanRefineJob(item: unknown): PlanRefineJobRecord | null {
  if (!item || typeof item !== 'object' || Array.isArray(item)) return null
  const raw = item as Record<string, unknown>
  const id = String(raw.id || '').trim()
  const taskId = String(raw.task_id || '').trim()
  const baseRevisionId = String(raw.base_revision_id || '').trim()
  if (!id || !taskId || !baseRevisionId) return null
  const statusRaw = String(raw.status || 'queued').trim()
  const status: PlanRefineJobRecord['status'] = (
    statusRaw === 'running' || statusRaw === 'completed' || statusRaw === 'failed' || statusRaw === 'cancelled'
  ) ? statusRaw : 'queued'
  const priorityRaw = String(raw.priority || 'normal').trim()
  const priority: PlanRefineJobRecord['priority'] = priorityRaw === 'high' ? 'high' : 'normal'
  return {
    id,
    task_id: taskId,
    base_revision_id: baseRevisionId,
    status,
    created_at: String(raw.created_at || ''),
    started_at: raw.started_at ? String(raw.started_at) : null,
    finished_at: raw.finished_at ? String(raw.finished_at) : null,
    feedback: String(raw.feedback || ''),
    instructions: raw.instructions ? String(raw.instructions) : null,
    priority,
    result_revision_id: raw.result_revision_id ? String(raw.result_revision_id) : null,
    error: raw.error ? String(raw.error) : null,
  }
}

function normalizeTaskPlan(payload: unknown): TaskPlanDocument {
  const root = payload && typeof payload === 'object' && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {}
  const revisionsRaw = Array.isArray(root.revisions) ? root.revisions : []
  const revisions = revisionsRaw.map((item) => normalizePlanRevision(item)).filter((item): item is PlanRevisionRecord => item !== null)
  const plansRaw = Array.isArray(root.plans) ? root.plans : []
  const plans = plansRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => ({ step: item.step ? String(item.step) : null, ts: item.ts ? String(item.ts) : null, content: item.content ? String(item.content) : '' }))
  const latestRaw = root.latest && typeof root.latest === 'object' && !Array.isArray(root.latest)
    ? root.latest as Record<string, unknown>
    : null
  return {
    task_id: String(root.task_id || ''),
    latest_revision_id: root.latest_revision_id ? String(root.latest_revision_id) : null,
    committed_revision_id: root.committed_revision_id ? String(root.committed_revision_id) : null,
    revisions,
    active_refine_job: normalizePlanRefineJob(root.active_refine_job || null),
    plans,
    latest: latestRaw ? { step: latestRaw.step ? String(latestRaw.step) : null, ts: latestRaw.ts ? String(latestRaw.ts) : null, content: latestRaw.content ? String(latestRaw.content) : '' } : null,
  }
}

function normalizePlanRefineJobs(payload: unknown): PlanRefineJobRecord[] {
  const root = payload && typeof payload === 'object' && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {}
  const jobsRaw = Array.isArray(root.jobs) ? root.jobs : []
  return jobsRaw
    .map((item) => normalizePlanRefineJob(item))
    .filter((item): item is PlanRefineJobRecord => item !== null)
}

function toLocaleTimestamp(value?: string | null): string {
  if (!value) return ''
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

function inferProjectId(projectDir: string): string {
  const normalized = projectDir.trim().replace(/[\\/]+$/, '')
  if (!normalized) return ''
  const parts = normalized.split(/[\\/]/).filter(Boolean)
  return parts[parts.length - 1] || ''
}

function repoNameFromPath(projectPath: string): string {
  const normalized = projectPath.trim().replace(/[\\/]+$/, '')
  if (!normalized) return ''
  const parts = normalized.split(/[\\/]/).filter(Boolean)
  return parts[parts.length - 1] || normalized
}

function summarizePlanDiff(nextText: string, prevText: string): { added: number; removed: number; preview: string[] } {
  const nextLines = nextText.split('\n')
  const prevSet = new Set(prevText.split('\n'))
  const nextSet = new Set(nextLines)
  let added = 0
  for (const line of nextLines) {
    if (!prevSet.has(line)) added += 1
  }
  let removed = 0
  for (const line of prevSet) {
    if (!nextSet.has(line)) removed += 1
  }
  const preview = nextLines
    .filter((line) => !prevSet.has(line))
    .map((line) => `+ ${line}`.trimEnd())
    .slice(0, 8)
  return { added, removed, preview }
}

function looksLikeJson(s: string): boolean {
  const c = s[0]
  return c === '{' || c === '[' || c === '"' || s.includes('":') || s.includes('_json_delta') || s.includes('session_id')
}

function renderStructuredStdout(raw: string): {
  text: string
  hasContent: boolean
  structured: boolean
  parsedLines: number
  streamEvents: number
} {
  const input = String(raw || '')
  if (!input.trim()) {
    return { text: '', hasContent: false, structured: false, parsedLines: 0, streamEvents: 0 }
  }
  const lines = input.split('\n')
  let parsedCount = 0
  let streamEvents = 0
  const deltaParts: string[] = []
  const toolNames: string[] = []
  const toolErrors: string[] = []
  const plainLines: string[] = []
  let assistantMessage = ''
  let resultMessage = ''
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    let obj: unknown
    try {
      obj = JSON.parse(trimmed)
    } catch {
      if (!looksLikeJson(trimmed)) plainLines.push(trimmed)
      continue
    }
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
      if (!looksLikeJson(trimmed)) plainLines.push(trimmed)
      continue
    }
    parsedCount += 1
    const record = obj as Record<string, unknown>
    const type = String(record.type || '')
    if (type === 'assistant') {
      const message = record.message
      if (message && typeof message === 'object' && !Array.isArray(message)) {
        const content = (message as Record<string, unknown>).content
        if (Array.isArray(content)) {
          const text = content
            .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
            .filter((item) => String(item.type || '') === 'text')
            .map((item) => String(item.text || ''))
            .join('')
            .trim()
          if (text) assistantMessage = text
        }
      }
      continue
    }
    if (type === 'result') {
      const result = record.result
      if (typeof result === 'string' && result.trim()) {
        resultMessage = result.trim()
      }
      continue
    }
    if (type === 'stream_event') {
      streamEvents += 1
      const event = record.event
      if (!event || typeof event !== 'object' || Array.isArray(event)) continue
      const e = event as Record<string, unknown>
      const eventType = String(e.type || '')
      if (eventType === 'content_block_start') {
        const block = e.content_block
        if (block && typeof block === 'object' && !Array.isArray(block)) {
          const b = block as Record<string, unknown>
          const blockType = String(b.type || '')
          if (blockType === 'tool_use') {
            const name = String(b.name || '')
            if (name) toolNames.push(name)
          }
          const startText = String(b.text || '')
          if (startText) deltaParts.push(startText)
        }
        continue
      }
      if (eventType === 'content_block_delta') {
        const delta = e.delta
        if (!delta || typeof delta !== 'object' || Array.isArray(delta)) continue
        const d = delta as Record<string, unknown>
        const deltaType = String(d.type || '')
        if (deltaType === 'text_delta') {
          const text = String(d.text || '')
          if (text) deltaParts.push(text)
        }
      }
      // All other stream events (message_start, message_delta, message_stop,
      // content_block_stop, input_json_delta, ping) are silently skipped.
      continue
    }
    if (type === 'user') {
      const message = record.message
      if (message && typeof message === 'object' && !Array.isArray(message)) {
        const content = (message as Record<string, unknown>).content
        if (Array.isArray(content)) {
          for (const item of content) {
            if (!item || typeof item !== 'object' || Array.isArray(item)) continue
            const ci = item as Record<string, unknown>
            if (String(ci.type || '') === 'tool_result' && ci.is_error) {
              const errText = String(ci.content || '').trim()
              if (errText && !toolErrors.includes(errText)) toolErrors.push(errText)
            }
          }
        }
      }
      // Also check top-level tool_use_result shorthand
      const toolUseResult = record.tool_use_result
      if (typeof toolUseResult === 'string' && toolUseResult.trim()) {
        const errText = toolUseResult.trim()
        if (!toolErrors.includes(errText)) toolErrors.push(errText)
      }
      continue
    }
    // Other top-level types (system, init, etc.) are silently skipped.
  }

  const structured = parsedCount > 0
  if (!structured) {
    return { text: input, hasContent: true, structured: false, parsedLines: parsedCount, streamEvents }
  }

  // Build the display text from extracted content, never falling back to raw JSON.
  const parts: string[] = []
  if (deltaParts.length > 0) {
    parts.push(deltaParts.join('').trim())
  }
  if (assistantMessage && !parts.length) {
    parts.push(assistantMessage)
  }
  if (toolNames.length > 0) {
    parts.push(`\nTools used: ${[...new Set(toolNames)].join(', ')}`)
  }
  if (toolErrors.length > 0) {
    parts.push(`\nTool errors:\n${toolErrors.join('\n')}`)
  }
  if (resultMessage) {
    parts.push(`\n---\nResult: ${resultMessage}`)
  }
  // Intentionally skip plain non-JSON lines in structured mode. They are
  // commonly orphaned tail fragments from truncated NDJSON streams.
  const text = parts.join('\n').trim()
  if (text) {
    return { text, hasContent: true, structured: true, parsedLines: parsedCount, streamEvents }
  }
  return { text: `(${parsedCount} events processed, ${streamEvents} stream events — no text output)`, hasContent: false, structured: true, parsedLines: parsedCount, streamEvents }
}

function mergeAppendOnlyText(previous: string, incoming: string): string {
  const prev = String(previous || '')
  const next = String(incoming || '')
  if (!next.trim()) return prev
  if (!prev) return next
  if (next === prev) return prev
  if (next.startsWith(prev)) return next
  if (prev.includes(next)) return prev

  // Find the longest overlap where previous suffix equals incoming prefix.
  const minStart = Math.max(0, prev.length - next.length)
  for (let start = minStart; start < prev.length; start += 1) {
    const suffix = prev.slice(start)
    if (next.startsWith(suffix)) {
      return prev + next.slice(suffix.length)
    }
  }

  // If there is no overlap, preserve previous output and append incoming text.
  return `${prev}\n${next}`.trim()
}

function formatProgressEntries(progress?: Record<string, unknown>): Array<{ key: string; value: string }> {
  if (!progress || typeof progress !== 'object') return []
  const root = progress as Record<string, unknown>
  const orderedKeys = ['status', 'phase', 'step', 'progress', 'percent', 'message', 'summary', 'last_heartbeat', 'run_id']
  const seen = new Set<string>()
  const out: Array<{ key: string; value: string }> = []
  const toValue = (value: unknown): string => {
    if (value == null) return ''
    if (typeof value === 'string') return value
    if (typeof value === 'number' || typeof value === 'boolean') return String(value)
    if (Array.isArray(value)) return value.map((item) => (typeof item === 'string' ? item : JSON.stringify(item))).join(', ')
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }
  for (const key of orderedKeys) {
    if (!(key in root)) continue
    const value = toValue(root[key])
    if (!value.trim()) continue
    out.push({ key, value })
    seen.add(key)
  }
  for (const [key, rawValue] of Object.entries(root)) {
    if (seen.has(key)) continue
    const value = toValue(rawValue)
    if (!value.trim()) continue
    out.push({ key, value })
    if (out.length >= 12) break
  }
  return out
}

export default function App() {
  const [route, setRoute] = useState<RouteKey>(() => routeFromHash(window.location.hash || localStorage.getItem(STORAGE_ROUTE) || '#/board'))
  const [projectDir, setProjectDir] = useState<string>(() => localStorage.getItem(STORAGE_PROJECT) || '')
  const [board, setBoard] = useState<BoardResponse>({ columns: {} })
  const [orchestrator, setOrchestrator] = useState<OrchestratorStatus | null>(null)
  const [reviewQueue, setReviewQueue] = useState<TaskRecord[]>([])
  const [agents, setAgents] = useState<AgentRecord[]>([])
  const [workerHealth, setWorkerHealth] = useState<WorkerHealthRecord[]>([])
  const [workerRoutingRows, setWorkerRoutingRows] = useState<WorkerRoutingRow[]>([])
  const [workerDefaultProvider, setWorkerDefaultProvider] = useState('codex')
  const [workerHealthRefreshing, setWorkerHealthRefreshing] = useState(false)
  const [projects, setProjects] = useState<ProjectRef[]>([])
  const [pinnedProjects, setPinnedProjects] = useState<PinnedProjectRef[]>([])
  const [quickActions, setQuickActions] = useState<QuickActionRecord[]>([])
  const [taskExplorerItems, setTaskExplorerItems] = useState<TaskRecord[]>([])
  const [executionBatches, setExecutionBatches] = useState<string[][]>([])
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null)
  const [activeProjectId, setActiveProjectId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  const [workOpen, setWorkOpen] = useState(false)
  const [createTab, setCreateTab] = useState<CreateTab>('task')
  const [selectedTaskId, setSelectedTaskId] = useState<string>('')
  const modalDismissedRef = useRef(false)
  const modalExplicitRef = useRef(false)
  const taskSelectTabRef = useRef<TaskDetailTab | undefined>(undefined)
  const [planningTaskId, setPlanningTaskId] = useState('')
  const [selectedTaskDetail, setSelectedTaskDetail] = useState<TaskRecord | null>(null)
  const [selectedTaskDetailLoading, setSelectedTaskDetailLoading] = useState(false)
  const [selectedTaskPlan, setSelectedTaskPlan] = useState<TaskPlanDocument | null>(null)
  const [selectedTaskPlanJobs, setSelectedTaskPlanJobs] = useState<PlanRefineJobRecord[]>([])
  const [selectedPlanRevisionId, setSelectedPlanRevisionId] = useState('')
  const [planManualContent, setPlanManualContent] = useState('')
  const [planManualFeedbackNote, setPlanManualFeedbackNote] = useState('')
  const [planRefineFeedback, setPlanRefineFeedback] = useState('')
  const [planJobLoading, setPlanJobLoading] = useState(false)
  const [planRefineStdout, setPlanRefineStdout] = useState('')
  const [planningWorkerTab, setPlanningWorkerTab] = useState<'plan' | 'manual'>('plan')
  const [planSavingManual, setPlanSavingManual] = useState(false)
  const [planCommitting, setPlanCommitting] = useState(false)
  const [planGenerateLoading, setPlanGenerateLoading] = useState(false)
  const [planGenerateStdout, setPlanGenerateStdout] = useState('')
  const [planGenerateSource, setPlanGenerateSource] = useState<'committed' | 'revision' | 'override' | 'latest'>('latest')
  const [planGenerateRevisionId, setPlanGenerateRevisionId] = useState('')
  const [planGenerateOverride, setPlanGenerateOverride] = useState('')
  const [planGenerateInferDeps, setPlanGenerateInferDeps] = useState(true)
  const [planActionMessage, setPlanActionMessage] = useState('')
  const [planActionError, setPlanActionError] = useState('')
  const [taskDetailTab, setTaskDetailTab] = useState<TaskDetailTab>('overview')
  const [boardCompact, setBoardCompact] = useState(false)
  // taskEditMode removed — configLocked (status-based) controls editability
  const [taskActionPending, setTaskActionPending] = useState<TaskActionKey | null>(null)
  const [taskActionMessage, setTaskActionMessage] = useState('')
  const [taskActionError, setTaskActionError] = useState('')
  const [editTaskTitle, setEditTaskTitle] = useState('')
  const [editTaskDescription, setEditTaskDescription] = useState('')
  const [editTaskType, setEditTaskType] = useState('feature')
  const [editTaskPriority, setEditTaskPriority] = useState('P2')
  const [editTaskLabels, setEditTaskLabels] = useState('')
  const [editTaskApprovalMode, setEditTaskApprovalMode] = useState<'human_review' | 'auto_approve'>('human_review')
  const [editTaskHitlMode, setEditTaskHitlMode] = useState('autopilot')

  const [newTaskTitle, setNewTaskTitle] = useState('')
  const [newTaskDescription, setNewTaskDescription] = useState('')
  const [newTaskType, setNewTaskType] = useState('feature')
  const [newTaskPriority, setNewTaskPriority] = useState('P2')
  const [newTaskLabels, setNewTaskLabels] = useState('')
  const [newTaskBlockedBy, setNewTaskBlockedBy] = useState('')
  const [newTaskApprovalMode, setNewTaskApprovalMode] = useState<'human_review' | 'auto_approve'>('human_review')
  const [newTaskHitlMode, setNewTaskHitlMode] = useState('autopilot')
  const [newTaskParentId, setNewTaskParentId] = useState('')
  const [newTaskPipelineTemplate, setNewTaskPipelineTemplate] = useState('')
  const [newTaskMetadata, setNewTaskMetadata] = useState('')
  const [newTaskWorkerModel, setNewTaskWorkerModel] = useState('')
  const [collaborationModes, setCollaborationModes] = useState<CollaborationMode[]>(DEFAULT_COLLABORATION_MODES)
  const [newDependencyId, setNewDependencyId] = useState('')
  const [dependencyActionLoading, setDependencyActionLoading] = useState(false)
  const [dependencyActionMessage, setDependencyActionMessage] = useState('')
  const [taskExplorerQuery, setTaskExplorerQuery] = useState('')
  const [taskExplorerStatus, setTaskExplorerStatus] = useState('')
  const [taskExplorerType, setTaskExplorerType] = useState('')
  const [taskExplorerPriority, setTaskExplorerPriority] = useState('')
  const [taskExplorerOnlyBlocked, setTaskExplorerOnlyBlocked] = useState(false)
  const [taskExplorerLoading, setTaskExplorerLoading] = useState(false)
  const [taskExplorerError, setTaskExplorerError] = useState('')
  const [taskExplorerPage, setTaskExplorerPage] = useState(1)
  const [taskExplorerPageSize, setTaskExplorerPageSize] = useState(6)
  const [collaborationTimeline, setCollaborationTimeline] = useState<CollaborationTimelineEvent[]>([])
  const [collaborationLoading, setCollaborationLoading] = useState(false)
  const [collaborationError, setCollaborationError] = useState('')
  const [selectedTaskLogs, setSelectedTaskLogs] = useState<TaskLogsSnapshot | null>(null)
  const [selectedTaskLogsError, setSelectedTaskLogsError] = useState('')
  const [selectedTaskLogsLoading, setSelectedTaskLogsLoading] = useState(false)
  const planManualSeedRef = useRef<{ taskId: string; workerText: string }>({ taskId: '', workerText: '' })
  const planRefineOutputRef = useRef<{ taskId: string; jobKey: string; text: string }>({ taskId: '', jobKey: '', text: '' })
  const planGenerateOutputRef = useRef<{ taskId: string; jobKey: string; text: string }>({ taskId: '', jobKey: '', text: '' })
  const logAccumRef = useRef<LogAccumState>(createEmptyLogAccum())
  const [stdoutHistory, setStdoutHistory] = useState('')
  const [stderrHistory, setStderrHistory] = useState('')
  const stdoutPreRef = useRef<HTMLPreElement>(null)
  const stderrPreRef = useRef<HTMLPreElement>(null)
  const taskLogsRequestSeqRef = useRef(0)
  const logAutoPinRef = useRef<Record<LogPaneKey, boolean>>({ stdout: true, stderr: true })
  const logScrollSnapshotRef = useRef<Record<LogPaneKey, { top: number; height: number; op: 'none' | 'prepend' | 'append' | 'reset' }>>({
    stdout: { top: 0, height: 0, op: 'none' },
    stderr: { top: 0, height: 0, op: 'none' },
  })

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
  const [topbarProjectPickerFocused, setTopbarProjectPickerFocused] = useState(false)

  const [settingsLoading, setSettingsLoading] = useState(false)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsError, setSettingsError] = useState('')
  const [settingsSuccess, setSettingsSuccess] = useState('')
  const [settingsConcurrency, setSettingsConcurrency] = useState(String(DEFAULT_SETTINGS.orchestrator.concurrency))
  const [settingsAutoDeps, setSettingsAutoDeps] = useState(DEFAULT_SETTINGS.orchestrator.auto_deps)
  const [settingsMaxReviewAttempts, setSettingsMaxReviewAttempts] = useState(String(DEFAULT_SETTINGS.orchestrator.max_review_attempts))
  const [settingsDefaultRole, setSettingsDefaultRole] = useState(DEFAULT_SETTINGS.agent_routing.default_role)
  const [settingsTaskTypeRoles, setSettingsTaskTypeRoles] = useState('')
  const [settingsRoleProviderOverrides, setSettingsRoleProviderOverrides] = useState('')
  const [settingsWorkerDefault, setSettingsWorkerDefault] = useState(DEFAULT_SETTINGS.workers.default)
  const [settingsProviderView, setSettingsProviderView] = useState<'codex' | 'ollama' | 'claude'>('codex')
  const [settingsWorkerRouting, setSettingsWorkerRouting] = useState('')
  const [settingsWorkerProviders, setSettingsWorkerProviders] = useState('')
  const [settingsCodexCommand, setSettingsCodexCommand] = useState('codex exec')
  const [settingsCodexModel, setSettingsCodexModel] = useState('')
  const [settingsCodexEffort, setSettingsCodexEffort] = useState('')
  const [settingsClaudeCommand, setSettingsClaudeCommand] = useState('claude -p')
  const [settingsClaudeModel, setSettingsClaudeModel] = useState('')
  const [settingsClaudeEffort, setSettingsClaudeEffort] = useState('')
  const [settingsOllamaEndpoint, setSettingsOllamaEndpoint] = useState('http://localhost:11434')
  const [settingsOllamaModel, setSettingsOllamaModel] = useState('')
  const [settingsOllamaTemperature, setSettingsOllamaTemperature] = useState('')
  const [settingsOllamaNumCtx, setSettingsOllamaNumCtx] = useState('')
  const [settingsProjectCommands, setSettingsProjectCommands] = useState('')
  const [settingsGateCritical, setSettingsGateCritical] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.critical))
  const [settingsGateHigh, setSettingsGateHigh] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.high))
  const [settingsGateMedium, setSettingsGateMedium] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.medium))
  const [settingsGateLow, setSettingsGateLow] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.low))

  const selectedTaskIdRef = useRef(selectedTaskId)
  const selectedQuickActionIdRef = useRef(selectedQuickActionId)
  const selectedImportJobIdRef = useRef(selectedImportJobId)
  const activeProjectIdRef = useRef(activeProjectId)
  const projectDirRef = useRef(projectDir)
  const taskDetailRequestSeqRef = useRef(0)
  const collaborationRequestSeqRef = useRef(0)
  const taskExplorerRequestSeqRef = useRef(0)
  const reloadAllSeqRef = useRef(0)
  const reloadTimerRef = useRef<number | null>(null)
  const realtimeRefreshInFlightRef = useRef(false)
  const realtimeRefreshPendingRef = useRef(false)
  const realtimeChannelsRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    selectedTaskIdRef.current = selectedTaskId
  }, [selectedTaskId])

  useEffect(() => {
    setTaskDetailTab(taskSelectTabRef.current || 'overview')
    taskSelectTabRef.current = undefined
    setTaskActionPending(null)
    setTaskActionError('')
    setTaskActionMessage('')
    setNewDependencyId('')
    setDependencyActionMessage('')
    setPlanRefineFeedback('')
    setPlanGenerateSource('latest')
    setPlanGenerateOverride('')
    setPlanGenerateInferDeps(true)
  }, [selectedTaskId])

  useEffect(() => {
    selectedQuickActionIdRef.current = selectedQuickActionId
  }, [selectedQuickActionId])

  useEffect(() => {
    selectedImportJobIdRef.current = selectedImportJobId
  }, [selectedImportJobId])

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId
  }, [activeProjectId])

  useEffect(() => {
    projectDirRef.current = projectDir
  }, [projectDir])

  useEffect(() => {
    return () => {
      if (reloadTimerRef.current !== null) {
        window.clearTimeout(reloadTimerRef.current)
        reloadTimerRef.current = null
      }
      realtimeChannelsRef.current.clear()
      realtimeRefreshPendingRef.current = false
      realtimeRefreshInFlightRef.current = false
    }
  }, [])

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

  async function loadProjectIdentity(): Promise<void> {
    const fallback = inferProjectId(projectDir)
    setActiveProjectId(fallback)
    try {
      const root = await requestJson<RootSnapshot>(buildApiUrl('/', projectDir))
      const resolved = String(root.project_id || '').trim()
      if (resolved) {
        setActiveProjectId(resolved)
      }
    } catch {
      // Keep fallback project identity when root metadata is unavailable.
    }
  }

  useEffect(() => {
    if (projectDir) {
      localStorage.setItem(STORAGE_PROJECT, projectDir)
    } else {
      localStorage.removeItem(STORAGE_PROJECT)
    }
    void loadProjectIdentity()
  }, [projectDir])

  useEffect(() => {
    const hasModalOpen = workOpen || browseOpen || (!!selectedTaskId && modalExplicitRef.current && !modalDismissedRef.current)
    document.documentElement.classList.toggle('modal-open', hasModalOpen)
    document.body.classList.toggle('modal-open', hasModalOpen)
    return () => {
      document.documentElement.classList.remove('modal-open')
      document.body.classList.remove('modal-open')
    }
  }, [workOpen, browseOpen, selectedTaskId, route])

  useEffect(() => {
    const columns = ['backlog', 'queued', 'in_progress', 'in_review', 'blocked', 'done', 'cancelled'] as const
    const allTasks = columns.flatMap((column) => board.columns[column] || [])
    if (!selectedTaskId && !modalDismissedRef.current && allTasks.length > 0) {
      setSelectedTaskId(allTasks[0].id)
    }
    if (selectedTaskId && allTasks.every((task) => task.id !== selectedTaskId)) {
      setSelectedTaskId(allTasks[0]?.id || '')
      setSelectedTaskDetail(null)
    }
  }, [board, selectedTaskId])

  useEffect(() => {
    if (route !== 'execution') return
    const prioritized = [
      ...(board.columns.in_progress || []),
      ...(board.columns.queued || []),
      ...(board.columns.backlog || []),
      ...(board.columns.in_review || []),
      ...(board.columns.blocked || []),
      ...(board.columns.done || []),
      ...(board.columns.cancelled || []),
    ]
    if (prioritized.length === 0) return
    if (selectedTaskId && prioritized.every((task) => task.id !== selectedTaskId)) {
      setSelectedTaskId(prioritized[0].id)
    }
  }, [route, board, selectedTaskId])

  useEffect(() => {
    if (route !== 'planning') return
    const allTasks: TaskRecord[] = Object.values(board.columns).flat()
    if (allTasks.length === 0) {
      if (planningTaskId) setPlanningTaskId('')
      return
    }

    const hasPlanningTask = !!(planningTaskId && allTasks.some((task) => task.id === planningTaskId))
    const hasSelectedTask = !!(selectedTaskId && allTasks.some((task) => task.id === selectedTaskId))
    const nextTaskId = hasPlanningTask
      ? planningTaskId
      : (hasSelectedTask ? selectedTaskId : allTasks[0].id)

    if (planningTaskId !== nextTaskId) {
      setPlanningTaskId(nextTaskId)
    }
    if (selectedTaskId !== nextTaskId) {
      setSelectedTaskId(nextTaskId)
    }
  }, [route, board, planningTaskId, selectedTaskId])

  function applySettings(payload: SystemSettings): void {
    setSettingsConcurrency(String(payload.orchestrator.concurrency))
    setSettingsAutoDeps(payload.orchestrator.auto_deps)
    setSettingsMaxReviewAttempts(String(payload.orchestrator.max_review_attempts))
    setSettingsDefaultRole(payload.agent_routing.default_role || 'general')
    const taskTypeRoles = payload.agent_routing.task_type_roles || {}
    setSettingsTaskTypeRoles(Object.keys(taskTypeRoles).length > 0 ? JSON.stringify(taskTypeRoles, null, 2) : '')
    const roleProviderOverrides = payload.agent_routing.role_provider_overrides || {}
    setSettingsRoleProviderOverrides(
      Object.keys(roleProviderOverrides).length > 0 ? JSON.stringify(roleProviderOverrides, null, 2) : ''
    )
    const workerDefault = payload.workers.default || 'codex'
    setSettingsWorkerDefault(workerDefault === 'ollama' || workerDefault === 'claude' ? workerDefault : 'codex')
    setSettingsProviderView(workerDefault === 'ollama' || workerDefault === 'claude' ? workerDefault : 'codex')
    const workerRouting = payload.workers.routing || {}
    setSettingsWorkerRouting(Object.keys(workerRouting).length > 0 ? JSON.stringify(workerRouting, null, 2) : '')
    const providers = payload.workers.providers || {}
    const advancedProviders = Object.fromEntries(
      Object.entries(providers).filter(([name]) => name !== 'codex' && name !== 'claude' && name !== 'ollama')
    )
    setSettingsWorkerProviders(
      Object.keys(advancedProviders).length > 0 ? JSON.stringify(advancedProviders, null, 2) : ''
    )
    const entries = Object.entries(providers)

    const codexEntry = entries.find(([name, provider]) => name === 'codex' && provider?.type === 'codex')
      || entries.find(([, provider]) => provider?.type === 'codex')
    if (codexEntry) {
      const [, provider] = codexEntry
      setSettingsCodexCommand(String(provider.command || 'codex exec'))
      setSettingsCodexModel(String(provider.model || ''))
      setSettingsCodexEffort(String(provider.reasoning_effort || ''))
    } else {
      setSettingsCodexCommand('codex exec')
      setSettingsCodexModel('')
      setSettingsCodexEffort('')
    }

    const claudeEntry = entries.find(([name, provider]) => name === 'claude' && provider?.type === 'claude')
      || entries.find(([, provider]) => provider?.type === 'claude')
    if (claudeEntry) {
      const [, provider] = claudeEntry
      setSettingsClaudeCommand(String(provider.command || 'claude -p'))
      setSettingsClaudeModel(String(provider.model || ''))
      setSettingsClaudeEffort(String(provider.reasoning_effort || ''))
    } else {
      setSettingsClaudeCommand('claude -p')
      setSettingsClaudeModel('')
      setSettingsClaudeEffort('')
    }

    const ollamaEntry = entries.find(([name, provider]) => name === 'ollama' && provider?.type === 'ollama')
      || entries.find(([, provider]) => provider?.type === 'ollama')
    if (ollamaEntry) {
      const [, provider] = ollamaEntry
      setSettingsOllamaEndpoint(String(provider.endpoint || 'http://localhost:11434'))
      setSettingsOllamaModel(String(provider.model || ''))
      setSettingsOllamaTemperature(
        provider.temperature === undefined || provider.temperature === null ? '' : String(provider.temperature)
      )
      setSettingsOllamaNumCtx(provider.num_ctx === undefined || provider.num_ctx === null ? '' : String(provider.num_ctx))
    } else {
      setSettingsOllamaEndpoint('http://localhost:11434')
      setSettingsOllamaModel('')
      setSettingsOllamaTemperature('')
      setSettingsOllamaNumCtx('')
    }
    const projectCommands = payload.project.commands || {}
    setSettingsProjectCommands(
      Object.keys(projectCommands).length > 0 ? JSON.stringify(projectCommands, null, 2) : ''
    )
    setSettingsGateCritical(String(payload.defaults.quality_gate.critical))
    setSettingsGateHigh(String(payload.defaults.quality_gate.high))
    setSettingsGateMedium(String(payload.defaults.quality_gate.medium))
    setSettingsGateLow(String(payload.defaults.quality_gate.low))
  }

  async function loadSettings(): Promise<void> {
    setSettingsLoading(true)
    setSettingsError('')
    setSettingsSuccess('')
    try {
      const payload = await requestJson<Partial<SystemSettings>>(buildApiUrl('/api/settings', projectDir))
      applySettings(normalizeSettings(payload))
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setSettingsError(`Failed to load settings (${detail})`)
      applySettings(DEFAULT_SETTINGS)
    } finally {
      setSettingsLoading(false)
    }
  }

  useEffect(() => {
    if (route !== 'settings' && route !== 'agents') return
    void loadSettings()
  }, [route, projectDir])

  useEffect(() => {
    const fetchModes = async () => {
      try {
        const data = await requestJson<{ modes?: CollaborationMode[] }>(buildApiUrl('/api/collaboration/modes', projectDir))
        const modes = (data.modes || []).map((mode) => ({
          mode: mode.mode,
          display_name: mode.display_name || humanizeLabel(mode.mode),
          description: mode.description || '',
        }))
        if (modes.length > 0) {
          setCollaborationModes(modes)
        }
      } catch {
        setCollaborationModes(DEFAULT_COLLABORATION_MODES)
      }
    }
    void fetchModes()
  }, [projectDir])

  async function loadTaskDetail(taskId: string): Promise<void> {
    if (!taskId) {
      setSelectedTaskDetail(null)
      return
    }
    const requestSeq = taskDetailRequestSeqRef.current + 1
    taskDetailRequestSeqRef.current = requestSeq
    setSelectedTaskDetailLoading(true)
    try {
      const detail = await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}`, projectDir))
      if (requestSeq !== taskDetailRequestSeqRef.current || selectedTaskIdRef.current !== taskId) {
        return
      }
      const task = detail.task
      setSelectedTaskDetail(task)
      void loadTaskPlan(taskId)
      setEditTaskTitle(task.title || '')
      setEditTaskDescription(task.description || '')
      setEditTaskType(task.task_type || 'feature')
      setEditTaskPriority(task.priority || 'P2')
      setEditTaskLabels((task.labels || []).join(', '))
      setEditTaskApprovalMode(task.approval_mode || 'human_review')
      setEditTaskHitlMode(task.hitl_mode || 'autopilot')
    } catch {
      if (requestSeq !== taskDetailRequestSeqRef.current || selectedTaskIdRef.current !== taskId) {
        return
      }
      setSelectedTaskDetail(null)
      setSelectedTaskPlan(null)
    } finally {
      if (requestSeq === taskDetailRequestSeqRef.current) {
        setSelectedTaskDetailLoading(false)
      }
    }
  }

  async function loadTaskPlan(taskId: string): Promise<void> {
    if (!taskId) {
      setSelectedTaskPlan(null)
      setSelectedTaskPlanJobs([])
      setSelectedPlanRevisionId('')
      return
    }
    try {
      const payload = await requestJson<unknown>(buildApiUrl(`/api/tasks/${taskId}/plan`, projectDir))
      if (selectedTaskIdRef.current !== taskId) {
        return
      }
      const planDoc = normalizeTaskPlan(payload)
      setSelectedTaskPlan(planDoc)
      void loadTaskPlanJobs(taskId)
      const committedOrLatest = planDoc.committed_revision_id || planDoc.latest_revision_id || ''
      setSelectedPlanRevisionId((prev) => {
        if (prev && planDoc.revisions.some((item) => item.id === prev)) {
          return prev
        }
        return committedOrLatest
      })
      setPlanGenerateRevisionId((prev) => {
        if (prev && planDoc.revisions.some((item) => item.id === prev)) {
          return prev
        }
        return committedOrLatest
      })
      setPlanActionError('')
    } catch (err) {
      if (selectedTaskIdRef.current !== taskId) {
        return
      }
      setSelectedTaskPlan(null)
      setSelectedTaskPlanJobs([])
      setPlanActionError(toErrorMessage('Failed to load planning data', err))
    }
  }

  async function loadTaskPlanJobs(taskId: string): Promise<void> {
    try {
      const payload = await requestJson<unknown>(buildApiUrl(`/api/tasks/${taskId}/plan/jobs`, projectDir))
      if (selectedTaskIdRef.current !== taskId) return
      setSelectedTaskPlanJobs(normalizePlanRefineJobs(payload))
    } catch {
      // Keep the last known history on transient failures to avoid UI dropouts.
      if (selectedTaskIdRef.current !== taskId) return
    }
  }

  useEffect(() => {
    const refineOut = planRefineOutputRef.current
    if (refineOut.taskId !== selectedTaskId) {
      planRefineOutputRef.current = { taskId: selectedTaskId || '', jobKey: '', text: '' }
      planGenerateOutputRef.current = { taskId: selectedTaskId || '', jobKey: '', text: '' }
      planManualSeedRef.current = { taskId: selectedTaskId || '', workerText: '' }
      setPlanRefineStdout('')
      setPlanGenerateStdout('')
      setPlanningWorkerTab('plan')
      setPlanManualContent('')
      setPlanManualFeedbackNote('')
    }
    if (!selectedTaskId) {
      setSelectedTaskPlan(null)
      setSelectedTaskPlanJobs([])
      setSelectedPlanRevisionId('')
      setPlanActionMessage('')
      setPlanActionError('')
      return
    }
    setPlanActionMessage('')
    setPlanActionError('')
    void loadTaskDetail(selectedTaskId)
  }, [selectedTaskId, projectDir])

  function appendPlanRefineStdout(taskId: string, jobKey: string, incoming: string): void {
    const nextText = String(incoming || '').trim()
    if (!taskId || !jobKey || !nextText) return
    const current = planRefineOutputRef.current
    if (current.taskId !== taskId || current.jobKey !== jobKey) {
      planRefineOutputRef.current = { taskId, jobKey, text: nextText }
      setPlanRefineStdout(nextText)
      return
    }
    const merged = mergeAppendOnlyText(current.text, nextText)
    if (merged !== current.text) {
      planRefineOutputRef.current = { taskId, jobKey, text: merged }
      setPlanRefineStdout(merged)
    }
  }

  function appendPlanGenerateStdout(taskId: string, jobKey: string, incoming: string): void {
    const nextText = String(incoming || '').trim()
    if (!taskId || !jobKey || !nextText) return
    const current = planGenerateOutputRef.current
    if (current.taskId !== taskId || current.jobKey !== jobKey) {
      planGenerateOutputRef.current = { taskId, jobKey, text: nextText }
      setPlanGenerateStdout(nextText)
      return
    }
    const merged = mergeAppendOnlyText(current.text, nextText)
    if (merged !== current.text) {
      planGenerateOutputRef.current = { taskId, jobKey, text: merged }
      setPlanGenerateStdout(merged)
    }
  }

  function openPlanningWorkerTab(taskId: string, nextTab: 'plan' | 'manual', workerPlanText: string): void {
    if (nextTab === 'manual') {
      const planText = String(workerPlanText || '').trim()
      if (planText) {
        setPlanManualContent((current) => {
          const currentTrimmed = current.trim()
          const seeded = planManualSeedRef.current
          const wasSeededForTask = seeded.taskId === taskId && seeded.workerText.trim().length > 0
          const sameAsSeeded = wasSeededForTask && currentTrimmed === seeded.workerText.trim()
          if (!currentTrimmed || sameAsSeeded) {
            planManualSeedRef.current = { taskId, workerText: planText }
            return planText
          }
          return current
        })
      }
    }
    setPlanningWorkerTab(nextTab)
  }


  async function loadCollaboration(taskId: string): Promise<void> {
    if (!taskId) {
      setCollaborationTimeline([])
      setCollaborationError('')
      return
    }
    const requestSeq = collaborationRequestSeqRef.current + 1
    collaborationRequestSeqRef.current = requestSeq
    setCollaborationLoading(true)
    setCollaborationError('')
    try {
      const timelinePayload = await requestJson<unknown>(buildApiUrl(`/api/collaboration/timeline/${taskId}`, projectDir))
      if (requestSeq !== collaborationRequestSeqRef.current || selectedTaskIdRef.current !== taskId) {
        return
      }
      setCollaborationTimeline(normalizeTimelineEvents(timelinePayload))
    } catch (err) {
      if (requestSeq !== collaborationRequestSeqRef.current || selectedTaskIdRef.current !== taskId) {
        return
      }
      setCollaborationTimeline([])
      const detail = err instanceof Error ? err.message : 'unknown error'
      setCollaborationError(`Failed to load activity (${detail})`)
    } finally {
      if (requestSeq === collaborationRequestSeqRef.current) {
        setCollaborationLoading(false)
      }
    }
  }

  function getLogPaneElement(pane: LogPaneKey): HTMLPreElement | null {
    return pane === 'stdout' ? stdoutPreRef.current : stderrPreRef.current
  }

  function updateLogAutoPin(pane: LogPaneKey): void {
    const el = getLogPaneElement(pane)
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    logAutoPinRef.current[pane] = distanceFromBottom < LOG_NEAR_BOTTOM_PX
  }

  function handleLogPaneScroll(pane: LogPaneKey): void {
    updateLogAutoPin(pane)
  }

  function snapshotLogScrollOps(stdoutOp: 'none' | 'prepend' | 'append' | 'reset', stderrOp: 'none' | 'prepend' | 'append' | 'reset'): void {
    const stdoutEl = stdoutPreRef.current
    const stderrEl = stderrPreRef.current
    logScrollSnapshotRef.current.stdout = {
      top: stdoutEl ? stdoutEl.scrollTop : 0,
      height: stdoutEl ? stdoutEl.scrollHeight : 0,
      op: stdoutOp,
    }
    logScrollSnapshotRef.current.stderr = {
      top: stderrEl ? stderrEl.scrollTop : 0,
      height: stderrEl ? stderrEl.scrollHeight : 0,
      op: stderrOp,
    }
  }

  async function loadTaskLogs(taskId: string, quiet = false): Promise<void> {
    if (!taskId) {
      setSelectedTaskLogs(null)
      setSelectedTaskLogsError('')
      return
    }
    const requestSeq = taskLogsRequestSeqRef.current + 1
    taskLogsRequestSeqRef.current = requestSeq
    if (!quiet) setSelectedTaskLogsLoading(true)
    let continueBackfill = false
    try {
      if (logAccumRef.current.taskId !== taskId) {
        logAccumRef.current = createEmptyLogAccum(taskId)
      }
      const accum = logAccumRef.current
      const backfillStdoutBefore = accum.stdoutBackfillOffset
      const backfillStderrBefore = accum.stderrBackfillOffset
      const params = new URLSearchParams()
      params.set('max_chars', String(LOG_CHUNK_CHARS))
      if (accum.phase === 'backfill') {
        params.set('backfill', 'true')
        const stdoutNeedsBackfill = accum.stdoutTailStart > 0 && accum.stdoutBackfillOffset > 0
        const stderrNeedsBackfill = accum.stderrTailStart > 0 && accum.stderrBackfillOffset > 0

        const stdoutReadTo = stdoutNeedsBackfill ? accum.stdoutBackfillOffset : accum.stdoutOffset
        const stderrReadTo = stderrNeedsBackfill ? accum.stderrBackfillOffset : accum.stderrOffset
        const stdoutOffset = stdoutNeedsBackfill
          ? Math.max(0, accum.stdoutBackfillOffset - LOG_CHUNK_CHARS)
          : accum.stdoutOffset
        const stderrOffset = stderrNeedsBackfill
          ? Math.max(0, accum.stderrBackfillOffset - LOG_CHUNK_CHARS)
          : accum.stderrOffset

        params.set('stdout_offset', String(stdoutOffset))
        params.set('stderr_offset', String(stderrOffset))
        params.set('stdout_read_to', String(stdoutReadTo))
        params.set('stderr_read_to', String(stderrReadTo))
      } else if (accum.phase === 'forward') {
        // Incremental: read from where we left off.
        params.set('stdout_offset', String(Math.max(0, accum.stdoutOffset)))
        params.set('stderr_offset', String(Math.max(0, accum.stderrOffset)))
      }
      const qs = params.toString()
      const url = buildApiUrl(`/api/tasks/${taskId}/logs${qs ? `?${qs}` : ''}`, projectDir)
      const payload = await requestJson<TaskLogsSnapshot>(url)
      if (requestSeq !== taskLogsRequestSeqRef.current || selectedTaskIdRef.current !== taskId) return

      const payloadLogId = String(payload.log_id || payload.started_at || '').trim()
      if (accum.logId && payloadLogId && accum.logId !== payloadLogId) {
        const reset = createEmptyLogAccum(taskId)
        reset.logId = payloadLogId
        logAccumRef.current = reset
        snapshotLogScrollOps('reset', 'reset')
        setStdoutHistory('')
        setStderrHistory('')
        void loadTaskLogs(taskId, true)
        return
      }
      if (!accum.logId && payloadLogId) {
        accum.logId = payloadLogId
      }

      setSelectedTaskLogs(payload)
      setSelectedTaskLogsError('')

      const prevStdout = accum.stdout
      const prevStderr = accum.stderr
      let stdoutOp: 'none' | 'prepend' | 'append' | 'reset' = 'none'
      let stderrOp: 'none' | 'prepend' | 'append' | 'reset' = 'none'

      const stdoutBackfillRequested = accum.phase === 'backfill' && accum.stdoutTailStart > 0 && accum.stdoutBackfillOffset > 0
      const stderrBackfillRequested = accum.phase === 'backfill' && accum.stderrTailStart > 0 && accum.stderrBackfillOffset > 0
      const nextStdoutBackfillOffset = stdoutBackfillRequested
        ? Math.max(0, accum.stdoutBackfillOffset - LOG_CHUNK_CHARS)
        : accum.stdoutBackfillOffset
      const nextStderrBackfillOffset = stderrBackfillRequested
        ? Math.max(0, accum.stderrBackfillOffset - LOG_CHUNK_CHARS)
        : accum.stderrBackfillOffset

      if (accum.phase === 'init') {
        const incomingStdout = payload.stdout || ''
        const incomingStderr = payload.stderr || ''
        accum.stdout = incomingStdout
        accum.stderr = incomingStderr
        if (incomingStdout !== prevStdout) stdoutOp = 'reset'
        if (incomingStderr !== prevStderr) stderrOp = 'reset'
        if (payload.stdout_offset != null) accum.stdoutOffset = payload.stdout_offset
        if (payload.stderr_offset != null) accum.stderrOffset = payload.stderr_offset
        accum.stdoutTailStart = payload.stdout_tail_start || 0
        accum.stderrTailStart = payload.stderr_tail_start || 0
        accum.stdoutBackfillOffset = accum.stdoutTailStart
        accum.stderrBackfillOffset = accum.stderrTailStart
        accum.phase = (accum.stdoutTailStart > 0 || accum.stderrTailStart > 0) ? 'backfill' : 'forward'
      } else if (accum.phase === 'backfill') {
        const stdoutNeedsBackfill = stdoutBackfillRequested
        const stderrNeedsBackfill = stderrBackfillRequested

        if (stdoutNeedsBackfill) {
          if (payload.stdout) {
            accum.stdout = payload.stdout + accum.stdout
            stdoutOp = 'prepend'
          }
          const reportedStart = typeof payload.stdout_chunk_start === 'number'
            ? payload.stdout_chunk_start
            : nextStdoutBackfillOffset
          accum.stdoutBackfillOffset = Math.max(0, Math.min(reportedStart, accum.stdoutBackfillOffset))
        }
        if (stderrNeedsBackfill) {
          if (payload.stderr) {
            accum.stderr = payload.stderr + accum.stderr
            stderrOp = 'prepend'
          }
          const reportedStart = typeof payload.stderr_chunk_start === 'number'
            ? payload.stderr_chunk_start
            : nextStderrBackfillOffset
          accum.stderrBackfillOffset = Math.max(0, Math.min(reportedStart, accum.stderrBackfillOffset))
        }

        const stdoutBackfillDone = accum.stdoutTailStart <= 0 || accum.stdoutBackfillOffset <= 0
        const stderrBackfillDone = accum.stderrTailStart <= 0 || accum.stderrBackfillOffset <= 0
        if (stdoutBackfillDone && stderrBackfillDone) {
          accum.phase = 'forward'
        }
      } else {
        if (payload.stdout) {
          accum.stdout += payload.stdout
          stdoutOp = 'append'
        }
        if (payload.stderr) {
          accum.stderr += payload.stderr
          stderrOp = 'append'
        }
        if (payload.stdout_offset != null) accum.stdoutOffset = payload.stdout_offset
        if (payload.stderr_offset != null) accum.stderrOffset = payload.stderr_offset
      }

      if (accum.stdout.length > LOG_HISTORY_MAX_CHARS) {
        accum.stdout = accum.stdout.slice(accum.stdout.length - LOG_HISTORY_MAX_CHARS)
        stdoutOp = 'reset'
      }
      if (accum.stderr.length > LOG_HISTORY_MAX_CHARS) {
        accum.stderr = accum.stderr.slice(accum.stderr.length - LOG_HISTORY_MAX_CHARS)
        stderrOp = 'reset'
      }

      snapshotLogScrollOps(stdoutOp, stderrOp)
      setStdoutHistory(accum.stdout)
      setStderrHistory(accum.stderr)
      const stdoutProgressed = accum.stdoutBackfillOffset < backfillStdoutBefore
      const stderrProgressed = accum.stderrBackfillOffset < backfillStderrBefore
      continueBackfill = accum.phase === 'backfill' && (stdoutProgressed || stderrProgressed)
    } catch (err) {
      if (requestSeq !== taskLogsRequestSeqRef.current || selectedTaskIdRef.current !== taskId) return
      setSelectedTaskLogs(null)
      if (!quiet) {
        const detail = err instanceof Error ? err.message : 'unknown error'
        setSelectedTaskLogsError(`Failed to load logs (${detail})`)
      }
    } finally {
      if (requestSeq === taskLogsRequestSeqRef.current && !quiet) setSelectedTaskLogsLoading(false)
    }
    if (continueBackfill && requestSeq === taskLogsRequestSeqRef.current && selectedTaskIdRef.current === taskId) {
      // Drain historical backfill quickly so rendered text stabilizes sooner.
      void loadTaskLogs(taskId, true)
    }
  }

  useEffect(() => {
    if (!selectedTaskId) {
      setCollaborationTimeline([])
      setCollaborationError('')
      setSelectedTaskLogs(null)
      setSelectedTaskLogsError('')
      logAccumRef.current = createEmptyLogAccum()
      taskLogsRequestSeqRef.current += 1
      logAutoPinRef.current = { stdout: true, stderr: true }
      logScrollSnapshotRef.current = {
        stdout: { top: 0, height: 0, op: 'none' },
        stderr: { top: 0, height: 0, op: 'none' },
      }
      setStdoutHistory('')
      setStderrHistory('')
      return
    }
    logAutoPinRef.current = { stdout: true, stderr: true }
    logScrollSnapshotRef.current = {
      stdout: { top: 0, height: 0, op: 'reset' },
      stderr: { top: 0, height: 0, op: 'reset' },
    }
    void loadCollaboration(selectedTaskId)
    void loadTaskLogs(selectedTaskId)
  }, [selectedTaskId, projectDir])

  useEffect(() => {
    if (!selectedTaskId) return
    const timer = window.setInterval(() => {
      void loadTaskLogs(selectedTaskId, true)
    }, 2_000)
    return () => window.clearInterval(timer)
  }, [selectedTaskId, projectDir])

  // Keep log panes pinned to bottom unless user scrolled up.
  useEffect(() => {
    const panes: LogPaneKey[] = ['stdout', 'stderr']
    for (const pane of panes) {
      const el = getLogPaneElement(pane)
      if (!el) continue
      const snap = logScrollSnapshotRef.current[pane]
      const shouldPin = logAutoPinRef.current[pane]
      if (shouldPin) {
        el.scrollTop = el.scrollHeight
      } else if (snap.op === 'prepend') {
        const delta = el.scrollHeight - snap.height
        if (delta !== 0) {
          el.scrollTop = Math.max(0, snap.top + delta)
        }
      }
      logScrollSnapshotRef.current[pane] = {
        top: el.scrollTop,
        height: el.scrollHeight,
        op: 'none',
      }
    }
  }, [stdoutHistory, stderrHistory])

  async function loadTaskExplorer(): Promise<void> {
    const requestSeq = taskExplorerRequestSeqRef.current + 1
    taskExplorerRequestSeqRef.current = requestSeq
    setTaskExplorerLoading(true)
    setTaskExplorerError('')
    try {
      const params: Record<string, string> = {}
      const effectiveStatus = taskExplorerOnlyBlocked ? 'blocked' : taskExplorerStatus
      if (effectiveStatus) params.status = effectiveStatus
      if (taskExplorerType) params.task_type = taskExplorerType
      if (taskExplorerPriority) params.priority = taskExplorerPriority
      const response = await requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/tasks', projectDir, params))
      const tasks = response.tasks || []
      const query = taskExplorerQuery.trim().toLowerCase()
      const filtered = query
        ? tasks.filter((task) => {
            const haystack = `${task.title} ${task.description || ''} ${task.id}`.toLowerCase()
            return haystack.includes(query)
          })
        : tasks
      if (requestSeq !== taskExplorerRequestSeqRef.current) {
        return
      }
      setTaskExplorerItems(filtered)
    } catch (err) {
      if (requestSeq !== taskExplorerRequestSeqRef.current) {
        return
      }
      setTaskExplorerItems([])
      const detail = err instanceof Error ? err.message : 'unknown error'
      setTaskExplorerError(`Failed to load task explorer (${detail})`)
    } finally {
      if (requestSeq === taskExplorerRequestSeqRef.current) {
        setTaskExplorerLoading(false)
      }
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
      const payload = await requestJson<{ job: ImportJobRecord }>(buildApiUrl(`/api/import/${jobId}`, projectDir))
      if (selectedImportJobIdRef.current !== jobId) return
      setSelectedImportJob(payload.job ?? null)
    } catch (err) {
      if (selectedImportJobIdRef.current !== jobId) return
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
    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      if (Date.now() - startedAt > 60_000) {
        window.clearInterval(timer)
        return
      }
      void loadImportJobDetail(selectedImportJobId)
    }, 2_000)
    return () => window.clearInterval(timer)
  }, [workOpen, createTab, selectedImportJobId, projectDir])

  useEffect(() => {
    const activeJob = selectedTaskPlan?.active_refine_job
    const isActive = activeJob && (activeJob.status === 'queued' || activeJob.status === 'running')
    if (!isActive && planJobLoading) {
      setPlanJobLoading(false)
      // Final log fetch to capture complete output
      const taskId = selectedTaskIdRef.current
      if (taskId) {
        void (async () => {
          try {
            const logs = await requestJson<TaskLogsSnapshot>(buildApiUrl(`/api/tasks/${taskId}/logs?max_chars=50000`, projectDir))
            if (selectedTaskIdRef.current !== taskId) return
            if (logs.step === 'plan_refine' && logs.stdout) {
              const rendered = renderStructuredStdout(logs.stdout)
              if (rendered.hasContent) {
                const jobKey = String(selectedTaskPlan?.active_refine_job?.id || logs.started_at || 'plan_refine')
                appendPlanRefineStdout(taskId, jobKey, rendered.text)
              }
            }
          } catch { /* ignore */ }
        })()
      }
    }
  }, [selectedTaskPlan?.active_refine_job?.status, planJobLoading, projectDir])

  useEffect(() => {
    if (!selectedTaskId) return
    const activeJob = selectedTaskPlan?.active_refine_job
    if (!activeJob) return
    if (!(activeJob.status === 'queued' || activeJob.status === 'running')) return
    const activeJobId = String(activeJob.id || '')
    async function pollRefine(): Promise<void> {
      void loadTaskPlan(selectedTaskId)
      try {
        const logs = await requestJson<TaskLogsSnapshot>(buildApiUrl(`/api/tasks/${selectedTaskId}/logs?max_chars=50000`, projectDir))
        if (logs.step === 'plan_refine' && logs.stdout) {
          const rendered = renderStructuredStdout(logs.stdout)
          if (rendered.hasContent) {
            const jobKey = String(activeJobId || logs.started_at || 'plan_refine')
            appendPlanRefineStdout(selectedTaskId, jobKey, rendered.text)
          }
        }
      } catch { /* ignore */ }
    }
    void pollRefine()
    const timer = window.setInterval(() => void pollRefine(), 2_000)
    return () => window.clearInterval(timer)
  }, [selectedTaskId, selectedTaskPlan?.active_refine_job?.id, selectedTaskPlan?.active_refine_job?.status, projectDir])

  useEffect(() => {
    if (!planGenerateLoading) return
    if (!selectedTaskId || !selectedTaskLogs) return
    if (selectedTaskLogs.step !== 'generate_tasks' || !selectedTaskLogs.stdout) return
    const rendered = renderStructuredStdout(selectedTaskLogs.stdout)
    if (!rendered.hasContent) return
    const jobKey = String(selectedTaskLogs.started_at || 'generate_tasks')
    appendPlanGenerateStdout(selectedTaskId, jobKey, rendered.text)
  }, [planGenerateLoading, selectedTaskId, selectedTaskLogs?.step, selectedTaskLogs?.stdout, selectedTaskLogs?.started_at])

  async function loadQuickActionDetail(quickActionId: string): Promise<void> {
    if (!quickActionId) {
      setSelectedQuickActionDetail(null)
      return
    }
    setSelectedQuickActionLoading(true)
    setSelectedQuickActionError('')
    setSelectedQuickActionErrorAt('')
    try {
      const payload = await requestJson<{ quick_action: QuickActionRecord }>(buildApiUrl(`/api/quick-actions/${quickActionId}`, projectDir))
      if (selectedQuickActionIdRef.current !== quickActionId) return
      setSelectedQuickActionDetail(payload.quick_action ?? null)
    } catch (err) {
      if (selectedQuickActionIdRef.current !== quickActionId) return
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

  function toErrorMessage(prefix: string, err: unknown): string {
    const detail = err instanceof Error ? err.message : 'unknown error'
    return `${prefix} (${detail})`
  }

  async function refreshTasksSurface(): Promise<void> {
    const refreshProjectDir = projectDirRef.current
    try {
      const [boardData, orchestratorData, reviewData, executionOrderData, metricsData] = await Promise.all([
        requestJson<BoardResponse>(buildApiUrl('/api/tasks/board', refreshProjectDir)),
        requestJson<OrchestratorStatus>(buildApiUrl('/api/orchestrator/status', refreshProjectDir)),
        requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/review-queue', refreshProjectDir)),
        requestJson<{ batches: string[][] }>(buildApiUrl('/api/tasks/execution-order', refreshProjectDir)),
        requestJson<unknown>(buildApiUrl('/api/metrics', refreshProjectDir)).catch(() => null),
      ])
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setBoard(boardData)
      setOrchestrator(orchestratorData)
      setReviewQueue(reviewData.tasks || [])
      setExecutionBatches(executionOrderData.batches || [])
      setMetrics(normalizeMetrics(metricsData))

      const selectedTask = String(selectedTaskIdRef.current || '').trim()
      if (selectedTask) {
        void loadTaskDetail(selectedTask)
        void loadTaskPlan(selectedTask)
      }
    } catch (err) {
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setError(toErrorMessage('Failed to refresh tasks surface', err))
    }
  }

  async function refreshAgentsSurface(): Promise<void> {
    const refreshProjectDir = projectDirRef.current
    try {
      const [agentData, healthData, routingData] = await Promise.all([
        requestJson<{ agents: AgentRecord[] }>(buildApiUrl('/api/agents', refreshProjectDir)),
        requestJson<unknown>(buildApiUrl('/api/workers/health', refreshProjectDir)).catch(() => ({ providers: [] })),
        requestJson<unknown>(buildApiUrl('/api/workers/routing', refreshProjectDir)).catch(() => ({ default: 'codex', rows: [] })),
      ])
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setAgents(agentData.agents || [])
      setWorkerHealth(normalizeWorkerHealth(healthData))
      const normalizedRouting = normalizeWorkerRouting(routingData)
      setWorkerDefaultProvider(normalizedRouting.defaultProvider)
      setWorkerRoutingRows(normalizedRouting.rows)
    } catch (err) {
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setError(toErrorMessage('Failed to refresh workers surface', err))
    }
  }

  async function refreshQuickActionsSurface(): Promise<void> {
    const refreshProjectDir = projectDirRef.current
    try {
      const payload = await requestJson<{ quick_actions: QuickActionRecord[] }>(buildApiUrl('/api/quick-actions', refreshProjectDir))
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setQuickActions(payload.quick_actions || [])

      const selectedQuickAction = String(selectedQuickActionIdRef.current || '').trim()
      if (selectedQuickAction) {
        void loadQuickActionDetail(selectedQuickAction)
      }
    } catch (err) {
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setError(toErrorMessage('Failed to refresh quick actions surface', err))
    }
  }

  async function reloadAll(): Promise<void> {
    const requestSeq = reloadAllSeqRef.current + 1
    reloadAllSeqRef.current = requestSeq
    setLoading(true)
    setError('')
    try {
      const [
        boardData,
        orchestratorData,
        reviewData,
        agentData,
        projectData,
        pinnedData,
        quickActionData,
        executionOrderData,
        metricsData,
        workerHealthData,
        workerRoutingData,
      ] = await Promise.all([
        requestJson<BoardResponse>(buildApiUrl('/api/tasks/board', projectDir)),
        requestJson<OrchestratorStatus>(buildApiUrl('/api/orchestrator/status', projectDir)),
        requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/review-queue', projectDir)),
        requestJson<{ agents: AgentRecord[] }>(buildApiUrl('/api/agents', projectDir)),
        requestJson<{ projects: ProjectRef[] }>(buildApiUrl('/api/projects', projectDir)),
        requestJson<{ items: PinnedProjectRef[] }>(buildApiUrl('/api/projects/pinned', projectDir)),
        requestJson<{ quick_actions: QuickActionRecord[] }>(buildApiUrl('/api/quick-actions', projectDir)),
        requestJson<{ batches: string[][] }>(buildApiUrl('/api/tasks/execution-order', projectDir)),
        requestJson<unknown>(buildApiUrl('/api/metrics', projectDir)).catch(() => null),
        requestJson<unknown>(buildApiUrl('/api/workers/health', projectDir)).catch(() => ({ providers: [] })),
        requestJson<unknown>(buildApiUrl('/api/workers/routing', projectDir)).catch(() => ({ default: 'codex', rows: [] })),
      ])
      if (requestSeq !== reloadAllSeqRef.current) {
        return
      }
      setBoard(boardData)
      setOrchestrator(orchestratorData)
      setReviewQueue(reviewData.tasks || [])
      setAgents(agentData.agents || [])
      setProjects(projectData.projects || [])
      setPinnedProjects(pinnedData.items || [])
      setQuickActions(quickActionData.quick_actions || [])
      setExecutionBatches(executionOrderData.batches || [])
      setMetrics(normalizeMetrics(metricsData))
      setWorkerHealth(normalizeWorkerHealth(workerHealthData))
      const normalizedRouting = normalizeWorkerRouting(workerRoutingData)
      setWorkerDefaultProvider(normalizedRouting.defaultProvider)
      setWorkerRoutingRows(normalizedRouting.rows)
    } catch (err) {
      if (requestSeq !== reloadAllSeqRef.current) {
        return
      }
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      if (requestSeq === reloadAllSeqRef.current) {
        setLoading(false)
      }
    }
  }

  async function flushRealtimeRefreshQueue(): Promise<void> {
    if (realtimeRefreshInFlightRef.current) {
      realtimeRefreshPendingRef.current = true
      return
    }
    realtimeRefreshInFlightRef.current = true
    try {
      do {
        realtimeRefreshPendingRef.current = false
        const channels = new Set(realtimeChannelsRef.current)
        realtimeChannelsRef.current.clear()
        if (channels.size === 0) {
          continue
        }

        const refreshTasks = channels.has('tasks') || channels.has('queue') || channels.has('review') || channels.has('notifications')
        const refreshAgents = channels.has('agents') || channels.has('notifications')
        const refreshQuickActions = channels.has('quick_actions')
        const ops: Array<Promise<void>> = []
        if (refreshTasks) ops.push(refreshTasksSurface())
        if (refreshAgents) ops.push(refreshAgentsSurface())
        if (refreshQuickActions) ops.push(refreshQuickActionsSurface())
        if (ops.length > 0) {
          await Promise.all(ops)
        }
      } while (realtimeRefreshPendingRef.current || realtimeChannelsRef.current.size > 0)
    } finally {
      realtimeRefreshInFlightRef.current = false
    }
  }

  function scheduleRealtimeRefresh(channel: string, delayMs = 160): void {
    if (!WS_RELOAD_CHANNELS.has(channel)) {
      return
    }
    realtimeChannelsRef.current.add(channel)

    if (realtimeRefreshInFlightRef.current) {
      realtimeRefreshPendingRef.current = true
      return
    }

    if (reloadTimerRef.current !== null) {
      return
    }
    reloadTimerRef.current = window.setTimeout(() => {
      reloadTimerRef.current = null
      void flushRealtimeRefreshQueue()
    }, delayMs)
  }

  useEffect(() => {
    void reloadAll()
  }, [projectDir])

  useEffect(() => {
    let stopped = false
    let socket: WebSocket | null = null
    let reconnectTimer: number | null = null
    let reconnectAttempts = 0
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
    const subscribedChannels = ['tasks', 'queue', 'agents', 'review', 'quick_actions', 'notifications', 'system']

    const scheduleReconnect = (): void => {
      if (stopped || reconnectTimer !== null) return
      const attempt = Math.min(reconnectAttempts, 6)
      const baseDelay = Math.min(30_000, 1_000 * (2 ** attempt))
      const jitter = Math.floor(Math.random() * 300)
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        reconnectAttempts += 1
        connect()
      }, baseDelay + jitter)
    }

    const handleMessage = (event: MessageEvent): void => {
      let payload: unknown
      try {
        payload = JSON.parse(String(event.data || '{}'))
      } catch {
        return
      }
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        return
      }
      const data = payload as Record<string, unknown>
      const channel = String(data.channel || '').trim()
      if (!channel || channel === 'system') {
        return
      }
      const eventProjectId = String(data.project_id || '').trim()
      const currentProjectId = String(activeProjectIdRef.current || '').trim()
      if (eventProjectId && (!currentProjectId || currentProjectId !== eventProjectId)) {
        return
      }

      if (channel === 'quick_actions') {
        const selectedQuickAction = String(selectedQuickActionIdRef.current || '').trim()
        const eventEntityId = String(data.entity_id || '').trim()
        if (selectedQuickAction && (!eventEntityId || eventEntityId === selectedQuickAction)) {
          void loadQuickActionDetail(selectedQuickAction)
        }
      }

      if (WS_RELOAD_CHANNELS.has(channel)) {
        scheduleRealtimeRefresh(channel, 120)
      }
    }

    const connect = (): void => {
      if (stopped) return
      socket = new WebSocket(wsUrl)
      socket.addEventListener('open', () => {
        reconnectAttempts = 0
        socket?.send(JSON.stringify({
          action: 'subscribe',
          channels: subscribedChannels,
          project_id: activeProjectIdRef.current || undefined,
        }))
      })
      socket.addEventListener('message', handleMessage)
      socket.addEventListener('error', () => {
        socket?.close()
      })
      socket.addEventListener('close', () => {
        scheduleReconnect()
      })
    }

    connect()

    return () => {
      stopped = true
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
      socket?.close()
    }
  }, [projectDir, activeProjectId])

  async function runTaskMutation(
    kind: TaskActionKey,
    mutation: () => Promise<void>,
    options?: {
      startMessage?: string
      successMessage?: string
      errorPrefix?: string
      clearEditModeOnSuccess?: boolean
    },
  ): Promise<void> {
    setTaskActionPending(kind)
    setTaskActionError('')
    if (options?.startMessage !== undefined) {
      setTaskActionMessage(options.startMessage)
    } else {
      setTaskActionMessage('')
    }
    try {
      await mutation()
      if (options?.successMessage) {
        setTaskActionMessage(options.successMessage)
      }
      // clearEditModeOnSuccess — no-op (config editability is status-driven)
    } catch (err) {
      setTaskActionError(toErrorMessage(options?.errorPrefix || `Failed to ${kind} task`, err))
    } finally {
      setTaskActionPending(null)
    }
  }

  async function submitTask(event: FormEvent, statusOverride?: 'queued' | 'backlog'): Promise<void> {
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
    try {
      await requestJson<{ task: TaskRecord }>(buildApiUrl('/api/tasks', projectDir), {
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
          hitl_mode: newTaskHitlMode,
          worker_model: newTaskWorkerModel.trim() || undefined,
          parent_id: newTaskParentId.trim() || undefined,
          pipeline_template: parsedPipelineTemplate.length > 0 ? parsedPipelineTemplate : undefined,
          metadata: parsedMetadata,
          status: statusOverride || 'queued',
        }),
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create task')
      return
    }
    setError('')
    setNewTaskTitle('')
    setNewTaskDescription('')
    setNewTaskType('feature')
    setNewTaskPriority('P2')
    setNewTaskLabels('')
    setNewTaskBlockedBy('')
    setNewTaskApprovalMode('human_review')
    setNewTaskHitlMode('autopilot')
    setNewTaskParentId('')
    setNewTaskPipelineTemplate('')
    setNewTaskMetadata('')
    setNewTaskWorkerModel('')
    setWorkOpen(false)
    await reloadAll()
  }

  async function previewImport(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!importText.trim()) return
    try {
      const preview = await requestJson<{ job_id: string; preview: PrdPreview }>(buildApiUrl('/api/import/prd/preview', projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: importText, default_priority: 'P2' }),
      })
      setImportJobId(preview.job_id)
      setImportPreview(preview.preview)
      setSelectedImportJobId(preview.job_id)
      setRecentImportJobIds((prev) => [preview.job_id, ...prev.filter((item) => item !== preview.job_id)].slice(0, 8))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to preview import')
    }
  }

  async function commitImport(): Promise<void> {
    if (!importJobId) return
    try {
      const commitResponse = await requestJson<{ created_task_ids: string[] }>(buildApiUrl('/api/import/prd/commit', projectDir), {
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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to commit import')
    }
  }

  async function submitQuickAction(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!quickPrompt.trim()) return
    try {
      const resp = await requestJson<{ quick_action: QuickActionRecord }>(buildApiUrl('/api/quick-actions', projectDir), {
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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit quick action')
    }
  }

  async function taskAction(taskId: string, action: 'run' | 'retry' | 'cancel'): Promise<void> {
    const actionPrefix = action === 'run'
      ? 'Failed to run task'
      : action === 'retry'
        ? 'Failed to retry task'
        : 'Failed to cancel task'
    const startMessage = action === 'run' ? 'Starting task run...' : undefined
    const successMessage = action === 'run'
      ? 'Run request completed.'
      : action === 'retry'
        ? 'Task queued for retry.'
        : 'Task cancelled.'
    await runTaskMutation(
      action,
      async () => {
        await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/${action}`, projectDir), {
          method: 'POST',
        })
        await reloadAll()
        if (selectedTaskIdRef.current === taskId) {
          await loadTaskDetail(taskId)
          await loadTaskLogs(taskId, true)
        }
      },
      {
        startMessage,
        successMessage,
        errorPrefix: actionPrefix,
      },
    )
  }

  async function transitionTask(taskId: string, targetStatus: string): Promise<void> {
    const status = targetStatus
    await runTaskMutation(
      'transition',
      async () => {
        await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/transition`, projectDir), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status }),
        })
        await reloadAll()
        if (selectedTaskIdRef.current === taskId) {
          await loadTaskDetail(taskId)
        }
      },
      {
        successMessage: `Task moved to ${humanizeLabel(status)}.`,
        errorPrefix: 'Failed to transition task',
      },
    )
  }

  async function addDependency(taskId: string): Promise<void> {
    if (!newDependencyId.trim()) return
    try {
      await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/dependencies`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ depends_on: newDependencyId.trim() }),
      })
      setNewDependencyId('')
      await reloadAll()
      if (selectedTaskIdRef.current === taskId) {
        await loadTaskDetail(taskId)
      }
    } catch (err) {
      setTaskActionError(err instanceof Error ? err.message : 'Failed to add dependency')
    }
  }

  async function removeDependency(taskId: string, depId: string): Promise<void> {
    try {
      await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/dependencies/${depId}`, projectDir), {
        method: 'DELETE',
      })
      await reloadAll()
      if (selectedTaskIdRef.current === taskId) {
        await loadTaskDetail(taskId)
      }
    } catch (err) {
      setTaskActionError(err instanceof Error ? err.message : 'Failed to remove dependency')
    }
  }

  async function analyzeDependencies(): Promise<void> {
    setDependencyActionLoading(true)
    setDependencyActionMessage('')
    try {
      const result = await requestJson<{ edges?: Array<{ from: string; to: string; reason?: string }> }>(
        buildApiUrl('/api/tasks/analyze-dependencies', projectDir),
        { method: 'POST' },
      )
      const edgeCount = result.edges?.length || 0
      setDependencyActionMessage(`Dependency analysis complete (${edgeCount} inferred edge${edgeCount === 1 ? '' : 's'}).`)
      await reloadAll()
      if (selectedTaskId) {
        await loadTaskDetail(selectedTaskId)
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setDependencyActionMessage(`Dependency analysis failed (${detail})`)
    } finally {
      setDependencyActionLoading(false)
    }
  }

  async function resetDependencyAnalysis(taskId: string): Promise<void> {
    setDependencyActionLoading(true)
    setDependencyActionMessage('')
    try {
      await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/reset-dep-analysis`, projectDir), {
        method: 'POST',
      })
      setDependencyActionMessage('Reset inferred dependency analysis for selected task.')
      await reloadAll()
      await loadTaskDetail(taskId)
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setDependencyActionMessage(`Reset dependency analysis failed (${detail})`)
    } finally {
      setDependencyActionLoading(false)
    }
  }

  async function approveGate(taskId: string, gate?: string | null): Promise<void> {
    try {
      await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/approve-gate`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gate: gate || undefined }),
      })
      await reloadAll()
      if (selectedTaskIdRef.current === taskId) {
        await loadTaskDetail(taskId)
      }
    } catch (err) {
      setTaskActionError(err instanceof Error ? err.message : 'Failed to approve gate')
    }
  }

  async function refineTaskPlan(taskId: string): Promise<void> {
    if (planJobLoading) return
    const activeJob = selectedTaskPlan?.active_refine_job
    if (activeJob && (activeJob.status === 'queued' || activeJob.status === 'running')) return
    if (!planRefineFeedback.trim()) {
      setPlanActionError('Refine feedback is required.')
      return
    }
    setPlanJobLoading(true)
    setPlanActionMessage('')
    setPlanActionError('')
    try {
      const baseRevisionId = selectedPlanRevisionId || selectedTaskPlan?.latest_revision_id || undefined
      await requestJson<{ job: PlanRefineJobRecord }>(buildApiUrl(`/api/tasks/${taskId}/plan/refine`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_revision_id: baseRevisionId,
          feedback: planRefineFeedback.trim(),
          priority: 'normal',
        }),
      })
      setPlanActionMessage('Plan refine job queued.')
      planRefineOutputRef.current = { taskId, jobKey: '', text: '' }
      setPlanRefineStdout('')
      await loadTaskPlan(taskId)
    } catch (err) {
      setPlanActionError(toErrorMessage('Failed to queue plan refine job', err))
      setPlanJobLoading(false)
    }
  }

  async function saveManualPlanRevision(taskId: string): Promise<void> {
    const manualContent = planManualContent.trim()
    if (!manualContent) {
      setPlanActionError('Manual revision content is required.')
      return
    }
    setPlanSavingManual(true)
    setPlanActionMessage('')
    setPlanActionError('')
    try {
      const resp = await requestJson<{ revision: PlanRevisionRecord }>(buildApiUrl(`/api/tasks/${taskId}/plan/revisions`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: manualContent,
          parent_revision_id: selectedPlanRevisionId || selectedTaskPlan?.latest_revision_id || undefined,
          feedback_note: planManualFeedbackNote.trim() || undefined,
        }),
      })
      setPlanManualContent(resp.revision.content || manualContent)
      planManualSeedRef.current = { taskId, workerText: manualContent }
      setSelectedPlanRevisionId(resp.revision.id)
      setPlanActionMessage('Manual plan revision saved.')
      await loadTaskPlan(taskId)
    } catch (err) {
      setPlanActionError(toErrorMessage('Failed to save manual plan revision', err))
    } finally {
      setPlanSavingManual(false)
    }
  }

  async function commitPlanRevision(taskId: string, revisionId: string): Promise<void> {
    if (!revisionId) {
      setPlanActionError('Select a plan revision to commit.')
      return
    }
    setPlanCommitting(true)
    setPlanActionMessage('')
    setPlanActionError('')
    try {
      await requestJson<{ committed_revision_id: string }>(buildApiUrl(`/api/tasks/${taskId}/plan/commit`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ revision_id: revisionId }),
      })
      setPlanActionMessage('Committed selected plan revision.')
      await loadTaskPlan(taskId)
    } catch (err) {
      setPlanActionError(toErrorMessage('Failed to commit plan revision', err))
    } finally {
      setPlanCommitting(false)
    }
  }

  async function generateTasksFromPlan(taskId: string): Promise<void> {
    setPlanGenerateLoading(true)
    planGenerateOutputRef.current = { taskId, jobKey: '', text: '' }
    setPlanGenerateStdout('')
    setPlanActionMessage('')
    setPlanActionError('')
    let aborted = false
    const captureGenerateLogs = async (): Promise<void> => {
      if (aborted || selectedTaskIdRef.current !== taskId) return
      try {
        const logs = await requestJson<TaskLogsSnapshot>(buildApiUrl(`/api/tasks/${taskId}/logs?max_chars=50000`, projectDir))
        if (aborted || selectedTaskIdRef.current !== taskId) return
        if (logs.step !== 'generate_tasks' || !logs.stdout) return
        const rendered = renderStructuredStdout(logs.stdout)
        if (!rendered.hasContent) return
        const jobKey = String(logs.started_at || logs.finished_at || 'generate_tasks')
        appendPlanGenerateStdout(taskId, jobKey, rendered.text)
      } catch {
        // Best-effort only.
      }
    }
    void captureGenerateLogs()
    const generateLogsTimer = window.setInterval(() => { void captureGenerateLogs() }, 1_000)
    try {
      const payload: Record<string, unknown> = {
        source: planGenerateSource,
        infer_deps: planGenerateInferDeps,
      }
      if (planGenerateSource === 'revision') {
        const revisionId = planGenerateRevisionId || selectedPlanRevisionId || selectedTaskPlan?.latest_revision_id || ''
        if (!revisionId) {
          throw new Error('Choose a revision source before generating.')
        }
        payload.revision_id = revisionId
      }
      if (planGenerateSource === 'override') {
        if (!planGenerateOverride.trim()) {
          throw new Error('Manual override plan text is required.')
        }
        payload.plan_override = planGenerateOverride
      }
      const result = await requestJson<{ created_task_ids: string[] }>(buildApiUrl(`/api/tasks/${taskId}/generate-tasks`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setPlanActionMessage(`Generated ${result.created_task_ids?.length || 0} task(s) from plan.`)
      try {
        const boardData = await requestJson<BoardResponse>(buildApiUrl('/api/tasks/board', projectDir))
        setBoard(boardData)
      } catch {
        // Keep user-facing flow resilient even if ancillary reloads fail.
      }
      await reloadAll()
      await loadTaskPlan(taskId)
    } catch (err) {
      setPlanActionError(toErrorMessage('Failed to generate tasks', err))
    } finally {
      aborted = true
      window.clearInterval(generateLogsTimer)
      await captureGenerateLogs()
      setPlanGenerateLoading(false)
    }
  }

  async function saveTaskEdits(taskId: string): Promise<void> {
    await runTaskMutation(
      'save',
      async () => {
        await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}`, projectDir), {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: editTaskTitle.trim(),
            description: editTaskDescription,
            task_type: editTaskType,
            priority: editTaskPriority,
            labels: editTaskLabels.split(',').map((item) => item.trim()).filter(Boolean),
            approval_mode: editTaskApprovalMode,
            hitl_mode: editTaskHitlMode,
          }),
        })
        await reloadAll()
        await loadTaskDetail(taskId)
      },
      {
        successMessage: 'Task configuration saved.',
        errorPrefix: 'Failed to save task configuration',
        clearEditModeOnSuccess: true,
      },
    )
  }

  function buildWorkerProvidersPayload(extraProviders: Record<string, WorkerProviderSettings>): Record<string, WorkerProviderSettings> {
    const providers: Record<string, WorkerProviderSettings> = {}

    const codexProvider: WorkerProviderSettings = {
      type: 'codex',
      command: settingsCodexCommand.trim() || 'codex exec',
    }
    const codexModel = settingsCodexModel.trim()
    if (codexModel) codexProvider.model = codexModel
    const codexEffort = settingsCodexEffort.trim().toLowerCase()
    if (codexEffort === 'low' || codexEffort === 'medium' || codexEffort === 'high') {
      codexProvider.reasoning_effort = codexEffort
    }
    providers.codex = codexProvider

    const claudeProvider: WorkerProviderSettings = {
      type: 'claude',
      command: settingsClaudeCommand.trim() || 'claude -p',
    }
    const claudeModel = settingsClaudeModel.trim()
    if (claudeModel) claudeProvider.model = claudeModel
    const claudeEffort = settingsClaudeEffort.trim().toLowerCase()
    if (claudeEffort === 'low' || claudeEffort === 'medium' || claudeEffort === 'high') {
      claudeProvider.reasoning_effort = claudeEffort
    }
    providers.claude = claudeProvider

    const ollamaEndpoint = settingsOllamaEndpoint.trim()
    const ollamaModel = settingsOllamaModel.trim()
    const shouldConfigureOllama = Boolean(ollamaModel) || settingsWorkerDefault === 'ollama'
    if (shouldConfigureOllama) {
      if (!ollamaEndpoint || !ollamaModel) {
        throw new Error('Ollama provider requires endpoint and model')
      }
      const ollamaProvider: WorkerProviderSettings = {
        type: 'ollama',
        endpoint: ollamaEndpoint,
        model: ollamaModel,
      }
      const temperature = Number(settingsOllamaTemperature)
      if (settingsOllamaTemperature.trim() && Number.isFinite(temperature)) {
        ollamaProvider.temperature = temperature
      }
      const numCtx = Number(settingsOllamaNumCtx)
      if (settingsOllamaNumCtx.trim() && Number.isFinite(numCtx) && numCtx > 0) {
        ollamaProvider.num_ctx = Math.floor(numCtx)
      }
      providers.ollama = ollamaProvider
    }

    for (const [name, provider] of Object.entries(extraProviders)) {
      providers[name] = provider
    }
    return providers
  }

  async function saveSettings(event: FormEvent): Promise<void> {
    event.preventDefault()
    setSettingsSaving(true)
    setSettingsError('')
    setSettingsSuccess('')
    try {
      const taskTypeRoles = parseStringMap(settingsTaskTypeRoles, 'Task type role map')
      const roleProviderOverrides = parseStringMap(settingsRoleProviderOverrides, 'Role provider overrides')
      const workerRouting = Object.fromEntries(
        workerRoutingRows
          .filter((row) => row.source === 'explicit' && row.step.trim() && row.provider.trim())
          .map((row) => [row.step.trim(), row.provider.trim()])
      )
      let advancedWorkerProviders: Record<string, WorkerProviderSettings> = {}
      try {
        advancedWorkerProviders = parseWorkerProviders(settingsWorkerProviders)
      } catch {
        advancedWorkerProviders = {}
      }
      const workerProviders = buildWorkerProvidersPayload(advancedWorkerProviders)
      const projectCommands = parseProjectCommands(settingsProjectCommands)
      const payload: SystemSettings = {
        orchestrator: {
          concurrency: Math.max(1, parseNonNegativeInt(settingsConcurrency, DEFAULT_SETTINGS.orchestrator.concurrency)),
          auto_deps: settingsAutoDeps,
          max_review_attempts: Math.max(1, parseNonNegativeInt(settingsMaxReviewAttempts, DEFAULT_SETTINGS.orchestrator.max_review_attempts)),
        },
        agent_routing: {
          default_role: settingsDefaultRole.trim() || 'general',
          task_type_roles: taskTypeRoles,
          role_provider_overrides: roleProviderOverrides,
        },
        defaults: {
          quality_gate: {
            critical: parseNonNegativeInt(settingsGateCritical, 0),
            high: parseNonNegativeInt(settingsGateHigh, 0),
            medium: parseNonNegativeInt(settingsGateMedium, 0),
            low: parseNonNegativeInt(settingsGateLow, 0),
          },
        },
        workers: {
          default: (settingsWorkerDefault === 'ollama' || settingsWorkerDefault === 'claude') ? settingsWorkerDefault : 'codex',
          default_model: '',
          routing: workerRouting,
          providers: workerProviders,
        },
        project: {
          commands: projectCommands,
        },
      }
      const updated = await requestJson<Partial<SystemSettings>>(buildApiUrl('/api/settings', projectDir), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      applySettings(normalizeSettings(updated))
      setSettingsSuccess('Settings saved.')
      await reloadAll()
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setSettingsError(`Failed to save settings (${detail})`)
    } finally {
      setSettingsSaving(false)
    }
  }

  async function saveWorkerMaps(event: FormEvent): Promise<void> {
    event.preventDefault()
    setSettingsSaving(true)
    setSettingsError('')
    setSettingsSuccess('')
    try {
      const workerRouting = parseStringMap(settingsWorkerRouting, 'Worker routing map')
      const advancedWorkerProviders = parseWorkerProviders(settingsWorkerProviders)
      const workerProviders = buildWorkerProvidersPayload(advancedWorkerProviders)
      const updated = await requestJson<Partial<SystemSettings>>(buildApiUrl('/api/settings', projectDir), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workers: {
            default: (settingsWorkerDefault === 'ollama' || settingsWorkerDefault === 'claude') ? settingsWorkerDefault : 'codex',
            default_model: '',
            routing: workerRouting,
            providers: workerProviders,
          },
        }),
      })
      applySettings(normalizeSettings(updated))
      setSettingsSuccess('Worker routing saved.')
      await reloadAll()
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setSettingsError(`Failed to save worker routing (${detail})`)
    } finally {
      setSettingsSaving(false)
    }
  }

  async function handleRecheckProviders(): Promise<void> {
    setWorkerHealthRefreshing(true)
    try {
      await refreshAgentsSurface()
    } finally {
      setWorkerHealthRefreshing(false)
    }
  }

  function updateWorkerRoute(step: string, provider: string): void {
    try {
      const current = parseStringMap(settingsWorkerRouting, 'Worker routing map')
      const next = { ...current }
      const normalizedStep = step.trim()
      if (!normalizedStep) return
      if (!provider.trim() || provider.trim() === workerDefaultProvider) {
        delete next[normalizedStep]
      } else {
        next[normalizedStep] = provider.trim()
      }
      setSettingsWorkerRouting(Object.keys(next).length > 0 ? JSON.stringify(next, null, 2) : '')
      setSettingsError('')
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'invalid routing JSON'
      setSettingsError(detail)
    }
  }

  function handleFormatJsonField(
    label: string,
    value: string,
    setter: (next: string) => void,
  ): void {
    try {
      setter(formatJsonObjectInput(value, label))
      setSettingsError('')
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'invalid JSON'
      setSettingsError(detail)
    }
  }

  async function promoteQuickAction(quickActionId: string): Promise<void> {
    try {
      await requestJson<{ task: TaskRecord; already_promoted: boolean }>(buildApiUrl(`/api/quick-actions/${quickActionId}/promote`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priority: 'P2' }),
      })
      await reloadAll()
      if (selectedQuickActionIdRef.current === quickActionId) {
        await loadQuickActionDetail(quickActionId)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to promote quick action')
    }
  }

  async function controlOrchestrator(action: 'pause' | 'resume' | 'drain' | 'stop'): Promise<void> {
    try {
      await requestJson<OrchestratorStatus>(buildApiUrl('/api/orchestrator/control', projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      })
      await reloadAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} orchestrator`)
    }
  }

  async function reviewAction(taskId: string, action: 'approve' | 'request-changes'): Promise<void> {
    try {
      const endpoint = action === 'approve' ? `/api/review/${taskId}/approve` : `/api/review/${taskId}/request-changes`
      await requestJson<{ task: TaskRecord }>(buildApiUrl(endpoint, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ guidance: reviewGuidance.trim() || undefined }),
      })
      setReviewGuidance('')
      await reloadAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit review')
    }
  }

  async function pinProjectPath(path: string, allowNonGitValue: boolean): Promise<void> {
    try {
      const pinned = await requestJson<{ project: ProjectRef }>(buildApiUrl('/api/projects/pinned', projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, allow_non_git: allowNonGitValue }),
      })
      setProjectDir(pinned.project?.path || path)
      await reloadAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pin project')
    }
  }

  async function unpinProject(projectId: string): Promise<void> {
    try {
      await requestJson<{ removed: boolean }>(buildApiUrl(`/api/projects/pinned/${projectId}`, projectDir), {
        method: 'DELETE',
      })
      await reloadAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unpin project')
    }
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
        buildApiUrl('/api/projects/browse', projectDir, nextPath ? { path: nextPath } : {})
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

  function handleRouteChange(nextRoute: RouteKey): void {
    modalDismissedRef.current = false
    window.location.hash = toHash(nextRoute)
    setRoute(nextRoute)
  }

  function handleTaskSelect(taskId: string, defaultTab?: TaskDetailTab): void {
    modalDismissedRef.current = false
    modalExplicitRef.current = true
    taskSelectTabRef.current = defaultTab
    setSelectedTaskId(taskId)
  }

  const boardColumns = ['backlog', 'queued', 'in_progress', 'in_review', 'blocked', 'done', 'cancelled']
  const allBoardTasks = boardColumns.flatMap((column) => board.columns[column] || [])
  const taskIndex = new Map<string, TaskRecord>()
  for (const task of allBoardTasks) {
    taskIndex.set(task.id, task)
  }
  if (selectedTaskDetail) {
    taskIndex.set(selectedTaskDetail.id, selectedTaskDetail)
  }
  const selectedTask = allBoardTasks.find((task) => task.id === selectedTaskId) ?? (selectedTaskId ? undefined : allBoardTasks[0])
  const selectedTaskView = selectedTaskDetail && selectedTaskDetail.id === selectedTaskId ? selectedTaskDetail : selectedTask
  const planRevisions = selectedTaskPlan?.revisions || []
  const blockerIds = selectedTaskView?.blocked_by || []
  const blockedIds = selectedTaskView?.blocks || []
  const isPlanTask = selectedTaskView?.task_type === 'plan' || selectedTaskView?.task_type === 'plan_only'
  const hasPlanContent = planRevisions.length > 0 || selectedTaskPlanJobs.length > 0 || !!selectedTaskPlan?.active_refine_job
  const isTaskInExecution = selectedTaskView?.status === 'in_progress' || selectedTaskLogs?.mode === 'active'
  const isTaskTerminal = selectedTaskView?.status === 'done' || selectedTaskView?.status === 'cancelled'
  const isTaskActionBusy = taskActionPending !== null
  const configLocked = !new Set(['backlog', 'blocked', 'cancelled']).has(selectedTaskView?.status || '')
  const taskStatus = selectedTaskView?.status || ''
  const unresolvedBlockers = blockerIds.filter((depId) => {
    const dep = taskIndex.get(depId)
    return !dep || (dep.status !== 'done' && dep.status !== 'cancelled')
  })
  const hasUnresolvedBlockers = unresolvedBlockers.length > 0
  const showViewPlan = isPlanTask && selectedTaskView?.status === 'done'

  const taskDetailContent = selectedTaskView ? (
      <div className="detail-card">
        {selectedTaskDetailLoading ? <p className="field-label">Loading full task detail...</p> : null}
        <p className="task-meta"><span className="task-id-chip" title={selectedTaskView.id} onClick={() => { void navigator.clipboard.writeText(selectedTaskView.id) }} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); void navigator.clipboard.writeText(selectedTaskView.id) } }}>{selectedTaskView.id.replace(/^task-/, '')}</span> · {selectedTaskView.priority} · {humanizeLabel(selectedTaskView.task_type || 'feature')}</p>
        <div className="detail-tabs" role="tablist" aria-label="Task detail sections">
          <button
            className={`detail-tab ${taskDetailTab === 'overview' ? 'is-active' : ''}`}
            aria-pressed={taskDetailTab === 'overview'}
            onClick={() => setTaskDetailTab('overview')}
          >
            Overview
          </button>
          <button
            className={`detail-tab ${taskDetailTab === 'logs' ? 'is-active' : ''}`}
            aria-pressed={taskDetailTab === 'logs'}
            onClick={() => setTaskDetailTab('logs')}
          >
            Logs
          </button>
          <button
            className={`detail-tab ${taskDetailTab === 'activity' ? 'is-active' : ''}`}
            aria-pressed={taskDetailTab === 'activity'}
            onClick={() => setTaskDetailTab('activity')}
          >
            Activity
          </button>
          <button
            className={`detail-tab ${taskDetailTab === 'dependencies' ? 'is-active' : ''}`}
            aria-pressed={taskDetailTab === 'dependencies'}
            onClick={() => setTaskDetailTab('dependencies')}
          >
            Dependencies
          </button>
          <button
            className={`detail-tab ${taskDetailTab === 'configuration' ? 'is-active' : ''}`}
            aria-pressed={taskDetailTab === 'configuration'}
            onClick={() => setTaskDetailTab('configuration')}
          >
            Configuration
          </button>
        </div>
        {taskDetailTab === 'overview' ? (
          <div className="task-detail-section-body">
            {selectedTaskView.description ? <RenderedMarkdown content={selectedTaskView.description} className="task-desc" /> : <p className="task-desc">No description.</p>}
            <p className="field-label">
              {'Depends on: '}
              {blockerIds.length > 0 ? (
                blockerIds.map((depId, idx) => {
                  const dep = describeTask(depId, taskIndex)
                  const resolved = dep.status === 'done' || dep.status === 'cancelled'
                  return (
                    <span key={`dep-inline-${depId}`}>
                      {idx > 0 ? ', ' : ''}
                      <button className="link-button" onClick={() => handleTaskSelect(depId)}>{dep.label}</button>
                      <span className={`status-pill status-pill-inline ${resolved ? 'status-running' : 'status-blocked'}`}>{humanizeLabel(dep.status)}</span>
                    </span>
                  )
                })
              ) : 'None'}
            </p>
            {showViewPlan ? (
              <button className="button button-primary" onClick={() => { setPlanningTaskId(selectedTaskView.id); handleRouteChange('planning') }}>
                View Plan
              </button>
            ) : null}
            {selectedTaskView.error?.trim() ? (() => {
              const stderrTail = (stderrHistory || '').trim()
              const logTail = (stdoutHistory || '').trim()
              const contextLines: string[] = []
              if (logTail) {
                const last = logTail.slice(-800)
                const fromNewline = last.indexOf('\n')
                contextLines.push(fromNewline > 0 ? last.slice(fromNewline + 1) : last)
              }
              if (stderrTail) {
                const last = stderrTail.slice(-400)
                const fromNewline = last.indexOf('\n')
                contextLines.push('stderr: ' + (fromNewline > 0 ? last.slice(fromNewline + 1) : last))
              }
              const context = contextLines.join('\n').trim()
              return (
                <div className="error-detail-box">
                  <p className="error-detail-label">Error</p>
                  <pre>{context ? `${selectedTaskView.error}\n\n${context}` : selectedTaskView.error}</pre>
                </div>
              )
            })() : null}
            {selectedTaskView.pending_gate ? (
              <div className="preview-box">
                <p className="field-label">
                  Pending gate: <strong>{humanizeLabel(selectedTaskView.pending_gate)}</strong>
                </p>
                <button className="button button-primary" onClick={() => void approveGate(selectedTaskView.id, selectedTaskView.pending_gate)}>
                  Approve gate
                </button>
              </div>
            ) : null}
            {Array.isArray(selectedTaskView.human_blocking_issues) && selectedTaskView.human_blocking_issues.length > 0 ? (
              <div className="preview-box">
                <p className="field-label">Human blocking issues</p>
                {selectedTaskView.human_blocking_issues.map((issue, index) => (
                  <div className="row-card" key={`task-human-issue-${index}`}>
                    <p className="task-title">{issue.summary}</p>
                    {issue.details ? <p className="task-desc">{issue.details}</p> : null}
                    {(issue.action || issue.blocking_on || issue.category || issue.severity) ? (
                      <p className="task-meta">
                        {issue.action ? `action: ${issue.action}` : null}
                        {issue.action && issue.blocking_on ? ' · ' : null}
                        {issue.blocking_on ? `blocking on: ${issue.blocking_on}` : null}
                        {(issue.action || issue.blocking_on) && issue.category ? ' · ' : null}
                        {issue.category ? `category: ${issue.category}` : null}
                        {(issue.action || issue.blocking_on || issue.category) && issue.severity ? ' · ' : null}
                        {issue.severity ? `severity: ${issue.severity}` : null}
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
        {taskDetailTab === 'configuration' ? (
          <div className="task-detail-section-body">
              <div className="form-stack">
                <label className="field-label" htmlFor="edit-task-approval-mode">Approval mode</label>
                <select
                  id="edit-task-approval-mode"
                  value={configLocked ? (selectedTaskView.approval_mode || 'human_review') : editTaskApprovalMode}
                  onChange={(event) => setEditTaskApprovalMode(event.target.value as 'human_review' | 'auto_approve')}
                  disabled={configLocked || taskActionPending === 'save'}
                >
                  <option value="human_review">{humanizeLabel('human_review')}</option>
                  <option value="auto_approve">{humanizeLabel('auto_approve')}</option>
                </select>
                <label className="field-label" htmlFor="edit-task-labels">Labels (comma-separated)</label>
                <input
                  id="edit-task-labels"
                  value={configLocked ? ((selectedTaskView.labels || []).join(', ')) : editTaskLabels}
                  onChange={(event) => setEditTaskLabels(event.target.value)}
                  disabled={configLocked || taskActionPending === 'save'}
                />
                <label className="field-label">HITL mode</label>
                {configLocked ? (
                  <p className="task-meta">{humanizeLabel(selectedTaskView.hitl_mode || 'default')}</p>
                ) : (
                  <HITLModeSelector
                    currentMode={editTaskHitlMode}
                    onModeChange={setEditTaskHitlMode}
                    projectDir={projectDir}
                  />
                )}
                {!configLocked ? (
                  <div className="inline-actions">
                    <button
                      className="button button-primary"
                      onClick={() => void saveTaskEdits(selectedTaskView.id)}
                      disabled={taskActionPending === 'save'}
                    >
                      {taskActionPending === 'save' ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                ) : null}
              </div>
          </div>
        ) : null}
        {taskDetailTab === 'dependencies' ? (
          <div className="task-detail-section-body">
            <div className="dependency-graph-panel">
              <p className="field-label">Dependency graph</p>
              <div className="dependency-graph-grid">
                <div className="dependency-graph-column">
                  <p className="field-label">Depends on</p>
                  {blockerIds.length > 0 ? (
                    blockerIds.map((depId) => {
                      const dep = describeTask(depId, taskIndex)
                      return (
                        <div className="dependency-node dependency-node-blocker" key={`blocker-${depId}`} onClick={() => handleTaskSelect(depId)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleTaskSelect(depId) } }} role="button" tabIndex={0} style={{ cursor: 'pointer' }}>
                          <p className="dependency-node-title">{dep.label}</p>
                          <p className="dependency-node-meta">{humanizeLabel(dep.status)} {'->'} depends on</p>
                        </div>
                      )
                    })
                  ) : (
                    <p className="empty">No dependencies</p>
                  )}
                </div>
                <div className="dependency-graph-column dependency-graph-center">
                  <p className="field-label">Selected task</p>
                  <div className="dependency-node dependency-node-current">
                    <p className="dependency-node-title">{selectedTaskView.title} ({selectedTaskView.id})</p>
                    <p className="dependency-node-meta">{humanizeLabel(selectedTaskView.status)}</p>
                  </div>
                </div>
                <div className="dependency-graph-column">
                  <p className="field-label">Blocks</p>
                  {blockedIds.length > 0 ? (
                    blockedIds.map((depId) => {
                      const dep = describeTask(depId, taskIndex)
                      return (
                        <div className="dependency-node dependency-node-dependent" key={`dependent-${depId}`} onClick={() => handleTaskSelect(depId)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleTaskSelect(depId) } }} role="button" tabIndex={0} style={{ cursor: 'pointer' }}>
                          <p className="dependency-node-title">{dep.label}</p>
                          <p className="dependency-node-meta">depends on current task · {humanizeLabel(dep.status)}</p>
                        </div>
                      )
                    })
                  ) : (
                    <p className="empty">No dependents</p>
                  )}
                </div>
              </div>
              {blockerIds.length > 0 || blockedIds.length > 0 ? (
                <div className="dependency-edge-list">
                  {blockerIds.map((depId) => (
                    <p key={`edge-in-${depId}`} className="dependency-edge">
                      {describeTask(depId, taskIndex).label} {'->'} {selectedTaskView.id}
                    </p>
                  ))}
                  {blockedIds.map((depId) => (
                    <p key={`edge-out-${depId}`} className="dependency-edge">
                      {selectedTaskView.id} {'->'} {describeTask(depId, taskIndex).label}
                    </p>
                  ))}
                </div>
              ) : null}
            </div>
            <div className="form-stack">
              <label className="field-label" htmlFor="task-blocker-input">Add dependency task ID</label>
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
              <div className="inline-actions">
                <button
                  className="button"
                  onClick={() => void analyzeDependencies()}
                  disabled={dependencyActionLoading}
                >
                  Analyze dependencies
                </button>
                <button
                  className="button"
                  onClick={() => void resetDependencyAnalysis(selectedTaskView.id)}
                  disabled={dependencyActionLoading}
                >
                  Reset inferred deps
                </button>
              </div>
              {dependencyActionMessage ? <p className="field-label">{dependencyActionMessage}</p> : null}
            </div>
          </div>
        ) : null}
        {taskDetailTab === 'activity' ? (
          <div className="task-detail-section-body">
            <div className="list-stack">
              {collaborationLoading ? <p className="field-label">Loading activity...</p> : null}
              {collaborationTimeline.slice(0, 8).map((event) => (
                <div className="row-card timeline-event-card" key={event.id}>
                  <div>
                    <p className="task-title">{event.summary || humanizeLabel(event.type)}</p>
                    <p className="task-meta">{humanizeLabel(event.type)} · {event.actor} · {toLocaleTimestamp(event.timestamp) || '-'}</p>
                  </div>
                  {event.details ? <RenderedMarkdown content={event.details} className="task-desc" /> : null}
                  {event.human_blocking_issues && event.human_blocking_issues.length > 0 ? (
                    <div className="list-stack">
                      <p className="field-label">Required human input</p>
                      {event.human_blocking_issues.map((issue, idx) => (
                        <p className="task-meta" key={`${event.id}-issue-${idx}`}>- {issue.summary}</p>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
              {collaborationTimeline.length === 0 && !collaborationLoading ? <p className="empty">No activity for this task yet.</p> : null}
              {collaborationError ? <p className="error-banner">{collaborationError}</p> : null}
            </div>
          </div>
        ) : null}
        {taskDetailTab === 'logs' ? (() => {
          const hasTaskLogs = !!(stdoutHistory || stderrHistory) || (!!selectedTaskLogs && (selectedTaskLogs.mode !== 'none' || !!selectedTaskLogs.stdout || !!selectedTaskLogs.stderr))
          const renderedStdout = renderStructuredStdout(stdoutHistory || '')
          const progressEntries = formatProgressEntries(selectedTaskLogs?.progress)
          return (
            <div className="task-detail-section-body">
              <p className="task-meta">
                Source: {humanizeLabel(selectedTaskLogs?.mode || 'none')}
                {selectedTaskLogs?.step ? ` · step: ${selectedTaskLogs.step}` : ''}
                {selectedTaskLogs?.started_at ? ` · started: ${toLocaleTimestamp(selectedTaskLogs.started_at) || selectedTaskLogs.started_at}` : ''}
                {selectedTaskLogs?.finished_at ? ` · finished: ${toLocaleTimestamp(selectedTaskLogs.finished_at) || selectedTaskLogs.finished_at}` : ''}
              </p>
              {selectedTaskLogsLoading ? <p className="field-label">Loading logs...</p> : null}
              {selectedTaskLogsError ? <p className="error-banner">{selectedTaskLogsError}</p> : null}
              {progressEntries.length > 0 ? (
                <div className="preview-box">
                  <p className="field-label">Run snapshot</p>
                  {progressEntries.map((item) => (
                    <p className="task-meta" key={`detail-log-progress-${item.key}`}>
                      <strong>{humanizeLabel(item.key)}:</strong> {item.value}
                    </p>
                  ))}
                </div>
              ) : null}
              {hasTaskLogs ? (
                <div className="task-log-grid">
                  <div className="task-log-pane">
                    <p className="field-label">Stdout{renderedStdout.structured ? ' (rendered)' : ''}</p>
                    {renderedStdout.structured ? (
                      <p className="task-meta">Parsed {renderedStdout.parsedLines} JSON lines · {renderedStdout.streamEvents} stream events</p>
                    ) : null}
                    <pre className="task-log-output" ref={stdoutPreRef} onScroll={() => handleLogPaneScroll('stdout')}>{renderedStdout.text || '(empty)'}</pre>
                  </div>
                  <div className="task-log-pane">
                    <p className="field-label">Stderr</p>
                    <pre className="task-log-output" ref={stderrPreRef} onScroll={() => handleLogPaneScroll('stderr')}>{stderrHistory || '(empty)'}</pre>
                  </div>
                </div>
              ) : (
                <p className="empty">No logs available.</p>
              )}
            </div>
          )
        })() : null}
      </div>
    ) : (
      <p className="empty">No tasks on board yet.</p>
    )

  function renderBoard(): JSX.Element {
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Board</h2>
          <label className="switch-label"><input type="checkbox" role="switch" checked={boardCompact} onChange={() => setBoardCompact((v) => !v)} /> Compact</label>
        </header>
        <div className="board-grid">
          {boardColumns.map((column) => (
            <article className="board-col" key={column}>
              <h3>{humanizeLabel(column)}</h3>
              <div className="card-list">
                {(board.columns[column] || []).map((task) => (
                  <button className={`task-card task-card-button${boardCompact ? ' task-card-compact' : ''}`} key={task.id} onClick={() => handleTaskSelect(task.id)}>
                    <p className="task-title">{task.title}</p>
                    {!boardCompact && <p className="task-meta">{task.priority} · {task.id.replace(/^task-/, '')}</p>}
                    {!boardCompact && task.description ? <p className="task-desc">{task.description}</p> : null}
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>
        <div className="board-summary">
          <span className="field-label">Queue: {orchestrator?.queue_depth ?? 0}</span>
          <span className="field-label">In progress: {orchestrator?.in_progress ?? 0}</span>
          <span className="field-label">Workers: {agents.length}</span>
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
    const blockedCount = (board.columns.blocked || []).length
    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Execution</h2>
          <div className="inline-actions">
            {orchestrator?.status === 'running' ? <button className="button" onClick={() => void controlOrchestrator('pause')}>Pause</button> : null}
            {orchestrator?.status === 'paused' || orchestrator?.status === 'stopped' ? <button className="button" onClick={() => void controlOrchestrator('resume')}>Resume</button> : null}
            {orchestrator?.status === 'running' && !orchestrator?.draining ? <button className="button" onClick={() => void controlOrchestrator('drain')}>Drain</button> : null}
            {orchestrator?.status !== 'stopped' ? <button className="button button-danger" onClick={() => void controlOrchestrator('stop')}>Stop</button> : null}
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
            <span>Blocked</span>
            <strong>{blockedCount}</strong>
          </div>
        </div>
        <div className="list-stack">
          <p className="field-label section-heading">Execution pipeline</p>
          {executionBatches.map((batch, index) => (
            <div className="wave-card" key={`batch-${index}`}>
              <div className="wave-label">
                <p className="wave-title">Wave {index + 1}</p>
                <p className="wave-count">{batch.length} {batch.length === 1 ? 'task' : 'tasks'}</p>
              </div>
              <div className="wave-tasks">
                {batch.map((taskId, i) => {
                  const task = taskById.get(taskId)
                  const label = task?.title || taskId
                  const status = task?.status || ''
                  return (
                    <span key={`batch-${index}-${taskId}`} className="wave-task-item">
                      {i > 0 ? <span className="wave-sep">|</span> : null}
                      <button className="link-button" onClick={() => handleTaskSelect(taskId, 'logs')}>{label}</button>
                      {status ? <span className={`status-pill status-pill-inline ${statusPillClass(status)}`}>{humanizeLabel(status)}</span> : null}
                    </span>
                  )
                })}
              </div>
            </div>
          ))}
          {executionBatches.length === 0 ? <p className="empty">No execution batches available.</p> : null}
        </div>
        <div className="list-stack">
          <p className="field-label section-heading">Runtime metrics</p>
          <div className="row-card">
            <p className="task-meta">
              API calls: {metrics?.api_calls ?? 0} ·
              wall time: {metrics?.wall_time_seconds ?? 0}s ·
              steps: {metrics?.phases_completed ?? 0}/{metrics?.phases_total ?? 0}
            </p>
            <p className="task-meta">
              tokens: {metrics?.tokens_used ?? 0} ·
              est cost: ${(metrics?.estimated_cost_usd ?? 0).toFixed(2)} ·
              files changed: {metrics?.files_changed ?? 0}
            </p>
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
    const healthOrder = ['codex', 'claude', 'ollama']
    const providerHealth = [...workerHealth].sort((a, b) => {
      const aIndex = healthOrder.indexOf(a.name)
      const bIndex = healthOrder.indexOf(b.name)
      const aRank = aIndex === -1 ? 999 : aIndex
      const bRank = bIndex === -1 ? 999 : bIndex
      if (aRank !== bRank) return aRank - bRank
      return a.name.localeCompare(b.name)
    })
    const inProgressTasks = board.columns?.in_progress || []
    const routingByStep = new Map(workerRoutingRows.map((row) => [row.step, row]))
    let editableRoutingMap: Record<string, string> = {}
    try {
      editableRoutingMap = parseStringMap(settingsWorkerRouting, 'Worker routing map')
    } catch {
      editableRoutingMap = {}
    }
    const dropdownProviders = Array.from(
      new Set(
        workerHealth
          .filter((item) => item.configured || item.healthy)
          .map((item) => item.name)
          .concat(workerDefaultProvider || 'codex')
      )
    ).sort((a, b) => a.localeCompare(b))

    const statusClass = (status: WorkerHealthRecord['status']): string => {
      if (status === 'connected') return 'status-pill status-running'
      if (status === 'not_configured') return 'status-pill status-paused'
      return 'status-pill status-failed'
    }

    const resolvedProviderForTask = (task: TaskRecord): string => {
      const step = String(task.current_step || '').trim() || (task.task_type === 'plan' ? 'plan' : 'implement')
      return routingByStep.get(step)?.provider || workerDefaultProvider
    }

    const stepLabel = (step: string): string => {
      const normalized = String(step || '').trim().toLowerCase()
      if (normalized === 'plan') return 'Task Planning'
      if (normalized === 'plan_impl') return 'Execution Plan'
      return humanizeLabel(normalized)
    }

    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Workers</h2>
        </header>
        <div className="agents-layout">
          <article className="settings-card agents-active-card">
            <h3>Provider Status</h3>
            <p className="field-label">Availability of configured worker providers.</p>
            <div className="inline-actions workers-recheck-actions">
              <button
                className={`button ${workerHealthRefreshing ? 'is-loading' : ''}`}
                onClick={() => void handleRecheckProviders()}
                disabled={workerHealthRefreshing}
                aria-busy={workerHealthRefreshing}
              >
                {workerHealthRefreshing ? 'Rechecking...' : 'Recheck providers'}
              </button>
            </div>
            <div className="list-stack">
              {providerHealth.map((provider) => (
                <div className="row-card" key={provider.name}>
                  <div>
                    <p className="task-title">{humanizeLabel(provider.name)}</p>
                    <p className="task-meta">
                      type: {provider.type}
                      {provider.model ? ` · model: ${provider.model}` : ''}
                    </p>
                    {provider.command ? <p className="task-meta">command: {provider.command}</p> : null}
                    {provider.endpoint ? <p className="task-meta">endpoint: {provider.endpoint}</p> : null}
                    <p className="task-meta">{provider.detail || 'No diagnostics message.'}</p>
                  </div>
                  <div className="inline-actions">
                    <span className={statusClass(provider.status)}>{humanizeLabel(provider.status)}</span>
                  </div>
                </div>
              ))}
              {providerHealth.length === 0 ? <p className="empty">No providers detected.</p> : null}
            </div>
          </article>

          <article className="settings-card agents-presence-card">
            <h3>Routing Table</h3>
            <p className="field-label">Default provider: {workerDefaultProvider}</p>
            <div className="list-stack">
              {workerRoutingRows.map((row) => (
                <div className="row-card" key={row.step}>
                  <div>
                    <p className="task-title">{stepLabel(row.step)}</p>
                    <p className="task-meta">{row.source === 'explicit' ? 'Explicit route' : 'Default route'}</p>
                  </div>
                  <div className="inline-actions">
                    <select
                      aria-label={`Route ${row.step} provider`}
                      value={editableRoutingMap[row.step] ?? ''}
                      onChange={(event) => updateWorkerRoute(row.step, event.target.value)}
                    >
                      <option value="">Use default ({workerDefaultProvider})</option>
                      {dropdownProviders.map((providerName) => (
                        <option key={`${row.step}-${providerName}`} value={providerName}>
                          {providerName}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              ))}
              {workerRoutingRows.length === 0 ? <p className="empty">No routing rules configured; default applies to all steps.</p> : null}
            </div>
            <details className="advanced-fields workers-advanced">
              <summary>Advanced</summary>
              <form className="advanced-fields-body form-stack" onSubmit={(event) => void saveWorkerMaps(event)}>
                <label className="field-label" htmlFor="settings-worker-routing">Worker routing map (JSON object: step {'->'} provider)</label>
                <div className="json-editor-group">
                  <textarea
                    id="settings-worker-routing"
                    className="json-editor-textarea"
                    rows={4}
                    value={settingsWorkerRouting}
                    onChange={(event) => setSettingsWorkerRouting(event.target.value)}
                    placeholder={WORKER_ROUTING_EXAMPLE}
                  />
                  <div className="inline-actions json-editor-actions">
                    <button
                      className="button"
                      type="button"
                      onClick={() => handleFormatJsonField('Worker routing map', settingsWorkerRouting, setSettingsWorkerRouting)}
                    >
                      Format
                    </button>
                    <button className="button" type="button" onClick={() => setSettingsWorkerRouting('')}>
                      Clear
                    </button>
                  </div>
                </div>
                <label
                  className="field-label"
                  htmlFor="settings-worker-providers"
                  title="Configure reasoning effort in your CLI setup first (Codex/Claude profile or config). Agent Orchestrator only passes flags supported by your installed CLI version."
                >
                  Worker providers (JSON object, optional advanced overrides)
                </label>
                <div className="json-editor-group">
                  <textarea
                    id="settings-worker-providers"
                    className="json-editor-textarea"
                    rows={8}
                    value={settingsWorkerProviders}
                    onChange={(event) => setSettingsWorkerProviders(event.target.value)}
                    placeholder={WORKER_PROVIDERS_EXAMPLE}
                  />
                  <div className="inline-actions json-editor-actions">
                    <button
                      className="button"
                      type="button"
                      onClick={() => handleFormatJsonField('Worker providers', settingsWorkerProviders, setSettingsWorkerProviders)}
                    >
                      Format
                    </button>
                    <button className="button" type="button" onClick={() => setSettingsWorkerProviders('')}>
                      Clear
                    </button>
                  </div>
                </div>
                <div className="inline-actions">
                  <button className="button button-primary" type="submit" disabled={settingsSaving}>
                    {settingsSaving ? 'Saving...' : 'Save worker routing'}
                  </button>
                  <button className="button" type="button" onClick={() => void loadSettings()} disabled={settingsLoading}>
                    {settingsLoading ? 'Loading...' : 'Reload'}
                  </button>
                </div>
                {settingsError ? <p className="error-banner">{settingsError}</p> : null}
                {settingsSuccess ? <p className="field-label">{settingsSuccess}</p> : null}
              </form>
            </details>
          </article>

          <article className="settings-card agents-catalog-card">
            <h3>Execution Snapshot</h3>
            <p className="field-label">
              Queue depth: {orchestrator?.queue_depth ?? 0} · In progress: {orchestrator?.in_progress ?? inProgressTasks.length}
            </p>
            <div className="list-stack">
              {inProgressTasks.map((task) => (
                <div className="row-card" key={task.id}>
                  <div>
                    <p className="task-title">{task.title}</p>
                    <p className="task-meta">
                      {task.id} · step: {task.current_step || 'implement'}
                    </p>
                  </div>
                  <div className="inline-actions">
                    <span className="status-pill status-running">{resolvedProviderForTask(task)}</span>
                  </div>
                </div>
              ))}
              {inProgressTasks.length === 0 ? <p className="empty">No tasks currently in progress.</p> : null}
              <p className="task-meta">Worker labels are derived from settings routing and default provider.</p>
            </div>
          </article>
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
          <article className="settings-card settings-card-projects">
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

          <article className="settings-card settings-card-diagnostics">
            <h3>Diagnostics</h3>
            <p>Schema version: 3</p>
            <p>Selected route: {humanizeLabel(route)}</p>
            <p>Project dir: {projectDir || 'current workspace'}</p>
          </article>

          <article className="settings-card settings-card-routing">
            <h3>Execution & Routing</h3>
            <form className="form-stack" onSubmit={(event) => void saveSettings(event)}>
              <label className="field-label" htmlFor="settings-concurrency">Orchestrator concurrency</label>
              <input
                id="settings-concurrency"
                value={settingsConcurrency}
                onChange={(event) => setSettingsConcurrency(event.target.value)}
                inputMode="numeric"
              />
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={settingsAutoDeps}
                  onChange={(event) => setSettingsAutoDeps(event.target.checked)}
                />
                Auto dependency analysis
              </label>
              <label className="field-label" htmlFor="settings-review-attempts">Max review attempts</label>
              <input
                id="settings-review-attempts"
                value={settingsMaxReviewAttempts}
                onChange={(event) => setSettingsMaxReviewAttempts(event.target.value)}
                inputMode="numeric"
              />
              <p className="settings-subheading">Role Routing</p>
              <label className="field-label" htmlFor="settings-default-role">Default role</label>
              <input
                id="settings-default-role"
                value={settingsDefaultRole}
                onChange={(event) => setSettingsDefaultRole(event.target.value)}
                placeholder="general"
              />
              <label className="field-label" htmlFor="settings-task-type-roles">Task type role map (JSON object)</label>
              <p className="field-label">Maps <code>task_type</code> {'->'} <code>role</code>. Unmapped task types use Default role.</p>
              <div className="json-editor-group">
                <textarea
                  id="settings-task-type-roles"
                  className="json-editor-textarea"
                  rows={4}
                  value={settingsTaskTypeRoles}
                  onChange={(event) => setSettingsTaskTypeRoles(event.target.value)}
                  placeholder={TASK_TYPE_ROLE_MAP_EXAMPLE}
                />
                <div className="inline-actions json-editor-actions">
                  <button
                    className="button"
                    type="button"
                    onClick={() => handleFormatJsonField('Task type role map', settingsTaskTypeRoles, setSettingsTaskTypeRoles)}
                  >
                    Format
                  </button>
                  <button className="button" type="button" onClick={() => setSettingsTaskTypeRoles('')}>
                    Clear
                  </button>
                </div>
              </div>
              <label className="field-label" htmlFor="settings-role-overrides">Role provider overrides (JSON object)</label>
              <p className="field-label">Maps <code>role</code> {'->'} <code>provider</code> when that role executes.</p>
              <div className="json-editor-group">
                <textarea
                  id="settings-role-overrides"
                  className="json-editor-textarea"
                  rows={4}
                  value={settingsRoleProviderOverrides}
                  onChange={(event) => setSettingsRoleProviderOverrides(event.target.value)}
                  placeholder={ROLE_PROVIDER_OVERRIDES_EXAMPLE}
                />
                <div className="inline-actions json-editor-actions">
                  <button
                    className="button"
                    type="button"
                    onClick={() => handleFormatJsonField('Role provider overrides', settingsRoleProviderOverrides, setSettingsRoleProviderOverrides)}
                  >
                    Format
                  </button>
                  <button className="button" type="button" onClick={() => setSettingsRoleProviderOverrides('')}>
                    Clear
                  </button>
                </div>
              </div>
              <p className="settings-subheading">Worker Routing</p>
              <label className="field-label" htmlFor="settings-worker-default">Default worker provider</label>
              <select
                id="settings-worker-default"
                value={settingsWorkerDefault}
                onChange={(event) => setSettingsWorkerDefault(event.target.value)}
              >
                <option value="codex">codex</option>
                <option value="ollama">ollama</option>
                <option value="claude">claude</option>
              </select>
              <label className="field-label" htmlFor="settings-provider-view">Configure provider</label>
              <select
                id="settings-provider-view"
                value={settingsProviderView}
                onChange={(event) => setSettingsProviderView(event.target.value as 'codex' | 'ollama' | 'claude')}
              >
                <option value="codex">codex</option>
                <option value="ollama">ollama</option>
                <option value="claude">claude</option>
              </select>
              <div className="provider-grid">
                {settingsProviderView === 'codex' ? (
                  <div className="provider-card">
                  <p className="field-label">Codex provider</p>
                  <label className="field-label" htmlFor="settings-codex-command">Codex command</label>
                  <input
                    id="settings-codex-command"
                    value={settingsCodexCommand}
                    onChange={(event) => setSettingsCodexCommand(event.target.value)}
                    placeholder="codex exec"
                  />
                  <label className="field-label" htmlFor="settings-codex-model">Codex model (optional)</label>
                  <input
                    id="settings-codex-model"
                    value={settingsCodexModel}
                    onChange={(event) => setSettingsCodexModel(event.target.value)}
                    placeholder="gpt-5.3-codex"
                  />
                  <label className="field-label" htmlFor="settings-codex-effort">Codex effort (optional)</label>
                  <select
                    id="settings-codex-effort"
                    value={settingsCodexEffort}
                    onChange={(event) => setSettingsCodexEffort(event.target.value)}
                  >
                    <option value="">(none)</option>
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                  </select>
                  </div>
                ) : null}
                {settingsProviderView === 'ollama' ? (
                  <div className="provider-card">
                  <p className="field-label">Ollama provider</p>
                  <label className="field-label" htmlFor="settings-ollama-endpoint">Ollama endpoint</label>
                  <input
                    id="settings-ollama-endpoint"
                    value={settingsOllamaEndpoint}
                    onChange={(event) => setSettingsOllamaEndpoint(event.target.value)}
                    placeholder="http://localhost:11434"
                  />
                  <label className="field-label" htmlFor="settings-ollama-model">Ollama model</label>
                  <input
                    id="settings-ollama-model"
                    value={settingsOllamaModel}
                    onChange={(event) => setSettingsOllamaModel(event.target.value)}
                    placeholder="llama3.1:8b"
                  />
                  <div className="inline-actions">
                    <input
                      aria-label="Ollama temperature"
                      value={settingsOllamaTemperature}
                      onChange={(event) => setSettingsOllamaTemperature(event.target.value)}
                      placeholder="temperature"
                    />
                    <input
                      aria-label="Ollama num ctx"
                      value={settingsOllamaNumCtx}
                      onChange={(event) => setSettingsOllamaNumCtx(event.target.value)}
                      placeholder="num_ctx"
                    />
                  </div>
                  </div>
                ) : null}
                {settingsProviderView === 'claude' ? (
                  <div className="provider-card">
                  <p className="field-label">Claude provider</p>
                  <label className="field-label" htmlFor="settings-claude-command">Claude command</label>
                  <input
                    id="settings-claude-command"
                    value={settingsClaudeCommand}
                    onChange={(event) => setSettingsClaudeCommand(event.target.value)}
                    placeholder="claude -p"
                  />
                  <label className="field-label" htmlFor="settings-claude-model">Claude model (optional)</label>
                  <input
                    id="settings-claude-model"
                    value={settingsClaudeModel}
                    onChange={(event) => setSettingsClaudeModel(event.target.value)}
                    placeholder="sonnet"
                  />
                  <label className="field-label" htmlFor="settings-claude-effort">Claude effort (optional)</label>
                  <select
                    id="settings-claude-effort"
                    value={settingsClaudeEffort}
                    onChange={(event) => setSettingsClaudeEffort(event.target.value)}
                  >
                    <option value="">(none)</option>
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                  </select>
                  </div>
                ) : null}
              </div>
              <p className="settings-subheading">Project Commands</p>
              <label className="field-label" htmlFor="settings-project-commands">Project commands by language (JSON object)</label>
              <p className="field-label">
                Used by workers during implement/review steps. Keys are language names (`python`, `typescript`, `go`) and each language supports `test`, `lint`, `typecheck`, `format`.
              </p>
              <div className="json-editor-group">
                <textarea
                  id="settings-project-commands"
                  className="json-editor-textarea"
                  rows={8}
                  value={settingsProjectCommands}
                  onChange={(event) => setSettingsProjectCommands(event.target.value)}
                  placeholder={PROJECT_COMMANDS_EXAMPLE}
                />
                <div className="inline-actions json-editor-actions">
                  <button
                    className="button"
                    type="button"
                    onClick={() => handleFormatJsonField('Project commands', settingsProjectCommands, setSettingsProjectCommands)}
                  >
                    Format
                  </button>
                  <button className="button" type="button" onClick={() => setSettingsProjectCommands('')}>
                    Clear
                  </button>
                </div>
              </div>
              <p className="settings-subheading">Quality Gate</p>
              <p className="field-label">
                Define how many unresolved findings can remain before a task can pass the quality gate. Use `0` to require all findings at that severity to be fixed.
              </p>
              <div className="quality-gate-grid">
                <div className="quality-gate-row">
                  <div>
                    <p className="quality-gate-label">
                      <span className="quality-severity-badge severity-critical">Critical</span>
                    </p>
                    <p className="field-label">Release-blocking or security-critical issues.</p>
                  </div>
                  <div className="quality-gate-input-wrap">
                    <label className="field-label" htmlFor="quality-gate-critical-input">Allowed unresolved</label>
                    <input
                      id="quality-gate-critical-input"
                      aria-label="Quality gate critical"
                      value={settingsGateCritical}
                      onChange={(event) => setSettingsGateCritical(event.target.value)}
                      inputMode="numeric"
                    />
                  </div>
                </div>
                <div className="quality-gate-row">
                  <div>
                    <p className="quality-gate-label">
                      <span className="quality-severity-badge severity-high">High</span>
                    </p>
                    <p className="field-label">Major correctness or reliability problems.</p>
                  </div>
                  <div className="quality-gate-input-wrap">
                    <label className="field-label" htmlFor="quality-gate-high-input">Allowed unresolved</label>
                    <input
                      id="quality-gate-high-input"
                      aria-label="Quality gate high"
                      value={settingsGateHigh}
                      onChange={(event) => setSettingsGateHigh(event.target.value)}
                      inputMode="numeric"
                    />
                  </div>
                </div>
                <div className="quality-gate-row">
                  <div>
                    <p className="quality-gate-label">
                      <span className="quality-severity-badge severity-medium">Medium</span>
                    </p>
                    <p className="field-label">Important issues that should be addressed soon.</p>
                  </div>
                  <div className="quality-gate-input-wrap">
                    <label className="field-label" htmlFor="quality-gate-medium-input">Allowed unresolved</label>
                    <input
                      id="quality-gate-medium-input"
                      aria-label="Quality gate medium"
                      value={settingsGateMedium}
                      onChange={(event) => setSettingsGateMedium(event.target.value)}
                      inputMode="numeric"
                    />
                  </div>
                </div>
                <div className="quality-gate-row">
                  <div>
                    <p className="quality-gate-label">
                      <span className="quality-severity-badge severity-low">Low</span>
                    </p>
                    <p className="field-label">Minor issues, cleanup, and polish improvements.</p>
                  </div>
                  <div className="quality-gate-input-wrap">
                    <label className="field-label" htmlFor="quality-gate-low-input">Allowed unresolved</label>
                    <input
                      id="quality-gate-low-input"
                      aria-label="Quality gate low"
                      value={settingsGateLow}
                      onChange={(event) => setSettingsGateLow(event.target.value)}
                      inputMode="numeric"
                    />
                  </div>
                </div>
              </div>
              <div className="inline-actions">
                <button className="button button-primary" type="submit" disabled={settingsSaving}>
                  {settingsSaving ? 'Saving...' : 'Save settings'}
                </button>
                <button className="button" type="button" onClick={() => void loadSettings()} disabled={settingsLoading}>
                  {settingsLoading ? 'Loading...' : 'Reload settings'}
                </button>
              </div>
              {settingsError ? <p className="error-banner">{settingsError}</p> : null}
              {settingsSuccess ? <p className="field-label">{settingsSuccess}</p> : null}
            </form>
          </article>
        </div>
      </section>
    )
  }

  function renderPlanning(): JSX.Element {
    const allTasks: TaskRecord[] = Object.values(board.columns).flat()
    const planningTask = allTasks.find((t) => t.id === planningTaskId) || null
    const planRevisions = selectedTaskPlan?.revisions || []
    const selectedPlanRevision = selectedPlanRevisionId
      ? (planRevisions.find((item) => item.id === selectedPlanRevisionId) || null)
      : null
    const latestPlanRevision = selectedTaskPlan?.latest_revision_id
      ? (planRevisions.find((item) => item.id === selectedTaskPlan.latest_revision_id) || null)
      : null
    const effectiveWorkerPlanRevision = selectedPlanRevision || latestPlanRevision
    const selectedPlanParentRevision = effectiveWorkerPlanRevision?.parent_revision_id
      ? planRevisions.find((item) => item.id === effectiveWorkerPlanRevision.parent_revision_id) || null
      : null
    const selectedPlanDiff = effectiveWorkerPlanRevision && selectedPlanParentRevision
      ? summarizePlanDiff(effectiveWorkerPlanRevision.content || '', selectedPlanParentRevision.content || '')
      : null
    const effectiveGenerateRevisionId = planGenerateRevisionId || selectedPlanRevisionId || selectedTaskPlan?.latest_revision_id || ''
    const workerPlanContent = (effectiveWorkerPlanRevision?.content || selectedTaskPlan?.latest?.content || '').trim()
    const isRefining = !!(selectedTaskPlan?.active_refine_job
      && (selectedTaskPlan.active_refine_job.status === 'queued' || selectedTaskPlan.active_refine_job.status === 'running'))
    const workerOutputDisplay = planRefineStdout || ' '
    const generateOutputDisplay = planGenerateStdout || ' '

    function selectPlanningTask(taskId: string): void {
      setPlanningTaskId(taskId)
      setSelectedTaskId(taskId)
      setPlanActionMessage('')
      setPlanActionError('')
    }

    function openCreatePlanTaskModal(): void {
      setCreateTab('task')
      setNewTaskType('plan')
      setWorkOpen(true)
    }

    return (
      <section className="panel">
        <header className="panel-head">
          <h2>Planning</h2>
          <div className="inline-actions">
            <button className="button button-primary" onClick={openCreatePlanTaskModal}>Create Plan</button>
          </div>
        </header>
        <div className="planning-layout">
          <aside className="planning-task-list">
            <p className="field-label">Select a task</p>
            <div className="list-stack">
              {allTasks.map((task) => (
                <button
                  key={task.id}
                  className={`task-card task-card-button ${planningTaskId === task.id ? 'is-selected' : ''}`}
                  onClick={() => selectPlanningTask(task.id)}
                >
                  <p className="task-title">{task.title}</p>
                  <p className="task-meta">{task.priority} · {humanizeLabel(task.status)} · {humanizeLabel(task.task_type || 'feature')}</p>
                </button>
              ))}
              {allTasks.length === 0 ? <p className="empty">No tasks yet.</p> : null}
            </div>
          </aside>
          <div className="planning-content">
            {planningTask ? (
              <div className="list-stack">
                <p className="task-title">{planningTask.title}</p>
                <p className="task-meta">{planningTask.id} · {humanizeLabel(planningTask.status)}</p>
                <div className="row-card">
                  <p className="task-meta">
                    Latest: {selectedTaskPlan?.latest_revision_id || '-'} · Committed: {selectedTaskPlan?.committed_revision_id || '-'}
                  </p>
                  {selectedTaskPlan?.active_refine_job ? (
                    <p className="task-meta">
                      Refine job: {selectedTaskPlan.active_refine_job.id} · {humanizeLabel(selectedTaskPlan.active_refine_job.status)}
                    </p>
                  ) : (
                    <p className="task-meta">No active refine job.</p>
                  )}
                </div>
                <label className="field-label" htmlFor="planning-revision-selector">Select revision</label>
                <select
                  id="planning-revision-selector"
                  value={selectedPlanRevisionId}
                  onChange={(event) => {
                    setSelectedPlanRevisionId(event.target.value)
                    setPlanGenerateRevisionId(event.target.value)
                  }}
                >
                  <option value="">(latest)</option>
                  {planRevisions.map((revision) => (
                    <option key={revision.id} value={revision.id}>
                      {revision.id} · {humanizeLabel(revision.source)} · {toLocaleTimestamp(revision.created_at) || revision.created_at}
                    </option>
                  ))}
                </select>
                <div className="form-stack">
                  <div className="detail-tabs planning-worker-tabs" role="tablist" aria-label="Worker plan panels">
                    <button
                      className={`detail-tab ${planningWorkerTab === 'plan' ? 'is-active' : ''}`}
                      aria-pressed={planningWorkerTab === 'plan'}
                      onClick={() => openPlanningWorkerTab(planningTask.id, 'plan', workerPlanContent)}
                    >
                      Worker Plan
                    </button>
                    <button
                      className={`detail-tab ${planningWorkerTab === 'manual' ? 'is-active' : ''}`}
                      aria-pressed={planningWorkerTab === 'manual'}
                      onClick={() => openPlanningWorkerTab(planningTask.id, 'manual', workerPlanContent)}
                    >
                      Manual Revision
                    </button>
                  </div>
                  {planningWorkerTab === 'plan' ? (
                    workerPlanContent ? (
                      <div className="preview-box">
                        {effectiveWorkerPlanRevision ? (
                          <p className="task-meta">
                            {humanizeLabel(effectiveWorkerPlanRevision.source)}
                            {effectiveWorkerPlanRevision.step ? ` · ${humanizeLabel(effectiveWorkerPlanRevision.step)}` : ''}
                            {effectiveWorkerPlanRevision.provider ? ` · ${effectiveWorkerPlanRevision.provider}` : ''}
                            {effectiveWorkerPlanRevision.model ? `/${effectiveWorkerPlanRevision.model}` : ''}
                            {' · '}
                            {humanizeLabel(effectiveWorkerPlanRevision.status)}
                            {!selectedPlanRevisionId ? ' · latest' : ''}
                          </p>
                        ) : null}
                        <RenderedMarkdown content={workerPlanContent} className="plan-content-field" />
                        {selectedPlanDiff ? (
                          <p className="task-meta">
                            Compared to parent: +{selectedPlanDiff.added} / -{selectedPlanDiff.removed} lines
                          </p>
                        ) : null}
                      </div>
                    ) : (
                      <p className="empty">No worker plan yet.</p>
                    )
                  ) : (
                    <div className="form-stack">
                      <textarea
                        id="planning-manual-content"
                        className="plan-content-field"
                        rows={20}
                        value={planManualContent}
                        onChange={(event) => setPlanManualContent(event.target.value)}
                        placeholder="Paste or edit full plan text."
                      />
                      <div className="inline-actions">
                        <button
                          className="button"
                          onClick={() => void saveManualPlanRevision(planningTask.id)}
                          disabled={planSavingManual}
                        >
                          {planSavingManual ? 'Saving...' : 'Save Revision'}
                        </button>
                        <button
                          className="button button-primary"
                          onClick={() => void commitPlanRevision(planningTask.id, selectedPlanRevisionId || selectedTaskPlan?.latest_revision_id || '')}
                          disabled={planCommitting}
                        >
                          {planCommitting ? 'Committing...' : 'Commit Selected Revision'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="form-stack">
                  <label className="field-label" htmlFor="planning-refine-feedback">Request changes from worker</label>
                  <div className="planning-refine-inline">
                    <input
                      id="planning-refine-feedback"
                      value={planRefineFeedback}
                      onChange={(event) => setPlanRefineFeedback(event.target.value)}
                      placeholder="Describe what should change in the plan."
                    />
                    <button
                      className="button"
                      onClick={() => void refineTaskPlan(planningTask.id)}
                      disabled={planJobLoading || isRefining || !planRefineFeedback.trim()}
                    >
                      {planJobLoading ? 'Requesting changes...' : isRefining ? 'Requesting changes...' : 'Refine'}
                    </button>
                  </div>
                  <div className="preview-box">
                    <p className="field-label">Worker output{isRefining ? ' (live)' : ''}</p>
                    <pre className="task-log-output plan-content-field planning-worker-output">{workerOutputDisplay}</pre>
                  </div>
                </div>

                <div className="form-stack">
                  <label className="field-label" htmlFor="planning-generate-source">Generate tasks from</label>
                  <select
                    id="planning-generate-source"
                    value={planGenerateSource}
                    onChange={(event) => setPlanGenerateSource(event.target.value as 'committed' | 'revision' | 'override' | 'latest')}
                  >
                    <option value="latest">Latest revision</option>
                    <option value="committed">Committed revision</option>
                    <option value="revision">Selected revision</option>
                    <option value="override">Manual override text</option>
                  </select>
                  {planGenerateSource === 'revision' ? (
                    <select
                      value={effectiveGenerateRevisionId}
                      onChange={(event) => setPlanGenerateRevisionId(event.target.value)}
                      aria-label="Generate from revision"
                    >
                      <option value="">Select revision</option>
                      {planRevisions.map((revision) => (
                        <option key={`gen-${revision.id}`} value={revision.id}>{revision.id}</option>
                      ))}
                    </select>
                  ) : null}
                  {planGenerateSource === 'override' ? (
                    <textarea
                      rows={4}
                      value={planGenerateOverride}
                      onChange={(event) => setPlanGenerateOverride(event.target.value)}
                      placeholder="Provide full plan text override."
                      aria-label="Manual generate override"
                    />
                  ) : null}
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={planGenerateInferDeps}
                      onChange={(event) => setPlanGenerateInferDeps(event.target.checked)}
                    />
                    Infer dependencies between generated tasks
                  </label>
                  <button
                    className="button button-primary"
                    onClick={() => void generateTasksFromPlan(planningTask.id)}
                    disabled={planGenerateLoading}
                  >
                    {planGenerateLoading ? 'Generating...' : 'Generate Tasks'}
                  </button>
                  <div className="preview-box">
                    <p className="field-label">Worker output{planGenerateLoading ? ' (live)' : ''}</p>
                    <pre className="task-log-output plan-content-field planning-worker-output">{generateOutputDisplay}</pre>
                  </div>
                </div>

                <div className="list-stack planning-job-history">
                  <p className="field-label">Refine job history</p>
                  {selectedTaskPlanJobs.map((job) => (
                    <div className="row-card" key={job.id}>
                      <p className="task-meta">
                        {job.id} · {humanizeLabel(job.status)} · base {job.base_revision_id}
                      </p>
                      <p className="task-meta">
                        created: {toLocaleTimestamp(job.created_at) || job.created_at}
                        {job.result_revision_id ? ` · result: ${job.result_revision_id}` : ''}
                      </p>
                      {job.feedback ? <p className="task-desc">{job.feedback}</p> : null}
                      {job.error ? <p className="task-meta">error: {job.error}</p> : null}
                    </div>
                  ))}
                  {selectedTaskPlanJobs.length === 0 ? <p className="empty">No refine jobs yet.</p> : null}
                </div>
                {planActionError ? <p className="error-banner">{planActionError}</p> : null}
                {planActionMessage ? <p className="field-label">{planActionMessage}</p> : null}
              </div>
            ) : (
              <p className="empty">Select a task to view its plan.</p>
            )}
          </div>
        </div>
      </section>
    )
  }

  function renderRoute(): JSX.Element {
    if (route === 'planning') return renderPlanning()
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
          <p className="kicker">agent-led execution</p>
          <h1>Agent Orchestrator</h1>
        </div>
        <div className="topbar-actions">
          <select
            className="topbar-project-select"
            value={projectDir}
            onFocus={() => setTopbarProjectPickerFocused(true)}
            onBlur={() => setTopbarProjectPickerFocused(false)}
            onChange={(event) => {
              setTopbarProjectPickerFocused(false)
              void handleTopbarProjectChange(event.target.value)
            }}
            aria-label="Active repo"
          >
            <option value="">Current workspace</option>
            {projects.map((project) => (
              <option key={`${project.id}-${project.path}`} value={project.path}>
                {(!topbarProjectPickerFocused && project.path === projectDir) ? repoNameFromPath(project.path) : project.path}
              </option>
            ))}
            <option value={ADD_REPO_VALUE}>Add repo...</option>
          </select>
          <button className="button" onClick={() => void reloadAll()} disabled={loading}>Refresh</button>
          <button className="button button-primary" onClick={() => setWorkOpen(true)}>Create Work</button>
        </div>
      </header>

      <div className="nav-mobile-select-wrap">
        <label className="field-label" htmlFor="mobile-route-select">View</label>
        <select
          id="mobile-route-select"
          className="nav-mobile-select"
          value={route}
          onChange={(event) => handleRouteChange(event.target.value as RouteKey)}
          aria-label="Main navigation"
        >
          {ROUTES.map((item) => (
            <option key={`mobile-route-${item.key}`} value={item.key}>{item.label}</option>
          ))}
        </select>
      </div>

      <nav className="nav-strip" aria-label="Main navigation">
        {ROUTES.map((item) => (
          <button
            key={item.key}
            className={`nav-pill ${route === item.key ? 'is-active' : ''}`}
            onClick={() => handleRouteChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main className="main-content">{renderRoute()}</main>

      {selectedTaskId && selectedTaskView && modalExplicitRef.current && !modalDismissedRef.current ? (
        <div className="modal-scrim" role="dialog" aria-modal="true" aria-label="Task detail" onClick={(event) => { if (event.target === event.currentTarget) { modalDismissedRef.current = true; modalExplicitRef.current = false; setSelectedTaskId('') } }} onKeyDown={(event) => { if (event.key === 'Escape') { modalDismissedRef.current = true; modalExplicitRef.current = false; setSelectedTaskId('') } }}>
          <div className="modal-card task-detail-modal">
            <header className="task-detail-modal-head">
              <h2>{selectedTaskView.title}</h2>
              <span className={`status-pill status-pill-prominent ${statusPillClass(taskStatus)}`}>{humanizeLabel(selectedTaskView.status)}</span>
              <button className="button" onClick={() => { modalDismissedRef.current = true; modalExplicitRef.current = false; setSelectedTaskId('') }}>Close</button>
            </header>
            <div className="task-detail-modal-body">
              {taskDetailContent}
            </div>
            <footer className="task-detail-modal-foot">
              {taskActionMessage ? <p className="field-label">{taskActionMessage}</p> : null}
              {taskActionError ? <p className="error-banner">{taskActionError}</p> : null}
              <div className="inline-actions">
                {taskStatus === 'backlog' ? (
                  <>
                    <button className="button button-primary" onClick={() => void transitionTask(selectedTaskView.id, 'queued')} disabled={isTaskActionBusy}>Queue</button>
                    <button className="button button-danger" onClick={() => void transitionTask(selectedTaskView.id, 'cancelled')} disabled={isTaskActionBusy}>Cancel</button>
                  </>
                ) : null}
                {taskStatus === 'queued' ? (
                  <>
                    <button className="button" onClick={() => void transitionTask(selectedTaskView.id, 'backlog')} disabled={isTaskActionBusy}>Move to Backlog</button>
                    <button className="button button-danger" onClick={() => void transitionTask(selectedTaskView.id, 'cancelled')} disabled={isTaskActionBusy}>Cancel</button>
                  </>
                ) : null}
                {taskStatus === 'in_progress' ? (
                  <button className="button button-danger" onClick={() => void transitionTask(selectedTaskView.id, 'cancelled')} disabled={isTaskActionBusy}>Cancel</button>
                ) : null}
                {taskStatus === 'in_review' ? (
                  <>
                    <button className="button button-primary" onClick={() => void transitionTask(selectedTaskView.id, 'done')} disabled={isTaskActionBusy}>Approve</button>
                    <button className="button" onClick={() => void transitionTask(selectedTaskView.id, 'blocked')} disabled={isTaskActionBusy}>Reject</button>
                    <button className="button button-danger" onClick={() => void transitionTask(selectedTaskView.id, 'cancelled')} disabled={isTaskActionBusy}>Cancel</button>
                  </>
                ) : null}
                {taskStatus === 'blocked' ? (
                  <>
                    {(selectedTaskView.retry_count || 0) > 0 ? (
                      <span className="field-label">Retried {selectedTaskView.retry_count} time{(selectedTaskView.retry_count || 0) > 1 ? 's' : ''}</span>
                    ) : null}
                    <button
                      className="button button-primary"
                      onClick={() => void transitionTask(selectedTaskView.id, 'queued')}
                      disabled={isTaskActionBusy || hasUnresolvedBlockers}
                      title={hasUnresolvedBlockers ? `Blocked by: ${unresolvedBlockers.map((id) => describeTask(id, taskIndex).label).join(', ')}` : undefined}
                    >
                      Retry
                    </button>
                    {hasUnresolvedBlockers ? (
                      <span className="field-label">Cannot retry: {unresolvedBlockers.length} unresolved {unresolvedBlockers.length === 1 ? 'dependency' : 'dependencies'} ({unresolvedBlockers.map((id) => describeTask(id, taskIndex).label).join(', ')})</span>
                    ) : null}
                    <button className="button button-danger" onClick={() => void transitionTask(selectedTaskView.id, 'cancelled')} disabled={isTaskActionBusy}>Cancel</button>
                  </>
                ) : null}
                {taskStatus === 'cancelled' ? (
                  <button className="button" onClick={() => void transitionTask(selectedTaskView.id, 'backlog')} disabled={isTaskActionBusy}>Move to Backlog</button>
                ) : null}
                {taskStatus === 'done' ? (
                  <span className="field-label">Task complete</span>
                ) : null}
              </div>
            </footer>
          </div>
        </div>
      ) : null}

      {error ? <p className="error-banner">{error}</p> : null}

      {workOpen ? (
        <div className="modal-scrim" role="dialog" aria-modal="true" aria-label="Create Work modal" onClick={(event) => { if (event.target === event.currentTarget) setWorkOpen(false) }} onKeyDown={(event) => { if (event.key === 'Escape') setWorkOpen(false) }}>
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
                <form id="create-task-form" className="form-stack create-task-form" onSubmit={(event) => void submitTask(event, 'queued')}>
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
                      <label className="field-label" htmlFor="task-hitl-mode">HITL mode</label>
                      <select
                        id="task-hitl-mode"
                        value={newTaskHitlMode}
                        onChange={(event) => setNewTaskHitlMode(event.target.value)}
                      >
                        {collaborationModes.map((mode) => (
                          <option key={mode.mode} value={mode.mode}>
                            {mode.display_name}
                          </option>
                        ))}
                      </select>
                      <label className="field-label" htmlFor="task-labels">Labels (comma-separated)</label>
                      <input
                        id="task-labels"
                        value={newTaskLabels}
                        onChange={(event) => setNewTaskLabels(event.target.value)}
                        placeholder="frontend, urgent"
                      />
                      <label className="field-label" htmlFor="task-blocked-by">Depends on task IDs (comma-separated)</label>
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
                      <label className="field-label" htmlFor="task-worker-model">Worker model override (optional)</label>
                      <input
                        id="task-worker-model"
                        value={newTaskWorkerModel}
                        onChange={(event) => setNewTaskWorkerModel(event.target.value)}
                        placeholder="gpt-5-codex"
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
                <button className="button button-primary" type="submit" form="create-task-form">Create & Queue</button>
                <button className="button" type="button" onClick={(event) => void submitTask(event, 'backlog')}>Add to Backlog</button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {browseOpen ? (
        <div className="modal-scrim" role="dialog" aria-modal="true" aria-label="Browse repositories" onClick={(event) => { if (event.target === event.currentTarget) setBrowseOpen(false) }} onKeyDown={(event) => { if (event.key === 'Escape') setBrowseOpen(false) }}>
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
