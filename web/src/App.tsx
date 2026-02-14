import { FormEvent, useEffect, useRef, useState } from 'react'
import { buildApiUrl, buildAuthHeaders } from './api'
import { ImportJobPanel } from './components/AppPanels/ImportJobPanel'
import { QuickActionDetailPanel } from './components/AppPanels/QuickActionDetailPanel'
import { TaskExplorerPanel } from './components/AppPanels/TaskExplorerPanel'
import HITLModeSelector from './components/HITLModeSelector/HITLModeSelector'
import ParallelPlanView from './components/ParallelPlanView'
import { humanizeLabel } from './ui/labels'
import './styles/orchestrator.css'

type RouteKey = 'board' | 'execution' | 'review' | 'agents' | 'settings'
type CreateTab = 'task' | 'import' | 'quick'

type TaskRecord = {
  id: string
  title: string
  description?: string
  task_type?: string
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

type PhaseSnapshot = {
  id: string
  name: string
  description?: string
  status: string
  deps: string[]
  progress: number
}

type PresenceUser = {
  id: string
  name: string
  role: string
  status: string
  activity: string
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

type AgentTypeRecord = {
  role: string
  display_name: string
  description: string
  task_type_affinity: string[]
  allowed_steps: string[]
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

const STORAGE_PROJECT = 'agent-orchestrator-project'
const STORAGE_ROUTE = 'agent-orchestrator-route'
const ADD_REPO_VALUE = '__add_new_repo__'
const WS_RELOAD_CHANNELS = new Set(['tasks', 'queue', 'agents', 'review', 'quick_actions', 'notifications'])

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
    max_review_attempts: 3,
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
      codex: { type: 'codex', command: 'codex' },
    },
  },
  project: {
    commands: {},
  },
}

function routeFromHash(hash: string): RouteKey {
  const cleaned = hash.replace(/^#\/?/, '').trim().toLowerCase()
  const found = ROUTES.find((route) => route.key === cleaned)
  return found?.key ?? 'board'
}

function toHash(route: RouteKey): string {
  return `#/${route}`
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

function parseWorkerProviders(input: string): Record<string, WorkerProviderSettings> {
  if (!input.trim()) {
    return { codex: { type: 'codex', command: 'codex' } }
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
    let type = String(record.type || (name === 'codex' ? 'codex' : name === 'claude' ? 'claude' : 'codex')).trim().toLowerCase()
    if (type === 'local') type = 'ollama'
    if (type !== 'codex' && type !== 'ollama' && type !== 'claude') {
      throw new Error(`Worker provider "${name}" has invalid type "${type}" (allowed: codex, ollama, claude)`)
    }
    if (type === 'codex' || type === 'claude') {
      const defaultCommand = type === 'codex' ? 'codex' : 'claude -p'
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
  if (!out.codex || out.codex.type !== 'codex') {
    out.codex = { type: 'codex', command: 'codex' }
  } else {
    out.codex.command = String(out.codex.command || 'codex').trim() || 'codex'
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
      const defaultCommand = type === 'codex' ? 'codex' : 'claude -p'
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
    providers.codex = { type: 'codex', command: 'codex' }
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

function presenceStatusClass(status: string): string {
  const normalized = status.toLowerCase().replace(/[^a-z0-9_-]+/g, '-')
  return `presence-${normalized || 'unknown'}`
}

function normalizePhases(payload: unknown): PhaseSnapshot[] {
  if (!Array.isArray(payload)) return []
  return payload
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((phase) => {
      const id = String(phase.id || '').trim()
      const name = String(phase.name || id || 'Unnamed phase').trim()
      const description = String(phase.description || '').trim()
      const status = String(phase.status || 'unknown').trim()
      const deps = Array.isArray(phase.deps) ? phase.deps.map((dep) => String(dep || '').trim()).filter(Boolean) : []
      const maybeProgress = Number(phase.progress)
      const progress = Number.isFinite(maybeProgress) ? Math.max(0, Math.min(1, maybeProgress)) : 0
      return {
        id,
        name,
        description,
        status,
        deps,
        progress,
      }
    })
    .filter((phase) => !!phase.id)
}

function normalizePresenceUsers(payload: unknown): PresenceUser[] {
  const usersRaw = Array.isArray(payload)
    ? payload
    : payload && typeof payload === 'object' && !Array.isArray(payload) && Array.isArray((payload as { users?: unknown[] }).users)
      ? (payload as { users: unknown[] }).users
      : []
  return usersRaw
    .filter((user): user is Record<string, unknown> => !!user && typeof user === 'object' && !Array.isArray(user))
    .map((user, index) => {
      const id = String(user.id || user.user_id || user.user || `user-${index + 1}`).trim()
      const name = String(user.name || user.display_name || user.user || id).trim()
      const role = String(user.role || '').trim()
      const status = String(user.status || user.state || 'online').trim()
      const activity = String(user.activity || user.current_task || user.focus || '').trim()
      return {
        id,
        name,
        role,
        status,
        activity,
      }
    })
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

function normalizeAgentTypes(payload: unknown): AgentTypeRecord[] {
  const itemsRaw = Array.isArray(payload)
    ? payload
    : payload && typeof payload === 'object' && !Array.isArray(payload) && Array.isArray((payload as { types?: unknown[] }).types)
      ? (payload as { types: unknown[] }).types
      : []
  return itemsRaw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item))
    .map((item) => {
      const role = String(item.role || '').trim()
      const displayName = String(item.display_name || role || '').trim()
      const description = String(item.description || '').trim()
      const taskTypeAffinity = Array.isArray(item.task_type_affinity)
        ? item.task_type_affinity.map((entry) => String(entry || '').trim()).filter(Boolean)
        : []
      const allowedSteps = Array.isArray(item.allowed_steps)
        ? item.allowed_steps.map((entry) => String(entry || '').trim()).filter(Boolean)
        : []
      return {
        role,
        display_name: displayName || role,
        description,
        task_type_affinity: taskTypeAffinity,
        allowed_steps: allowedSteps,
      }
    })
    .filter((item) => !!item.role)
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
  const [phases, setPhases] = useState<PhaseSnapshot[]>([])
  const [presenceUsers, setPresenceUsers] = useState<PresenceUser[]>([])
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null)
  const [agentTypes, setAgentTypes] = useState<AgentTypeRecord[]>([])
  const [activeProjectId, setActiveProjectId] = useState('')
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
  const [selectedTaskTransition, setSelectedTaskTransition] = useState('ready')
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
  const [collaborationFeedback, setCollaborationFeedback] = useState<CollaborationFeedbackItem[]>([])
  const [collaborationComments, setCollaborationComments] = useState<CollaborationCommentItem[]>([])
  const [collaborationLoading, setCollaborationLoading] = useState(false)
  const [collaborationError, setCollaborationError] = useState('')
  const [feedbackSummary, setFeedbackSummary] = useState('')
  const [feedbackDetails, setFeedbackDetails] = useState('')
  const [feedbackType, setFeedbackType] = useState('general')
  const [feedbackPriority, setFeedbackPriority] = useState('should')
  const [feedbackTargetFile, setFeedbackTargetFile] = useState('')
  const [commentFilePath, setCommentFilePath] = useState('')
  const [commentLineNumber, setCommentLineNumber] = useState('0')
  const [commentBody, setCommentBody] = useState('')

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

  const [settingsLoading, setSettingsLoading] = useState(false)
  const [settingsSaving, setSettingsSaving] = useState(false)
  const [settingsError, setSettingsError] = useState('')
  const [settingsSuccess, setSettingsSuccess] = useState('')
  const [settingsConcurrency, setSettingsConcurrency] = useState(String(DEFAULT_SETTINGS.orchestrator.concurrency))
  const [settingsAutoDeps, setSettingsAutoDeps] = useState(DEFAULT_SETTINGS.orchestrator.auto_deps)
  const [settingsMaxReviewAttempts, setSettingsMaxReviewAttempts] = useState(String(DEFAULT_SETTINGS.orchestrator.max_review_attempts))
  const [settingsDefaultRole, setSettingsDefaultRole] = useState(DEFAULT_SETTINGS.agent_routing.default_role)
  const [settingsTaskTypeRoles, setSettingsTaskTypeRoles] = useState('{}')
  const [settingsRoleProviderOverrides, setSettingsRoleProviderOverrides] = useState('{}')
  const [settingsWorkerDefault, setSettingsWorkerDefault] = useState(DEFAULT_SETTINGS.workers.default)
  const [settingsWorkerDefaultModel, setSettingsWorkerDefaultModel] = useState(DEFAULT_SETTINGS.workers.default_model)
  const [settingsWorkerRouting, setSettingsWorkerRouting] = useState('{}')
  const [settingsWorkerProviders, setSettingsWorkerProviders] = useState(JSON.stringify(DEFAULT_SETTINGS.workers.providers, null, 2))
  const [settingsClaudeProviderName, setSettingsClaudeProviderName] = useState('claude')
  const [settingsClaudeCommand, setSettingsClaudeCommand] = useState('claude -p')
  const [settingsClaudeModel, setSettingsClaudeModel] = useState('')
  const [settingsClaudeEffort, setSettingsClaudeEffort] = useState('')
  const [settingsProjectCommands, setSettingsProjectCommands] = useState(JSON.stringify(DEFAULT_SETTINGS.project.commands, null, 2))
  const [settingsGateCritical, setSettingsGateCritical] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.critical))
  const [settingsGateHigh, setSettingsGateHigh] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.high))
  const [settingsGateMedium, setSettingsGateMedium] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.medium))
  const [settingsGateLow, setSettingsGateLow] = useState(String(DEFAULT_SETTINGS.defaults.quality_gate.low))

  const selectedTaskIdRef = useRef(selectedTaskId)
  const selectedQuickActionIdRef = useRef(selectedQuickActionId)
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
    selectedQuickActionIdRef.current = selectedQuickActionId
  }, [selectedQuickActionId])

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

  function applySettings(payload: SystemSettings): void {
    setSettingsConcurrency(String(payload.orchestrator.concurrency))
    setSettingsAutoDeps(payload.orchestrator.auto_deps)
    setSettingsMaxReviewAttempts(String(payload.orchestrator.max_review_attempts))
    setSettingsDefaultRole(payload.agent_routing.default_role || 'general')
    setSettingsTaskTypeRoles(JSON.stringify(payload.agent_routing.task_type_roles || {}, null, 2))
    setSettingsRoleProviderOverrides(JSON.stringify(payload.agent_routing.role_provider_overrides || {}, null, 2))
    setSettingsWorkerDefault(payload.workers.default || 'codex')
    setSettingsWorkerDefaultModel(payload.workers.default_model || '')
    setSettingsWorkerRouting(JSON.stringify(payload.workers.routing || {}, null, 2))
    setSettingsWorkerProviders(JSON.stringify(payload.workers.providers || {}, null, 2))
    const entries = Object.entries(payload.workers.providers || {})
    const claudeEntry = entries.find(([name, provider]) => name === 'claude' && provider?.type === 'claude')
      || entries.find(([, provider]) => provider?.type === 'claude')
    if (claudeEntry) {
      const [name, provider] = claudeEntry
      setSettingsClaudeProviderName(name)
      setSettingsClaudeCommand(String(provider.command || 'claude -p'))
      setSettingsClaudeModel(String(provider.model || ''))
      setSettingsClaudeEffort(String(provider.reasoning_effort || ''))
    } else {
      setSettingsClaudeProviderName('claude')
      setSettingsClaudeCommand('claude -p')
      setSettingsClaudeModel('')
      setSettingsClaudeEffort('')
    }
    setSettingsProjectCommands(JSON.stringify(payload.project.commands || {}, null, 2))
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
    if (route !== 'settings') return
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

  useEffect(() => {
    try {
      const providers = parseWorkerProviders(settingsWorkerProviders)
      const entries = Object.entries(providers)
      const claudeEntry = entries.find(([name, provider]) => name === 'claude' && provider.type === 'claude')
        || entries.find(([, provider]) => provider.type === 'claude')
      if (!claudeEntry) return
      const [name, provider] = claudeEntry
      setSettingsClaudeProviderName(name)
      setSettingsClaudeCommand(String(provider.command || 'claude -p'))
      setSettingsClaudeModel(String(provider.model || ''))
      setSettingsClaudeEffort(String(provider.reasoning_effort || ''))
    } catch {
      // Ignore JSON parse errors while user is editing.
    }
  }, [settingsWorkerProviders])

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
    } finally {
      if (requestSeq === taskDetailRequestSeqRef.current) {
        setSelectedTaskDetailLoading(false)
      }
    }
  }

  useEffect(() => {
    if (!selectedTaskId) return
    void loadTaskDetail(selectedTaskId)
  }, [selectedTaskId, projectDir])

  async function loadCollaboration(taskId: string): Promise<void> {
    if (!taskId) {
      setCollaborationTimeline([])
      setCollaborationFeedback([])
      setCollaborationComments([])
      setCollaborationError('')
      return
    }
    const requestSeq = collaborationRequestSeqRef.current + 1
    collaborationRequestSeqRef.current = requestSeq
    setCollaborationLoading(true)
    setCollaborationError('')
    try {
      const [timelinePayload, feedbackPayload, commentsPayload] = await Promise.all([
        requestJson<unknown>(buildApiUrl(`/api/collaboration/timeline/${taskId}`, projectDir)),
        requestJson<unknown>(buildApiUrl(`/api/collaboration/feedback/${taskId}`, projectDir)),
        requestJson<unknown>(buildApiUrl(`/api/collaboration/comments/${taskId}`, projectDir)),
      ])
      if (requestSeq !== collaborationRequestSeqRef.current || selectedTaskIdRef.current !== taskId) {
        return
      }
      setCollaborationTimeline(normalizeTimelineEvents(timelinePayload))
      setCollaborationFeedback(normalizeFeedbackItems(feedbackPayload))
      setCollaborationComments(normalizeComments(commentsPayload))
    } catch (err) {
      if (requestSeq !== collaborationRequestSeqRef.current || selectedTaskIdRef.current !== taskId) {
        return
      }
      setCollaborationTimeline([])
      setCollaborationFeedback([])
      setCollaborationComments([])
      const detail = err instanceof Error ? err.message : 'unknown error'
      setCollaborationError(`Failed to load collaboration context (${detail})`)
    } finally {
      if (requestSeq === collaborationRequestSeqRef.current) {
        setCollaborationLoading(false)
      }
    }
  }

  useEffect(() => {
    if (!selectedTaskId) {
      setCollaborationTimeline([])
      setCollaborationFeedback([])
      setCollaborationComments([])
      setCollaborationError('')
      return
    }
    void loadCollaboration(selectedTaskId)
  }, [selectedTaskId, projectDir])

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
      const payload = await requestJson<{ quick_action: QuickActionRecord }>(buildApiUrl(`/api/quick-actions/${quickActionId}`, projectDir))
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

  function toErrorMessage(prefix: string, err: unknown): string {
    const detail = err instanceof Error ? err.message : 'unknown error'
    return `${prefix} (${detail})`
  }

  async function refreshTasksSurface(): Promise<void> {
    const refreshProjectDir = projectDirRef.current
    try {
      const [boardData, orchestratorData, reviewData, executionOrderData, phasesData, metricsData] = await Promise.all([
        requestJson<BoardResponse>(buildApiUrl('/api/tasks/board', refreshProjectDir)),
        requestJson<OrchestratorStatus>(buildApiUrl('/api/orchestrator/status', refreshProjectDir)),
        requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/review-queue', refreshProjectDir)),
        requestJson<{ batches: string[][] }>(buildApiUrl('/api/tasks/execution-order', refreshProjectDir)),
        requestJson<unknown>(buildApiUrl('/api/phases', refreshProjectDir)).catch(() => []),
        requestJson<unknown>(buildApiUrl('/api/metrics', refreshProjectDir)).catch(() => null),
      ])
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setBoard(boardData)
      setOrchestrator(orchestratorData)
      setReviewQueue(reviewData.tasks || [])
      setExecutionBatches(executionOrderData.batches || [])
      setPhases(normalizePhases(phasesData))
      setMetrics(normalizeMetrics(metricsData))

      const selectedTask = String(selectedTaskIdRef.current || '').trim()
      if (selectedTask) {
        void loadTaskDetail(selectedTask)
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
      const [agentData, presenceData, agentTypesData] = await Promise.all([
        requestJson<{ agents: AgentRecord[] }>(buildApiUrl('/api/agents', refreshProjectDir)),
        requestJson<unknown>(buildApiUrl('/api/collaboration/presence', refreshProjectDir)).catch(() => ({ users: [] })),
        requestJson<unknown>(buildApiUrl('/api/agents/types', refreshProjectDir)).catch(() => ({ types: [] })),
      ])
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setAgents(agentData.agents || [])
      setPresenceUsers(normalizePresenceUsers(presenceData))
      setAgentTypes(normalizeAgentTypes(agentTypesData))
    } catch (err) {
      if (refreshProjectDir !== projectDirRef.current) {
        return
      }
      setError(toErrorMessage('Failed to refresh agents surface', err))
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
        phasesData,
        presenceData,
        metricsData,
        agentTypesData,
      ] = await Promise.all([
        requestJson<BoardResponse>(buildApiUrl('/api/tasks/board', projectDir)),
        requestJson<OrchestratorStatus>(buildApiUrl('/api/orchestrator/status', projectDir)),
        requestJson<{ tasks: TaskRecord[] }>(buildApiUrl('/api/review-queue', projectDir)),
        requestJson<{ agents: AgentRecord[] }>(buildApiUrl('/api/agents', projectDir)),
        requestJson<{ projects: ProjectRef[] }>(buildApiUrl('/api/projects', projectDir)),
        requestJson<{ items: PinnedProjectRef[] }>(buildApiUrl('/api/projects/pinned', projectDir)),
        requestJson<{ quick_actions: QuickActionRecord[] }>(buildApiUrl('/api/quick-actions', projectDir)),
        requestJson<{ batches: string[][] }>(buildApiUrl('/api/tasks/execution-order', projectDir)),
        requestJson<unknown>(buildApiUrl('/api/phases', projectDir)).catch(() => []),
        requestJson<unknown>(buildApiUrl('/api/collaboration/presence', projectDir)).catch(() => ({ users: [] })),
        requestJson<unknown>(buildApiUrl('/api/metrics', projectDir)).catch(() => null),
        requestJson<unknown>(buildApiUrl('/api/agents/types', projectDir)).catch(() => ({ types: [] })),
      ])
      if (requestSeq !== reloadAllSeqRef.current) {
        return
      }
      setBoard(boardData)
      setOrchestrator(orchestratorData)
      setReviewQueue(reviewData.tasks)
      setAgents(agentData.agents)
      setProjects(projectData.projects)
      setPinnedProjects(pinnedData.items || [])
      setQuickActions(quickActionData.quick_actions || [])
      setExecutionBatches(executionOrderData.batches || [])
      setPhases(normalizePhases(phasesData))
      setPresenceUsers(normalizePresenceUsers(presenceData))
      setMetrics(normalizeMetrics(metricsData))
      setAgentTypes(normalizeAgentTypes(agentTypesData))
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
    const preview = await requestJson<{ job_id: string; preview: PrdPreview }>(buildApiUrl('/api/import/prd/preview', projectDir), {
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
  }

  async function submitQuickAction(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (!quickPrompt.trim()) return
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
  }

  async function taskAction(taskId: string, action: 'run' | 'retry' | 'cancel'): Promise<void> {
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/${action}`, projectDir), {
      method: 'POST',
    })
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
    }
  }

  async function transitionTask(taskId: string): Promise<void> {
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/transition`, projectDir), {
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
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/dependencies`, projectDir), {
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
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/dependencies/${depId}`, projectDir), {
      method: 'DELETE',
    })
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
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
    await requestJson<{ task: TaskRecord }>(buildApiUrl(`/api/tasks/${taskId}/approve-gate`, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gate: gate || undefined }),
    })
    await reloadAll()
    if (selectedTaskId === taskId) {
      await loadTaskDetail(taskId)
    }
  }

  async function submitFeedback(taskId: string): Promise<void> {
    if (!feedbackSummary.trim()) return
    setCollaborationError('')
    try {
      await requestJson<{ feedback: CollaborationFeedbackItem }>(buildApiUrl('/api/collaboration/feedback', projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: taskId,
          feedback_type: feedbackType,
          priority: feedbackPriority,
          summary: feedbackSummary.trim(),
          details: feedbackDetails.trim(),
          target_file: feedbackTargetFile.trim() || undefined,
        }),
      })
      setFeedbackSummary('')
      setFeedbackDetails('')
      setFeedbackTargetFile('')
      await loadCollaboration(taskId)
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setCollaborationError(`Failed to add feedback (${detail})`)
    }
  }

  async function dismissFeedback(taskId: string, feedbackId: string): Promise<void> {
    setCollaborationError('')
    try {
      await requestJson<{ feedback: CollaborationFeedbackItem }>(buildApiUrl(`/api/collaboration/feedback/${feedbackId}/dismiss`, projectDir), {
        method: 'POST',
      })
      await loadCollaboration(taskId)
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setCollaborationError(`Failed to dismiss feedback (${detail})`)
    }
  }

  async function submitComment(taskId: string): Promise<void> {
    if (!commentFilePath.trim() || !commentBody.trim()) return
    setCollaborationError('')
    try {
      await requestJson<{ comment: CollaborationCommentItem }>(buildApiUrl('/api/collaboration/comments', projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: taskId,
          file_path: commentFilePath.trim(),
          line_number: Math.max(0, parseNonNegativeInt(commentLineNumber, 0)),
          body: commentBody.trim(),
        }),
      })
      setCommentBody('')
      await loadCollaboration(taskId)
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setCollaborationError(`Failed to add comment (${detail})`)
    }
  }

  async function resolveComment(taskId: string, commentId: string): Promise<void> {
    setCollaborationError('')
    try {
      await requestJson<{ comment: CollaborationCommentItem }>(buildApiUrl(`/api/collaboration/comments/${commentId}/resolve`, projectDir), {
        method: 'POST',
      })
      await loadCollaboration(taskId)
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setCollaborationError(`Failed to resolve comment (${detail})`)
    }
  }

  async function saveTaskEdits(taskId: string): Promise<void> {
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
  }

  async function saveSettings(event: FormEvent): Promise<void> {
    event.preventDefault()
    setSettingsSaving(true)
    setSettingsError('')
    setSettingsSuccess('')
    try {
      const taskTypeRoles = parseStringMap(settingsTaskTypeRoles, 'Task type role map')
      const roleProviderOverrides = parseStringMap(settingsRoleProviderOverrides, 'Role provider overrides')
      const workerRouting = parseStringMap(settingsWorkerRouting, 'Worker routing map')
      const workerProviders = parseWorkerProviders(settingsWorkerProviders)
      const claudeProviderName = settingsClaudeProviderName.trim()
      if (claudeProviderName) {
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
        workerProviders[claudeProviderName] = claudeProvider
      }
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
          default: settingsWorkerDefault.trim() || 'codex',
          default_model: settingsWorkerDefaultModel.trim(),
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

  async function promoteQuickAction(quickActionId: string): Promise<void> {
    await requestJson<{ task: TaskRecord; already_promoted: boolean }>(buildApiUrl(`/api/quick-actions/${quickActionId}/promote`, projectDir), {
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
    await requestJson<OrchestratorStatus>(buildApiUrl('/api/orchestrator/control', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    })
    await reloadAll()
  }

  async function reviewAction(taskId: string, action: 'approve' | 'request-changes'): Promise<void> {
    const endpoint = action === 'approve' ? `/api/review/${taskId}/approve` : `/api/review/${taskId}/request-changes`
    await requestJson<{ task: TaskRecord }>(buildApiUrl(endpoint, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ guidance: reviewGuidance.trim() || undefined }),
    })
    setReviewGuidance('')
    await reloadAll()
  }

  async function spawnAgent(): Promise<void> {
    await requestJson<{ agent: AgentRecord }>(buildApiUrl('/api/agents/spawn', projectDir), {
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
    await requestJson<{ agent: AgentRecord }>(buildApiUrl(`/api/agents/${agentId}/${action}`, projectDir), { method: 'POST' })
    await reloadAll()
  }

  async function pinProjectPath(path: string, allowNonGitValue: boolean): Promise<void> {
    const pinned = await requestJson<{ project: ProjectRef }>(buildApiUrl('/api/projects/pinned', projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, allow_non_git: allowNonGitValue }),
    })
    setProjectDir(pinned.project.path)
    await reloadAll()
  }

  async function unpinProject(projectId: string): Promise<void> {
    await requestJson<{ removed: boolean }>(buildApiUrl(`/api/projects/pinned/${projectId}`, projectDir), {
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

  function renderBoard(): JSX.Element {
    const columns = ['backlog', 'ready', 'in_progress', 'in_review', 'blocked', 'done']
    const allTasks = columns.flatMap((column) => board.columns[column] || [])
    const taskIndex = new Map<string, TaskRecord>()
    for (const task of allTasks) {
      taskIndex.set(task.id, task)
    }
    if (selectedTaskDetail) {
      taskIndex.set(selectedTaskDetail.id, selectedTaskDetail)
    }
    const selectedTask = allTasks.find((task) => task.id === selectedTaskId) || allTasks[0]
    const selectedTaskView = selectedTaskDetail && selectedTask && selectedTaskDetail.id === selectedTask.id ? selectedTaskDetail : selectedTask
    const blockerIds = selectedTaskView?.blocked_by || []
    const blockedIds = selectedTaskView?.blocks || []
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
                <div className="dependency-graph-panel">
                  <p className="field-label">Dependency graph</p>
                  <div className="dependency-graph-grid">
                    <div className="dependency-graph-column">
                      <p className="field-label">Blocked by</p>
                      {blockerIds.length > 0 ? (
                        blockerIds.map((depId) => {
                          const dep = describeTask(depId, taskIndex)
                          return (
                            <div className="dependency-node dependency-node-blocker" key={`blocker-${depId}`}>
                              <p className="dependency-node-title">{dep.label}</p>
                              <p className="dependency-node-meta">{humanizeLabel(dep.status)} {'->'} depends on</p>
                            </div>
                          )
                        })
                      ) : (
                        <p className="empty">No blockers</p>
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
                            <div className="dependency-node dependency-node-dependent" key={`dependent-${depId}`}>
                              <p className="dependency-node-title">{dep.label}</p>
                              <p className="dependency-node-meta">blocked until done · {humanizeLabel(dep.status)}</p>
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
                  <label className="field-label">HITL mode</label>
                  <HITLModeSelector
                    currentMode={editTaskHitlMode}
                    onModeChange={setEditTaskHitlMode}
                    projectDir={projectDir}
                  />
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
                <div className="list-stack">
                  <p className="field-label">Collaboration timeline</p>
                  {collaborationLoading ? <p className="field-label">Loading collaboration activity...</p> : null}
                  {collaborationTimeline.slice(0, 8).map((event) => (
                    <div className="row-card" key={event.id}>
                      <div>
                        <p className="task-title">{event.summary || humanizeLabel(event.type)}</p>
                        <p className="task-meta">{humanizeLabel(event.type)} · {event.actor} · {toLocaleTimestamp(event.timestamp) || '-'}</p>
                      </div>
                      {event.details ? <p className="task-desc">{event.details}</p> : null}
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
                  {collaborationTimeline.length === 0 && !collaborationLoading ? <p className="empty">No collaboration events for this task yet.</p> : null}
                </div>
                <div className="list-stack">
                  <p className="field-label">Feedback</p>
                  <div className="form-stack">
                    <label className="field-label" htmlFor="feedback-summary">Summary</label>
                    <input
                      id="feedback-summary"
                      value={feedbackSummary}
                      onChange={(event) => setFeedbackSummary(event.target.value)}
                      placeholder="What should change?"
                    />
                    <div className="inline-actions">
                      <select value={feedbackType} onChange={(event) => setFeedbackType(event.target.value)} aria-label="Feedback type">
                        <option value="general">General</option>
                        <option value="bug">Bug</option>
                        <option value="nit">Nit</option>
                        <option value="question">Question</option>
                      </select>
                      <select value={feedbackPriority} onChange={(event) => setFeedbackPriority(event.target.value)} aria-label="Feedback priority">
                        <option value="must">Must</option>
                        <option value="should">Should</option>
                        <option value="could">Could</option>
                      </select>
                    </div>
                    <label className="field-label" htmlFor="feedback-details">Details</label>
                    <textarea
                      id="feedback-details"
                      rows={3}
                      value={feedbackDetails}
                      onChange={(event) => setFeedbackDetails(event.target.value)}
                    />
                    <label className="field-label" htmlFor="feedback-file">Target file (optional)</label>
                    <input
                      id="feedback-file"
                      value={feedbackTargetFile}
                      onChange={(event) => setFeedbackTargetFile(event.target.value)}
                      placeholder="src/path/file.ts"
                    />
                    <button className="button" onClick={() => void submitFeedback(selectedTaskView.id)}>
                      Add feedback
                    </button>
                  </div>
                  {collaborationFeedback.map((item) => (
                    <div className="row-card" key={item.id}>
                      <div>
                        <p className="task-title">{item.summary}</p>
                        <p className="task-meta">
                          {humanizeLabel(item.feedback_type)} · {humanizeLabel(item.priority)} · {humanizeLabel(item.status)}
                        </p>
                        {item.details ? <p className="task-desc">{item.details}</p> : null}
                        {item.target_file ? <p className="task-meta">file: {item.target_file}</p> : null}
                      </div>
                      {item.status !== 'addressed' ? (
                        <button className="button" onClick={() => void dismissFeedback(selectedTaskView.id, item.id)}>
                          Dismiss
                        </button>
                      ) : null}
                    </div>
                  ))}
                  {collaborationFeedback.length === 0 ? <p className="empty">No feedback yet.</p> : null}
                </div>
                <div className="list-stack">
                  <p className="field-label">Comments</p>
                  <div className="form-stack">
                    <label className="field-label" htmlFor="comment-file-path">File path</label>
                    <input
                      id="comment-file-path"
                      value={commentFilePath}
                      onChange={(event) => setCommentFilePath(event.target.value)}
                      placeholder="src/path/file.ts"
                    />
                    <label className="field-label" htmlFor="comment-line-number">Line number</label>
                    <input
                      id="comment-line-number"
                      value={commentLineNumber}
                      onChange={(event) => setCommentLineNumber(event.target.value)}
                      inputMode="numeric"
                    />
                    <label className="field-label" htmlFor="comment-body">Comment</label>
                    <textarea
                      id="comment-body"
                      rows={3}
                      value={commentBody}
                      onChange={(event) => setCommentBody(event.target.value)}
                    />
                    <button className="button" onClick={() => void submitComment(selectedTaskView.id)}>
                      Add comment
                    </button>
                  </div>
                  {collaborationComments.map((comment) => (
                    <div className="row-card" key={comment.id}>
                      <div>
                        <p className="task-title">{comment.file_path}:{comment.line_number}</p>
                        <p className="task-meta">{comment.author || 'human'} · {toLocaleTimestamp(comment.created_at) || '-'}</p>
                        <p className="task-desc">{comment.body}</p>
                        {comment.resolved ? <p className="task-meta">Resolved</p> : null}
                      </div>
                      {!comment.resolved ? (
                        <button className="button" onClick={() => void resolveComment(selectedTaskView.id, comment.id)}>
                          Resolve
                        </button>
                      ) : null}
                    </div>
                  ))}
                  {collaborationComments.length === 0 ? <p className="empty">No comments yet.</p> : null}
                </div>
                {collaborationError ? <p className="error-banner">{collaborationError}</p> : null}
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
          <p className="field-label">Runtime metrics</p>
          <div className="row-card">
            <p className="task-meta">
              API calls: {metrics?.api_calls ?? 0} ·
              wall time: {metrics?.wall_time_seconds ?? 0}s ·
              phases: {metrics?.phases_completed ?? 0}/{metrics?.phases_total ?? 0}
            </p>
            <p className="task-meta">
              tokens: {metrics?.tokens_used ?? 0} ·
              est cost: ${(metrics?.estimated_cost_usd ?? 0).toFixed(2)} ·
              files changed: {metrics?.files_changed ?? 0}
            </p>
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
        <ParallelPlanView projectDir={projectDir} />
        <div className="list-stack phase-list">
          <p className="field-label">Phase timeline</p>
          {phases.map((phase) => {
            const progressPercent = Math.round((phase.progress || 0) * 100)
            return (
              <div className="phase-card" key={phase.id}>
                <div className="phase-head">
                  <p className="task-title">{phase.name}</p>
                  <p className="task-meta">{humanizeLabel(phase.status)}</p>
                </div>
                {phase.description ? <p className="task-desc">{phase.description}</p> : null}
                <div className="phase-progress-track">
                  <span className="phase-progress-fill" style={{ width: `${progressPercent}%` }} />
                </div>
                <p className="task-meta">
                  {progressPercent}% complete · {phase.deps.length > 0 ? `deps: ${phase.deps.join(', ')}` : 'no blockers'}
                </p>
              </div>
            )
          })}
          {phases.length === 0 ? <p className="empty">No phases available.</p> : null}
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
          <p className="field-label">Collaboration presence</p>
          {presenceUsers.length > 0 ? (
            <div className="presence-grid">
              {presenceUsers.map((user) => (
                <div className="presence-card" key={user.id}>
                  <div className="presence-head">
                    <span className={`presence-dot ${presenceStatusClass(user.status)}`} />
                    <p className="task-title">{user.name}</p>
                  </div>
                  <p className="task-meta">
                    {user.role ? `${humanizeLabel(user.role)} · ` : ''}{humanizeLabel(user.status)}
                  </p>
                  {user.activity ? <p className="task-desc">{user.activity}</p> : null}
                </div>
              ))}
            </div>
          ) : (
            <p className="empty">No active collaborators.</p>
          )}
          <p className="field-label">Agent type catalog</p>
          {agentTypes.length > 0 ? (
            <div className="presence-grid">
              {agentTypes.map((agentType) => (
                <div className="presence-card" key={agentType.role}>
                  <p className="task-title">{agentType.display_name}</p>
                  <p className="task-meta">{agentType.role}</p>
                  {agentType.description ? <p className="task-desc">{agentType.description}</p> : null}
                  <p className="task-meta">
                    affinity: {agentType.task_type_affinity.length > 0 ? agentType.task_type_affinity.join(', ') : 'none'}
                  </p>
                  <p className="task-meta">
                    steps: {agentType.allowed_steps.length > 0 ? agentType.allowed_steps.join(', ') : 'n/a'}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty">No agent types returned.</p>
          )}
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

          <article className="settings-card">
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
              <label className="field-label" htmlFor="settings-default-role">Default role</label>
              <input
                id="settings-default-role"
                value={settingsDefaultRole}
                onChange={(event) => setSettingsDefaultRole(event.target.value)}
                placeholder="general"
              />
              <label className="field-label" htmlFor="settings-task-type-roles">Task type role map (JSON object)</label>
              <textarea
                id="settings-task-type-roles"
                rows={4}
                value={settingsTaskTypeRoles}
                onChange={(event) => setSettingsTaskTypeRoles(event.target.value)}
                placeholder='{"bug":"debugger","docs":"researcher"}'
              />
              <label className="field-label" htmlFor="settings-role-overrides">Role provider overrides (JSON object)</label>
              <textarea
                id="settings-role-overrides"
                rows={4}
                value={settingsRoleProviderOverrides}
                onChange={(event) => setSettingsRoleProviderOverrides(event.target.value)}
                placeholder='{"reviewer":"openai"}'
              />
              <p className="field-label">Worker provider catalog</p>
              <label className="field-label" htmlFor="settings-worker-default">Default worker provider</label>
              <input
                id="settings-worker-default"
                value={settingsWorkerDefault}
                onChange={(event) => setSettingsWorkerDefault(event.target.value)}
                placeholder="codex"
              />
              <label className="field-label" htmlFor="settings-worker-default-model">Default worker model (optional)</label>
              <input
                id="settings-worker-default-model"
                value={settingsWorkerDefaultModel}
                onChange={(event) => setSettingsWorkerDefaultModel(event.target.value)}
                placeholder="gpt-5-codex"
              />
              <p className="field-label">Claude provider quick edit</p>
              <label className="field-label" htmlFor="settings-claude-provider-name">Claude provider name</label>
              <input
                id="settings-claude-provider-name"
                value={settingsClaudeProviderName}
                onChange={(event) => setSettingsClaudeProviderName(event.target.value)}
                placeholder="claude"
              />
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
              <label className="field-label" htmlFor="settings-worker-routing">Worker routing map (JSON object: step {'->'} provider)</label>
              <textarea
                id="settings-worker-routing"
                rows={4}
                value={settingsWorkerRouting}
                onChange={(event) => setSettingsWorkerRouting(event.target.value)}
                placeholder='{"plan":"codex","implement":"ollama-dev","review":"codex"}'
              />
              <label
                className="field-label"
                htmlFor="settings-worker-providers"
                title="Configure reasoning effort in your CLI setup first (Codex/Claude profile or config). Agent Orchestrator only passes flags supported by your installed CLI version."
              >
                Worker providers (JSON object)
              </label>
              <textarea
                id="settings-worker-providers"
                rows={8}
                value={settingsWorkerProviders}
                onChange={(event) => setSettingsWorkerProviders(event.target.value)}
                placeholder='{"codex":{"type":"codex","command":"codex"},"ollama-dev":{"type":"ollama","endpoint":"http://localhost:11434","model":"llama3.1:8b"}}'
              />
              <label className="field-label" htmlFor="settings-project-commands">Project commands by language (JSON object)</label>
              <textarea
                id="settings-project-commands"
                rows={8}
                value={settingsProjectCommands}
                onChange={(event) => setSettingsProjectCommands(event.target.value)}
                placeholder='{"python":{"test":".venv/bin/pytest -n auto","lint":".venv/bin/ruff check .","typecheck":".venv/bin/mypy . --strict","format":".venv/bin/ruff format ."}}'
              />
              <p className="field-label">Default quality gate thresholds</p>
              <div className="inline-actions">
                <input
                  aria-label="Quality gate critical"
                  value={settingsGateCritical}
                  onChange={(event) => setSettingsGateCritical(event.target.value)}
                  inputMode="numeric"
                  placeholder="critical"
                />
                <input
                  aria-label="Quality gate high"
                  value={settingsGateHigh}
                  onChange={(event) => setSettingsGateHigh(event.target.value)}
                  inputMode="numeric"
                  placeholder="high"
                />
                <input
                  aria-label="Quality gate medium"
                  value={settingsGateMedium}
                  onChange={(event) => setSettingsGateMedium(event.target.value)}
                  inputMode="numeric"
                  placeholder="medium"
                />
                <input
                  aria-label="Quality gate low"
                  value={settingsGateLow}
                  onChange={(event) => setSettingsGateLow(event.target.value)}
                  inputMode="numeric"
                  placeholder="low"
                />
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
          <h1>Agent Orchestrator</h1>
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
