export const STORAGE_KEY_TOKEN = 'feature-prd-runner-auth-token'

type QueryValue = string | number | boolean | null | undefined

export function getAuthToken(): string | null {
  return localStorage.getItem(STORAGE_KEY_TOKEN)
}

export function buildAuthHeaders(extra: HeadersInit = {}): HeadersInit {
  const headers: Record<string, string> = {}

  // Normalize extra headers into a plain object
  if (extra instanceof Headers) {
    extra.forEach((value, key) => {
      headers[key] = value
    })
  } else if (Array.isArray(extra)) {
    for (const [key, value] of extra) {
      headers[key] = value
    }
  } else {
    Object.assign(headers, extra)
  }

  const token = getAuthToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  return headers
}

export function buildApiUrl(
  path: string,
  projectDir?: string,
  query: Record<string, QueryValue> = {}
): string {
  const [base, existingQuery = ''] = path.split('?', 2)
  const params = new URLSearchParams(existingQuery)

  if (projectDir) {
    params.set('project_dir', projectDir)
  }

  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue
    params.set(key, String(value))
  }

  const qs = params.toString()
  return qs ? `${base}?${qs}` : base
}

export function buildWsUrl(pathname: string, projectDir?: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = new URL(`${protocol}//${window.location.host}${pathname}`)
  if (projectDir) {
    url.searchParams.set('project_dir', projectDir)
  }
  return url.toString()
}

// --- Feature gap API helpers ---

export async function fetchExplain(taskId: string, projectDir?: string) {
  const res = await fetch(buildApiUrl(`/api/tasks/${taskId}/explain`, projectDir), {
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchInspect(taskId: string, projectDir?: string) {
  const res = await fetch(buildApiUrl(`/api/tasks/${taskId}/inspect`, projectDir), {
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchTrace(taskId: string, projectDir?: string, limit?: number) {
  const res = await fetch(
    buildApiUrl(`/api/tasks/${taskId}/trace`, projectDir, { limit }),
    { headers: buildAuthHeaders() },
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchDryRun(projectDir?: string, prdFile?: string) {
  const res = await fetch(buildApiUrl('/api/dry-run', projectDir, { prd_file: prdFile }), {
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchDoctor(projectDir?: string, checkCodex?: boolean) {
  const res = await fetch(buildApiUrl('/api/doctor', projectDir, { check_codex: checkCodex }), {
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchWorkers(projectDir?: string) {
  const res = await fetch(buildApiUrl('/api/workers', projectDir), {
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function testWorker(workerName: string, projectDir?: string) {
  const res = await fetch(buildApiUrl(`/api/workers/${workerName}/test`, projectDir), {
    method: 'POST',
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function sendCorrection(
  taskId: string,
  correction: { issue: string; file_path?: string; suggested_fix?: string },
  projectDir?: string,
) {
  const res = await fetch(buildApiUrl(`/api/tasks/${taskId}/correct`, projectDir), {
    method: 'POST',
    headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(correction),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function sendRequirement(
  requirement: { requirement: string; task_id?: string; priority?: string },
  projectDir?: string,
) {
  const res = await fetch(buildApiUrl('/api/requirements', projectDir), {
    method: 'POST',
    headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(requirement),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchTaskLogs(
  taskId: string,
  projectDir?: string,
  step?: string,
  lines?: number,
) {
  const res = await fetch(
    buildApiUrl(`/api/tasks/${taskId}/logs`, projectDir, { step, lines }),
    { headers: buildAuthHeaders() },
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export function getMetricsExportUrl(projectDir?: string, format: string = 'csv') {
  return buildApiUrl('/api/metrics/export', projectDir, { format })
}

export async function fetchExecutionOrder(projectDir?: string) {
  const res = await fetch(buildApiUrl('/api/v2/tasks/execution-order', projectDir), {
    headers: buildAuthHeaders(),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

