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
