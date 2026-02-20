import { useEffect, useMemo, useRef, useState } from 'react'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'
import 'xterm/css/xterm.css'

type TerminalSessionRecord = {
  id: string
  project_id?: string
  status: string
  shell: string
  cwd: string
  started_at?: string | null
  finished_at?: string | null
  exit_code?: number | null
  cols?: number
  rows?: number
}

type TerminalPanelProps = {
  projectDir: string
}

function buildApiUrl(path: string, projectDir?: string): string {
  const trimmed = (projectDir || '').trim()
  if (!trimmed) return path
  const joiner = path.includes('?') ? '&' : '?'
  return `${path}${joiner}project_dir=${encodeURIComponent(trimmed)}`
}

export function TerminalPanel({ projectDir }: TerminalPanelProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null)
  const terminalRef = useRef<Terminal | null>(null)
  const fitRef = useRef<FitAddon | null>(null)
  const sessionRef = useRef<TerminalSessionRecord | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const logsOffsetRef = useRef<number>(0)
  const resizeTimerRef = useRef<number | null>(null)
  const [session, setSession] = useState<TerminalSessionRecord | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const wsUrl = useMemo(() => {
    return `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
  }, [])

  useEffect(() => {
    sessionRef.current = session
  }, [session])

  useEffect(() => {
    const term = new Terminal({
      cursorBlink: true,
      fontFamily: 'IBM Plex Mono, monospace',
      fontSize: 13,
      convertEol: false,
      scrollback: 3000,
      allowProposedApi: false,
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    terminalRef.current = term
    fitRef.current = fit
    if (containerRef.current) {
      term.open(containerRef.current)
      fit.fit()
    }

    const disposeData = term.onData((data) => {
      const current = sessionRef.current
      if (!current) return
      void fetch(buildApiUrl(`/api/terminal/session/${current.id}/input`, projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data }),
      }).catch(() => {
        // ignore transient failures; status poll will surface hard errors
      })
    })

    return () => {
      disposeData.dispose()
      term.dispose()
      terminalRef.current = null
      fitRef.current = null
    }
  }, [projectDir])

  function setSessionState(next: TerminalSessionRecord | null): void {
    sessionRef.current = next
    setSession(next)
    const projectId = String(next?.project_id || '').trim()
    const socket = wsRef.current
    if (socket && socket.readyState === WebSocket.OPEN && projectId) {
      socket.send(JSON.stringify({ action: 'subscribe', channels: ['terminal'], project_id: projectId }))
    }
  }

  async function readBackfill(sessionId: string): Promise<void> {
    const response = await fetch(
      buildApiUrl(`/api/terminal/session/${sessionId}/logs?offset=${logsOffsetRef.current}&max_bytes=262144`, projectDir),
    )
    if (!response.ok) return
    const payload = await response.json() as { output?: string; offset?: number }
    const output = String(payload.output || '')
    if (output) {
      terminalRef.current?.write(output)
    }
    logsOffsetRef.current = Number(payload.offset || logsOffsetRef.current || 0)
  }

  async function refreshActive(): Promise<void> {
    const response = await fetch(buildApiUrl('/api/terminal/session', projectDir))
    if (!response.ok) {
      setSessionState(null)
      return
    }
    const payload = await response.json() as { session?: TerminalSessionRecord | null }
    const found = payload.session || null
    setSessionState(found)
    if (found) {
      await readBackfill(found.id)
    }
  }

  async function startOrAttach(): Promise<void> {
    setLoading(true)
    setError('')
    try {
      fitRef.current?.fit()
      const cols = terminalRef.current?.cols || 120
      const rows = terminalRef.current?.rows || 36
      const response = await fetch(buildApiUrl('/api/terminal/session', projectDir), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cols, rows }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `HTTP ${response.status}`)
      }
      const payload = await response.json() as { session: TerminalSessionRecord }
      const next = payload.session
      setSessionState(next)
      terminalRef.current?.clear()
      logsOffsetRef.current = 0
      await readBackfill(next.id)
      if (next.status === 'running') {
        terminalRef.current?.focus()
      }
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'unknown error'
      setError(`Failed to start terminal session (${detail})`)
    } finally {
      setLoading(false)
    }
  }

  async function stopSession(): Promise<void> {
    const current = sessionRef.current
    if (!current) return
    await fetch(buildApiUrl(`/api/terminal/session/${current.id}/stop`, projectDir), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal: 'TERM' }),
    }).catch(() => undefined)
    await refreshActive()
  }

  useEffect(() => {
    let stopped = false
    let reconnectTimer: number | null = null
    let reconnectAttempts = 0

    const scheduleReconnect = (): void => {
      if (stopped || reconnectTimer !== null) return
      const attempt = Math.min(reconnectAttempts, 6)
      const delay = Math.min(30000, 1000 * (2 ** attempt))
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        reconnectAttempts += 1
        connect()
      }, delay)
    }

    const connect = (): void => {
      if (stopped) return
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws
      ws.addEventListener('open', () => {
        reconnectAttempts = 0
        const projectId = String(sessionRef.current?.project_id || '').trim()
        ws.send(JSON.stringify({ action: 'subscribe', channels: ['terminal'], project_id: projectId || undefined }))
      })
      ws.addEventListener('message', (event) => {
        let payload: Record<string, unknown> | null = null
        try {
          const parsed = JSON.parse(String(event.data || '{}'))
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) payload = parsed as Record<string, unknown>
        } catch {
          payload = null
        }
        if (!payload) return
        if (String(payload.channel || '') !== 'terminal') return
        const sessionId = String(payload.entity_id || '')
        const current = sessionRef.current
        if (!current || !sessionId || current.id !== sessionId) return
        const eventType = String(payload.type || '')
        const eventPayload = (payload.payload && typeof payload.payload === 'object' && !Array.isArray(payload.payload))
          ? payload.payload as Record<string, unknown>
          : {}
        if (eventType === 'terminal.output') {
          const chunk = String(eventPayload.chunk || '')
          if (chunk) terminalRef.current?.write(chunk)
          const offset = Number(eventPayload.offset || 0)
          if (!Number.isNaN(offset) && offset > 0) logsOffsetRef.current = offset
          return
        }
        if (eventType === 'terminal.exited' || eventType === 'terminal.error' || eventType === 'terminal.started') {
          void refreshActive()
        }
      })
      ws.addEventListener('error', () => ws.close())
      ws.addEventListener('close', () => scheduleReconnect())
    }

    connect()
    void refreshActive()

    return () => {
      stopped = true
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [projectDir, wsUrl])

  useEffect(() => {
    const onResize = (): void => {
      if (resizeTimerRef.current !== null) window.clearTimeout(resizeTimerRef.current)
      resizeTimerRef.current = window.setTimeout(() => {
        fitRef.current?.fit()
        const current = sessionRef.current
        if (!current) return
        const cols = terminalRef.current?.cols || current.cols || 120
        const rows = terminalRef.current?.rows || current.rows || 36
        void fetch(buildApiUrl(`/api/terminal/session/${current.id}/resize`, projectDir), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cols, rows }),
        }).catch(() => undefined)
      }, 120)
    }
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      if (resizeTimerRef.current !== null) {
        window.clearTimeout(resizeTimerRef.current)
      }
    }
  }, [projectDir])

  return (
    <div className="form-stack">
      <p className="hint">Interactive shell session. Commands run directly in project directory.</p>
      <div className="inline-actions">
        <button className="button button-primary" onClick={() => void startOrAttach()} disabled={loading}>
          {session ? 'Attach / Restart' : 'Start Terminal'}
        </button>
        <button className="button" onClick={() => void refreshActive()} disabled={loading}>
          Reconnect
        </button>
        <button className="button" onClick={() => void stopSession()} disabled={!session || session.status !== 'running'}>
          Stop
        </button>
        <button className="button" onClick={() => terminalRef.current?.clear()}>
          Clear View
        </button>
      </div>
      {error ? <p className="error-banner">{error}</p> : null}
      {session ? (
        <p className="task-meta">
          {session.status} · shell: {session.shell || '-'} · cwd: {session.cwd || '-'} · started: {session.started_at || '-'} · exit: {session.exit_code ?? '-'}
        </p>
      ) : (
        <p className="task-meta">No active terminal session.</p>
      )}
      <div
        ref={containerRef}
        style={{ height: 360, width: '100%', background: '#111', borderRadius: 8, padding: 4 }}
      />
    </div>
  )
}
