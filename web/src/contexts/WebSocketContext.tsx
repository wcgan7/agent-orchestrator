/**
 * Unified WebSocket context — single multiplexed connection for all real-time data.
 *
 * Usage:
 *   <WebSocketProvider>
 *     <App />
 *   </WebSocketProvider>
 *
 *   // Inside any component:
 *   const { subscribe, send, status } = useWebSocket()
 *   useChannel('tasks', (event, data) => { ... })
 */

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  ReactNode,
} from 'react'
import { buildWsUrl } from '../api'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected'

export interface WsMessage {
  channel: string
  event: string
  data: any
  event_id?: number
}

type ChannelHandler = (event: string, data: any) => void

interface WebSocketContextType {
  status: ConnectionStatus
  subscribe: (channel: string) => void
  unsubscribe: (channel: string) => void
  send: (action: string, payload?: Record<string, any>) => void
  addHandler: (channel: string, handler: ChannelHandler) => () => void
  lastEventId: number
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

interface WebSocketProviderProps {
  children: ReactNode
  projectDir?: string
}

const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30000

export function WebSocketProvider({ children, projectDir }: WebSocketProviderProps) {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected')
  const wsRef = useRef<WebSocket | null>(null)
  const handlersRef = useRef<Map<string, Set<ChannelHandler>>>(new Map())
  const subscribedRef = useRef<Set<string>>(new Set())
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastEventIdRef = useRef(0)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')
    const url = buildWsUrl('/ws', projectDir)
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('connected')
      reconnectAttemptRef.current = 0

      // Re-subscribe to previously subscribed channels
      if (subscribedRef.current.size > 0) {
        ws.send(JSON.stringify({
          action: 'subscribe',
          channels: Array.from(subscribedRef.current),
        }))
      }
    }

    ws.onmessage = (ev) => {
      try {
        const msg: WsMessage = JSON.parse(ev.data)
        if (msg.event_id) {
          lastEventIdRef.current = msg.event_id
        }
        const handlers = handlersRef.current.get(msg.channel)
        if (handlers) {
          handlers.forEach((h) => {
            try { h(msg.event, msg.data) } catch { /* handler error */ }
          })
        }
        // Also dispatch to wildcard '*' handlers
        const wildcardHandlers = handlersRef.current.get('*')
        if (wildcardHandlers) {
          wildcardHandlers.forEach((h) => {
            try { h(msg.event, { ...msg.data, _channel: msg.channel }) } catch { /* handler error */ }
          })
        }
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      wsRef.current = null
      setStatus('reconnecting')
      scheduleReconnect()
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [projectDir])

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) return
    const delay = Math.min(
      RECONNECT_BASE_MS * Math.pow(2, reconnectAttemptRef.current),
      RECONNECT_MAX_MS,
    )
    reconnectAttemptRef.current++
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null
      connect()
    }, delay)
  }, [connect])

  // Connect on mount
  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const subscribe = useCallback((channel: string) => {
    subscribedRef.current.add(channel)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'subscribe', channels: [channel] }))
    }
  }, [])

  const unsubscribe = useCallback((channel: string) => {
    subscribedRef.current.delete(channel)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'unsubscribe', channels: [channel] }))
    }
  }, [])

  const send = useCallback((action: string, payload: Record<string, any> = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, ...payload }))
    }
  }, [])

  const addHandler = useCallback((channel: string, handler: ChannelHandler) => {
    if (!handlersRef.current.has(channel)) {
      handlersRef.current.set(channel, new Set())
    }
    handlersRef.current.get(channel)!.add(handler)

    // Return cleanup function
    return () => {
      handlersRef.current.get(channel)?.delete(handler)
    }
  }, [])

  const value: WebSocketContextType = {
    status,
    subscribe,
    unsubscribe,
    send,
    addHandler,
    lastEventId: lastEventIdRef.current,
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

const _noop = () => {}
const _noopCleanup = () => _noop

/** Safe fallback when used outside a WebSocketProvider (e.g. in tests). */
const _fallback: WebSocketContextType = {
  status: 'disconnected',
  subscribe: _noop,
  unsubscribe: _noop,
  send: _noop,
  addHandler: _noopCleanup,
  lastEventId: 0,
}

export function useWebSocket(): WebSocketContextType {
  const ctx = useContext(WebSocketContext)
  return ctx ?? _fallback
}

/**
 * Subscribe to a channel and call `handler` on every event.
 * Automatically subscribes on mount and cleans up on unmount.
 */
export function useChannel(channel: string, handler: ChannelHandler) {
  const { subscribe, addHandler } = useWebSocket()

  useEffect(() => {
    subscribe(channel)
    const cleanup = addHandler(channel, handler)
    return () => {
      cleanup()
      // Don't unsubscribe — other components may still need it.
      // The server ignores duplicate unsubscribes anyway.
    }
  }, [channel, handler, subscribe, addHandler])
}
