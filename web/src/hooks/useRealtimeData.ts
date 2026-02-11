import { useState, useEffect, useCallback, useRef } from 'react'
import { useChannel } from '../contexts/WebSocketContext'

interface UseRealtimeDataOptions<T> {
  /** WebSocket channel to subscribe to */
  channel: string
  /** URL to fetch initial (and refreshed) data from */
  fetchUrl: string
  /** Optional headers to include in the fetch request */
  fetchHeaders?: HeadersInit
  /** Optional transform applied to the raw JSON before storing */
  transform?: (raw: unknown) => T
}

interface UseRealtimeDataResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => void
}

/**
 * Hook that fetches data on mount and subscribes to a WebSocket channel
 * for real-time updates.  On "update" or "snapshot" events the fetch is
 * re-triggered so the component always shows fresh server state.
 *
 * Falls back gracefully if the WebSocket is not connected -- the initial
 * fetch still runs, there is just no live push.
 */
export function useRealtimeData<T = unknown>(
  options: UseRealtimeDataOptions<T>,
): UseRealtimeDataResult<T> {
  const { channel, fetchUrl, fetchHeaders, transform } = options

  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Keep transform in a ref so the callback identity is stable
  const transformRef = useRef(transform)
  transformRef.current = transform

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(fetchUrl, {
        headers: fetchHeaders,
      })
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const raw = await response.json()
      const result = transformRef.current ? transformRef.current(raw) : (raw as T)
      setData(result)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }, [fetchUrl, fetchHeaders])

  // Initial fetch on mount / when URL changes
  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Subscribe to the WebSocket channel -- re-fetch on update/snapshot events
  useChannel(
    channel,
    useCallback(
      (event: string) => {
        if (event === 'update' || event === 'snapshot') {
          fetchData()
        }
      },
      [fetchData],
    ),
  )

  return { data, loading, error, refetch: fetchData }
}
