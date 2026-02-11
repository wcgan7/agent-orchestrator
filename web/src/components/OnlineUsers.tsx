/**
 * Online users indicator — shows a dot with count of currently active users.
 * Fetches from /api/v2/collaboration/presence on mount and on WebSocket events.
 */

import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'

interface OnlineUser {
  username: string
  viewing?: string
  task_id?: string
}

export default function OnlineUsers({ projectDir }: { projectDir?: string }) {
  const [users, setUsers] = useState<OnlineUser[]>([])
  const [showList, setShowList] = useState(false)

  const fetchPresence = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/v2/collaboration/presence', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setUsers(data.users || [])
      }
    } catch {
      // ignore
    }
  }, [projectDir])

  useEffect(() => {
    fetchPresence()
  }, [fetchPresence])

  useChannel('presence', useCallback(() => {
    fetchPresence()
  }, [fetchPresence]))

  if (users.length === 0) return null

  return (
    <div className="online-users" style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        className="online-users-btn"
        onClick={() => setShowList(!showList)}
        title={`${users.length} user(s) online`}
        style={{
          display: 'flex', alignItems: 'center', gap: '4px',
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--color-text-secondary)', fontSize: 'var(--text-sm)',
          padding: '4px 8px', borderRadius: 'var(--radius-md)',
        }}
      >
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: 'var(--color-success-500)', display: 'inline-block',
        }} />
        {users.length}
      </button>
      {showList && (
        <div style={{
          position: 'absolute', top: '100%', right: 0, marginTop: 4,
          background: 'var(--color-bg-primary)', border: '1px solid var(--color-border-default)',
          borderRadius: 'var(--radius-md)', padding: '8px 0',
          boxShadow: 'var(--shadow-lg)', zIndex: 100, minWidth: 160,
        }}>
          {users.map((u) => (
            <div key={u.username} style={{
              padding: '4px 12px', fontSize: 'var(--text-sm)',
              color: 'var(--color-text-primary)',
            }}>
              <span>{u.username}</span>
              {u.viewing && (
                <span style={{ marginLeft: 8, fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                  {u.viewing}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
