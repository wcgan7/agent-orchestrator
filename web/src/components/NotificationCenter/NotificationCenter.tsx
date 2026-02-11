/**
 * Notification bell with a dropdown history panel.
 *
 * Subscribes to the WebSocket `notifications` channel and maintains a
 * persistent list of up to 50 recent notifications. Optionally shows
 * desktop (browser) notifications when permission has been granted.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useChannel } from '../../contexts/WebSocketContext'
import './NotificationCenter.css'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  /** ISO 8601 timestamp */
  timestamp: string
  read: boolean
}

const MAX_NOTIFICATIONS = 50

// ---------------------------------------------------------------------------
// Sound alert — short beep via Web Audio API (no external files needed)
// ---------------------------------------------------------------------------

let _audioCtx: AudioContext | null = null

function playNotificationSound(type: Notification['type'] = 'info') {
  try {
    if (!_audioCtx) {
      _audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
    }
    const ctx = _audioCtx
    if (ctx.state === 'suspended') {
      ctx.resume()
    }
    const oscillator = ctx.createOscillator()
    const gain = ctx.createGain()
    oscillator.connect(gain)
    gain.connect(ctx.destination)

    // Different tones for severity
    const freq = type === 'error' ? 440 : type === 'warning' ? 520 : type === 'success' ? 660 : 600
    oscillator.frequency.setValueAtTime(freq, ctx.currentTime)
    oscillator.type = 'sine'
    gain.gain.setValueAtTime(0.08, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3)

    oscillator.start(ctx.currentTime)
    oscillator.stop(ctx.currentTime + 0.3)
  } catch {
    // Audio not available — silently skip
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function typeIcon(type: Notification['type']): string {
  switch (type) {
    case 'success':
      return '\u2713' // check mark
    case 'error':
      return '\u2717' // cross mark
    case 'warning':
      return '!'
    case 'info':
    default:
      return 'i'
  }
}

/**
 * Return a human-friendly relative time string (e.g. "3m ago", "2h ago").
 */
function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const seconds = Math.max(0, Math.floor(diff / 1000))

  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function NotificationCenter() {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [open, setOpen] = useState(false)
  const [soundEnabled, setSoundEnabled] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const desktopPermRef = useRef<NotificationPermission>(
    typeof window !== 'undefined' && 'Notification' in window
      ? window.Notification.permission
      : 'denied',
  )

  // -----------------------------------------------------------------------
  // Desktop notification permission
  // -----------------------------------------------------------------------

  const requestDesktopPermission = useCallback(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) return
    if (desktopPermRef.current !== 'default') return
    window.Notification.requestPermission().then((perm) => {
      desktopPermRef.current = perm
    })
  }, [])

  const showDesktopNotification = useCallback((n: Notification) => {
    if (desktopPermRef.current !== 'granted') return
    try {
      new window.Notification(n.title, {
        body: n.message,
        tag: n.id,
      })
    } catch {
      // Some browsers restrict Notification constructor usage
    }
  }, [])

  // -----------------------------------------------------------------------
  // WebSocket subscription
  // -----------------------------------------------------------------------

  useChannel(
    'notifications',
    useCallback(
      (_event: string, data: any) => {
        const incoming: Notification = {
          id: data.id ?? crypto.randomUUID(),
          type: data.type ?? 'info',
          title: data.title ?? 'Notification',
          message: data.message ?? '',
          timestamp: data.timestamp ?? new Date().toISOString(),
          read: false,
        }

        setNotifications((prev) => [incoming, ...prev].slice(0, MAX_NOTIFICATIONS))
        showDesktopNotification(incoming)
        if (soundEnabled) {
          playNotificationSound(incoming.type)
        }
      },
      [showDesktopNotification, soundEnabled],
    ),
  )

  // -----------------------------------------------------------------------
  // Close dropdown when clicking outside
  // -----------------------------------------------------------------------

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // -----------------------------------------------------------------------
  // Actions
  // -----------------------------------------------------------------------

  const unreadCount = notifications.filter((n) => !n.read).length

  const handleToggle = () => {
    setOpen((prev) => !prev)
    // Request desktop permission on first user interaction with the bell
    requestDesktopPermission()
  }

  const markAllRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })))
  }

  const clearAll = () => {
    setNotifications([])
    setOpen(false)
  }

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="notification-center" ref={containerRef}>
      <button
        className="notification-bell-btn"
        onClick={handleToggle}
        aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ''}`}
        aria-expanded={open}
        aria-haspopup="true"
      >
        {/* Bell icon (SVG) */}
        <svg
          className="notification-bell-icon"
          width="20"
          height="20"
          viewBox="0 0 20 20"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <path
            d="M10 2a1 1 0 0 1 1 1v.268A5.002 5.002 0 0 1 15 8v3.5l1.354 1.354a.5.5 0 0 1-.354.854H4a.5.5 0 0 1-.354-.854L5 11.5V8a5.002 5.002 0 0 1 4-4.9V3a1 1 0 0 1 1-1ZM8 15a2 2 0 1 0 4 0H8Z"
            fill="currentColor"
          />
        </svg>

        {unreadCount > 0 && (
          <span className="notification-badge">{unreadCount > 99 ? '99+' : unreadCount}</span>
        )}
      </button>

      {open && (
        <div className="notification-dropdown" role="region" aria-label="Notifications">
          <div className="notification-dropdown-header">
            <h3 className="notification-dropdown-title">Notifications</h3>
            <div className="notification-dropdown-actions">
              <button
                className={`notification-action-btn ${soundEnabled ? '' : 'notification-sound-off'}`}
                onClick={() => setSoundEnabled((prev) => !prev)}
                title={soundEnabled ? 'Mute sound alerts' : 'Unmute sound alerts'}
                aria-label={soundEnabled ? 'Mute notifications' : 'Unmute notifications'}
              >
                {soundEnabled ? '\u{1F50A}' : '\u{1F507}'}
              </button>
              <button className="notification-action-btn" onClick={markAllRead} disabled={unreadCount === 0}>
                Mark all read
              </button>
              <button className="notification-action-btn" onClick={clearAll} disabled={notifications.length === 0}>
                Clear all
              </button>
            </div>
          </div>

          <div className="notification-list">
            {notifications.length === 0 ? (
              <div className="notification-empty">No notifications</div>
            ) : (
              notifications.map((n) => (
                <div
                  key={n.id}
                  className={`notification-item ${n.read ? '' : 'notification-item-unread'} notification-item-${n.type}`}
                  onClick={() =>
                    setNotifications((prev) =>
                      prev.map((item) => (item.id === n.id ? { ...item, read: true } : item)),
                    )
                  }
                >
                  <div className={`notification-type-icon notification-type-${n.type}`}>
                    {typeIcon(n.type)}
                  </div>
                  <div className="notification-content">
                    <div className="notification-title">{n.title}</div>
                    <div className="notification-message">{n.message}</div>
                  </div>
                  <div className="notification-time">{relativeTime(n.timestamp)}</div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}
