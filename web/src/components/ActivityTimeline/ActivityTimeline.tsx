/**
 * Activity timeline — unified chronological view of all actions on a task.
 */

import { useState, useEffect, useCallback } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import './ActivityTimeline.css'

interface ActivityEvent {
  id: string
  type: string        // status_change | agent_output | feedback | comment | file_change
  timestamp: string
  actor: string       // username or agent_id
  actor_type: string  // "human" | "agent" | "system"
  summary: string
  details?: string
  metadata?: Record<string, any>
}

interface Props {
  taskId: string
  projectDir?: string
}

const EVENT_ICONS: Record<string, string> = {
  status_change: '\u2192',
  agent_output: '\u2699',
  feedback: '\u270E',
  comment: '\u{1F4AC}',
  file_change: '\u{1F4C4}',
  commit: '\u2713',
  reasoning: '\u{1F9E0}',
  error: '\u26A0',
  assignment: '\u{1F464}',
}

export default function ActivityTimeline({ taskId, projectDir }: Props) {
  const [events, setEvents] = useState<ActivityEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null)

  const fetchActivity = useCallback(async () => {
    try {
      // Use unified timeline API endpoint
      const resp = await fetch(
        buildApiUrl(`/api/v2/collaboration/timeline/${taskId}`, projectDir),
        { headers: buildAuthHeaders() }
      )

      if (resp.ok) {
        const data = await resp.json()
        setEvents(data.events || [])
      }
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [taskId, projectDir])

  useEffect(() => {
    fetchActivity()
  }, [fetchActivity])

  if (loading) {
    return <div className="timeline-loading">Loading activity...</div>
  }

  return (
    <div className="activity-timeline">
      <h3 className="timeline-title">Activity</h3>
      {events.length === 0 ? (
        <div className="timeline-empty">No activity recorded yet.</div>
      ) : (
        <div className="timeline-events">
          {events.map(event => (
            <div
              key={event.id}
              className={`timeline-event actor-${event.actor_type}`}
              onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
            >
              <div className="timeline-event-dot">
                <span className="timeline-event-icon">
                  {EVENT_ICONS[event.type] || '\u2022'}
                </span>
              </div>
              <div className="timeline-event-content">
                <div className="timeline-event-header">
                  <span className="timeline-event-actor">{event.actor}</span>
                  <span className={`timeline-event-type type-${event.type}`}>{event.type.replace('_', ' ')}</span>
                  <span className="timeline-event-time">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="timeline-event-summary">{event.summary}</div>
                {expandedEvent === event.id && event.details && (
                  <div className="timeline-event-details">{event.details}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
