/**
 * Activity timeline — unified chronological view of all actions on a task.
 */

import { useState, useEffect, useCallback } from 'react'
import { Box, Card, Chip, Stack, Typography } from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../../api'

interface ActivityEvent {
  id: string
  type: string
  timestamp: string
  actor: string
  actor_type: string
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

const ACTOR_COLORS: Record<string, string> = {
  human: '#2563eb',
  agent: '#7c3aed',
  system: '#6b7280',
}

export default function ActivityTimeline({ taskId, projectDir }: Props) {
  const [events, setEvents] = useState<ActivityEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null)

  const fetchActivity = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl(`/api/v3/collaboration/timeline/${taskId}`, projectDir),
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
    return <Typography className="timeline-loading" color="text.secondary">Loading activity...</Typography>
  }

  return (
    <Box className="activity-timeline" sx={{ p: 1.5 }}>
      <Typography className="timeline-title" variant="h6" sx={{ fontSize: '1rem', mb: 1.5 }}>
        Activity
      </Typography>

      {events.length === 0 ? (
        <Typography className="timeline-empty" color="text.secondary">No activity recorded yet.</Typography>
      ) : (
        <Stack className="timeline-events" spacing={1}>
          {events.map(event => (
            <Card
              key={event.id}
              className={`timeline-event actor-${event.actor_type}`}
              variant="outlined"
              onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
              sx={{
                p: 1,
                cursor: 'pointer',
                borderLeft: '3px solid',
                borderLeftColor: ACTOR_COLORS[event.actor_type] || 'divider',
              }}
            >
              <Stack direction="row" spacing={1} alignItems="flex-start">
                <Box
                  className="timeline-event-dot"
                  sx={{
                    width: 22,
                    height: 22,
                    borderRadius: '50%',
                    bgcolor: 'background.default',
                    border: '1px solid',
                    borderColor: 'divider',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    mt: 0.25,
                  }}
                >
                  <Typography className="timeline-event-icon" variant="caption">
                    {EVENT_ICONS[event.type] || '\u2022'}
                  </Typography>
                </Box>

                <Box className="timeline-event-content" sx={{ minWidth: 0, flex: 1 }}>
                  <Stack className="timeline-event-header" direction="row" spacing={0.75} alignItems="center" useFlexGap flexWrap="wrap">
                    <Typography className="timeline-event-actor" variant="caption" sx={{ fontWeight: 700 }}>
                      {event.actor}
                    </Typography>
                    <Chip
                      className={`timeline-event-type type-${event.type}`}
                      label={event.type.replace('_', ' ')}
                      size="small"
                      variant="outlined"
                      sx={{ height: 20, textTransform: 'capitalize' }}
                    />
                    <Typography className="timeline-event-time" variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </Typography>
                  </Stack>

                  <Typography className="timeline-event-summary" variant="body2" sx={{ mt: 0.5 }}>
                    {event.summary}
                  </Typography>

                  {expandedEvent === event.id && event.details && (
                    <Typography className="timeline-event-details" variant="body2" color="text.secondary" sx={{ mt: 0.75, whiteSpace: 'pre-wrap' }}>
                      {event.details}
                    </Typography>
                  )}
                </Box>
              </Stack>
            </Card>
          ))}
        </Stack>
      )}
    </Box>
  )
}
