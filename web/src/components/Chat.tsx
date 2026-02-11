import { useState, useEffect, useRef, useCallback } from 'react'
import './Chat.css'
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'

interface ChatMessage {
  id: string
  type: string
  content: string
  timestamp: string
  from_human: boolean
  metadata: Record<string, any>
}

interface ChatProps {
  runId?: string
  projectDir?: string
}

type OutgoingMessageType = 'guidance' | 'clarification_request' | 'requirement' | 'correction'

const Chat = ({ runId, projectDir }: ChatProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [messageType, setMessageType] = useState<OutgoingMessageType>('guidance')
  const [requirementPriority, setRequirementPriority] = useState<'high' | 'medium' | 'low'>('medium')
  const [requirementTaskId, setRequirementTaskId] = useState('')
  const [correctionTaskId, setCorrectionTaskId] = useState('')
  const [correctionFile, setCorrectionFile] = useState('')
  const [correctionSuggestedFix, setCorrectionSuggestedFix] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen) {
      fetchMessages()
    }
  }, [runId, projectDir, isOpen])

  // Real-time updates via WebSocket instead of polling
  useChannel('notifications', useCallback((_event: string, _data: any) => {
    if (isOpen) fetchMessages()
  }, [isOpen]))

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const fetchMessages = async () => {
    try {
      const response = await fetch(
        buildApiUrl('/api/messages', projectDir, { run_id: runId }),
        { headers: buildAuthHeaders() }
      )
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }
      const data = await response.json()
      setMessages(Array.isArray(data) ? data : [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch messages')
    }
  }

  const sendMessage = async () => {
    if (!inputValue.trim() || sending) return

    if (messageType === 'correction' && !correctionTaskId.trim()) {
      setError('Task ID is required for corrections')
      return
    }

    setSending(true)
    setError(null)

    try {
      const metadata: Record<string, any> = {}
      if (messageType === 'requirement') {
        metadata.priority = requirementPriority
        if (requirementTaskId.trim()) {
          metadata.task_id = requirementTaskId.trim()
        }
      } else if (messageType === 'correction') {
        metadata.task_id = correctionTaskId.trim()
        metadata.issue = inputValue.trim()
        if (correctionFile.trim()) {
          metadata.file = correctionFile.trim()
        }
        if (correctionSuggestedFix.trim()) {
          metadata.suggested_fix = correctionSuggestedFix.trim()
        }
      } else if (messageType === 'clarification_request') {
        metadata.expects_response = true
      }

      const response = await fetch(buildApiUrl('/api/messages', projectDir, { run_id: runId }), {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          content: inputValue,
          type: messageType,
          metadata,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }

      setInputValue('')
      if (messageType === 'requirement') {
        setRequirementTaskId('')
      } else if (messageType === 'correction') {
        setCorrectionTaskId('')
        setCorrectionFile('')
        setCorrectionSuggestedFix('')
      }
      // Immediately fetch messages to show the sent message
      await fetchMessages()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
    } finally {
      setSending(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const formatTimestamp = (timestamp: string): string => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch {
      return ''
    }
  }

  const getMessageTypeLabel = (type: string): string => {
    switch (type) {
      case 'guidance':
        return 'Guidance'
      case 'clarification_request':
        return 'Question'
      case 'clarification_response':
        return 'Answer'
      case 'requirement':
        return 'Requirement'
      case 'correction':
        return 'Correction'
      default:
        return type
    }
  }

  if (!isOpen) {
    return (
      <button
        className="chat-toggle"
        onClick={() => setIsOpen(true)}
        title="Open chat"
        aria-label="Open chat"
      >
        💬 Chat
      </button>
    )
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h3>Live Collaboration Chat</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="chat-close"
          aria-label="Close chat"
        >
          ×
        </button>
      </div>

      {!runId ? (
        <div className="chat-empty">
          <p>No active run</p>
          <p className="hint">Chat is available when a run is active</p>
        </div>
      ) : (
        <>
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="chat-empty">
                <p>No messages yet</p>
                <p className="hint">Send a message to start collaborating with the worker</p>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`chat-message ${msg.from_human ? 'human' : 'worker'}`}
                >
                  <div className="message-header">
                    <span className="message-sender">
                      {msg.from_human ? '👤 You' : '🤖 Worker'}
                    </span>
                    <span className="message-type">{getMessageTypeLabel(msg.type)}</span>
                    <span className="message-time">{formatTimestamp(msg.timestamp)}</span>
                  </div>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {error && (
            <div className="chat-error">
              Error: {error}
            </div>
          )}

          <div className="chat-input" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <label style={{ fontSize: '0.75rem', color: '#666' }}>
                Type
              </label>
              <select
                value={messageType}
                onChange={(e) => setMessageType(e.target.value as OutgoingMessageType)}
                disabled={sending}
                aria-label="Message type"
                style={{ padding: '0.25rem 0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.75rem' }}
              >
                <option value="guidance">Guidance</option>
                <option value="clarification_request">Question</option>
                <option value="requirement">Requirement</option>
                <option value="correction">Correction</option>
              </select>
            </div>

            {messageType === 'requirement' && (
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <input
                  value={requirementTaskId}
                  onChange={(e) => setRequirementTaskId(e.target.value)}
                  placeholder="Optional task_id"
                  disabled={sending}
                  aria-label="Requirement task id"
                  style={{
                    flex: 1,
                    minWidth: '160px',
                    padding: '0.5rem',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    fontSize: '0.75rem',
                    fontFamily: 'inherit',
                  }}
                />
                <select
                  value={requirementPriority}
                  onChange={(e) => setRequirementPriority(e.target.value as any)}
                  disabled={sending}
                  aria-label="Requirement priority"
                  style={{ padding: '0.25rem 0.5rem', border: '1px solid #ddd', borderRadius: '4px', fontSize: '0.75rem' }}
                >
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
            )}

            {messageType === 'correction' && (
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <input
                  value={correctionTaskId}
                  onChange={(e) => setCorrectionTaskId(e.target.value)}
                  placeholder="task_id (required)"
                  disabled={sending}
                  aria-label="Correction task id"
                  style={{
                    flex: 1,
                    minWidth: '160px',
                    padding: '0.5rem',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    fontSize: '0.75rem',
                    fontFamily: 'inherit',
                  }}
                />
                <input
                  value={correctionFile}
                  onChange={(e) => setCorrectionFile(e.target.value)}
                  placeholder="Optional file path"
                  disabled={sending}
                  aria-label="Correction file path"
                  style={{
                    flex: 2,
                    minWidth: '180px',
                    padding: '0.5rem',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    fontSize: '0.75rem',
                    fontFamily: 'inherit',
                  }}
                />
              </div>
            )}

            {messageType === 'correction' && (
              <input
                value={correctionSuggestedFix}
                onChange={(e) => setCorrectionSuggestedFix(e.target.value)}
                placeholder="Optional suggested fix"
                disabled={sending}
                aria-label="Correction suggested fix"
                style={{
                  padding: '0.5rem',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  fontSize: '0.75rem',
                  fontFamily: 'inherit',
                }}
              />
            )}

            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-end' }}>
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message to the worker..."
                disabled={sending}
                rows={2}
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || sending || (messageType === 'correction' && !correctionTaskId.trim())}
              >
                {sending ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>

          <div className="chat-hints">
            <p>💡 Tips:</p>
            <ul>
              <li>Provide guidance to steer the worker's approach</li>
              <li>Ask questions about the current implementation</li>
              <li>Inject requirements or corrections mid-run</li>
            </ul>
          </div>
        </>
      )}
    </div>
  )
}

export default Chat
