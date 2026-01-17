import { useState, useEffect, useRef } from 'react'
import './Chat.css'

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

const Chat = ({ runId, projectDir }: ChatProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen) {
      fetchMessages()
      const interval = setInterval(fetchMessages, 3000)
      return () => clearInterval(interval)
    }
  }, [runId, projectDir, isOpen])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const fetchMessages = async () => {
    try {
      const params = new URLSearchParams()
      if (projectDir) params.append('project_dir', projectDir)
      if (runId) params.append('run_id', runId)

      const headers: HeadersInit = {}
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`/api/messages?${params}`, { headers })
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

    setSending(true)
    setError(null)

    try {
      const params = new URLSearchParams()
      if (projectDir) params.append('project_dir', projectDir)
      if (runId) params.append('run_id', runId)

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }
      const token = localStorage.getItem('feature-prd-runner-auth-token')
      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`/api/messages?${params}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          content: inputValue,
          type: 'guidance',
          metadata: {},
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`)
      }

      setInputValue('')
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

          <div className="chat-input">
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
              disabled={!inputValue.trim() || sending}
            >
              {sending ? 'Sending...' : 'Send'}
            </button>
          </div>

          <div className="chat-hints">
            <p>💡 Tips:</p>
            <ul>
              <li>Provide guidance to steer the worker's approach</li>
              <li>Ask questions about the current implementation</li>
              <li>Request explanations or clarifications</li>
            </ul>
          </div>
        </>
      )}
    </div>
  )
}

export default Chat
