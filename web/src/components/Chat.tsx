import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  IconButton,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline'
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
    if (isOpen) fetchMessages()
  }, [runId, projectDir, isOpen])

  useChannel('notifications', useCallback(() => {
    if (isOpen) fetchMessages()
  }, [isOpen]))

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const fetchMessages = async () => {
    try {
      const response = await fetch(
        buildApiUrl('/api/v3/messages', projectDir, { run_id: runId }),
        { headers: buildAuthHeaders() }
      )
      if (!response.ok) throw new Error(`HTTP error ${response.status}`)
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
        if (requirementTaskId.trim()) metadata.task_id = requirementTaskId.trim()
      } else if (messageType === 'correction') {
        metadata.task_id = correctionTaskId.trim()
        metadata.issue = inputValue.trim()
        if (correctionFile.trim()) metadata.file = correctionFile.trim()
        if (correctionSuggestedFix.trim()) metadata.suggested_fix = correctionSuggestedFix.trim()
      } else if (messageType === 'clarification_request') {
        metadata.expects_response = true
      }

      const response = await fetch(buildApiUrl('/api/v3/messages', projectDir, { run_id: runId }), {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ content: inputValue, type: messageType, metadata }),
      })

      if (!response.ok) throw new Error(`HTTP error ${response.status}`)

      setInputValue('')
      if (messageType === 'requirement') {
        setRequirementTaskId('')
      } else if (messageType === 'correction') {
        setCorrectionTaskId('')
        setCorrectionFile('')
        setCorrectionSuggestedFix('')
      }
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
      return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch {
      return ''
    }
  }

  const getMessageTypeLabel = (type: string): string => {
    switch (type) {
      case 'guidance': return 'Guidance'
      case 'clarification_request': return 'Question'
      case 'clarification_response': return 'Answer'
      case 'requirement': return 'Requirement'
      case 'correction': return 'Correction'
      default: return type
    }
  }

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        title="Open chat"
        aria-label="Open chat"
        variant="contained"
        startIcon={<ChatBubbleOutlineIcon />}
        sx={{
          position: 'fixed',
          bottom: { xs: 16, sm: 24 },
          right: { xs: 16, sm: 24 },
          zIndex: (theme) => theme.zIndex.modal + 1,
          borderRadius: 6,
          px: 2,
          py: 1,
          boxShadow: 6,
        }}
      >
        Chat
      </Button>
    )
  }

  return (
    <Box
      role="dialog"
      aria-label="Live collaboration chat panel"
      sx={{
        position: 'fixed',
        bottom: { xs: 16, sm: 24 },
        right: { xs: 16, sm: 24 },
        width: { xs: 'calc(100vw - 32px)', sm: 420 },
        maxWidth: '100vw',
        height: { xs: 'calc(100vh - 32px)', sm: 620 },
        maxHeight: 'calc(100vh - 32px)',
        bgcolor: 'background.paper',
        borderRadius: 3,
        boxShadow: 10,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        zIndex: (theme) => theme.zIndex.modal + 1,
      }}
    >
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{
          px: 2,
          py: 1.5,
          color: 'common.white',
          background: 'linear-gradient(135deg, #0ea5e9 0%, #0369a1 100%)',
        }}
      >
        <Typography variant="h3" sx={{ fontSize: '1rem', color: 'inherit' }}>Live Collaboration Chat</Typography>
        <IconButton
          onClick={() => setIsOpen(false)}
          aria-label="Close chat"
          size="small"
          sx={{ color: 'inherit' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </Stack>

      {!runId ? (
        <Stack alignItems="center" justifyContent="center" sx={{ flex: 1, textAlign: 'center', px: 3, color: 'text.secondary' }}>
          <Typography>No active run</Typography>
          <Typography variant="body2" color="text.disabled">Chat is available when a run is active</Typography>
        </Stack>
      ) : (
        <>
          <Stack spacing={1.5} sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
            {messages.length === 0 ? (
              <Stack alignItems="center" justifyContent="center" sx={{ flex: 1, textAlign: 'center', px: 3, color: 'text.secondary' }}>
                <Typography>No messages yet</Typography>
                <Typography variant="body2" color="text.disabled">
                  Send a message to start collaborating with the worker
                </Typography>
              </Stack>
            ) : (
              messages.map((msg) => (
                <Box
                  key={msg.id}
                  className={`chat-message ${msg.from_human ? 'human' : 'worker'}`}
                  sx={{ alignSelf: msg.from_human ? 'flex-end' : 'flex-start', maxWidth: '86%' }}
                >
                  <Stack direction="row" spacing={0.75} alignItems="center" sx={{ mb: 0.5 }}>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {msg.from_human ? 'You' : 'Worker'}
                    </Typography>
                    <Chip size="small" label={getMessageTypeLabel(msg.type)} sx={{ textTransform: 'uppercase', fontSize: '0.625rem', height: 20 }} />
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                      {formatTimestamp(msg.timestamp)}
                    </Typography>
                  </Stack>
                  <Box
                    sx={{
                      px: 1.5,
                      py: 1,
                      borderRadius: 2,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      bgcolor: msg.from_human ? 'info.main' : 'background.default',
                      color: msg.from_human ? 'common.white' : 'text.primary',
                      border: msg.from_human ? 'none' : '1px solid',
                      borderColor: msg.from_human ? 'transparent' : 'divider',
                    }}
                  >
                    {msg.content}
                  </Box>
                </Box>
              ))
            )}
            <div ref={messagesEndRef} />
          </Stack>

          {error && <Alert severity="error" sx={{ mx: 2, mb: 1 }}>Error: {error}</Alert>}

          <Stack spacing={1} sx={{ p: 2, borderTop: 1, borderColor: 'divider', bgcolor: 'background.default' }}>
            <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
              <Typography variant="caption" color="text.secondary">Type</Typography>
              <Box
                component="select"
                value={messageType}
                onChange={(e) => setMessageType(e.target.value as OutgoingMessageType)}
                disabled={sending}
                aria-label="Message type"
                sx={{
                  minWidth: 170,
                  borderRadius: 1,
                  borderColor: 'divider',
                  borderStyle: 'solid',
                  borderWidth: 1,
                  bgcolor: 'background.paper',
                  color: 'text.primary',
                  px: 1,
                  py: 0.75,
                }}
              >
                <option value="guidance">Guidance</option>
                <option value="clarification_request">Question</option>
                <option value="requirement">Requirement</option>
                <option value="correction">Correction</option>
              </Box>
            </Stack>

            {messageType === 'requirement' && (
              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                <TextField
                  value={requirementTaskId}
                  onChange={(e) => setRequirementTaskId(e.target.value)}
                  placeholder="Optional task_id"
                  disabled={sending}
                  inputProps={{ 'aria-label': 'Requirement task id' }}
                  size="small"
                  sx={{ flex: 1, minWidth: 160 }}
                />
                <Box
                  component="select"
                  value={requirementPriority}
                  onChange={(e) => setRequirementPriority(e.target.value as 'high' | 'medium' | 'low')}
                  disabled={sending}
                  aria-label="Requirement priority"
                  sx={{
                    minWidth: 120,
                    borderRadius: 1,
                    borderColor: 'divider',
                    borderStyle: 'solid',
                    borderWidth: 1,
                    bgcolor: 'background.paper',
                    color: 'text.primary',
                    px: 1,
                    py: 0.75,
                  }}
                >
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </Box>
              </Stack>
            )}

            {messageType === 'correction' && (
              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                <TextField
                  value={correctionTaskId}
                  onChange={(e) => setCorrectionTaskId(e.target.value)}
                  placeholder="task_id (required)"
                  disabled={sending}
                  inputProps={{ 'aria-label': 'Correction task id' }}
                  size="small"
                  sx={{ flex: 1, minWidth: 160 }}
                />
                <TextField
                  value={correctionFile}
                  onChange={(e) => setCorrectionFile(e.target.value)}
                  placeholder="Optional file path"
                  disabled={sending}
                  inputProps={{ 'aria-label': 'Correction file path' }}
                  size="small"
                  sx={{ flex: 2, minWidth: 180 }}
                />
              </Stack>
            )}

            {messageType === 'correction' && (
              <TextField
                value={correctionSuggestedFix}
                onChange={(e) => setCorrectionSuggestedFix(e.target.value)}
                placeholder="Optional suggested fix"
                disabled={sending}
                inputProps={{ 'aria-label': 'Correction suggested fix' }}
                size="small"
                fullWidth
              />
            )}

            <Stack direction="row" spacing={1} alignItems="flex-end">
              <TextField
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Type a message to the worker..."
                disabled={sending}
                multiline
                minRows={2}
                fullWidth
              />
              <Button
                onClick={sendMessage}
                disabled={!inputValue.trim() || sending || (messageType === 'correction' && !correctionTaskId.trim())}
                variant="contained"
              >
                {sending ? 'Sending...' : 'Send'}
              </Button>
            </Stack>
          </Stack>

          <Box sx={{ p: 1.5, borderTop: 1, borderColor: 'divider', bgcolor: 'action.hover' }}>
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5, fontWeight: 600 }}>
              Tips
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Provide guidance, ask questions, or inject requirements/corrections mid-run.
            </Typography>
          </Box>
        </>
      )}
    </Box>
  )
}

export default Chat
