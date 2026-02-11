/**
 * Command Palette — Cmd/Ctrl+K global action center.
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import {
  Dialog,
  DialogContent,
  InputAdornment,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  TextField,
  Typography,
  Chip,
  Box,
} from '@mui/material'
import SearchIcon from '@mui/icons-material/Search'

export interface Command {
  id: string
  label: string
  category: string
  icon?: string
  shortcut?: string
  action: () => void
}

interface Props {
  commands: Command[]
  isOpen: boolean
  onClose: () => void
}

export default function CommandPalette({ commands, isOpen, onClose }: Props) {
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)

  const filtered = useMemo(() => {
    if (!query) return commands
    const q = query.toLowerCase()
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(q) ||
        cmd.category.toLowerCase().includes(q) ||
        (cmd.shortcut || '').toLowerCase().includes(q)
    )
  }, [commands, query])

  const grouped = useMemo(() => {
    const groups: Record<string, Command[]> = {}
    for (const cmd of filtered) {
      if (!groups[cmd.category]) groups[cmd.category] = []
      groups[cmd.category].push(cmd)
    }
    return groups
  }, [filtered])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 10)
    }
  }, [isOpen])

  const executeAtIndex = useCallback((idx: number) => {
    const cmd = filtered[idx]
    if (!cmd) return
    cmd.action()
    onClose()
  }, [filtered, onClose])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex((i) => Math.min(i + 1, filtered.length - 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex((i) => Math.max(i - 1, 0))
        break
      case 'Enter':
        e.preventDefault()
        executeAtIndex(selectedIndex)
        break
      case 'Escape':
        e.preventDefault()
        onClose()
        break
    }
  }, [executeAtIndex, filtered.length, onClose, selectedIndex])

  useEffect(() => {
    const item = listRef.current?.querySelector(`[data-index="${selectedIndex}"]`)
    item?.scrollIntoView({ block: 'nearest' })
  }, [selectedIndex])

  let flatIndex = -1

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      fullWidth
      maxWidth="sm"
      aria-labelledby="command-palette-title"
      PaperProps={{
        sx: {
          mt: '8vh',
          borderRadius: 2,
          alignSelf: 'flex-start',
        },
      }}
    >
      <DialogContent sx={{ p: 0 }} onKeyDown={handleKeyDown}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography id="command-palette-title" variant="overline" color="text.secondary">
            Command Palette
          </Typography>
          <TextField
            inputRef={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type a command..."
            fullWidth
            size="small"
            sx={{ mt: 1 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: <Chip size="small" label="Esc" variant="outlined" />,
            }}
          />
        </Box>

        <List ref={listRef} sx={{ maxHeight: 420, overflowY: 'auto', py: 0.5 }}>
          {Object.entries(grouped).map(([category, cmds]) => (
            <Box key={category}>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ px: 2, py: 0.75, display: 'block', textTransform: 'uppercase', letterSpacing: 0.5 }}
              >
                {category}
              </Typography>
              {cmds.map((cmd) => {
                flatIndex++
                const idx = flatIndex
                return (
                  <ListItemButton
                    key={cmd.id}
                    selected={idx === selectedIndex}
                    data-index={idx}
                    onClick={() => executeAtIndex(idx)}
                    onMouseEnter={() => setSelectedIndex(idx)}
                  >
                    <ListItemText
                      primary={
                        <Stack direction="row" spacing={1.25} alignItems="center" useFlexGap>
                          <Typography variant="body2" sx={{ minWidth: 18, textAlign: 'center' }}>
                            {cmd.icon || '>'}
                          </Typography>
                          <Typography variant="body2">{cmd.label}</Typography>
                        </Stack>
                      }
                    />
                    {cmd.shortcut && (
                      <Chip size="small" label={cmd.shortcut} variant="outlined" sx={{ fontSize: 11 }} />
                    )}
                  </ListItemButton>
                )
              })}
            </Box>
          ))}

          {filtered.length === 0 && (
            <Box sx={{ px: 2, py: 4 }}>
              <Typography variant="body2" color="text.secondary" textAlign="center">
                No commands found
              </Typography>
            </Box>
          )}
        </List>
      </DialogContent>
    </Dialog>
  )
}

/**
 * Hook to register Cmd+K / Ctrl+K keybinding.
 */
export function useCommandPalette() {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setIsOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return { isOpen, open: () => setIsOpen(true), close: () => setIsOpen(false) }
}
