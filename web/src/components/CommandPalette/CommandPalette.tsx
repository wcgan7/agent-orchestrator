/**
 * Command Palette — Cmd+K powered quick actions.
 */

import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import './CommandPalette.css'

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
  const listRef = useRef<HTMLDivElement>(null)

  // Filter and group commands
  const filtered = useMemo(() => {
    if (!query) return commands
    const q = query.toLowerCase()
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(q) ||
        cmd.category.toLowerCase().includes(q)
    )
  }, [commands, query])

  // Group by category
  const grouped = useMemo(() => {
    const groups: Record<string, Command[]> = {}
    for (const cmd of filtered) {
      if (!groups[cmd.category]) groups[cmd.category] = []
      groups[cmd.category].push(cmd)
    }
    return groups
  }, [filtered])

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  // Focus input on open
  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
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
          if (filtered[selectedIndex]) {
            filtered[selectedIndex].action()
            onClose()
          }
          break
        case 'Escape':
          e.preventDefault()
          onClose()
          break
      }
    },
    [filtered, selectedIndex, onClose]
  )

  // Scroll selected item into view
  useEffect(() => {
    const item = listRef.current?.querySelector(`[data-index="${selectedIndex}"]`)
    item?.scrollIntoView({ block: 'nearest' })
  }, [selectedIndex])

  if (!isOpen) return null

  let flatIndex = -1

  return (
    <div className="palette-overlay" onClick={onClose}>
      <div className="palette-panel" onClick={(e) => e.stopPropagation()} onKeyDown={handleKeyDown}>
        <div className="palette-input-wrapper">
          <span className="palette-icon">&#x2318;</span>
          <input
            ref={inputRef}
            className="palette-input"
            type="text"
            placeholder="Type a command..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd className="palette-esc">esc</kbd>
        </div>
        <div className="palette-results" ref={listRef}>
          {Object.entries(grouped).map(([category, cmds]) => (
            <div key={category} className="palette-group">
              <div className="palette-group-label">{category}</div>
              {cmds.map((cmd) => {
                flatIndex++
                const idx = flatIndex
                return (
                  <div
                    key={cmd.id}
                    className={`palette-item ${idx === selectedIndex ? 'palette-item-selected' : ''}`}
                    data-index={idx}
                    onClick={() => {
                      cmd.action()
                      onClose()
                    }}
                    onMouseEnter={() => setSelectedIndex(idx)}
                  >
                    <span className="palette-item-icon">{cmd.icon || '>'}</span>
                    <span className="palette-item-label">{cmd.label}</span>
                    {cmd.shortcut && (
                      <kbd className="palette-item-shortcut">{cmd.shortcut}</kbd>
                    )}
                  </div>
                )
              })}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="palette-empty">No commands found</div>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Hook to register Cmd+K / Ctrl+K keybinding.
 */
export function useCommandPalette() {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return { isOpen, open: () => setIsOpen(true), close: () => setIsOpen(false) }
}
