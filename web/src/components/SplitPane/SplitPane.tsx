/**
 * Resizable split-pane layout component.
 *
 * Renders two panels separated by a draggable handle. The left panel width
 * is persisted to localStorage so it survives page reloads.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import './SplitPane.css'

const STORAGE_KEY = 'split-pane-width'

interface SplitPaneProps {
  left: React.ReactNode
  right: React.ReactNode
  /** Initial left panel width as a percentage (0-100). Default 50. */
  defaultLeftWidth?: number
  /** Minimum left panel width as a percentage. Default 20. */
  minLeftWidth?: number
  /** Maximum left panel width as a percentage. Default 80. */
  maxLeftWidth?: number
  className?: string
}

function readPersistedWidth(fallback: number): number {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored !== null) {
      const parsed = parseFloat(stored)
      if (!Number.isNaN(parsed)) return parsed
    }
  } catch {
    // localStorage may be unavailable (e.g. incognito quota exceeded)
  }
  return fallback
}

function persistWidth(value: number): void {
  try {
    localStorage.setItem(STORAGE_KEY, String(value))
  } catch {
    // ignore write failures
  }
}

export default function SplitPane({
  left,
  right,
  defaultLeftWidth = 50,
  minLeftWidth = 20,
  maxLeftWidth = 80,
  className,
}: SplitPaneProps) {
  const [leftWidth, setLeftWidth] = useState<number>(() =>
    clamp(readPersistedWidth(defaultLeftWidth), minLeftWidth, maxLeftWidth),
  )
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)

  /** Clamp a value between min and max. */
  function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value))
  }

  // -----------------------------------------------------------------------
  // Drag handlers
  // -----------------------------------------------------------------------

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!draggingRef.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const offsetX = e.clientX - rect.left
      const pct = (offsetX / rect.width) * 100
      const clamped = clamp(pct, minLeftWidth, maxLeftWidth)
      setLeftWidth(clamped)
    },
    [minLeftWidth, maxLeftWidth],
  )

  const handleMouseUp = useCallback(() => {
    if (!draggingRef.current) return
    draggingRef.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    // Persist the final width
    setLeftWidth((current) => {
      persistWidth(current)
      return current
    })
  }, [])

  // Attach/detach global listeners while dragging.
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    draggingRef.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  const rootClassName = ['split-pane', className].filter(Boolean).join(' ')

  return (
    <div ref={containerRef} className={rootClassName}>
      <div className="split-left" style={{ width: `${leftWidth}%` }}>
        {left}
      </div>
      <div
        className={`split-handle ${draggingRef.current ? 'split-handle-active' : ''}`}
        onMouseDown={handleMouseDown}
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={Math.round(leftWidth)}
        aria-valuemin={minLeftWidth}
        aria-valuemax={maxLeftWidth}
      />
      <div className="split-right" style={{ width: `${100 - leftWidth}%` }}>
        {right}
      </div>
    </div>
  )
}
