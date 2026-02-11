/**
 * Inline code review — line-level commenting on diffs, similar to GitHub PR reviews.
 */

import { useState, useCallback, useEffect } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'

interface DiffLine {
  number: number
  type: 'added' | 'removed' | 'context' | 'hunk'
  content: string
}

interface InlineComment {
  id: string
  file_path: string
  line_number: number
  body: string
  author: string
  author_type: string
  resolved: boolean
  created_at: string
  parent_id?: string
  replies?: InlineComment[]
}

interface Props {
  taskId: string
  filePath: string
  diff: string
  projectDir?: string
  onCommentAdded?: () => void
}

type ViewMode = 'unified' | 'split'

const colors = {
  border: '#d6dbe6',
  bg: '#ffffff',
  bgAlt: '#f6f8fb',
  text: '#1f2937',
  muted: '#6b7280',
  added: '#dcfce7',
  removed: '#fee2e2',
  hunk: '#eef2ff',
  primary: '#0284c7',
  success: '#16a34a',
  keyword: '#0550ae',
  string: '#0a3069',
}

const baseStyles: Record<string, React.CSSProperties> = {
  review: {
    border: `1px solid ${colors.border}`,
    borderRadius: 8,
    overflow: 'hidden',
    background: colors.bg,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    background: colors.bgAlt,
    borderBottom: `1px solid ${colors.border}`,
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  path: {
    fontSize: 13,
    color: colors.text,
    fontFamily: '"IBM Plex Mono", monospace',
  },
  count: {
    fontSize: 12,
    color: colors.muted,
  },
  toggleBtn: {
    fontSize: 12,
    padding: '3px 8px',
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
    background: colors.bg,
    color: colors.text,
    cursor: 'pointer',
  },
  diff: {
    fontFamily: '"IBM Plex Mono", monospace',
    fontSize: 12,
    overflowX: 'auto',
  },
  diffLine: {
    display: 'flex',
    minHeight: 20,
    lineHeight: '20px',
    borderLeft: '3px solid transparent',
    cursor: 'pointer',
  },
  splitRow: {
    display: 'flex',
  },
  splitCell: {
    flex: 1,
    minWidth: 0,
    borderLeftWidth: 2,
  },
  lineNumber: {
    width: 48,
    minWidth: 48,
    textAlign: 'right',
    padding: '0 8px',
    color: colors.muted,
    userSelect: 'none',
  },
  lineMarker: {
    width: 16,
    minWidth: 16,
    textAlign: 'center',
    color: colors.muted,
    userSelect: 'none',
  },
  lineContent: {
    flex: 1,
    whiteSpace: 'pre',
    paddingRight: 8,
    color: colors.text,
  },
  commentIndicator: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 18,
    height: 18,
    borderRadius: 999,
    background: colors.primary,
    color: '#fff',
    fontSize: 10,
    fontFamily: '"IBM Plex Sans", sans-serif',
    marginRight: 8,
    marginTop: 1,
  },
  comment: {
    padding: '8px 12px',
    margin: '0 8px 4px 64px',
    background: colors.bgAlt,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
    fontFamily: '"IBM Plex Sans", sans-serif',
  },
  commentResolved: {
    opacity: 0.6,
    borderLeft: `2px solid ${colors.success}`,
  },
  commentHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
    fontSize: 12,
  },
  commentAuthor: {
    fontWeight: 600,
    color: colors.text,
  },
  commentTime: {
    color: colors.muted,
  },
  commentBody: {
    fontSize: 13,
    color: colors.text,
    whiteSpace: 'pre-wrap',
  },
  miniBtn: {
    padding: '1px 8px',
    fontSize: 10,
    borderRadius: 4,
    border: `1px solid ${colors.border}`,
    background: colors.bg,
    color: colors.text,
    cursor: 'pointer',
  },
  miniBtnPrimary: {
    background: colors.primary,
    borderColor: colors.primary,
    color: '#fff',
  },
  miniBtnSuccess: {
    background: '#dcfce7',
    borderColor: '#86efac',
    color: '#166534',
  },
  repliesWrap: {
    marginLeft: 24,
    paddingLeft: 12,
    borderLeft: `2px solid ${colors.border}`,
  },
  commentForm: {
    padding: '8px 12px',
    margin: '4px 8px 4px 64px',
    background: colors.bg,
    border: `1px solid #93c5fd`,
    borderRadius: 6,
    fontFamily: '"IBM Plex Sans", sans-serif',
  },
  textarea: {
    width: '100%',
    padding: 8,
    border: `1px solid ${colors.border}`,
    borderRadius: 6,
    fontSize: 13,
    color: colors.text,
    background: colors.bg,
    resize: 'vertical',
    boxSizing: 'border-box',
  },
  commentActions: {
    display: 'flex',
    gap: 8,
    marginTop: 4,
  },
}

function diffLineStyle(type: DiffLine['type'] | 'empty'): React.CSSProperties {
  const base = { ...baseStyles.diffLine }
  if (type === 'added') {
    return { ...base, background: colors.added, borderLeftColor: '#4ade80' }
  }
  if (type === 'removed') {
    return { ...base, background: colors.removed, borderLeftColor: '#f87171' }
  }
  if (type === 'hunk') {
    return { ...base, background: colors.hunk, color: colors.muted, cursor: 'default', padding: '2px 0' }
  }
  if (type === 'empty') {
    return { ...base, background: colors.bgAlt, opacity: 0.45, cursor: 'default' }
  }
  return base
}

/* -------------------------------------------------------------------------- */
/*  Syntax highlighting                                                       */
/* -------------------------------------------------------------------------- */

const KEYWORDS = new Set([
  'function', 'const', 'let', 'var', 'class', 'import', 'export', 'return',
  'if', 'else', 'for', 'while', 'try', 'catch', 'async', 'await', 'def',
  'self', 'from',
])

function highlightSyntax(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let i = 0
  let buf = ''
  let key = 0

  const flush = () => {
    if (buf) {
      nodes.push(buf)
      buf = ''
    }
  }

  while (i < text.length) {
    const ch = text[i]

    if (ch === '"' || ch === "'") {
      flush()
      const quote = ch
      let str = ch
      i++
      while (i < text.length && text[i] !== quote) {
        if (text[i] === '\\' && i + 1 < text.length) {
          str += text[i] + text[i + 1]
          i += 2
        } else {
          str += text[i]
          i++
        }
      }
      if (i < text.length) {
        str += text[i]
        i++
      }
      nodes.push(
        <span key={`s${key++}`} style={{ color: colors.string }}>{str}</span>
      )
      continue
    }

    if (/[a-zA-Z_]/.test(ch)) {
      let word = ''
      while (i < text.length && /[a-zA-Z0-9_]/.test(text[i])) {
        word += text[i]
        i++
      }
      if (KEYWORDS.has(word)) {
        flush()
        nodes.push(
          <span key={`k${key++}`} style={{ color: colors.keyword, fontWeight: 600 }}>{word}</span>
        )
      } else {
        buf += word
      }
      continue
    }

    buf += ch
    i++
  }

  flush()
  return nodes
}

/* -------------------------------------------------------------------------- */
/*  Diff parser                                                               */
/* -------------------------------------------------------------------------- */

function parseDiff(diff: string): DiffLine[] {
  if (!diff) return []
  const lines = diff.split('\n')
  const result: DiffLine[] = []
  let lineNum = 0

  for (const line of lines) {
    if (line.startsWith('@@')) {
      const match = line.match(/@@ -\d+(?:,\d+)? \+(\d+)/)
      if (match) lineNum = parseInt(match[1], 10) - 1
      result.push({ number: 0, type: 'hunk', content: line })
    } else if (line.startsWith('+') && !line.startsWith('+++')) {
      lineNum++
      result.push({ number: lineNum, type: 'added', content: line.slice(1) })
    } else if (line.startsWith('-') && !line.startsWith('---')) {
      result.push({ number: 0, type: 'removed', content: line.slice(1) })
    } else if (!line.startsWith('---') && !line.startsWith('+++')) {
      lineNum++
      result.push({ number: lineNum, type: 'context', content: line.startsWith(' ') ? line.slice(1) : line })
    }
  }
  return result
}

function groupCommentsIntoThreads(flat: InlineComment[]): InlineComment[] {
  const map = new Map<string, InlineComment>()
  const roots: InlineComment[] = []

  for (const c of flat) {
    map.set(c.id, { ...c, replies: [] })
  }

  for (const c of flat) {
    const node = map.get(c.id)!
    if (c.parent_id && map.has(c.parent_id)) {
      map.get(c.parent_id)!.replies!.push(node)
    } else {
      roots.push(node)
    }
  }

  return roots
}

export default function InlineReview({ taskId, filePath, diff, projectDir, onCommentAdded }: Props) {
  const [comments, setComments] = useState<InlineComment[]>([])
  const [activeLineIdx, setActiveLineIdx] = useState<number | null>(null)
  const [commentBody, setCommentBody] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('unified')
  const [replyingTo, setReplyingTo] = useState<string | null>(null)
  const [replyBody, setReplyBody] = useState('')

  const diffLines = parseDiff(diff)

  const fetchComments = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl(`/api/v3/collaboration/comments/${taskId}`, projectDir, { file_path: filePath }),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setComments(data.comments || [])
      }
    } catch {
      // ignore
    }
  }, [taskId, filePath, projectDir])

  useEffect(() => {
    fetchComments()
  }, [fetchComments])

  const handleAddComment = async (lineNumber: number, lineType: string) => {
    if (!commentBody.trim()) return
    setSubmitting(true)
    try {
      await fetch(
        buildApiUrl('/api/v3/collaboration/comments', projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            task_id: taskId,
            file_path: filePath,
            line_number: lineNumber,
            body: commentBody,
            line_type: lineType,
          }),
        }
      )
      setCommentBody('')
      setActiveLineIdx(null)
      fetchComments()
      onCommentAdded?.()
    } finally {
      setSubmitting(false)
    }
  }

  const handleAddReply = async (parentComment: InlineComment) => {
    if (!replyBody.trim()) return
    setSubmitting(true)
    try {
      await fetch(
        buildApiUrl('/api/v3/collaboration/comments', projectDir),
        {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            task_id: taskId,
            file_path: filePath,
            line_number: parentComment.line_number,
            body: replyBody,
            parent_id: parentComment.id,
          }),
        }
      )
      setReplyBody('')
      setReplyingTo(null)
      fetchComments()
      onCommentAdded?.()
    } finally {
      setSubmitting(false)
    }
  }

  const handleResolve = async (commentId: string) => {
    await fetch(
      buildApiUrl(`/api/v3/collaboration/comments/${commentId}/resolve`, projectDir),
      { method: 'POST', headers: buildAuthHeaders() }
    )
    fetchComments()
  }

  const getCommentsForLine = (lineNumber: number) => comments.filter(c => c.line_number === lineNumber)
  const getThreadedCommentsForLine = (lineNumber: number) => groupCommentsIntoThreads(getCommentsForLine(lineNumber))

  const renderComment = (c: InlineComment) => (
    <div key={c.id}>
      <div
        className={`inline-comment ${c.resolved ? 'resolved' : ''}`}
        style={{ ...baseStyles.comment, ...(c.resolved ? baseStyles.commentResolved : {}) }}
      >
        <div className="inline-comment-header" style={baseStyles.commentHeader}>
          <span className="inline-comment-author" style={baseStyles.commentAuthor}>{c.author || 'user'}</span>
          <span className="inline-comment-time" style={baseStyles.commentTime}>{new Date(c.created_at).toLocaleTimeString()}</span>
          {!c.resolved && (
            <button
              className="inline-comment-resolve"
              style={{ ...baseStyles.miniBtn, ...baseStyles.miniBtnSuccess, marginLeft: 'auto' }}
              onClick={(e) => { e.stopPropagation(); handleResolve(c.id) }}
            >
              Resolve
            </button>
          )}
        </div>
        <div className="inline-comment-body" style={baseStyles.commentBody}>{c.body}</div>
        <button
          className="inline-comment-reply"
          style={{ ...baseStyles.miniBtn, marginTop: 4, color: colors.primary, borderColor: '#93c5fd' }}
          onClick={(e) => {
            e.stopPropagation()
            setReplyingTo(replyingTo === c.id ? null : c.id)
            setReplyBody('')
          }}
        >
          Reply
        </button>
      </div>

      {replyingTo === c.id && (
        <div className="inline-comment-form" style={{ ...baseStyles.commentForm, marginLeft: 24 }}>
          <textarea
            className="inline-comment-input"
            style={baseStyles.textarea}
            placeholder="Write a reply..."
            value={replyBody}
            onChange={(e) => setReplyBody(e.target.value)}
            rows={2}
            autoFocus
          />
          <div className="inline-comment-actions" style={baseStyles.commentActions}>
            <button
              className="inline-comment-submit"
              style={{ ...baseStyles.miniBtn, ...baseStyles.miniBtnPrimary }}
              onClick={() => handleAddReply(c)}
              disabled={submitting || !replyBody.trim()}
            >
              {submitting ? 'Adding...' : 'Reply'}
            </button>
            <button
              className="inline-comment-cancel"
              style={baseStyles.miniBtn}
              onClick={() => { setReplyingTo(null); setReplyBody('') }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {c.replies && c.replies.length > 0 && (
        <div className="inline-comment-replies" style={baseStyles.repliesWrap}>
          {c.replies.map(reply => renderComment(reply))}
        </div>
      )}
    </div>
  )

  interface SplitRow {
    left: DiffLine | null
    right: DiffLine | null
    originalIdx: number
  }

  function buildSplitRows(): SplitRow[] {
    const rows: SplitRow[] = []
    let i = 0
    while (i < diffLines.length) {
      const line = diffLines[i]

      if (line.type === 'hunk') {
        rows.push({ left: line, right: line, originalIdx: i })
        i++
        continue
      }

      if (line.type === 'context') {
        rows.push({ left: line, right: line, originalIdx: i })
        i++
        continue
      }

      const removedBlock: { line: DiffLine; idx: number }[] = []
      const addedBlock: { line: DiffLine; idx: number }[] = []

      while (i < diffLines.length && diffLines[i].type === 'removed') {
        removedBlock.push({ line: diffLines[i], idx: i })
        i++
      }
      while (i < diffLines.length && diffLines[i].type === 'added') {
        addedBlock.push({ line: diffLines[i], idx: i })
        i++
      }

      const maxLen = Math.max(removedBlock.length, addedBlock.length)
      for (let j = 0; j < maxLen; j++) {
        rows.push({
          left: j < removedBlock.length ? removedBlock[j].line : null,
          right: j < addedBlock.length ? addedBlock[j].line : null,
          originalIdx: j < addedBlock.length
            ? addedBlock[j].idx
            : j < removedBlock.length
              ? removedBlock[j].idx
              : i,
        })
      }
    }
    return rows
  }

  const renderUnifiedView = () => (
    <div className="inline-review-diff" style={baseStyles.diff}>
      {diffLines.map((line, idx) => {
        const threadedComments = line.number > 0 ? getThreadedCommentsForLine(line.number) : []
        return (
          <div key={idx}>
            <div
              className={`inline-diff-line diff-${line.type}`}
              style={diffLineStyle(line.type)}
              onClick={() => line.type !== 'hunk' && setActiveLineIdx(activeLineIdx === idx ? null : idx)}
            >
              <span className="line-number" style={baseStyles.lineNumber}>
                {line.type !== 'hunk' && line.type !== 'removed' ? line.number || '' : ''}
              </span>
              <span className="line-marker" style={baseStyles.lineMarker}>
                {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
              </span>
              <span className="line-content" style={baseStyles.lineContent}>{highlightSyntax(line.content || ' ')}</span>
              {threadedComments.length > 0 && (
                <span className="line-comment-indicator" style={baseStyles.commentIndicator} title={`${threadedComments.length} comment(s)`}>
                  {threadedComments.length}
                </span>
              )}
            </div>

            {threadedComments.map(c => renderComment(c))}

            {activeLineIdx === idx && (
              <div className="inline-comment-form" style={baseStyles.commentForm}>
                <textarea
                  className="inline-comment-input"
                  style={baseStyles.textarea}
                  placeholder="Write a review comment..."
                  value={commentBody}
                  onChange={(e) => setCommentBody(e.target.value)}
                  rows={2}
                  autoFocus
                />
                <div className="inline-comment-actions" style={baseStyles.commentActions}>
                  <button
                    className="inline-comment-submit"
                    style={{ ...baseStyles.miniBtn, ...baseStyles.miniBtnPrimary }}
                    onClick={() => handleAddComment(line.number || 0, line.type)}
                    disabled={submitting || !commentBody.trim()}
                  >
                    {submitting ? 'Adding...' : 'Add Comment'}
                  </button>
                  <button
                    className="inline-comment-cancel"
                    style={baseStyles.miniBtn}
                    onClick={() => { setActiveLineIdx(null); setCommentBody('') }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )

  const renderSplitView = () => {
    const splitRows = buildSplitRows()

    return (
      <div className="inline-review-diff inline-diff-split" style={baseStyles.diff}>
        {splitRows.map((row, rIdx) => {
          const lineForComments = row.right ?? row.left
          const lineNumber = lineForComments?.number ?? 0
          const threadedComments = lineNumber > 0 ? getThreadedCommentsForLine(lineNumber) : []
          const idx = row.originalIdx

          return (
            <div key={rIdx}>
              <div className="inline-diff-split-row" style={baseStyles.splitRow}>
                <div
                  className={`inline-diff-line inline-diff-split-cell ${row.left ? `diff-${row.left.type}` : 'diff-empty'}`}
                  style={{ ...baseStyles.splitCell, ...diffLineStyle(row.left ? row.left.type : 'empty') }}
                  onClick={() => row.left && row.left.type !== 'hunk' && setActiveLineIdx(activeLineIdx === idx ? null : idx)}
                >
                  {row.left ? (
                    <>
                      <span className="line-number" style={baseStyles.lineNumber}>
                        {row.left.type === 'removed' ? '' : row.left.type !== 'hunk' ? row.left.number || '' : ''}
                      </span>
                      <span className="line-marker" style={baseStyles.lineMarker}>
                        {row.left.type === 'removed' ? '-' : ' '}
                      </span>
                      <span className="line-content" style={baseStyles.lineContent}>{highlightSyntax(row.left.content || ' ')}</span>
                    </>
                  ) : (
                    <span className="line-content" style={baseStyles.lineContent}>&nbsp;</span>
                  )}
                </div>

                <div
                  className={`inline-diff-line inline-diff-split-cell ${row.right ? `diff-${row.right.type}` : 'diff-empty'}`}
                  style={{
                    ...baseStyles.splitCell,
                    ...diffLineStyle(row.right ? row.right.type : 'empty'),
                    borderLeft: `1px solid ${colors.border}`,
                  }}
                  onClick={() => row.right && row.right.type !== 'hunk' && setActiveLineIdx(activeLineIdx === idx ? null : idx)}
                >
                  {row.right ? (
                    <>
                      <span className="line-number" style={baseStyles.lineNumber}>
                        {row.right.type !== 'hunk' && row.right.type !== 'removed' ? row.right.number || '' : ''}
                      </span>
                      <span className="line-marker" style={baseStyles.lineMarker}>
                        {row.right.type === 'added' ? '+' : ' '}
                      </span>
                      <span className="line-content" style={baseStyles.lineContent}>{highlightSyntax(row.right.content || ' ')}</span>
                    </>
                  ) : (
                    <span className="line-content" style={baseStyles.lineContent}>&nbsp;</span>
                  )}
                </div>
              </div>

              {threadedComments.map(c => renderComment(c))}

              {activeLineIdx === idx && (
                <div className="inline-comment-form" style={baseStyles.commentForm}>
                  <textarea
                    className="inline-comment-input"
                    style={baseStyles.textarea}
                    placeholder="Write a review comment..."
                    value={commentBody}
                    onChange={(e) => setCommentBody(e.target.value)}
                    rows={2}
                    autoFocus
                  />
                  <div className="inline-comment-actions" style={baseStyles.commentActions}>
                    <button
                      className="inline-comment-submit"
                      style={{ ...baseStyles.miniBtn, ...baseStyles.miniBtnPrimary }}
                      onClick={() => handleAddComment(lineForComments?.number || 0, lineForComments?.type || 'context')}
                      disabled={submitting || !commentBody.trim()}
                    >
                      {submitting ? 'Adding...' : 'Add Comment'}
                    </button>
                    <button
                      className="inline-comment-cancel"
                      style={baseStyles.miniBtn}
                      onClick={() => { setActiveLineIdx(null); setCommentBody('') }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <div className="inline-review" style={baseStyles.review}>
      <div className="inline-review-header" style={baseStyles.header}>
        <code className="inline-review-path" style={baseStyles.path}>{filePath}</code>
        <div className="inline-review-header-actions" style={baseStyles.headerActions}>
          <button
            className="inline-review-view-toggle"
            style={baseStyles.toggleBtn}
            onClick={() => setViewMode(viewMode === 'unified' ? 'split' : 'unified')}
            title={viewMode === 'unified' ? 'Switch to split view' : 'Switch to unified view'}
          >
            {viewMode === 'unified' ? 'Split' : 'Unified'}
          </button>
          <span className="inline-review-comment-count" style={baseStyles.count}>
            {comments.filter(c => !c.resolved).length} open comments
          </span>
        </div>
      </div>

      {viewMode === 'unified' ? renderUnifiedView() : renderSplitView()}
    </div>
  )
}
