/**
 * Inline code review — line-level commenting on diffs, similar to GitHub PR reviews.
 */

import { useState, useCallback, useEffect } from 'react'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import './InlineReview.css'

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

/* -------------------------------------------------------------------------- */
/*  Syntax highlighting                                                       */
/* -------------------------------------------------------------------------- */

const KEYWORDS = new Set([
  'function', 'const', 'let', 'var', 'class', 'import', 'export', 'return',
  'if', 'else', 'for', 'while', 'try', 'catch', 'async', 'await', 'def',
  'self', 'from',
])

/**
 * Very basic keyword + string highlighting.
 * Returns an array of React nodes so we can embed <span> elements.
 */
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

    // --- strings (single / double quote) ---
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
        <span key={`s${key++}`} className="syntax-string">{str}</span>
      )
      continue
    }

    // --- word boundary: check for keyword ---
    if (/[a-zA-Z_]/.test(ch)) {
      let word = ''
      while (i < text.length && /[a-zA-Z0-9_]/.test(text[i])) {
        word += text[i]
        i++
      }
      // Only highlight if it is a standalone keyword (not part of a larger identifier)
      if (KEYWORDS.has(word)) {
        flush()
        nodes.push(
          <span key={`k${key++}`} className="syntax-keyword">{word}</span>
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

/* -------------------------------------------------------------------------- */
/*  Thread grouping helper                                                    */
/* -------------------------------------------------------------------------- */

function groupCommentsIntoThreads(flat: InlineComment[]): InlineComment[] {
  const map = new Map<string, InlineComment>()
  const roots: InlineComment[] = []

  // First pass: clone comments and initialise replies array
  for (const c of flat) {
    map.set(c.id, { ...c, replies: [] })
  }

  // Second pass: nest children under parents
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

/* -------------------------------------------------------------------------- */
/*  Component                                                                 */
/* -------------------------------------------------------------------------- */

export default function InlineReview({ taskId, filePath, diff, projectDir, onCommentAdded }: Props) {
  const [comments, setComments] = useState<InlineComment[]>([])
  const [activeLineIdx, setActiveLineIdx] = useState<number | null>(null)
  const [commentBody, setCommentBody] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('unified')
  const [replyingTo, setReplyingTo] = useState<string | null>(null)
  const [replyBody, setReplyBody] = useState('')

  const diffLines = parseDiff(diff)

  /* ---- data fetching ---- */

  const fetchComments = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl(`/api/v2/collaboration/comments/${taskId}`, projectDir, { file_path: filePath }),
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

  /* ---- add a top-level comment ---- */

  const handleAddComment = async (lineNumber: number, lineType: string) => {
    if (!commentBody.trim()) return
    setSubmitting(true)
    try {
      await fetch(
        buildApiUrl('/api/v2/collaboration/comments', projectDir),
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

  /* ---- add a reply ---- */

  const handleAddReply = async (parentComment: InlineComment) => {
    if (!replyBody.trim()) return
    setSubmitting(true)
    try {
      await fetch(
        buildApiUrl('/api/v2/collaboration/comments', projectDir),
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

  /* ---- resolve ---- */

  const handleResolve = async (commentId: string) => {
    await fetch(
      buildApiUrl(`/api/v2/collaboration/comments/${commentId}/resolve`, projectDir),
      { method: 'POST', headers: buildAuthHeaders() }
    )
    fetchComments()
  }

  /* ---- helpers ---- */

  const getCommentsForLine = (lineNumber: number) =>
    comments.filter(c => c.line_number === lineNumber)

  const getThreadedCommentsForLine = (lineNumber: number) =>
    groupCommentsIntoThreads(getCommentsForLine(lineNumber))

  /* ---- render a single comment (recursive for replies) ---- */

  const renderComment = (c: InlineComment) => (
    <div key={c.id}>
      <div className={`inline-comment ${c.resolved ? 'resolved' : ''}`}>
        <div className="inline-comment-header">
          <span className="inline-comment-author">{c.author || 'user'}</span>
          <span className="inline-comment-time">
            {new Date(c.created_at).toLocaleTimeString()}
          </span>
          {!c.resolved && (
            <button
              className="inline-comment-resolve"
              onClick={(e) => { e.stopPropagation(); handleResolve(c.id) }}
            >
              Resolve
            </button>
          )}
        </div>
        <div className="inline-comment-body">{c.body}</div>
        <button
          className="inline-comment-reply"
          onClick={(e) => {
            e.stopPropagation()
            setReplyingTo(replyingTo === c.id ? null : c.id)
            setReplyBody('')
          }}
        >
          Reply
        </button>
      </div>

      {/* Reply form */}
      {replyingTo === c.id && (
        <div className="inline-comment-form" style={{ marginLeft: 24 }}>
          <textarea
            className="inline-comment-input"
            placeholder="Write a reply..."
            value={replyBody}
            onChange={(e) => setReplyBody(e.target.value)}
            rows={2}
            autoFocus
          />
          <div className="inline-comment-actions">
            <button
              className="inline-comment-submit"
              onClick={() => handleAddReply(c)}
              disabled={submitting || !replyBody.trim()}
            >
              {submitting ? 'Adding...' : 'Reply'}
            </button>
            <button
              className="inline-comment-cancel"
              onClick={() => { setReplyingTo(null); setReplyBody('') }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Nested replies */}
      {c.replies && c.replies.length > 0 && (
        <div className="inline-comment-replies">
          {c.replies.map(reply => renderComment(reply))}
        </div>
      )}
    </div>
  )

  /* ---- split-view helpers ---- */

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

      // Collect consecutive removed / added blocks and pair them
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

  /* ==================================================================== */
  /*  Unified view                                                        */
  /* ==================================================================== */

  const renderUnifiedView = () => (
    <div className="inline-review-diff">
      {diffLines.map((line, idx) => {
        const threadedComments = line.number > 0 ? getThreadedCommentsForLine(line.number) : []
        return (
          <div key={idx}>
            <div
              className={`inline-diff-line diff-${line.type}`}
              onClick={() => line.type !== 'hunk' && setActiveLineIdx(activeLineIdx === idx ? null : idx)}
            >
              <span className="line-number">
                {line.type !== 'hunk' && line.type !== 'removed' ? line.number || '' : ''}
              </span>
              <span className="line-marker">
                {line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' '}
              </span>
              <span className="line-content">{highlightSyntax(line.content || ' ')}</span>
              {threadedComments.length > 0 && (
                <span className="line-comment-indicator" title={`${threadedComments.length} comment(s)`}>
                  {threadedComments.length}
                </span>
              )}
            </div>

            {/* Show existing threaded comments */}
            {threadedComments.map(c => renderComment(c))}

            {/* Comment form */}
            {activeLineIdx === idx && (
              <div className="inline-comment-form">
                <textarea
                  className="inline-comment-input"
                  placeholder="Write a review comment..."
                  value={commentBody}
                  onChange={(e) => setCommentBody(e.target.value)}
                  rows={2}
                  autoFocus
                />
                <div className="inline-comment-actions">
                  <button
                    className="inline-comment-submit"
                    onClick={() => handleAddComment(line.number || 0, line.type)}
                    disabled={submitting || !commentBody.trim()}
                  >
                    {submitting ? 'Adding...' : 'Add Comment'}
                  </button>
                  <button
                    className="inline-comment-cancel"
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

  /* ==================================================================== */
  /*  Split view                                                          */
  /* ==================================================================== */

  const renderSplitView = () => {
    const splitRows = buildSplitRows()

    return (
      <div className="inline-review-diff inline-diff-split">
        {splitRows.map((row, rIdx) => {
          const lineForComments = row.right ?? row.left
          const lineNumber = lineForComments?.number ?? 0
          const threadedComments = lineNumber > 0 ? getThreadedCommentsForLine(lineNumber) : []
          const idx = row.originalIdx

          return (
            <div key={rIdx}>
              <div className="inline-diff-split-row">
                {/* Left column (removed / context) */}
                <div
                  className={`inline-diff-line inline-diff-split-cell ${row.left ? `diff-${row.left.type}` : 'diff-empty'}`}
                  onClick={() => row.left && row.left.type !== 'hunk' && setActiveLineIdx(activeLineIdx === idx ? null : idx)}
                >
                  {row.left ? (
                    <>
                      <span className="line-number">
                        {row.left.type === 'removed' ? '' : row.left.type !== 'hunk' ? row.left.number || '' : ''}
                      </span>
                      <span className="line-marker">
                        {row.left.type === 'removed' ? '-' : ' '}
                      </span>
                      <span className="line-content">{highlightSyntax(row.left.content || ' ')}</span>
                    </>
                  ) : (
                    <span className="line-content">&nbsp;</span>
                  )}
                </div>

                {/* Right column (added / context) */}
                <div
                  className={`inline-diff-line inline-diff-split-cell ${row.right ? `diff-${row.right.type}` : 'diff-empty'}`}
                  onClick={() => row.right && row.right.type !== 'hunk' && setActiveLineIdx(activeLineIdx === idx ? null : idx)}
                >
                  {row.right ? (
                    <>
                      <span className="line-number">
                        {row.right.type !== 'hunk' && row.right.type !== 'removed' ? row.right.number || '' : ''}
                      </span>
                      <span className="line-marker">
                        {row.right.type === 'added' ? '+' : ' '}
                      </span>
                      <span className="line-content">{highlightSyntax(row.right.content || ' ')}</span>
                    </>
                  ) : (
                    <span className="line-content">&nbsp;</span>
                  )}
                </div>
              </div>

              {/* Show existing threaded comments */}
              {threadedComments.map(c => renderComment(c))}

              {/* Comment form */}
              {activeLineIdx === idx && (
                <div className="inline-comment-form">
                  <textarea
                    className="inline-comment-input"
                    placeholder="Write a review comment..."
                    value={commentBody}
                    onChange={(e) => setCommentBody(e.target.value)}
                    rows={2}
                    autoFocus
                  />
                  <div className="inline-comment-actions">
                    <button
                      className="inline-comment-submit"
                      onClick={() => handleAddComment(
                        lineForComments?.number || 0,
                        lineForComments?.type || 'context'
                      )}
                      disabled={submitting || !commentBody.trim()}
                    >
                      {submitting ? 'Adding...' : 'Add Comment'}
                    </button>
                    <button
                      className="inline-comment-cancel"
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

  /* ==================================================================== */
  /*  Main render                                                         */
  /* ==================================================================== */

  return (
    <div className="inline-review">
      <div className="inline-review-header">
        <code className="inline-review-path">{filePath}</code>
        <div className="inline-review-header-actions">
          <button
            className="inline-review-view-toggle"
            onClick={() => setViewMode(viewMode === 'unified' ? 'split' : 'unified')}
            title={viewMode === 'unified' ? 'Switch to split view' : 'Switch to unified view'}
          >
            {viewMode === 'unified' ? 'Split' : 'Unified'}
          </button>
          <span className="inline-review-comment-count">
            {comments.filter(c => !c.resolved).length} open comments
          </span>
        </div>
      </div>

      {viewMode === 'unified' ? renderUnifiedView() : renderSplitView()}
    </div>
  )
}
