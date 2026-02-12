import { humanizeLabel } from '../../ui/labels'

type TaskExplorerItem = {
  id: string
  title: string
  status: string
  priority: string
}

type TaskExplorerPanelProps = {
  query: string
  status: string
  taskType: string
  priority: string
  onlyBlocked: boolean
  loading: boolean
  error: string
  items: TaskExplorerItem[]
  page: number
  pageSize: number
  totalItems: number
  statusOptions: string[]
  typeOptions: string[]
  onQueryChange: (value: string) => void
  onStatusChange: (value: string) => void
  onTypeChange: (value: string) => void
  onPriorityChange: (value: string) => void
  onOnlyBlockedChange: (value: boolean) => void
  onPageChange: (nextPage: number) => void
  onPageSizeChange: (nextPageSize: number) => void
  onSelectTask: (taskId: string) => void
  onRetry: () => void
}

export function TaskExplorerPanel({
  query,
  status,
  taskType,
  priority,
  onlyBlocked,
  loading,
  error,
  items,
  page,
  pageSize,
  totalItems,
  statusOptions,
  typeOptions,
  onQueryChange,
  onStatusChange,
  onTypeChange,
  onPriorityChange,
  onOnlyBlockedChange,
  onPageChange,
  onPageSizeChange,
  onSelectTask,
  onRetry,
}: TaskExplorerPanelProps): JSX.Element {
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize))
  const canPrev = page > 1
  const canNext = page < totalPages
  return (
    <>
      <p className="field-label">Task Explorer</p>
      <div className="inline-actions">
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="search by title/id"
          aria-label="Task explorer search"
        />
        <select value={status} onChange={(event) => onStatusChange(event.target.value)}>
          <option value="">All Statuses</option>
          {statusOptions.map((optionStatus) => (
            <option key={optionStatus} value={optionStatus}>{humanizeLabel(optionStatus)}</option>
          ))}
        </select>
        <select value={taskType} onChange={(event) => onTypeChange(event.target.value)}>
          <option value="">All Types</option>
          {typeOptions.map((optionType) => (
            <option key={optionType} value={optionType}>{humanizeLabel(optionType)}</option>
          ))}
        </select>
        <select value={priority} onChange={(event) => onPriorityChange(event.target.value)}>
          <option value="">All Priorities</option>
          <option value="P0">P0</option>
          <option value="P1">P1</option>
          <option value="P2">P2</option>
          <option value="P3">P3</option>
        </select>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={onlyBlocked}
            onChange={(event) => onOnlyBlockedChange(event.target.checked)}
          />
          Only blocked
        </label>
      </div>
      {loading ? <p className="field-label">Loading explorer...</p> : null}
      {error ? (
        <div className="inline-actions">
          <p className="error-banner">{error}</p>
          <button className="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}
      {items.map((task) => (
        <button key={`task-explorer-${task.id}`} className="task-card task-card-button" onClick={() => onSelectTask(task.id)}>
          <p className="task-title">{task.title}</p>
          <p className="task-meta">{humanizeLabel(task.status)} · {task.priority} · {task.id}</p>
        </button>
      ))}
      {!loading && !error && totalItems > 0 ? (
        <div className="inline-actions">
          <button className="button" onClick={() => onPageChange(page - 1)} disabled={!canPrev}>Prev</button>
          <p className="field-label">Page {page} of {totalPages}</p>
          <button className="button" onClick={() => onPageChange(page + 1)} disabled={!canNext}>Next</button>
          <label className="field-label" htmlFor="task-explorer-page-size">Per page</label>
          <select
            id="task-explorer-page-size"
            value={String(pageSize)}
            onChange={(event) => onPageSizeChange(Number(event.target.value) || 6)}
          >
            <option value="6">6</option>
            <option value="10">10</option>
            <option value="20">20</option>
          </select>
        </div>
      ) : null}
      {!loading && !error && items.length === 0 ? <p className="empty">No matching tasks.</p> : null}
    </>
  )
}
