import { humanizeLabel } from '../../ui/labels'

type QuickActionRecord = {
  id: string
  prompt: string
  status: string
  started_at?: string | null
  finished_at?: string | null
  result_summary?: string | null
  promoted_task_id?: string | null
}

type QuickActionDetailPanelProps = {
  quickActions: QuickActionRecord[]
  selectedQuickActionId: string
  selectedQuickActionDetail: QuickActionRecord | null
  selectedQuickActionLoading: boolean
  selectedQuickActionError: string
  onSelectQuickAction: (quickActionId: string) => void
  onPromoteQuickAction: (quickActionId: string) => void
  onRefreshQuickActionDetail: () => void
  onRetryLoadQuickActionDetail: () => void
}

export function QuickActionDetailPanel({
  quickActions,
  selectedQuickActionId,
  selectedQuickActionDetail,
  selectedQuickActionLoading,
  selectedQuickActionError,
  onSelectQuickAction,
  onPromoteQuickAction,
  onRefreshQuickActionDetail,
  onRetryLoadQuickActionDetail,
}: QuickActionDetailPanelProps): JSX.Element {
  return (
    <div className="form-stack">
      <div className="list-stack">
        <p className="field-label">Recent quick actions</p>
        {quickActions.slice(0, 6).map((action) => (
          <div className="row-card" key={action.id}>
            <div>
              <p className="task-title">{action.prompt}</p>
              <p className="task-meta">{action.id} · {humanizeLabel(action.status)}</p>
            </div>
            <div className="inline-actions">
              <button className="button" onClick={() => onSelectQuickAction(action.id)}>View</button>
              <button
                className="button"
                onClick={() => onPromoteQuickAction(action.id)}
                disabled={Boolean(action.promoted_task_id)}
              >
                {action.promoted_task_id ? 'Promoted' : 'Promote'}
              </button>
            </div>
          </div>
        ))}
        {quickActions.length === 0 ? <p className="empty">No quick actions yet.</p> : null}
      </div>
      {selectedQuickActionId ? (
        <div className="preview-box">
          <p className="field-label">Quick action detail</p>
          {selectedQuickActionLoading ? <p>Loading quick action...</p> : null}
          {selectedQuickActionError ? (
            <div className="inline-actions">
              <p className="error-banner">{selectedQuickActionError}</p>
              <button className="button" onClick={onRetryLoadQuickActionDetail}>Retry</button>
            </div>
          ) : null}
          {selectedQuickActionDetail ? (
            <div className="form-stack">
              <p className="task-meta">ID: {selectedQuickActionDetail.id}</p>
              <p className="task-meta">Status: {humanizeLabel(selectedQuickActionDetail.status)}</p>
              <p className="task-meta">Started: {selectedQuickActionDetail.started_at || '-'}</p>
              <p className="task-meta">Finished: {selectedQuickActionDetail.finished_at || '-'}</p>
              <p className="task-meta">Result: {selectedQuickActionDetail.result_summary || '-'}</p>
              <p className="task-meta">Promoted task: {selectedQuickActionDetail.promoted_task_id || '-'}</p>
              <button className="button" onClick={onRefreshQuickActionDetail}>Refresh detail</button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
