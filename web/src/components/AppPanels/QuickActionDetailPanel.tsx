import { humanizeLabel } from '../../ui/labels'

type QuickActionRecord = {
  id: string
  prompt: string
  status: string
  started_at?: string | null
  finished_at?: string | null
  result_summary?: string | null
  promoted_task_id?: string | null
  kind?: string | null
  command?: string | null
  exit_code?: number | null
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

function StatusIndicator({ status }: { status: string }): JSX.Element {
  if (status === 'queued' || status === 'running') {
    return <span className="badge" title={humanizeLabel(status)}>⟳ {humanizeLabel(status)}</span>
  }
  if (status === 'failed') {
    return <span className="badge badge-danger" title="Failed">✗ Failed</span>
  }
  return <span className="badge">{humanizeLabel(status)}</span>
}

function KindBadge({ kind }: { kind?: string | null }): JSX.Element | null {
  if (!kind) return null
  const label = kind === 'shortcut' ? 'Shortcut' : kind === 'agent' ? 'Agent' : kind
  return <span className="badge">{label}</span>
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
              <p className="task-meta">
                {action.id} · <StatusIndicator status={action.status} />
                {action.kind ? <> · <KindBadge kind={action.kind} /></> : null}
              </p>
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
              <p className="task-meta">Status: <StatusIndicator status={selectedQuickActionDetail.status} /></p>
              {selectedQuickActionDetail.kind ? (
                <p className="task-meta">Kind: <KindBadge kind={selectedQuickActionDetail.kind} /></p>
              ) : null}
              {selectedQuickActionDetail.command ? (
                <p className="task-meta">Command: <code>{selectedQuickActionDetail.command}</code></p>
              ) : null}
              <p className="task-meta">Started: {selectedQuickActionDetail.started_at || '-'}</p>
              <p className="task-meta">Finished: {selectedQuickActionDetail.finished_at || '-'}</p>
              {selectedQuickActionDetail.exit_code != null ? (
                <p className="task-meta">Exit code: {selectedQuickActionDetail.exit_code}</p>
              ) : null}
              {selectedQuickActionDetail.result_summary ? (
                <div>
                  <p className="field-label">Output</p>
                  <pre className="output-block">{selectedQuickActionDetail.result_summary}</pre>
                </div>
              ) : (
                <p className="task-meta">Result: -</p>
              )}
              <p className="task-meta">Promoted task: {selectedQuickActionDetail.promoted_task_id || '-'}</p>
              <button className="button" onClick={onRefreshQuickActionDetail}>Refresh detail</button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
