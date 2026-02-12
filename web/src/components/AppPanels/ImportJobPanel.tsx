import { humanizeLabel } from '../../ui/labels'

type PreviewNode = {
  id: string
  title: string
  priority: string
}

type PreviewEdge = {
  from: string
  to: string
}

type PrdPreview = {
  nodes: PreviewNode[]
  edges: PreviewEdge[]
}

type ImportJobRecord = {
  id: string
  title?: string
  status?: string
  created_at?: string
  created_task_ids?: string[]
  tasks?: Array<{ title?: string; priority?: string }>
}

type ImportJobPanelProps = {
  importJobId: string
  importPreview: PrdPreview | null
  recentImportJobIds: string[]
  selectedImportJobId: string
  selectedImportJob: ImportJobRecord | null
  selectedImportJobLoading: boolean
  selectedImportJobError: string
  selectedCreatedTaskIds: string[]
  onCommitImport: () => void
  onSelectImportJob: (jobId: string) => void
  onRefreshImportJob: () => void
  onRetryLoadImportJob: () => void
}

export function ImportJobPanel({
  importJobId,
  importPreview,
  recentImportJobIds,
  selectedImportJobId,
  selectedImportJob,
  selectedImportJobLoading,
  selectedImportJobError,
  selectedCreatedTaskIds,
  onCommitImport,
  onSelectImportJob,
  onRefreshImportJob,
  onRetryLoadImportJob,
}: ImportJobPanelProps): JSX.Element {
  return (
    <div className="form-stack">
      {importJobId ? (
        <div className="preview-box">
          <p>Preview ready: {importJobId}</p>
          <div className="inline-actions">
            <button className="button" onClick={() => onSelectImportJob(importJobId)}>View details</button>
          </div>
          {importPreview ? (
            <div className="import-preview-graph">
              <p className="field-label">Preview Graph</p>
              {importPreview.nodes.map((node) => (
                <div key={node.id} className="import-node">
                  <span>{node.id}</span>
                  <span>{node.title}</span>
                  <span>{humanizeLabel(node.priority)}</span>
                </div>
              ))}
              {importPreview.edges.length > 0 ? (
                <p className="field-label">
                  Edges: {importPreview.edges.map((edge) => `${edge.from} -> ${edge.to}`).join(', ')}
                </p>
              ) : (
                <p className="field-label">Edges: none</p>
              )}
            </div>
          ) : null}
          <button className="button button-primary" onClick={onCommitImport}>Commit to board</button>
        </div>
      ) : null}
      <div className="list-stack">
        <p className="field-label">Recent import jobs</p>
        {recentImportJobIds.map((jobId) => (
          <div className="row-card" key={jobId}>
            <p className="task-meta">{jobId}</p>
            <button className="button" onClick={() => onSelectImportJob(jobId)}>Open</button>
          </div>
        ))}
        {recentImportJobIds.length === 0 ? <p className="empty">No import jobs yet.</p> : null}
      </div>
      {selectedImportJobId ? (
        <div className="preview-box">
          <p className="field-label">Import job detail</p>
          {selectedImportJobLoading ? <p>Loading job...</p> : null}
          {selectedImportJobError ? (
            <div className="inline-actions">
              <p className="error-banner">{selectedImportJobError}</p>
              <button className="button" onClick={onRetryLoadImportJob}>Retry</button>
            </div>
          ) : null}
          {selectedImportJob ? (
            <div className="form-stack">
              <p className="task-meta">ID: {selectedImportJob.id}</p>
              <p className="task-meta">Status: {selectedImportJob.status ? humanizeLabel(selectedImportJob.status) : '-'}</p>
              <p className="task-meta">Title: {selectedImportJob.title || '-'}</p>
              <p className="task-meta">Created: {selectedImportJob.created_at || '-'}</p>
              <p className="task-meta">Tasks: {(selectedImportJob.tasks || []).length}</p>
              <p className="task-meta">Created task IDs: {selectedCreatedTaskIds.join(', ') || '-'}</p>
              <button className="button" onClick={onRefreshImportJob}>Refresh job</button>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
