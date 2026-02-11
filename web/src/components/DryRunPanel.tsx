import { useState } from 'react'
import { fetchDryRun } from '../api'
import './DryRunPanel.css'

interface Props {
  projectDir?: string
}

interface DryRunResult {
  project_dir: string
  state_dir: string
  would_write_repo_files: boolean
  would_spawn_codex: boolean
  would_run_tests: boolean
  would_checkout_branch: boolean
  next: Record<string, any> | null
  warnings: string[]
  errors: string[]
}

export default function DryRunPanel({ projectDir }: Props) {
  const [result, setResult] = useState<DryRunResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDryRun = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchDryRun(projectDir)
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Dry run failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card dry-run-panel">
      <div className="dry-run-trigger">
        <h2>Preview Next Action</h2>
        <button className="btn btn-primary" onClick={handleDryRun} disabled={loading}>
          {loading ? 'Checking...' : 'Dry Run'}
        </button>
      </div>

      {error && <div className="dry-run-errors"><p>{error}</p></div>}

      {result && (
        <div className="dry-run-result">
          {result.next && (
            <div className="dry-run-next">
              <h4>Next Action</h4>
              <div className="dry-run-next-card">
                <div><strong>{result.next.action}</strong></div>
                {result.next.task_id && <div>Task: <code>{result.next.task_id}</code></div>}
                {result.next.step && <div>Step: <code>{result.next.step}</code></div>}
                {result.next.description && <div>{result.next.description}</div>}
              </div>
            </div>
          )}

          <div className="dry-run-flags">
            <span className={`dry-run-flag ${result.would_spawn_codex ? 'active' : 'inactive'}`}>
              {result.would_spawn_codex ? '\u2713' : '\u2717'} Spawn Codex
            </span>
            <span className={`dry-run-flag ${result.would_run_tests ? 'active' : 'inactive'}`}>
              {result.would_run_tests ? '\u2713' : '\u2717'} Run Tests
            </span>
            <span className={`dry-run-flag ${result.would_checkout_branch ? 'active' : 'inactive'}`}>
              {result.would_checkout_branch ? '\u2713' : '\u2717'} Checkout Branch
            </span>
          </div>

          {result.warnings.length > 0 && (
            <div className="dry-run-warnings">
              <h4>Warnings</h4>
              <ul>{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            </div>
          )}

          {result.errors.length > 0 && (
            <div className="dry-run-errors">
              <h4>Errors</h4>
              <ul>{result.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
