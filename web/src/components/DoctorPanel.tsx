import { useState } from 'react'
import { fetchDoctor } from '../api'
import './DoctorPanel.css'

interface Props {
  projectDir?: string
}

interface DoctorResult {
  checks: Record<string, { status: string; path?: string; command?: string; reason?: string }>
  warnings: string[]
  errors: string[]
  exit_code: number
}

const STATUS_ICON: Record<string, string> = {
  pass: '\u2705',
  fail: '\u274C',
  skip: '\u2796',
}

export default function DoctorPanel({ projectDir }: Props) {
  const [result, setResult] = useState<DoctorResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDoctor = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchDoctor(projectDir, true)
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Doctor check failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card doctor-panel">
      <div className="doctor-trigger">
        <h2>System Check</h2>
        <button className="btn btn-primary" onClick={handleDoctor} disabled={loading}>
          {loading ? 'Checking...' : 'Run Doctor'}
        </button>
      </div>

      {error && <div className="doctor-messages errors"><p>{error}</p></div>}

      {result && (
        <>
          <div className="doctor-checks">
            {Object.entries(result.checks).map(([name, check]) => (
              <div className="doctor-check" key={name}>
                <span className={`doctor-check-icon ${check.status}`}>
                  {STATUS_ICON[check.status] || '?'}
                </span>
                <span className="doctor-check-name">{name}</span>
                <span className="doctor-check-detail">
                  {check.path || check.command || check.reason || ''}
                </span>
              </div>
            ))}
          </div>

          {result.warnings.length > 0 && (
            <div className="doctor-messages warnings">
              <h4>Warnings</h4>
              <ul>{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            </div>
          )}

          {result.errors.length > 0 && (
            <div className="doctor-messages errors">
              <h4>Errors</h4>
              <ul>{result.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </div>
          )}

          <div className={`doctor-summary ${result.exit_code === 0 ? 'pass' : 'fail'}`}>
            {result.exit_code === 0 ? 'All checks passed' : `${result.errors.length} error(s) found`}
          </div>
        </>
      )}
    </div>
  )
}
