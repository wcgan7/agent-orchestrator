import { useState, useEffect } from 'react'
import { fetchWorkers, testWorker } from '../api'
import './WorkersPanel.css'

interface Props {
  projectDir?: string
}

interface WorkerInfo {
  name: string
  type: string
  detail: string
  model?: string
  endpoint?: string
  command?: string
}

interface WorkersData {
  default_worker: string
  routing: Record<string, string>
  providers: WorkerInfo[]
  config_error?: string
}

interface TestResult {
  success: boolean
  message: string
}

export default function WorkersPanel({ projectDir }: Props) {
  const [data, setData] = useState<WorkersData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [testing, setTesting] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchWorkers(projectDir)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [projectDir])

  const handleTest = async (workerName: string) => {
    setTesting(workerName)
    try {
      const result = await testWorker(workerName, projectDir)
      setTestResults((prev) => ({ ...prev, [workerName]: result }))
    } catch (err: any) {
      setTestResults((prev) => ({
        ...prev,
        [workerName]: { success: false, message: err.message },
      }))
    } finally {
      setTesting(null)
    }
  }

  if (loading) return <div className="card workers-panel"><h2>Workers</h2><p>Loading...</p></div>
  if (error) return <div className="card workers-panel"><h2>Workers</h2><p className="workers-error">{error}</p></div>
  if (!data) return null

  return (
    <div className="card workers-panel">
      <h2>Workers</h2>

      {data.config_error && (
        <p className="workers-error">Config error: {data.config_error}</p>
      )}

      <div className="workers-default">
        Default worker: <strong>{data.default_worker}</strong>
      </div>

      {Object.keys(data.routing).length > 0 && (
        <div className="workers-routing">
          <h4>Routing</h4>
          <div className="workers-routing-table">
            {Object.entries(data.routing).map(([step, provider]) => (
              <div key={step} style={{ display: 'contents' }}>
                <span className="workers-routing-step">{step}</span>
                <span className="workers-routing-provider">{provider}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="workers-list">
        <h4>Providers</h4>
        {data.providers.map((p) => (
          <div className="workers-provider" key={p.name}>
            <div className="workers-provider-info">
              <div>
                <span className="workers-provider-name">{p.name}</span>
                <span className="workers-provider-type">{p.type}</span>
              </div>
              {p.detail && <div className="workers-provider-detail">{p.detail}</div>}
            </div>
            <button
              className="workers-test-btn"
              onClick={() => handleTest(p.name)}
              disabled={testing === p.name}
            >
              {testing === p.name ? 'Testing...' : 'Test'}
            </button>
            {testResults[p.name] && (
              <span className={`workers-test-result ${testResults[p.name].success ? 'success' : 'fail'}`}>
                {testResults[p.name].success ? '\u2713' : '\u2717'} {testResults[p.name].message}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
