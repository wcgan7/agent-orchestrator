/**
 * AgentControls — lifecycle control buttons (pause / resume / terminate).
 * Extracted from AgentCard for independent reuse.
 */

interface AgentControlsProps {
  agentId: string
  status: string
  onAction: (agentId: string, action: string) => void
}

export function AgentControls({ agentId, status, onAction }: AgentControlsProps) {
  return (
    <div className="agent-card-actions">
      {status === 'running' && (
        <button onClick={() => onAction(agentId, 'pause')}>Pause</button>
      )}
      {status === 'paused' && (
        <button onClick={() => onAction(agentId, 'resume')}>Resume</button>
      )}
      {status !== 'terminated' && (
        <button className="btn-danger" onClick={() => onAction(agentId, 'terminate')}>
          Terminate
        </button>
      )}
    </div>
  )
}
