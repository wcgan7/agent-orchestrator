/**
 * AgentStream — renders the tail of an agent's output log.
 * Extracted from AgentCard for independent reuse.
 */

interface AgentStreamProps {
  outputTail: string[]
  maxLines?: number
}

export function AgentStream({ outputTail, maxLines = 10 }: AgentStreamProps) {
  if (outputTail.length === 0) {
    return null
  }

  return (
    <div className="agent-output">
      <div className="agent-output-label">Output</div>
      <pre className="agent-output-content">
        {outputTail.slice(-maxLines).join('\n')}
      </pre>
    </div>
  )
}
