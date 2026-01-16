import { useState, useEffect, useCallback } from 'react'
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Position,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

interface Phase {
  id: string
  name: string
  description: string
  status: string
  deps: string[]
  progress: number
}

export default function DependencyGraph() {
  const [phases, setPhases] = useState<Phase[]>([])
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  useEffect(() => {
    fetchPhases()
    const interval = setInterval(fetchPhases, 10000) // Poll every 10 seconds
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (phases.length > 0) {
      buildGraph(phases)
    }
  }, [phases])

  const fetchPhases = async () => {
    try {
      const response = await fetch('/api/phases')
      if (response.ok) {
        const data = await response.json()
        setPhases(data)
      }
    } catch (err) {
      console.error('Failed to fetch phases:', err)
    }
  }

  const buildGraph = (phaseList: Phase[]) => {
    // Calculate layout using a simple hierarchical approach
    const layout = calculateLayout(phaseList)

    // Create nodes
    const newNodes: Node[] = phaseList.map((phase) => {
      const position = layout[phase.id] || { x: 0, y: 0 }
      const statusColor = getStatusColor(phase.status)

      return {
        id: phase.id,
        type: 'default',
        position,
        data: {
          label: (
            <div style={{ textAlign: 'center', padding: '8px' }}>
              <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '4px' }}>
                {phase.name || phase.id}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#666' }}>
                {phase.status}
              </div>
              {phase.progress > 0 && (
                <div
                  style={{
                    marginTop: '4px',
                    height: '4px',
                    background: '#e0e0e0',
                    borderRadius: '2px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${phase.progress * 100}%`,
                      height: '100%',
                      background: statusColor,
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              )}
            </div>
          ),
        },
        style: {
          background: statusColor,
          color: '#fff',
          border: '2px solid rgba(0,0,0,0.1)',
          borderRadius: '8px',
          padding: 0,
          minWidth: '150px',
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      }
    })

    // Create edges from dependencies
    const newEdges: Edge[] = []
    phaseList.forEach((phase) => {
      phase.deps.forEach((depId) => {
        newEdges.push({
          id: `${depId}-${phase.id}`,
          source: depId,
          target: phase.id,
          type: 'smoothstep',
          animated: phase.status === 'running',
          style: { stroke: '#888', strokeWidth: 2 },
        })
      })
    })

    setNodes(newNodes)
    setEdges(newEdges)
  }

  const calculateLayout = (phaseList: Phase[]): Record<string, { x: number; y: number }> => {
    // Simple hierarchical layout based on dependency depth
    const layout: Record<string, { x: number; y: number }> = {}
    const levels: Record<string, number> = {}

    // Calculate depth for each phase
    const calculateDepth = (phaseId: string, visited = new Set<string>()): number => {
      if (levels[phaseId] !== undefined) {
        return levels[phaseId]
      }

      // Detect cycles
      if (visited.has(phaseId)) {
        return 0
      }

      visited.add(phaseId)

      const phase = phaseList.find((p) => p.id === phaseId)
      if (!phase || phase.deps.length === 0) {
        levels[phaseId] = 0
        return 0
      }

      const maxDepth = Math.max(...phase.deps.map((depId) => calculateDepth(depId, new Set(visited))))
      levels[phaseId] = maxDepth + 1
      return maxDepth + 1
    }

    // Calculate depth for all phases
    phaseList.forEach((phase) => {
      calculateDepth(phase.id)
    })

    // Group phases by level
    const levelGroups: Record<number, string[]> = {}
    Object.entries(levels).forEach(([phaseId, level]) => {
      if (!levelGroups[level]) {
        levelGroups[level] = []
      }
      levelGroups[level].push(phaseId)
    })

    // Position nodes
    const horizontalSpacing = 250
    const verticalSpacing = 120

    Object.entries(levelGroups).forEach(([level, phaseIds]) => {
      const levelNum = parseInt(level)
      phaseIds.forEach((phaseId, index) => {
        layout[phaseId] = {
          x: levelNum * horizontalSpacing,
          y: index * verticalSpacing + (levelNum % 2 === 0 ? 0 : 60), // Stagger alternate levels
        }
      })
    })

    return layout
  }

  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'done':
      case 'completed':
        return '#4caf50'
      case 'running':
      case 'in_progress':
        return '#2196f3'
      case 'blocked':
      case 'failed':
        return '#f44336'
      case 'pending':
      case 'ready':
        return '#9e9e9e'
      default:
        return '#757575'
    }
  }

  if (phases.length === 0) {
    return (
      <div className="card">
        <h2>Phase Dependency Graph</h2>
        <div className="empty-state">
          <p>No phases available to visualize</p>
          <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
            Dependency graph will appear once phases are defined
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <h2>Phase Dependency Graph</h2>
      <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.5rem' }}>
        {phases.length} phases • Dependencies shown as arrows
      </div>

      <div style={{ height: '500px', border: '1px solid #ddd', borderRadius: '4px', background: '#fafafa' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          attributionPosition="bottom-left"
        >
          <Background />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const statusMatch = node.data.label?.props?.children?.[1]?.props?.children
              return getStatusColor(statusMatch || 'pending')
            }}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
        </ReactFlow>
      </div>

      <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f5f5f5', borderRadius: '4px', fontSize: '0.75rem' }}>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: '16px', height: '16px', background: '#4caf50', borderRadius: '4px' }} />
            <span>Completed</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: '16px', height: '16px', background: '#2196f3', borderRadius: '4px' }} />
            <span>Running</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: '16px', height: '16px', background: '#f44336', borderRadius: '4px' }} />
            <span>Blocked/Failed</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: '16px', height: '16px', background: '#9e9e9e', borderRadius: '4px' }} />
            <span>Pending/Ready</span>
          </div>
        </div>
        <div style={{ marginTop: '0.5rem', color: '#666' }}>
          Use mouse wheel to zoom • Drag to pan • Click and drag nodes to reposition
        </div>
      </div>
    </div>
  )
}
