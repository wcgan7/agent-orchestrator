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
import { buildApiUrl, buildAuthHeaders } from '../api'
import { useChannel } from '../contexts/WebSocketContext'
import EmptyState from './EmptyState'
import './DependencyGraph.css'

interface Phase {
  id: string
  name: string
  description: string
  status: string
  deps: string[]
  progress: number
}

interface Props {
  projectDir?: string
}

export default function DependencyGraph({ projectDir }: Props) {
  const [phases, setPhases] = useState<Phase[]>([])
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])

  const normalizePhases = (value: unknown): Phase[] => {
    if (!Array.isArray(value)) return []
    const out: Phase[] = []
    for (const item of value) {
      if (!item || typeof item !== 'object') continue
      const raw = item as Record<string, unknown>
      const id = typeof raw.id === 'string' ? raw.id : ''
      if (!id) continue
      const depsRaw = raw.deps
      const deps = Array.isArray(depsRaw)
        ? depsRaw.map((d) => String(d)).filter(Boolean)
        : []
      const progress =
        typeof raw.progress === 'number' && Number.isFinite(raw.progress)
          ? raw.progress
          : 0
      out.push({
        id,
        name: typeof raw.name === 'string' ? raw.name : '',
        description: typeof raw.description === 'string' ? raw.description : '',
        status: typeof raw.status === 'string' ? raw.status : '',
        deps,
        progress,
      })
    }
    return out
  }

  useEffect(() => {
    fetchPhases()
  }, [projectDir])

  useChannel('phases', useCallback(() => {
    fetchPhases()
  }, [projectDir]))

  useEffect(() => {
    if (phases.length > 0) {
      buildGraph(phases)
    }
  }, [phases])

  const fetchPhases = async () => {
    try {
      const response = await fetch(buildApiUrl('/api/phases', projectDir), {
        headers: buildAuthHeaders(),
      })
      if (response.ok) {
        const data = await response.json()
        setPhases(normalizePhases(data))
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
          status: phase.status,
          label: (
            <div className="dependency-graph-node-label">
              <div className="dependency-graph-node-name">
                {phase.name || phase.id}
              </div>
              <div className="dependency-graph-node-status">
                {phase.status}
              </div>
              {phase.progress > 0 && (
                <div className="dependency-graph-node-progress">
                  <div
                    className="dependency-graph-node-progress-fill"
                    style={{ width: `${phase.progress * 100}%` }}
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
      const deps = Array.isArray(phase.deps) ? phase.deps : []
      deps.forEach((depId) => {
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
      const deps = phase && Array.isArray(phase.deps) ? phase.deps : []
      if (!phase || deps.length === 0) {
        levels[phaseId] = 0
        return 0
      }

      const maxDepth = Math.max(...deps.map((depId) => calculateDepth(depId, new Set(visited))))
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
    // These colors need to be hardcoded for ReactFlow nodes
    // They should match the CSS variables in variables.css
    switch (status.toLowerCase()) {
      case 'done':
      case 'completed':
        return '#22c55e' // --color-success-500
      case 'running':
      case 'in_progress':
        return '#3b82f6' // --color-primary-500
      case 'blocked':
      case 'failed':
        return '#ef4444' // --color-error-500
      case 'pending':
      case 'ready':
        return '#6b7280' // --color-gray-500
      default:
        return '#6b7280' // --color-gray-500
    }
  }

  if (phases.length === 0) {
    return (
      <div className="card">
        <h2>Phase Dependency Graph</h2>
        <EmptyState
          icon={<span>🔗</span>}
          title="No phases available"
          description="Dependency graph will appear once phases are defined"
          size="sm"
        />
      </div>
    )
  }

  return (
    <div className="card">
      <h2>Phase Dependency Graph</h2>
      <div className="dependency-graph-info">
        {phases.length} phases • Dependencies shown as arrows
      </div>

      <div className="dependency-graph-container">
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
              const status = (node.data as { status?: string }).status
              return getStatusColor(status || 'pending')
            }}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
        </ReactFlow>
      </div>

      <div className="dependency-graph-legend">
        <div className="dependency-graph-legend-items">
          <div className="dependency-graph-legend-item">
            <div className="dependency-graph-legend-color dependency-graph-legend-color-completed" />
            <span>Completed</span>
          </div>
          <div className="dependency-graph-legend-item">
            <div className="dependency-graph-legend-color dependency-graph-legend-color-running" />
            <span>Running</span>
          </div>
          <div className="dependency-graph-legend-item">
            <div className="dependency-graph-legend-color dependency-graph-legend-color-blocked" />
            <span>Blocked/Failed</span>
          </div>
          <div className="dependency-graph-legend-item">
            <div className="dependency-graph-legend-color dependency-graph-legend-color-pending" />
            <span>Pending/Ready</span>
          </div>
        </div>
        <div className="dependency-graph-legend-help">
          Use mouse wheel to zoom • Drag to pan • Click and drag nodes to reposition
        </div>
      </div>
    </div>
  )
}
