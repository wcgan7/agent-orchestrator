import { useState, useEffect, useCallback } from 'react'
import {
  Alert,
  Box,
  Chip,
  Stack,
  Typography,
} from '@mui/material'
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
      const response = await fetch(buildApiUrl('/api/v3/phases', projectDir), {
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
    const layout = calculateLayout(phaseList)

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
            <div style={{ textAlign: 'center', padding: '8px' }}>
              <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '4px' }}>{phase.name || phase.id}</div>
              <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.85)' }}>{phase.status}</div>
              {phase.progress > 0 && (
                <div
                  style={{
                    marginTop: '4px',
                    height: '4px',
                    background: 'rgba(255,255,255,0.3)',
                    borderRadius: '4px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${phase.progress * 100}%`,
                      height: '100%',
                      background: 'rgba(255,255,255,0.9)',
                      transition: 'width 200ms ease',
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
    const layout: Record<string, { x: number; y: number }> = {}
    const levels: Record<string, number> = {}

    const calculateDepth = (phaseId: string, visited = new Set<string>()): number => {
      if (levels[phaseId] !== undefined) return levels[phaseId]
      if (visited.has(phaseId)) return 0

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

    phaseList.forEach((phase) => calculateDepth(phase.id))

    const levelGroups: Record<number, string[]> = {}
    Object.entries(levels).forEach(([phaseId, level]) => {
      if (!levelGroups[level]) levelGroups[level] = []
      levelGroups[level].push(phaseId)
    })

    const horizontalSpacing = 250
    const verticalSpacing = 120

    Object.entries(levelGroups).forEach(([level, phaseIds]) => {
      const levelNum = parseInt(level)
      phaseIds.forEach((phaseId, index) => {
        layout[phaseId] = {
          x: levelNum * horizontalSpacing,
          y: index * verticalSpacing + (levelNum % 2 === 0 ? 0 : 60),
        }
      })
    })

    return layout
  }

  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'done':
      case 'completed':
        return '#22c55e'
      case 'running':
      case 'in_progress':
        return '#3b82f6'
      case 'blocked':
      case 'failed':
        return '#ef4444'
      case 'pending':
      case 'ready':
        return '#6b7280'
      default:
        return '#6b7280'
    }
  }

  if (phases.length === 0) {
    return (
      <Box>
        <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1.5 }}>Phase Dependency Graph</Typography>
        <EmptyState
          icon={<span>🔗</span>}
          title="No phases available"
          description="Dependency graph will appear once phases are defined"
          size="sm"
        />
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h2" sx={{ fontSize: '1.125rem', mb: 1 }}>Phase Dependency Graph</Typography>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
        <Chip size="small" variant="outlined" label={`${phases.length} phases`} />
        <Typography variant="caption" color="text.secondary">
          Dependencies shown as arrows
        </Typography>
      </Stack>

      <Box sx={{ height: 500, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'background.default' }}>
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
      </Box>

      <Box sx={{ mt: 1.5, p: 1.25, bgcolor: 'background.default', borderRadius: 2 }}>
        <Stack direction="row" spacing={1.5} useFlexGap flexWrap="wrap">
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 16, height: 16, borderRadius: 0.75, bgcolor: 'success.main' }} />
            <Typography variant="caption">Completed</Typography>
          </Stack>
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 16, height: 16, borderRadius: 0.75, bgcolor: 'info.main' }} />
            <Typography variant="caption">Running</Typography>
          </Stack>
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 16, height: 16, borderRadius: 0.75, bgcolor: 'error.main' }} />
            <Typography variant="caption">Blocked/Failed</Typography>
          </Stack>
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 16, height: 16, borderRadius: 0.75, bgcolor: 'text.disabled' }} />
            <Typography variant="caption">Pending/Ready</Typography>
          </Stack>
        </Stack>
        <Alert severity="info" sx={{ mt: 1 }}>
          Use mouse wheel to zoom, drag to pan, and drag nodes to reposition.
        </Alert>
      </Box>
    </Box>
  )
}
