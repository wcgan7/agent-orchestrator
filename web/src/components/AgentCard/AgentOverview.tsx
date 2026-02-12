/**
 * Agent overview grid — aggregate stats across all agents.
 *
 * Shows summary cards (total agents, running, idle, errored, total cost,
 * total tokens) and a per-type breakdown row. Designed to sit above or
 * alongside the individual AgentCard list.
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  Box,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
import { buildApiUrl, buildAuthHeaders } from '../../api'
import { useChannel } from '../../contexts/WebSocketContext'

interface AgentData {
  id: string
  agent_type: string
  display_name: string
  status: string
  task_id: string | null
  tokens_used: number
  cost_usd: number
  elapsed_seconds: number
}

interface Props {
  projectDir?: string
}

export function AgentOverview({ projectDir }: Props) {
  const [agents, setAgents] = useState<AgentData[]>([])

  const fetchAgents = useCallback(async () => {
    try {
      const resp = await fetch(
        buildApiUrl('/api/v3/agents', projectDir),
        { headers: buildAuthHeaders() }
      )
      if (resp.ok) {
        const data = await resp.json()
        setAgents(data.agents || [])
      }
    } catch {
      // retry on next cycle
    }
  }, [projectDir])

  useEffect(() => {
    fetchAgents()
  }, [fetchAgents])

  useChannel('agents', useCallback((_event: string, _data: any) => {
    fetchAgents()
  }, [fetchAgents]))

  const stats = useMemo(() => {
    const total = agents.length
    const running = agents.filter(a => a.status === 'running').length
    const paused = agents.filter(a => a.status === 'paused').length
    const idle = agents.filter(a => a.status === 'idle').length
    const failed = agents.filter(a => a.status === 'failed').length
    const totalTokens = agents.reduce((sum, a) => sum + a.tokens_used, 0)
    const totalCost = agents.reduce((sum, a) => sum + a.cost_usd, 0)
    const totalTime = agents.reduce((sum, a) => sum + a.elapsed_seconds, 0)

    return { total, running, paused, idle, failed, totalTokens, totalCost, totalTime }
  }, [agents])

  const byType = useMemo(() => {
    const groups: Record<string, { count: number; running: number; tokens: number; cost: number }> = {}
    for (const a of agents) {
      if (!groups[a.agent_type]) {
        groups[a.agent_type] = { count: 0, running: 0, tokens: 0, cost: 0 }
      }
      groups[a.agent_type].count++
      if (a.status === 'running') groups[a.agent_type].running++
      groups[a.agent_type].tokens += a.tokens_used
      groups[a.agent_type].cost += a.cost_usd
    }
    return groups
  }, [agents])

  const formatTokens = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : `${(n / 1000).toFixed(0)}k`

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.floor(seconds)}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 1.5 }}>Agent Overview</Typography>

      {/* Summary cards */}
      <Grid container spacing={1.25} sx={{ mb: 2 }}>
        {[
          { label: 'Total Agents', value: stats.total, color: 'text.primary' },
          { label: 'Running', value: stats.running, color: 'success.main' },
          { label: 'Idle', value: stats.idle, color: 'text.secondary' },
          { label: 'Failed', value: stats.failed, color: 'error.main' },
          { label: 'Total Tokens', value: formatTokens(stats.totalTokens), color: 'text.primary' },
          { label: 'Total Cost', value: `$${stats.totalCost.toFixed(2)}`, color: 'text.primary' },
        ].map((item) => (
          <Grid key={item.label} size={{ xs: 6, md: 4, lg: 2 }}>
            <Card variant="outlined">
              <CardContent sx={{ py: 1.25, '&:last-child': { pb: 1.25 }, textAlign: 'center' }}>
                <Typography variant="h6" sx={{ color: item.color }}>{item.value}</Typography>
                <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>
                  {item.label}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Per-type breakdown */}
      {Object.keys(byType).length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
            By Type
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Role</TableCell>
                <TableCell>Count</TableCell>
                <TableCell>Running</TableCell>
                <TableCell>Tokens</TableCell>
                <TableCell>Cost</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(byType)
                .sort(([, a], [, b]) => b.cost - a.cost)
                .map(([type, data]) => (
                  <TableRow key={type}>
                    <TableCell sx={{ fontWeight: 500, textTransform: 'capitalize' }}>{type}</TableCell>
                    <TableCell>{data.count}</TableCell>
                    <TableCell>{data.running}</TableCell>
                    <TableCell>{formatTokens(data.tokens)}</TableCell>
                    <TableCell>${data.cost.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </Box>
      )}

      {/* Empty state */}
      {agents.length === 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          No agents in pool. Spawn agents to see overview stats.
        </Typography>
      )}

      {/* Uptime */}
      {stats.totalTime > 0 && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right' }}>
          Total agent uptime: {formatTime(stats.totalTime)}
        </Typography>
      )}
    </Box>
  )
}
