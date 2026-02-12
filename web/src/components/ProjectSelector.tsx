import { useState, useEffect } from 'react'
import { Box, Chip, Paper, Typography } from '@mui/material'

interface Project {
  name: string
  path: string
  status: string
  last_run: string | null
  phases_total: number
  phases_completed: number
}

interface Props {
  currentProject: string | null
  onProjectChange: (projectPath: string) => void
}

export default function ProjectSelector({ currentProject, onProjectChange }: Props) {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [showDropdown, setShowDropdown] = useState(false)

  useEffect(() => {
    fetchProjects()
  }, [])

  const fetchProjects = async () => {
    try {
      const response = await fetch('/api/v3/projects')
      if (response.ok) {
        const data = await response.json()
        setProjects(Array.isArray(data) ? data : [])
      }
    } catch (err) {
      console.error('Failed to fetch projects:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleProjectSelect = (projectPath: string) => {
    onProjectChange(projectPath)
    setShowDropdown(false)
  }

  const getCurrentProjectName = () => {
    if (!currentProject) return 'No project selected'
    const project = projects.find((p) => p.path === currentProject)
    return project ? project.name : currentProject.split('/').pop() || 'Unknown'
  }

  const getStatusClass = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'active'
      case 'idle':
        return 'idle'
      case 'error':
        return 'error'
      default:
        return 'default'
    }
  }

  if (loading) {
    return (
      <Typography variant="body2" sx={{ px: 1, color: 'text.secondary' }}>
        Loading projects...
      </Typography>
    )
  }

  if (projects.length === 0) {
    return (
      <Typography variant="body2" sx={{ px: 1, color: 'text.secondary' }}>
        No projects found
      </Typography>
    )
  }

  return (
    <Box sx={{ position: 'relative' }}>
      <Box
        component="button"
        onClick={() => setShowDropdown(!showDropdown)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1.5,
          py: 0.75,
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
          bgcolor: 'background.paper',
          cursor: 'pointer',
          fontSize: '0.875rem',
          fontWeight: 500,
          color: 'text.primary',
          transition: 'background-color 0.2s ease, border-color 0.2s ease',
          '&:hover': {
            bgcolor: 'action.hover',
            borderColor: 'text.secondary',
          },
        }}
      >
        <Box component="span" sx={{ fontSize: '1rem' }}>📁</Box>
        <Box component="span">{getCurrentProjectName()}</Box>
        <Box component="span" sx={{ ml: 1, fontSize: '0.75rem' }}>▼</Box>
      </Box>

      {showDropdown && (
        <>
          <Box
            onClick={() => setShowDropdown(false)}
            sx={{
              position: 'fixed',
              inset: 0,
              zIndex: (theme) => theme.zIndex.modal - 1,
            }}
          />

          <Paper
            elevation={12}
            sx={{
              position: 'absolute',
              top: 'calc(100% + 6px)',
              left: 0,
              minWidth: 300,
              maxWidth: 500,
              maxHeight: 400,
              overflowY: 'auto',
              border: 1,
              borderColor: 'divider',
              zIndex: (theme) => theme.zIndex.modal,
            }}
          >
            {projects.map((project) => (
              <Box
                key={project.path}
                onClick={() => handleProjectSelect(project.path)}
                sx={{
                  p: 1.5,
                  borderBottom: 1,
                  borderColor: 'divider',
                  cursor: 'pointer',
                  bgcolor: project.path === currentProject ? 'action.selected' : 'background.paper',
                  transition: 'background-color 0.2s ease',
                  '&:hover': {
                    bgcolor: project.path === currentProject ? 'action.selected' : 'action.hover',
                  },
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box sx={{ flex: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {project.name}
                      </Typography>
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: {
                            active: 'info.main',
                            idle: 'success.main',
                            error: 'error.main',
                            default: 'grey.500',
                          }[getStatusClass(project.status)],
                        }}
                      />
                    </Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                      {project.path}
                    </Typography>
                    {project.phases_total > 0 && (
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        {project.phases_completed} / {project.phases_total} phases completed
                      </Typography>
                    )}
                  </Box>
                  {project.path === currentProject && (
                    <Chip label="Active" size="small" color="success" sx={{ height: 20 }} />
                  )}
                </Box>
              </Box>
            ))}

            <Box sx={{ p: 1.5, fontSize: '0.75rem', color: 'text.secondary', borderTop: 1, borderColor: 'divider', bgcolor: 'action.hover' }}>
              {projects.length} project{projects.length !== 1 ? 's' : ''} found
            </Box>
          </Paper>
        </>
      )}
    </Box>
  )
}
