import { useState, useEffect } from 'react'

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
      const response = await fetch('/api/projects')
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

  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'active':
        return '#2196f3'
      case 'idle':
        return '#4caf50'
      case 'error':
        return '#f44336'
      default:
        return '#9e9e9e'
    }
  }

  if (loading) {
    return (
      <div style={{ padding: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
        Loading projects...
      </div>
    )
  }

  if (projects.length === 0) {
    return (
      <div style={{ padding: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
        No projects found
      </div>
    )
  }

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.5rem 1rem',
          background: '#fff',
          border: '1px solid #ddd',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '0.875rem',
          fontWeight: 500,
        }}
      >
        <span style={{ fontSize: '1.25rem' }}>📁</span>
        <span>{getCurrentProjectName()}</span>
        <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem' }}>▼</span>
      </button>

      {showDropdown && (
        <>
          {/* Backdrop */}
          <div
            onClick={() => setShowDropdown(false)}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 999,
            }}
          />

          {/* Dropdown */}
          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 4px)',
              left: 0,
              minWidth: '300px',
              maxWidth: '500px',
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: '4px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: 1000,
              maxHeight: '400px',
              overflowY: 'auto',
            }}
          >
            {projects.map((project) => (
              <div
                key={project.path}
                onClick={() => handleProjectSelect(project.path)}
                style={{
                  padding: '0.75rem',
                  borderBottom: '1px solid #f0f0f0',
                  cursor: 'pointer',
                  background: project.path === currentProject ? '#f5f5f5' : '#fff',
                  transition: 'background 0.2s',
                }}
                onMouseEnter={(e) => {
                  if (project.path !== currentProject) {
                    e.currentTarget.style.background = '#fafafa'
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background =
                    project.path === currentProject ? '#f5f5f5' : '#fff'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>
                        {project.name}
                      </span>
                      <div
                        style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          background: getStatusColor(project.status),
                        }}
                      />
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.25rem' }}>
                      {project.path}
                    </div>
                    {project.phases_total > 0 && (
                      <div style={{ fontSize: '0.75rem', color: '#666' }}>
                        {project.phases_completed} / {project.phases_total} phases completed
                      </div>
                    )}
                  </div>
                  {project.path === currentProject && (
                    <div
                      style={{
                        fontSize: '0.75rem',
                        padding: '0.25rem 0.5rem',
                        background: '#4caf50',
                        color: '#fff',
                        borderRadius: '4px',
                      }}
                    >
                      Active
                    </div>
                  )}
                </div>
              </div>
            ))}

            <div
              style={{
                padding: '0.75rem',
                fontSize: '0.75rem',
                color: '#666',
                borderTop: '1px solid #e0e0e0',
                background: '#fafafa',
              }}
            >
              {projects.length} project{projects.length !== 1 ? 's' : ''} found
            </div>
          </div>
        </>
      )}
    </div>
  )
}
