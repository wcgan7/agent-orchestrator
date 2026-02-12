import { useState } from 'react'
import { Alert, Box, Button, Paper, TextField, Typography } from '@mui/material'

interface Props {
  onLoginSuccess: (token: string, username: string) => void
}

export default function Login({ onLoginSuccess }: Props) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/v3/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Login failed')
      }

      const data = await response.json()
      onLoginSuccess(data.access_token, data.username)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: 'background.default',
        px: 2,
      }}
    >
      <Paper elevation={8} sx={{ width: '100%', maxWidth: 420, p: 4 }}>
        <Typography variant="h4" sx={{ mb: 1, fontWeight: 700 }}>
          Feature PRD Runner
        </Typography>
        <Typography variant="body2" sx={{ mb: 4, color: 'text.secondary' }}>
          Sign in to access the dashboard
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <TextField
            id="username"
            type="text"
            label="Username"
            fullWidth
            margin="normal"
            size="small"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoFocus
            disabled={loading}
          />

          <TextField
            id="password"
            type="password"
            label="Password"
            fullWidth
            margin="normal"
            size="small"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            disabled={loading}
          />

          <Button
            type="submit"
            variant="contained"
            fullWidth
            disabled={loading}
            sx={{ mt: 2.5, py: 1 }}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </Button>
        </form>

        <Box sx={{ mt: 3, p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            <strong>Default credentials (if auth enabled):</strong>
            <br />
            Username: admin
            <br />
            Password: admin
            <br />
            <br />
            <em>Configure via DASHBOARD_USERNAME and DASHBOARD_PASSWORD env vars</em>
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}
