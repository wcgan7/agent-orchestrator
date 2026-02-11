import MenuIcon from '@mui/icons-material/Menu'
import SearchIcon from '@mui/icons-material/Search'
import {
  AppBar,
  Badge,
  Box,
  Chip,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery,
} from '@mui/material'
import type { Theme } from '@mui/material/styles'
import { ReactNode, useMemo, useState } from 'react'
import type { AppNavSection, CockpitView, GlobalStatusSummary } from '../../types/ui'

const SHELL_NAV_WIDTH = 248
const SHELL_RAIL_WIDTH = 320

interface AppShellProps {
  title: string
  sections: AppNavSection[]
  activeSection: CockpitView
  onSectionChange: (section: CockpitView) => void
  statusSummary: GlobalStatusSummary
  commandHint: string
  onOpenCommandPalette: () => void
  commandBarCenter: ReactNode
  commandBarRight: ReactNode
  rightRail: ReactNode
  children: ReactNode
}

function NavList({
  sections,
  activeSection,
  onSectionChange,
}: {
  sections: AppNavSection[]
  activeSection: CockpitView
  onSectionChange: (section: CockpitView) => void
}) {
  return (
    <Box sx={{ px: 1.5, py: 2 }}>
      <Typography variant="overline" sx={{ px: 1.5, color: 'text.secondary' }}>
        Navigation
      </Typography>
      <List sx={{ mt: 1 }}>
        {sections.map((section) => (
          <ListItemButton
            key={section.id}
            selected={section.id === activeSection}
            onClick={() => onSectionChange(section.id)}
            sx={{
              mb: 0.5,
              borderRadius: 1.5,
              alignItems: 'flex-start',
              '&.Mui-selected': {
                backgroundColor: 'rgba(6, 182, 212, 0.16)',
              },
            }}
          >
            <ListItemText
              primary={section.label}
              secondary={section.description}
              primaryTypographyProps={{ fontWeight: 600 }}
              secondaryTypographyProps={{ fontSize: 12 }}
            />
          </ListItemButton>
        ))}
      </List>
    </Box>
  )
}

export default function AppShell({
  title,
  sections,
  activeSection,
  onSectionChange,
  statusSummary,
  commandHint,
  onOpenCommandPalette,
  commandBarCenter,
  commandBarRight,
  rightRail,
  children,
}: AppShellProps) {
  const isMobile = useMediaQuery((theme: Theme) => theme.breakpoints.down('md'))
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [mobileRailOpen, setMobileRailOpen] = useState(false)

  const railWidth = useMemo(() => (isMobile ? 0 : SHELL_RAIL_WIDTH), [isMobile])

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar
        position="fixed"
        color="inherit"
        elevation={0}
        sx={{
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar sx={{ gap: 2, minHeight: 68 }}>
          {isMobile && (
            <IconButton aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}>
              <MenuIcon />
            </IconButton>
          )}
          <Stack direction="row" spacing={1.5} alignItems="center" sx={{ minWidth: 190 }}>
            <Typography variant="h6" sx={{ letterSpacing: 0.2 }}>
              {title}
            </Typography>
            <Chip
              size="small"
              color={statusSummary.color}
              label={statusSummary.label}
              aria-label={`Run status ${statusSummary.label}`}
            />
          </Stack>

          <Box sx={{ flex: 1, minWidth: 0 }}>{commandBarCenter}</Box>

          <Tooltip title="Command palette">
            <Chip
              icon={<SearchIcon />}
              label={commandHint}
              variant="outlined"
              clickable
              onClick={onOpenCommandPalette}
              sx={{ mr: 1 }}
            />
          </Tooltip>

          {commandBarRight}

          {isMobile && (
            <IconButton aria-label="Open activity rail" onClick={() => setMobileRailOpen(true)}>
              <Badge color="warning" variant="dot">
                <MenuIcon />
              </Badge>
            </IconButton>
          )}
        </Toolbar>
      </AppBar>

      {isMobile ? (
        <Drawer open={mobileNavOpen} onClose={() => setMobileNavOpen(false)}>
          <Box sx={{ width: SHELL_NAV_WIDTH, mt: 8 }}>
            <NavList
              sections={sections}
              activeSection={activeSection}
              onSectionChange={(next) => {
                onSectionChange(next)
                setMobileNavOpen(false)
              }}
            />
          </Box>
        </Drawer>
      ) : (
        <Drawer
          variant="permanent"
          sx={{
            width: SHELL_NAV_WIDTH,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: SHELL_NAV_WIDTH,
              boxSizing: 'border-box',
              borderRight: 1,
              borderColor: 'divider',
            },
          }}
        >
          <Toolbar />
          <NavList sections={sections} activeSection={activeSection} onSectionChange={onSectionChange} />
        </Drawer>
      )}

      <Box
        component="main"
        sx={{
          flex: 1,
          px: { xs: 2, md: 3 },
          pt: { xs: 10, md: 11 },
          pb: 3,
          mr: { md: `${railWidth}px` },
          minWidth: 0,
        }}
      >
        {children}
      </Box>

      {isMobile ? (
        <Drawer anchor="right" open={mobileRailOpen} onClose={() => setMobileRailOpen(false)}>
          <Box sx={{ width: SHELL_RAIL_WIDTH, mt: 8, p: 2 }}>{rightRail}</Box>
        </Drawer>
      ) : (
        <Box
          component="aside"
          sx={{
            position: 'fixed',
            right: 0,
            top: 68,
            bottom: 0,
            width: SHELL_RAIL_WIDTH,
            borderLeft: 1,
            borderColor: 'divider',
            bgcolor: 'background.paper',
            overflowY: 'auto',
            p: 2,
          }}
          aria-label="Activity rail"
        >
          {rightRail}
        </Box>
      )}
    </Box>
  )
}
