import { createTheme } from '@mui/material/styles'
import { cockpitTokens } from './tokens'

export function createCockpitTheme(mode: 'light' | 'dark') {
  const dark = mode === 'dark'

  return createTheme({
    spacing: cockpitTokens.spacing,
    shape: {
      borderRadius: cockpitTokens.shape.borderRadius,
    },
    palette: {
      mode,
      primary: {
        main: cockpitTokens.color.cyan[500],
        dark: cockpitTokens.color.cyan[600],
        light: cockpitTokens.color.cyan[400],
      },
      info: {
        main: cockpitTokens.color.cyan[500],
      },
      success: {
        main: cockpitTokens.color.green[500],
      },
      warning: {
        main: cockpitTokens.color.amber[500],
      },
      error: {
        main: cockpitTokens.color.red[500],
      },
      background: dark
        ? {
            default: cockpitTokens.color.slate[900],
            paper: cockpitTokens.color.slate[800],
          }
        : {
            default: cockpitTokens.color.slate[50],
            paper: '#ffffff',
          },
      text: dark
        ? {
            primary: cockpitTokens.color.slate[50],
            secondary: cockpitTokens.color.slate[300],
          }
        : {
            primary: cockpitTokens.color.slate[900],
            secondary: cockpitTokens.color.slate[600],
          },
      divider: dark ? cockpitTokens.color.slate[700] : cockpitTokens.color.slate[200],
    },
    typography: {
      fontFamily: cockpitTokens.typography.fontFamily,
      button: {
        textTransform: 'none',
        fontWeight: 600,
      },
      h1: { fontWeight: 700 },
      h2: { fontWeight: 700 },
      h3: { fontWeight: 700 },
      h4: { fontWeight: 700 },
      h5: { fontWeight: 700 },
      h6: { fontWeight: 700 },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            fontFamily: cockpitTokens.typography.fontFamily,
          },
          code: {
            fontFamily: cockpitTokens.typography.fontFamilyMono,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 14,
            border: dark
              ? `1px solid ${cockpitTokens.color.slate[700]}`
              : `1px solid ${cockpitTokens.color.slate[200]}`,
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            borderRadius: 0,
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontWeight: 600,
            letterSpacing: 0.2,
          },
        },
      },
    },
  })
}
