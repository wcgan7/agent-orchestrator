/**
 * Theme context — light/dark/system mode with localStorage persistence.
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'

export type ThemeMode = 'light' | 'dark' | 'system'

interface ThemeContextType {
  theme: ThemeMode
  setTheme: (mode: ThemeMode) => void
  toggleTheme: () => void
  effectiveTheme: 'light' | 'dark'
}

const STORAGE_KEY = 'feature-prd-runner-theme'
const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

function getSystemPreference(): 'light' | 'dark' {
  if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

function resolveEffective(mode: ThemeMode): 'light' | 'dark' {
  if (mode === 'system') return getSystemPreference()
  return mode
}

interface ThemeProviderProps {
  children: ReactNode
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setThemeState] = useState<ThemeMode>(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as ThemeMode | null
    return stored || 'system'
  })

  const effectiveTheme = resolveEffective(theme)

  // Apply data-theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme === 'system' ? 'system' : effectiveTheme)
  }, [theme, effectiveTheme])

  // Listen for system preference changes
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = () => {
      if (theme === 'system') {
        // Force re-render by toggling state
        setThemeState('system')
      }
    }
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [theme])

  const setTheme = useCallback((mode: ThemeMode) => {
    setThemeState(mode)
    localStorage.setItem(STORAGE_KEY, mode)
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme(effectiveTheme === 'light' ? 'dark' : 'light')
  }, [effectiveTheme, setTheme])

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggleTheme, effectiveTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme(): ThemeContextType {
  const ctx = useContext(ThemeContext)
  if (!ctx) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return ctx
}
