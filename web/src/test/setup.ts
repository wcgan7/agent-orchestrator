import { expect, afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'

// Silence noisy React act(...) warnings in tests (they do not affect runtime behavior).
// Keep other console.error output intact to avoid hiding real failures.
const originalConsoleError = console.error
console.error = (...args: unknown[]) => {
  const first = args[0]
  if (typeof first === 'string' && first.includes('not wrapped in act')) {
    return
  }
  originalConsoleError(...(args as Parameters<typeof console.error>))
}

// Cleanup after each test
afterEach(() => {
  cleanup()
})

// Mock localStorage
const storageBacking: Record<string, string> = {}
const localStorageMock = {
  getItem: (key: string) => {
    return Object.prototype.hasOwnProperty.call(storageBacking, key)
      ? storageBacking[key]
      : null
  },
  setItem: (key: string, value: string) => {
    storageBacking[key] = String(value)
  },
  removeItem: (key: string) => {
    delete storageBacking[key]
  },
  clear: () => {
    for (const key of Object.keys(storageBacking)) {
      delete storageBacking[key]
    }
  },
}

global.localStorage = localStorageMock as Storage

// JSDOM doesn't implement matchMedia (needed by ThemeContext)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
})

// JSDOM doesn't implement ResizeObserver (needed by @xyflow/react)
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).ResizeObserver = ResizeObserverMock

// JSDOM doesn't implement scrollIntoView
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}

// Avoid real WebSocket connections in tests (LiveLog uses this).
class WebSocketMock {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  url: string
  readyState = WebSocketMock.OPEN
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    queueMicrotask(() => this.onopen?.(new Event('open')))
  }

  send(_data: any) {}
  close() {
    this.readyState = WebSocketMock.CLOSED
    queueMicrotask(() => this.onclose?.(new Event('close')))
  }

  addEventListener(_type: string, _listener: any) {}
  removeEventListener(_type: string, _listener: any) {}
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).WebSocket = WebSocketMock

// Mock fetch
global.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  return new Response(JSON.stringify({}), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}
