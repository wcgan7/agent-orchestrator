import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import NotificationCenter from './NotificationCenter'

// Capture the handler passed to useChannel so we can simulate incoming notifications
let channelHandler: ((event: string, data: any) => void) | null = null

vi.mock('../../contexts/WebSocketContext', () => ({
  useChannel: vi.fn((_channel: string, handler: (event: string, data: any) => void) => {
    channelHandler = handler
  }),
  useWebSocket: vi.fn(() => ({
    status: 'disconnected',
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
    send: vi.fn(),
    addHandler: vi.fn(() => vi.fn()),
    lastEventId: 0,
  })),
}))

// Mock AudioContext to avoid errors in tests
const mockAudioContext = {
  state: 'running',
  resume: vi.fn(),
  createOscillator: vi.fn(() => ({
    connect: vi.fn(),
    frequency: { setValueAtTime: vi.fn() },
    type: 'sine',
    start: vi.fn(),
    stop: vi.fn(),
  })),
  createGain: vi.fn(() => ({
    connect: vi.fn(),
    gain: {
      setValueAtTime: vi.fn(),
      exponentialRampToValueAtTime: vi.fn(),
    },
  })),
  destination: {},
  currentTime: 0,
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(globalThis as any).AudioContext = vi.fn(() => mockAudioContext)

// Mock crypto.randomUUID
if (!globalThis.crypto) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(globalThis as any).crypto = {}
}
const originalRandomUUID = globalThis.crypto.randomUUID
globalThis.crypto.randomUUID = () => 'test-uuid-1234' as `${string}-${string}-${string}-${string}-${string}`

function simulateNotification(data: Record<string, any>) {
  act(() => {
    channelHandler!('notification', data)
  })
}

describe('NotificationCenter', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    vi.clearAllMocks()
    channelHandler = null
    user = userEvent.setup()
  })

  afterAll(() => {
    if (originalRandomUUID) {
      globalThis.crypto.randomUUID = originalRandomUUID
    }
  })

  it('renders the bell icon button', () => {
    render(<NotificationCenter />)

    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    expect(bellBtn).toBeInTheDocument()
    // SVG bell icon should be present
    expect(bellBtn.querySelector('svg')).toBeInTheDocument()
  })

  it('shows no badge when there are no notifications', () => {
    render(<NotificationCenter />)

    // No badge element should be present
    const badge = document.querySelector('.notification-badge')
    expect(badge).toBeNull()
  })

  it('shows badge count when notifications arrive', () => {
    render(<NotificationCenter />)

    expect(channelHandler).not.toBeNull()

    simulateNotification({
      id: 'n-1',
      type: 'info',
      title: 'Build Complete',
      message: 'Build succeeded for task-1',
      timestamp: new Date().toISOString(),
    })

    // Badge should show count
    const badge = document.querySelector('.notification-badge')
    expect(badge).not.toBeNull()
    expect(badge!.textContent).toBe('1')
  })

  it('opens dropdown when clicking the bell', async () => {
    render(<NotificationCenter />)

    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtn)

    // Dropdown should be open
    expect(screen.getByText('Notifications')).toBeInTheDocument()
    expect(screen.getByText('Mark all read')).toBeInTheDocument()
    expect(screen.getByText('Clear all')).toBeInTheDocument()
  })

  it('shows "No notifications" in empty dropdown', async () => {
    render(<NotificationCenter />)

    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtn)

    expect(screen.getByText('No notifications')).toBeInTheDocument()
  })

  it('displays incoming notifications in the dropdown', async () => {
    render(<NotificationCenter />)

    simulateNotification({
      id: 'n-1',
      type: 'success',
      title: 'Deploy Succeeded',
      message: 'Deployed to staging',
      timestamp: new Date().toISOString(),
    })

    simulateNotification({
      id: 'n-2',
      type: 'error',
      title: 'Test Failed',
      message: 'Unit tests failed on main',
      timestamp: new Date().toISOString(),
    })

    // Badge should show 2
    const badge = document.querySelector('.notification-badge')
    expect(badge).not.toBeNull()
    expect(badge!.textContent).toBe('2')

    // Open dropdown
    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtn)

    // Both notifications should appear
    expect(screen.getByText('Deploy Succeeded')).toBeInTheDocument()
    expect(screen.getByText('Deployed to staging')).toBeInTheDocument()
    expect(screen.getByText('Test Failed')).toBeInTheDocument()
    expect(screen.getByText('Unit tests failed on main')).toBeInTheDocument()
  })

  it('marks all notifications as read', async () => {
    render(<NotificationCenter />)

    simulateNotification({
      id: 'n-1',
      type: 'info',
      title: 'Info',
      message: 'Something happened',
      timestamp: new Date().toISOString(),
    })

    // Badge should show 1
    const badge = document.querySelector('.notification-badge')
    expect(badge).not.toBeNull()
    expect(badge!.textContent).toBe('1')

    // Open dropdown
    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtn)

    // Click "Mark all read"
    const markAllBtn = screen.getByText('Mark all read')
    await user.click(markAllBtn)

    // Badge should disappear (unread count = 0)
    const badgeAfter = document.querySelector('.notification-badge')
    expect(badgeAfter).toBeNull()
  })

  it('clears all notifications', async () => {
    render(<NotificationCenter />)

    simulateNotification({
      id: 'n-1',
      type: 'warning',
      title: 'Warning',
      message: 'Something warning',
      timestamp: new Date().toISOString(),
    })

    // Open dropdown
    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtn)

    expect(screen.getByText('Warning')).toBeInTheDocument()

    // Click "Clear all"
    const clearBtn = screen.getByText('Clear all')
    await user.click(clearBtn)

    // After clear, dropdown closes. Click bell again.
    const bellBtnAgain = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtnAgain)

    expect(screen.getByText('No notifications')).toBeInTheDocument()
  })

  it('marks individual notification as read when clicked', async () => {
    render(<NotificationCenter />)

    simulateNotification({
      id: 'n-1',
      type: 'info',
      title: 'Click Me',
      message: 'Read this notification',
      timestamp: new Date().toISOString(),
    })

    // Open dropdown
    const bellBtn = screen.getByRole('button', { name: /notifications/i })
    await user.click(bellBtn)

    // Click on the notification item
    const notifItem = screen.getByText('Click Me').closest('.notification-item')!
    fireEvent.click(notifItem)

    // After clicking, unread count should become 0 so badge disappears
    const badgeAfter = document.querySelector('.notification-badge')
    expect(badgeAfter).toBeNull()
  })
})
