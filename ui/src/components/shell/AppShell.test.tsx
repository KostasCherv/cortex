import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import { AppShell } from './AppShell'

function renderAppShell() {
  return render(
    <ThemeProvider>
      <AppShell />
    </ThemeProvider>,
  )
}

const { getSessionMock, onAuthStateChangeMock } = vi.hoisted(() => ({
  getSessionMock: vi.fn(),
  onAuthStateChangeMock: vi.fn(() => ({ data: { subscription: { unsubscribe: vi.fn() } } })),
}))

vi.mock('@/lib/supabase', () => ({
  supabase: {
    auth: {
      getSession: getSessionMock,
      onAuthStateChange: onAuthStateChangeMock,
      signInWithOAuth: vi.fn(),
      signOut: vi.fn(),
    },
  },
}))

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client')
  return {
    ...actual,
    checkHealth: vi.fn().mockResolvedValue({ status: 'ok' }),
    listRagAgents: vi.fn().mockResolvedValue({ agents: [] }),
    listRagResources: vi.fn().mockResolvedValue({ resources: [] }),
  }
})

const LANDING_HEADING = /a research platform for multi-step web research/i

describe('AppShell', () => {
  it('does not flash the landing page while the initial session check is pending', () => {
    getSessionMock.mockReturnValue(new Promise(() => {})) // never resolves

    renderAppShell()

    expect(screen.queryByRole('heading', { name: LANDING_HEADING })).not.toBeInTheDocument()
    expect(screen.queryByText(/sign in to start a workspace chat/i)).not.toBeInTheDocument()
  })

  it('shows the landing page once the session check resolves with no session', async () => {
    getSessionMock.mockResolvedValue({ data: { session: null } })

    renderAppShell()

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: LANDING_HEADING })).toBeInTheDocument()
    })
  })

  it('does not show the landing page once the session check resolves with an active session', async () => {
    getSessionMock.mockResolvedValue({
      data: { session: { access_token: 'token', user: { email: 'user@example.com' } } },
    })

    renderAppShell()

    await waitFor(() => {
      expect(screen.queryByText(/sign in to start a workspace chat/i)).not.toBeInTheDocument()
    })
    expect(screen.queryByRole('heading', { name: LANDING_HEADING })).not.toBeInTheDocument()
  })
})
