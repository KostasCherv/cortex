import { fireEvent, render, screen, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ThemeContext } from '@/components/layout/theme-context'
import { AgentRail } from './AgentRail'

const { getBillingUsageMock } = vi.hoisted(() => ({
  getBillingUsageMock: vi.fn(),
}))

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client')
  return {
    ...actual,
    getBillingUsage: getBillingUsageMock,
  }
})

function renderRail(overrides: Partial<ComponentProps<typeof AgentRail>> = {}) {
  return render(
    <ThemeContext.Provider value={{ theme: 'dark', toggle: vi.fn() }}>
      <AgentRail
        health="online"
        authSession={{ access_token: 'token', user: { email: 'user@example.com' } } as never}
        activeView={{ type: 'chat' }}
        ragAgents={[]}
        activeSessionId={null}
        sessionRefreshToken={0}
        onViewChange={vi.fn()}
        onSessionSelect={vi.fn()}
        onSignIn={vi.fn()}
        onSignOut={vi.fn()}
        onEditAgent={vi.fn()}
        onAgentDeleted={vi.fn()}
        onNewAgent={vi.fn()}
        onNewResearch={vi.fn()}
        onNewChat={vi.fn()}
        {...overrides}
      />
    </ThemeContext.Provider>,
  )
}

describe('AgentRail', () => {
  beforeEach(() => {
    getBillingUsageMock.mockReset()
    getBillingUsageMock.mockResolvedValue({
      plan: 'free',
      usage: { research_queries_count: 0, total_questions_count: 0 },
      limits: { research_queries_daily: 10, total_questions_daily: 10 },
      subscription: null,
    })
  })

  it('renders without crashing', () => {
    expect(() => renderRail()).not.toThrow()
  })

  it('renders a mobile hamburger button that is clickable', () => {
    renderRail()
    const menuButton = screen.getByRole('button', { name: /open menu/i })
    expect(menuButton).toBeInTheDocument()
    expect(() => fireEvent.click(menuButton)).not.toThrow()
  })

  it('opens the mobile drawer to reveal nav items when the hamburger is clicked', () => {
    renderRail()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /open menu/i }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('Chat')).toBeInTheDocument()
    expect(within(dialog).getByText('Research')).toBeInTheDocument()
    expect(within(dialog).getByText('Memory')).toBeInTheDocument()
    expect(within(dialog).getByText('Resources')).toBeInTheDocument()
  })

  it('closes the mobile drawer after selecting a nav item', () => {
    const onViewChange = vi.fn()
    renderRail({ onViewChange })

    fireEvent.click(screen.getByRole('button', { name: /open menu/i }))
    const dialog = screen.getByRole('dialog')
    fireEvent.click(within(dialog).getByText('Research'))

    expect(onViewChange).toHaveBeenCalledWith({ type: 'research' })
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('gives the mobile and desktop account controls distinct accessible names', () => {
    // Both copies are always mounted (CSS toggles which is visible per breakpoint),
    // so byRole/byLabelText must resolve to exactly one match for each — this is
    // what broke the "Account menu" Playwright smoke test after the mobile drawer
    // rework duplicated these controls without disambiguating their labels.
    renderRail()
    expect(screen.getAllByRole('button', { name: 'Account menu' })).toHaveLength(1)
    expect(screen.getAllByRole('button', { name: 'Account menu (mobile)' })).toHaveLength(1)
    expect(screen.getAllByRole('button', { name: /^Switch to (dark|light) theme$/ })).toHaveLength(1)
    expect(screen.getAllByRole('button', { name: /^Switch to (dark|light) theme \(mobile\)$/ })).toHaveLength(1)
  })

  it('gives the mobile and desktop sign-in buttons distinct accessible names when signed out', () => {
    renderRail({ authSession: null })
    expect(screen.getAllByRole('button', { name: 'Sign in' })).toHaveLength(1)
    expect(screen.getAllByRole('button', { name: 'Sign in (mobile)' })).toHaveLength(1)
  })
})
