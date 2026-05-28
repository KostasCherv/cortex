import { render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ThemeContext } from '@/components/layout/theme-context'
import { AgentRail } from './AgentRail'
import type { SavedSoftwareDevPlanSummary } from '@/types'

const {
  listSoftwareDevPlansMock,
  getBillingUsageMock,
} = vi.hoisted(() => ({
  listSoftwareDevPlansMock: vi.fn(),
  getBillingUsageMock: vi.fn(),
}))

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client')
  return {
    ...actual,
    getBillingUsage: getBillingUsageMock,
    listSoftwareDevPlans: listSoftwareDevPlansMock,
  }
})

function renderRail(overrides: Partial<ComponentProps<typeof AgentRail>> = {}) {
  return render(
    <ThemeContext.Provider value={{ theme: 'dark', toggle: vi.fn() }}>
      <AgentRail
        health="online"
        authSession={{ access_token: 'token', user: { email: 'user@example.com' } } as never}
        activeView={{ type: 'software-planner' }}
        ragAgents={[]}
        activeSessionId={null}
        sessionRefreshToken={0}
        plannerRefreshToken={0}
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
    listSoftwareDevPlansMock.mockReset()
    listSoftwareDevPlansMock.mockResolvedValue({
      plans: [
        {
          plan_id: 'plan-1',
          title: 'Persist planner history',
          summary: 'Save plans and expose history',
          prompt_preview: 'Persist planner history in the UI',
          created_at: '2026-05-28T10:00:00Z',
          updated_at: '2026-05-28T10:00:00Z',
        } satisfies SavedSoftwareDevPlanSummary,
      ],
    })
  })

  it('shows saved planner conversations in the sidebar when planner view is active', async () => {
    renderRail()

    await waitFor(() => {
      expect(listSoftwareDevPlansMock).toHaveBeenCalledWith('token')
    })

    expect(await screen.findByRole('button', { name: /persist planner history/i })).toBeInTheDocument()
    expect(screen.getByText('Persist planner history in the UI')).toBeInTheDocument()
  })

  it('opens the selected saved plan from the sidebar', async () => {
    const onViewChange = vi.fn()
    renderRail({ onViewChange })

    const savedPlanButton = await screen.findByRole('button', { name: /persist planner history/i })
    savedPlanButton.click()

    expect(onViewChange).toHaveBeenCalledWith({ type: 'software-planner', planId: 'plan-1' })
  })
})
