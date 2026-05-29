import { render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ThemeContext } from '@/components/layout/theme-context'
import { AgentRail } from './AgentRail'
import type { SavedPRDSummary } from '@/types'
import type { ItinerarySessionSummary } from '@/types'

const {
  listPRDsMock,
  listItinerarySessionsMock,
  getBillingUsageMock,
} = vi.hoisted(() => ({
  listPRDsMock: vi.fn(),
  listItinerarySessionsMock: vi.fn(),
  getBillingUsageMock: vi.fn(),
}))

vi.mock('@/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/api/client')>('@/api/client')
  return {
    ...actual,
    getBillingUsage: getBillingUsageMock,
    listPRDs: listPRDsMock,
    listItinerarySessions: listItinerarySessionsMock,
  }
})

function renderRail(overrides: Partial<ComponentProps<typeof AgentRail>> = {}) {
  return render(
    <ThemeContext.Provider value={{ theme: 'dark', toggle: vi.fn() }}>
      <AgentRail
        health="online"
        authSession={{ access_token: 'token', user: { email: 'user@example.com' } } as never}
        activeView={{ type: 'itinerary-planner' }}
        ragAgents={[]}
        activeSessionId={null}
        sessionRefreshToken={0}
        plannerRefreshToken={0}
        itineraryRefreshToken={0}
        onViewChange={vi.fn()}
        onSessionSelect={vi.fn()}
        onSignIn={vi.fn()}
        onSignOut={vi.fn()}
        onEditAgent={vi.fn()}
        onAgentDeleted={vi.fn()}
        onNewAgent={vi.fn()}
        onNewResearch={vi.fn()}
        onNewChat={vi.fn()}
        onNewItinerary={vi.fn()}
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
    listPRDsMock.mockReset()
    listPRDsMock.mockResolvedValue({
      plans: [
        {
          plan_id: 'plan-1',
          title: 'Mobile onboarding PRD',
          summary: 'Streamline user onboarding for mobile',
          prompt_preview: 'Redesign the mobile onboarding flow',
          created_at: '2026-05-28T10:00:00Z',
          updated_at: '2026-05-28T10:00:00Z',
        } satisfies SavedPRDSummary,
      ],
    })
    listPRDsMock.mockReset()
    listPRDsMock.mockResolvedValue({
      plans: [
        {
          plan_id: 'plan-1',
          title: 'Mobile onboarding PRD',
          summary: 'Streamline user onboarding for mobile',
          prompt_preview: 'Redesign the mobile onboarding flow',
          created_at: '2026-05-28T10:00:00Z',
          updated_at: '2026-05-28T10:00:00Z',
        } satisfies SavedPRDSummary,
      ],
    })
    listItinerarySessionsMock.mockReset()
    listItinerarySessionsMock.mockResolvedValue({
      sessions: [
        {
          session_id: 'itin-1',
          owner_id: 'test-user',
          workspace_id: 'test-user',
          title: 'Paris spring city break',
          status: 'generated',
          current_version_id: 'ver-1',
          prompt_preview: 'Plan a Paris trip',
          last_message_preview: 'Make it cheaper',
          created_at: '2026-05-29T10:00:00Z',
          updated_at: '2026-05-29T10:10:00Z',
        } satisfies ItinerarySessionSummary,
      ],
    })
  })

  it('shows saved itinerary sessions in the sidebar when itinerary planner view is active', async () => {
    renderRail()

    await waitFor(() => {
      expect(listItinerarySessionsMock).toHaveBeenCalledWith('token')
    })

    expect(await screen.findByRole('button', { name: /paris spring city break/i })).toBeInTheDocument()
    expect(screen.getByText('Make it cheaper')).toBeInTheDocument()
  })

  it('opens the selected itinerary session from the sidebar', async () => {
    const onViewChange = vi.fn()
    renderRail({ onViewChange })

    const savedSessionButton = await screen.findByRole('button', { name: /paris spring city break/i })
    savedSessionButton.click()

    expect(onViewChange).toHaveBeenCalledWith({ type: 'itinerary-planner', sessionId: 'itin-1' })
  })
})
