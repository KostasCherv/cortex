import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ItineraryPlannerPage } from './ItineraryPlannerPage'
import type { ItineraryPlannerResponse, ItinerarySessionDetail, ItinerarySessionSummary } from '@/types'

const {
  createItinerarySessionMock,
  getItinerarySessionMock,
  postItinerarySessionMessageMock,
} = vi.hoisted(() => ({
  createItinerarySessionMock: vi.fn(),
  getItinerarySessionMock: vi.fn(),
  postItinerarySessionMessageMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  createItinerarySession: createItinerarySessionMock,
  getItinerarySession: getItinerarySessionMock,
  postItinerarySessionMessage: postItinerarySessionMessageMock,
}))

function buildSessionSummary(overrides: Partial<ItinerarySessionSummary> = {}): ItinerarySessionSummary {
  return {
    session_id: 'itin-1',
    owner_id: 'test-user',
    workspace_id: 'test-user',
    title: 'Paris spring city break',
    status: 'collecting_requirements',
    current_version_id: null,
    prompt_preview: 'Plan a Paris trip',
    last_message_preview: 'Need a 4 day trip',
    created_at: '2026-05-29T10:00:00+00:00',
    updated_at: '2026-05-29T10:00:00+00:00',
    ...overrides,
  }
}

function buildSessionDetail(overrides: Partial<ItinerarySessionDetail> = {}): ItinerarySessionDetail {
  return {
    ...buildSessionSummary({
      status: 'generated',
      current_version_id: 'ver-1',
    }),
    requirements: {
      destination: 'Paris',
      start_date: '2026-06-10',
      end_date: '2026-06-14',
      trip_length_days: 4,
      traveler_count: 2,
      party_type: 'couple',
      budget_band: 'mid-range',
      interests: ['art', 'cafes'],
      constraints: ['avoid rushed mornings'],
      pace: 'relaxed',
    },
    messages: [
      {
        message_id: 'msg-1',
        session_id: 'itin-1',
        role: 'assistant',
        content: 'Tell me where you want to go.',
        metadata: {},
        created_at: '2026-05-29T10:00:00+00:00',
      },
    ],
    versions: [
      {
        version_id: 'ver-1',
        session_id: 'itin-1',
        version_number: 1,
        revision_summary: 'Initial itinerary',
        markdown: '# Paris spring city break\n',
        itinerary: {
          title: 'Paris spring city break',
          summary: 'A relaxed four-day Paris plan.',
          destination: 'Paris',
          budget_band: 'mid-range',
          days: [
            {
              day_number: 1,
              title: 'Arrival and Left Bank',
              morning: ['Check in and breakfast'],
              afternoon: ['Musee d’Orsay'],
              evening: ['Seine walk'],
              notes: ['Keep it light'],
            },
          ],
          tips: ['Book the museum ahead'],
          revision_summary: null,
        },
        created_at: '2026-05-29T10:05:00+00:00',
      },
    ],
    current_version: {
      version_id: 'ver-1',
      session_id: 'itin-1',
      version_number: 1,
      revision_summary: 'Initial itinerary',
      markdown: '# Paris spring city break\n',
      itinerary: {
        title: 'Paris spring city break',
        summary: 'A relaxed four-day Paris plan.',
        destination: 'Paris',
        budget_band: 'mid-range',
        days: [
          {
            day_number: 1,
            title: 'Arrival and Left Bank',
            morning: ['Check in and breakfast'],
            afternoon: ['Musee d’Orsay'],
            evening: ['Seine walk'],
            notes: ['Keep it light'],
          },
        ],
        tips: ['Book the museum ahead'],
        revision_summary: null,
      },
      created_at: '2026-05-29T10:05:00+00:00',
    },
    ...overrides,
  }
}

function buildPlannerResponse(overrides: Partial<ItineraryPlannerResponse> = {}): ItineraryPlannerResponse {
  const session = buildSessionDetail()
  return {
    session,
    assistant_message: {
      message_id: 'msg-2',
      session_id: session.session_id,
      role: 'assistant',
      content: 'I generated your itinerary.',
      metadata: { action: 'generate_itinerary' },
      created_at: '2026-05-29T10:10:00+00:00',
    },
    current_itinerary: session.current_version?.itinerary ?? null,
    new_version: session.current_version ?? null,
    created_new_version: true,
    missing_fields: [],
    ...overrides,
  }
}

describe('ItineraryPlannerPage', () => {
  beforeEach(() => {
    createItinerarySessionMock.mockReset()
    getItinerarySessionMock.mockReset()
    postItinerarySessionMessageMock.mockReset()
  })

  it('shows an empty state until a session is started or selected from the sidebar', () => {
    render(<ItineraryPlannerPage authSession={{ access_token: 'token' } as never} />)

    expect(screen.getByText('Start a new itinerary chat or choose one from the sidebar.')).toBeInTheDocument()
  })

  it('creates a new session on first submit and renders the returned itinerary draft', async () => {
    const created = buildSessionSummary()
    const response = buildPlannerResponse()
    const onSessionsChanged = vi.fn()
    const onSessionActivated = vi.fn()
    createItinerarySessionMock.mockResolvedValue(created)
    postItinerarySessionMessageMock.mockResolvedValue(response)

    render(
      <ItineraryPlannerPage
        authSession={{ access_token: 'token' } as never}
        onSessionsChanged={onSessionsChanged}
        onSessionActivated={onSessionActivated}
      />,
    )

    fireEvent.change(screen.getByLabelText(/describe your trip/i), {
      target: { value: 'Plan a 4 day Paris trip for two people.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(createItinerarySessionMock).toHaveBeenCalledWith('token')
    })
    await waitFor(() => {
      expect(postItinerarySessionMessageMock).toHaveBeenCalledWith(
        'itin-1',
        'Plan a 4 day Paris trip for two people.',
        'token',
      )
    })

    expect(await screen.findByText('Paris spring city break')).toBeInTheDocument()
    expect(screen.getByText('A relaxed four-day Paris plan.')).toBeInTheDocument()
    expect(onSessionsChanged).toHaveBeenCalled()
    expect(onSessionActivated).toHaveBeenCalledWith('itin-1')
  })

  it('loads the selected itinerary session from the sidebar', async () => {
    const detail = buildSessionDetail({
      title: 'Tokyo food week',
      current_version: {
        version_id: 'ver-2',
        session_id: 'itin-2',
        version_number: 2,
        revision_summary: 'Added more market time',
        markdown: '# Tokyo food week\n',
        itinerary: {
          title: 'Tokyo food week',
          summary: 'A food-first Tokyo itinerary.',
          destination: 'Tokyo',
          budget_band: 'mid-range',
          days: [],
          tips: ['Reserve omakase early'],
          revision_summary: null,
        },
        created_at: '2026-05-29T11:00:00+00:00',
      },
      versions: [
        {
          version_id: 'ver-2',
          session_id: 'itin-2',
          version_number: 2,
          revision_summary: 'Added more market time',
          markdown: '# Tokyo food week\n',
          itinerary: {
            title: 'Tokyo food week',
            summary: 'A food-first Tokyo itinerary.',
            destination: 'Tokyo',
            budget_band: 'mid-range',
            days: [],
            tips: ['Reserve omakase early'],
            revision_summary: null,
          },
          created_at: '2026-05-29T11:00:00+00:00',
        },
      ],
    })
    getItinerarySessionMock.mockResolvedValue(detail)

    render(
      <ItineraryPlannerPage
        authSession={{ access_token: 'token' } as never}
        activeSessionId="itin-2"
      />,
    )

    await waitFor(() => {
      expect(getItinerarySessionMock).toHaveBeenCalledWith('itin-2', 'token')
    })

    expect(await screen.findByText('Tokyo food week')).toBeInTheDocument()
    expect(screen.getByText('A food-first Tokyo itinerary.')).toBeInTheDocument()
    expect(screen.getByText('Added more market time')).toBeInTheDocument()
  })

  it('posts follow-up edits into the active session and updates the latest version', async () => {
    const detail = buildSessionDetail()
    const revised = buildPlannerResponse({
      session: buildSessionDetail({
        title: 'Paris spring city break',
        current_version: {
          version_id: 'ver-2',
          session_id: 'itin-1',
          version_number: 2,
          revision_summary: 'Made the itinerary cheaper',
          markdown: '# Paris spring city break\n',
          itinerary: {
            title: 'Paris spring city break',
            summary: 'A revised Paris plan with lower spend.',
            destination: 'Paris',
            budget_band: 'budget-conscious',
            days: [],
            tips: ['Use metro passes'],
            revision_summary: null,
          },
          created_at: '2026-05-29T10:20:00+00:00',
        },
        versions: [
          ...detail.versions,
          {
            version_id: 'ver-2',
            session_id: 'itin-1',
            version_number: 2,
            revision_summary: 'Made the itinerary cheaper',
            markdown: '# Paris spring city break\n',
            itinerary: {
              title: 'Paris spring city break',
              summary: 'A revised Paris plan with lower spend.',
              destination: 'Paris',
              budget_band: 'budget-conscious',
              days: [],
              tips: ['Use metro passes'],
              revision_summary: null,
            },
            created_at: '2026-05-29T10:20:00+00:00',
          },
        ],
      }),
      assistant_message: {
        message_id: 'msg-3',
        session_id: 'itin-1',
        role: 'assistant',
        content: 'I updated your itinerary. It is cheaper now.',
        metadata: { action: 'revise_itinerary' },
        created_at: '2026-05-29T10:20:00+00:00',
      },
      current_itinerary: {
        title: 'Paris spring city break',
        summary: 'A revised Paris plan with lower spend.',
        destination: 'Paris',
        budget_band: 'budget-conscious',
        days: [],
        tips: ['Use metro passes'],
        revision_summary: null,
      },
    })
    getItinerarySessionMock.mockResolvedValue(detail)
    postItinerarySessionMessageMock.mockResolvedValue(revised)

    render(
      <ItineraryPlannerPage
        authSession={{ access_token: 'token' } as never}
        activeSessionId="itin-1"
      />,
    )

    await screen.findByText('Paris spring city break')

    fireEvent.change(screen.getByLabelText(/describe your trip/i), {
      target: { value: 'Make it cheaper.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(postItinerarySessionMessageMock).toHaveBeenCalledWith('itin-1', 'Make it cheaper.', 'token')
    })

    expect(await screen.findByText('A revised Paris plan with lower spend.')).toBeInTheDocument()
    expect(screen.getByText('Made the itinerary cheaper')).toBeInTheDocument()
  })
})
