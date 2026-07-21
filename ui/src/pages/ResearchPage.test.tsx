import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ResearchPage } from './ResearchPage'
import type { SessionDetail } from '@/types'

const { getSessionMock, getBillingUsageMock } = vi.hoisted(() => ({
  getSessionMock: vi.fn(),
  getBillingUsageMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  createCheckoutSession: vi.fn(),
  createPortalSession: vi.fn(),
  createSession: vi.fn(),
  getBillingUsage: getBillingUsageMock,
  getSession: getSessionMock,
  startSessionResearch: vi.fn(),
  streamSessionRun: vi.fn(),
  submitRunFeedback: vi.fn(),
}))

const completedWithoutReport: SessionDetail = {
  session_id: 'session-1',
  title: 'Completed run',
  created_at: '2026-05-26T10:00:00Z',
  conversation: [],
  runs: [
    {
      run_id: 'run-1',
      query: 'What changed?',
      source_urls: [],
      report: '',
      status: 'completed',
      latest_node: 'report_node',
      latest_event_at: '2026-05-26T10:01:00Z',
      partial_report: '',
      created_at: '2026-05-26T10:00:00Z',
    },
  ],
}

const runningWithPartialReport: SessionDetail = {
  session_id: 'session-2',
  title: 'Running run',
  created_at: '2026-05-26T10:00:00Z',
  conversation: [],
  runs: [
    {
      run_id: 'run-2',
      query: 'Still running',
      source_urls: [],
      report: '',
      status: 'running',
      latest_node: 'summarize_node',
      latest_event_at: '2026-05-26T10:01:00Z',
      partial_report: 'Partial draft',
      created_at: '2026-05-26T10:00:00Z',
    },
  ],
}

const completedWithReport: SessionDetail = {
  session_id: 'session-3',
  title: 'Finished run',
  created_at: '2026-05-26T10:00:00Z',
  conversation: [],
  runs: [
    {
      run_id: 'run-3',
      query: 'Done',
      source_urls: [],
      report: '# Final report',
      status: 'completed',
      latest_node: 'report_node',
      latest_event_at: '2026-05-26T10:01:00Z',
      partial_report: '',
      created_at: '2026-05-26T10:00:00Z',
    },
  ],
}

describe('ResearchPage', () => {
  beforeEach(() => {
    getSessionMock.mockReset()
    getBillingUsageMock.mockReset()
    getBillingUsageMock.mockResolvedValue({
      plan: 'free',
      date: '2026-05-26',
      limits: { research_queries_daily: 10, total_questions_daily: 50 },
      usage: { research_queries_count: 1, total_questions_count: 2 },
      resets_at: '2026-05-27T00:00:00Z',
      subscription: null,
    })
  })

  it('shows a loading indicator instead of the empty composer while switching to a session', async () => {
    getSessionMock.mockReturnValue(new Promise(() => {})) // never resolves

    render(
      <ResearchPage
        authSession={{ access_token: 'token' } as never}
        activeSessionId="session-9"
        onSessionActivated={() => {}}
        onSessionsChanged={() => {}}
      />,
    )

    await waitFor(() => expect(getSessionMock).toHaveBeenCalledWith('session-9', 'token'))

    expect(screen.getByText(/loading discussion/i)).toBeInTheDocument()
    expect(
      screen.queryByPlaceholderText(/compare model context protocol/i),
    ).not.toBeInTheDocument()
  })

  it('shows a loader instead of the progress card while a completed session is still loading its report', async () => {
    getSessionMock.mockResolvedValue(completedWithoutReport)

    render(
      <ResearchPage
        authSession={{ access_token: 'token' } as never}
        activeSessionId="session-1"
        onSessionActivated={() => {}}
        onSessionsChanged={() => {}}
      />,
    )

    await waitFor(() => expect(getSessionMock).toHaveBeenCalledWith('session-1', 'token'))

    expect(screen.getByText(/loading final report/i)).toBeInTheDocument()
    expect(screen.queryByText('Progress')).not.toBeInTheDocument()
  })

  it('keeps the progress card visible while a run is still active', async () => {
    getSessionMock.mockResolvedValue(runningWithPartialReport)

    render(
      <ResearchPage
        authSession={{ access_token: 'token' } as never}
        activeSessionId="session-2"
        onSessionActivated={() => {}}
        onSessionsChanged={() => {}}
      />,
    )

    await waitFor(() => expect(getSessionMock).toHaveBeenCalledWith('session-2', 'token'))

    expect(screen.getByText('Progress')).toBeInTheDocument()
    expect(screen.queryByText(/loading final report/i)).not.toBeInTheDocument()
  })

  it('does not show the completed loader once the final report is available', async () => {
    getSessionMock.mockResolvedValue(completedWithReport)

    render(
      <ResearchPage
        authSession={{ access_token: 'token' } as never}
        activeSessionId="session-3"
        onSessionActivated={() => {}}
        onSessionsChanged={() => {}}
      />,
    )

    await waitFor(() => expect(getSessionMock).toHaveBeenCalledWith('session-3', 'token'))

    expect(screen.queryByText(/loading final report/i)).not.toBeInTheDocument()
    expect(screen.getByText('Final report')).toBeInTheDocument()
  })
})
