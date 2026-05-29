import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SoftwarePlannerPage } from './SoftwarePlannerPage'
import type { SavedPRD } from '@/types'

const { generatePRDMock, getPRDMock } = vi.hoisted(() => ({
  generatePRDMock: vi.fn(),
  getPRDMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  generatePRD: generatePRDMock,
  getPRD: getPRDMock,
}))

function buildSavedPlan(overrides: Partial<SavedPRD> = {}): SavedPRD {
  return {
    plan_id: 'plan-1',
    prompt: 'Build a mobile onboarding flow.',
    prompt_preview: 'Build a mobile onboarding flow.',
    created_at: '2026-05-28T10:00:00+00:00',
    updated_at: '2026-05-28T10:00:00+00:00',
    plan: {
      title: 'Mobile Onboarding PRD',
      executive_summary: 'Streamline user onboarding for mobile.',
      problem_statement: 'New users drop off during onboarding.',
      goals: ['Reduce onboarding drop-off by 30%', 'Improve first-session activation'],
      non_goals: ['Redesign the entire mobile app'],
      target_users: ['New mobile users', 'Enterprise admins'],
      user_stories: [
        'As a new user, I want to complete onboarding in under 2 minutes so that I can start using the product quickly.',
      ],
      requirements: [
        { id: 'REQ-001', description: 'Onboarding wizard with 3 steps', priority: 'Must Have', rationale: 'Core flow' },
        { id: 'REQ-002', description: 'Progress indicator', priority: 'Should Have', rationale: 'Reduces anxiety' },
      ],
      success_metrics: ['30% reduction in drop-off', 'NPS +10', 'Activation rate > 60%'],
      milestones: [
        { id: 'M1', title: 'MVP', description: 'Basic onboarding flow', deliverables: ['Wizard component', 'API endpoint'] },
      ],
      out_of_scope: ['Localization'],
      risks: ['High engineering effort if scope creeps'],
      assumptions: ['Users have the latest app version'],
      open_questions: ['Should onboarding be skippable?'],
    },
    markdown: '# Mobile Onboarding PRD\n',
    suggested_filename: '2026-05-28-mobile-onboarding-prd.md',
    planning_brief: {
      problem_statement: 'Users drop off during onboarding.',
      desired_outcome: 'Increase activation rate.',
      constraints: ['Must ship in Q3'],
      assumptions: ['Users are on iOS or Android'],
      open_questions: [],
    },
    ...overrides,
  }
}

function createDeferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe('SoftwarePlannerPage', () => {
  beforeEach(() => {
    generatePRDMock.mockReset()
    getPRDMock.mockReset()
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:test-url'),
      revokeObjectURL: vi.fn(),
    })
  })

  it('keeps saved plans out of the planner page and relies on the sidebar for selection', () => {
    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    expect(screen.queryByText('Saved plans')).not.toBeInTheDocument()
    expect(screen.getByText('Generate a new plan or choose one from the sidebar to view its details.')).toBeInTheDocument()
  })

  it('loads the externally selected saved plan', async () => {
    const detail = buildSavedPlan({
      plan_id: 'plan-sidebar',
      prompt_preview: 'Load me from the sidebar',
      plan: {
        ...buildSavedPlan().plan,
        title: 'Sidebar selected plan',
        executive_summary: 'Selected from the rail',
      },
      markdown: '# Sidebar selected plan\n',
    })
    getPRDMock.mockResolvedValue(detail)

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} activePlanId="plan-sidebar" />)

    await waitFor(() => {
      expect(getPRDMock).toHaveBeenCalledWith('plan-sidebar', 'token')
    })

    expect(screen.getAllByText('Sidebar selected plan').length).toBeGreaterThan(0)
    expect(screen.getByText('Load me from the sidebar')).toBeInTheDocument()
  })

  it('ignores stale saved-plan detail responses when the sidebar selection changes quickly', async () => {
    const firstDetail = buildSavedPlan({
      plan_id: 'plan-1',
      prompt_preview: 'First prompt preview',
      plan: { ...buildSavedPlan().plan, title: 'First saved plan', executive_summary: 'First summary' },
      markdown: '# First saved plan\n',
    })
    const secondDetail = buildSavedPlan({
      plan_id: 'plan-2',
      prompt_preview: 'Second prompt preview',
      plan: { ...buildSavedPlan().plan, title: 'Second saved plan', executive_summary: 'Second summary' },
      markdown: '# Second saved plan\n',
    })
    const firstDeferred = createDeferred<SavedPRD>()
    const secondDeferred = createDeferred<SavedPRD>()
    getPRDMock.mockImplementation((planId: string) =>
      planId === 'plan-1' ? firstDeferred.promise : secondDeferred.promise,
    )

    const { rerender } = render(
      <SoftwarePlannerPage authSession={{ access_token: 'token' } as never} activePlanId="plan-1" />,
    )

    await waitFor(() => {
      expect(getPRDMock).toHaveBeenCalledWith('plan-1', 'token')
    })

    rerender(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} activePlanId="plan-2" />)

    await waitFor(() => {
      expect(getPRDMock).toHaveBeenCalledWith('plan-2', 'token')
    })

    await act(async () => {
      secondDeferred.resolve(secondDetail)
      await secondDeferred.promise
    })

    expect(screen.getAllByText('Second saved plan').length).toBeGreaterThan(0)

    await act(async () => {
      firstDeferred.resolve(firstDetail)
      await firstDeferred.promise
    })

    await waitFor(() => {
      expect(screen.getAllByText('Second saved plan').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('First summary')).not.toBeInTheDocument()
  })

  it('renders a newly generated plan and notifies the sidebar to refresh', async () => {
    const newPlan = buildSavedPlan()
    const onPlansChanged = vi.fn()
    generatePRDMock.mockResolvedValue(newPlan)

    render(
      <SoftwarePlannerPage
        authSession={{ access_token: 'token' } as never}
        onPlansChanged={onPlansChanged}
      />,
    )

    fireEvent.change(screen.getByLabelText(/what product idea should we document/i), {
      target: { value: 'Build a mobile onboarding flow.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /generate prd/i }))

    await waitFor(() => {
      expect(generatePRDMock).toHaveBeenCalledWith(
        'Build a mobile onboarding flow.',
        'token',
      )
    })

    const plannerTitles = await screen.findAllByText('Mobile Onboarding PRD')
    expect(plannerTitles.length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /download markdown/i })).toBeInTheDocument()
    expect(onPlansChanged).toHaveBeenCalledTimes(1)
  })

  it('downloads markdown for the selected saved plan', async () => {
    const savedPlan = buildSavedPlan()
    generatePRDMock.mockResolvedValue(savedPlan)

    const anchor = document.createElement('a')
    const clickSpy = vi.spyOn(anchor, 'click').mockImplementation(() => {})
    const createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        return anchor
      }
      return document.createElementNS('http://www.w3.org/1999/xhtml', tagName) as HTMLElement
    })

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    fireEvent.change(screen.getByLabelText(/what product idea should we document/i), {
      target: { value: 'Build a mobile onboarding flow.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /generate prd/i }))

    const downloadButton = await screen.findByRole('button', { name: /download markdown/i })
    fireEvent.click(downloadButton)

    expect(anchor.download).toBe(savedPlan.suggested_filename)
    expect(clickSpy).toHaveBeenCalled()
    expect(URL.createObjectURL).toHaveBeenCalled()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-url')

    createElementSpy.mockRestore()
  })
})
