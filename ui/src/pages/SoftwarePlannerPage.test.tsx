import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SoftwarePlannerPage } from './SoftwarePlannerPage'
import type { SavedSoftwareDevPlan, SavedSoftwareDevPlanSummary } from '@/types'

const { generateSoftwareDevPlanMock, getSoftwareDevPlanMock, listSoftwareDevPlansMock } = vi.hoisted(() => ({
  generateSoftwareDevPlanMock: vi.fn(),
  getSoftwareDevPlanMock: vi.fn(),
  listSoftwareDevPlansMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  generateSoftwareDevPlan: generateSoftwareDevPlanMock,
  getSoftwareDevPlan: getSoftwareDevPlanMock,
  listSoftwareDevPlans: listSoftwareDevPlansMock,
}))

function buildSavedPlan(overrides: Partial<SavedSoftwareDevPlan> = {}): SavedSoftwareDevPlan {
  return {
    plan_id: 'plan-1',
    prompt: 'Build a software-dev implementation planner.',
    prompt_preview: 'Build a software-dev implementation planner.',
    created_at: '2026-05-28T10:00:00+00:00',
    updated_at: '2026-05-28T10:00:00+00:00',
    plan: {
      title: 'Planner for feature delivery',
      summary: 'Adds a staged planning workflow.',
      goal: 'Turn a feature request into an implementation plan.',
      repo_fit: 'Fits the existing backend/frontend split.',
      architecture: 'Dedicated planner service plus UI page.',
      recommended_approach: 'Use staged prompts and a synchronous endpoint.',
      file_map: [{ path: 'src/planner.py', reason: 'Planner orchestration lives here.' }],
      data_api_ui_impacts: ['Adds a planner API route and shell view.'],
      phases: [
        {
          id: 'phase-1',
          title: 'Backend',
          objective: 'Implement planner service and endpoint.',
          files: ['src/planner.py', 'src/api/endpoints.py'],
          deliverables: ['Planner service', 'Planner API route'],
          verification: ['pytest tests/test_planner.py tests/test_api.py -q'],
        },
      ],
      validation: ['pytest tests/test_planner.py tests/test_api.py -q'],
      risks: ['Structured output may require retries.'],
      assumptions: ['Synchronous generation is acceptable for v1.'],
      open_questions: ['Should plan history be persisted later?'],
      out_of_scope: ['Automated implementation.'],
    },
    markdown: '# Planner for feature delivery\n',
    suggested_filename: '2026-05-28-planner-for-feature-delivery-implementation-plan.md',
    planning_brief: {
      problem_statement: 'Need implementation planning.',
      desired_outcome: 'Return a markdown plan.',
      constraints: ['Stay repo-grounded.'],
      assumptions: ['Users are signed in.'],
      open_questions: [],
    },
    repo_analysis: {
      summary: 'Existing endpoint and shell patterns are reusable.',
      relevant_files: [{ path: 'src/api/endpoints.py', reason: 'Planner endpoint belongs here.' }],
      existing_patterns: ['Prompt-template driven generation.'],
      constraints: ['Plan only in v1.'],
      unknowns: [],
    },
    planning_options: {
      approaches: [
        {
          name: 'Dedicated planner module',
          summary: 'Keep planner logic separate from RAG helpers.',
          tradeoffs: ['Slightly larger diff, but cleaner boundaries.'],
          file_impact: ['src/planner.py', 'ui/src/pages/SoftwarePlannerPage.tsx'],
        },
      ],
      recommended_approach: 'Dedicated planner module',
      rationale: 'Clear ownership and easier iteration.',
      out_of_scope: ['True distributed multi-agent execution.'],
    },
    ...overrides,
  }
}

function toSummary(plan: SavedSoftwareDevPlan): SavedSoftwareDevPlanSummary {
  return {
    plan_id: plan.plan_id,
    title: plan.plan.title,
    summary: plan.plan.summary,
    prompt_preview: plan.prompt_preview,
    created_at: plan.created_at,
    updated_at: plan.updated_at,
  }
}

describe('SoftwarePlannerPage', () => {
  beforeEach(() => {
    generateSoftwareDevPlanMock.mockReset()
    getSoftwareDevPlanMock.mockReset()
    listSoftwareDevPlansMock.mockReset()
    listSoftwareDevPlansMock.mockResolvedValue({ plans: [] })
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:test-url'),
      revokeObjectURL: vi.fn(),
    })
  })

  it('loads history on mount', async () => {
    const savedPlan = buildSavedPlan()
    listSoftwareDevPlansMock.mockResolvedValue({ plans: [toSummary(savedPlan)] })

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    await waitFor(() => {
      expect(listSoftwareDevPlansMock).toHaveBeenCalledWith('token')
    })

    expect(await screen.findByText('Planner for feature delivery')).toBeInTheDocument()
    expect(screen.getByText('Build a software-dev implementation planner.')).toBeInTheDocument()
  })

  it('prepends and selects a newly generated saved plan', async () => {
    const olderPlan = buildSavedPlan({
      plan_id: 'plan-older',
      created_at: '2026-05-27T10:00:00+00:00',
      updated_at: '2026-05-27T10:00:00+00:00',
      prompt_preview: 'Older prompt preview',
      plan: {
        ...buildSavedPlan().plan,
        title: 'Older planner',
        summary: 'Older summary',
      },
    })
    const newPlan = buildSavedPlan()
    listSoftwareDevPlansMock.mockResolvedValue({ plans: [toSummary(olderPlan)] })
    generateSoftwareDevPlanMock.mockResolvedValue(newPlan)

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    await screen.findByText('Older planner')

    fireEvent.change(screen.getByLabelText(/what should the planner design/i), {
      target: { value: 'Build a software-dev implementation planner.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /generate plan/i }))

    await waitFor(() => {
      expect(generateSoftwareDevPlanMock).toHaveBeenCalledWith(
        'Build a software-dev implementation planner.',
        'token',
      )
    })

    const plannerTitles = await screen.findAllByText('Planner for feature delivery')
    expect(plannerTitles.length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /download markdown/i })).toBeInTheDocument()

    const historyButtons = screen.getAllByRole('button')
    const savedPlanButton = historyButtons.find((button) => button.textContent?.includes('Planner for feature delivery'))
    expect(savedPlanButton?.textContent).toContain('Build a software-dev implementation planner.')
  })

  it('clicking a history item loads and renders its detail', async () => {
    const summary = toSummary(
      buildSavedPlan({
        plan_id: 'plan-2',
        prompt_preview: 'Open the older saved plan',
        plan: {
          ...buildSavedPlan().plan,
          title: 'Saved planner from history',
          summary: 'History summary',
        },
      }),
    )
    const detail = buildSavedPlan({
      plan_id: 'plan-2',
      prompt_preview: 'Open the older saved plan',
      plan: {
        ...buildSavedPlan().plan,
        title: 'Saved planner from history',
        summary: 'History summary',
      },
      markdown: '# Saved planner from history\n',
    })
    listSoftwareDevPlansMock.mockResolvedValue({ plans: [summary] })
    getSoftwareDevPlanMock.mockResolvedValue(detail)

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    fireEvent.click(await screen.findByRole('button', { name: /saved planner from history/i }))

    await waitFor(() => {
      expect(getSoftwareDevPlanMock).toHaveBeenCalledWith('plan-2', 'token')
    })

    expect(screen.getAllByText('Saved planner from history').length).toBeGreaterThan(0)
    expect(screen.getByText('Dedicated planner module')).toBeInTheDocument()
  })

  it('renders the empty history state', async () => {
    listSoftwareDevPlansMock.mockResolvedValue({ plans: [] })

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    expect(await screen.findByText('No saved plans yet.')).toBeInTheDocument()
  })

  it('downloads markdown for the selected saved plan', async () => {
    const savedPlan = buildSavedPlan()
    generateSoftwareDevPlanMock.mockResolvedValue(savedPlan)

    const anchor = document.createElement('a')
    const clickSpy = vi.spyOn(anchor, 'click').mockImplementation(() => {})
    const createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        return anchor
      }
      return document.createElementNS('http://www.w3.org/1999/xhtml', tagName) as HTMLElement
    })

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    fireEvent.change(screen.getByLabelText(/what should the planner design/i), {
      target: { value: 'Build a software-dev implementation planner.' },
    })
    fireEvent.click(screen.getByRole('button', { name: /generate plan/i }))

    const downloadButton = await screen.findByRole('button', { name: /download markdown/i })
    fireEvent.click(downloadButton)

    expect(anchor.download).toBe(savedPlan.suggested_filename)
    expect(clickSpy).toHaveBeenCalled()
    expect(URL.createObjectURL).toHaveBeenCalled()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-url')

    createElementSpy.mockRestore()
  })
})
