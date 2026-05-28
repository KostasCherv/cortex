import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SoftwarePlannerPage } from './SoftwarePlannerPage'
import type { SavedSoftwareDevPlan } from '@/types'

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
    generateSoftwareDevPlanMock.mockReset()
    getSoftwareDevPlanMock.mockReset()
    listSoftwareDevPlansMock.mockReset()
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:test-url'),
      revokeObjectURL: vi.fn(),
    })
  })

  it('keeps saved plans out of the planner page and relies on the sidebar for selection', () => {
    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />)

    expect(listSoftwareDevPlansMock).not.toHaveBeenCalled()
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
        summary: 'Selected from the rail',
      },
      markdown: '# Sidebar selected plan\n',
    })
    getSoftwareDevPlanMock.mockResolvedValue(detail)

    render(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} activePlanId="plan-sidebar" />)

    await waitFor(() => {
      expect(getSoftwareDevPlanMock).toHaveBeenCalledWith('plan-sidebar', 'token')
    })

    expect(screen.getAllByText('Sidebar selected plan').length).toBeGreaterThan(0)
    expect(screen.getByText('Load me from the sidebar')).toBeInTheDocument()
  })

  it('ignores stale saved-plan detail responses when the sidebar selection changes quickly', async () => {
    const firstDetail = buildSavedPlan({
      plan_id: 'plan-1',
      prompt_preview: 'First prompt preview',
      plan: { ...buildSavedPlan().plan, title: 'First saved plan', summary: 'First summary' },
      markdown: '# First saved plan\n',
    })
    const secondDetail = buildSavedPlan({
      plan_id: 'plan-2',
      prompt_preview: 'Second prompt preview',
      plan: { ...buildSavedPlan().plan, title: 'Second saved plan', summary: 'Second summary' },
      markdown: '# Second saved plan\n',
    })
    const firstDeferred = createDeferred<SavedSoftwareDevPlan>()
    const secondDeferred = createDeferred<SavedSoftwareDevPlan>()
    getSoftwareDevPlanMock.mockImplementation((planId: string) =>
      planId === 'plan-1' ? firstDeferred.promise : secondDeferred.promise,
    )

    const { rerender } = render(
      <SoftwarePlannerPage authSession={{ access_token: 'token' } as never} activePlanId="plan-1" />,
    )

    await waitFor(() => {
      expect(getSoftwareDevPlanMock).toHaveBeenCalledWith('plan-1', 'token')
    })

    rerender(<SoftwarePlannerPage authSession={{ access_token: 'token' } as never} activePlanId="plan-2" />)

    await waitFor(() => {
      expect(getSoftwareDevPlanMock).toHaveBeenCalledWith('plan-2', 'token')
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
    generateSoftwareDevPlanMock.mockResolvedValue(newPlan)

    render(
      <SoftwarePlannerPage
        authSession={{ access_token: 'token' } as never}
        onPlansChanged={onPlansChanged}
      />,
    )

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
    expect(onPlansChanged).toHaveBeenCalledTimes(1)
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
