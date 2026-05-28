import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SoftwarePlannerPage } from './SoftwarePlannerPage'
import type { SoftwareDevPlanResponse } from '@/types'

const { generateSoftwareDevPlanMock } = vi.hoisted(() => ({
  generateSoftwareDevPlanMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  generateSoftwareDevPlan: generateSoftwareDevPlanMock,
}))

const responseFixture: SoftwareDevPlanResponse = {
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
}

describe('SoftwarePlannerPage', () => {
  beforeEach(() => {
    generateSoftwareDevPlanMock.mockReset()
  })

  it('submits the prompt and renders the returned plan', async () => {
    generateSoftwareDevPlanMock.mockResolvedValue(responseFixture)

    render(
      <SoftwarePlannerPage authSession={{ access_token: 'token' } as never} />,
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

    const planHeadings = await screen.findAllByText('Planner for feature delivery')
    expect(planHeadings).toHaveLength(2)
    expect(screen.getByRole('button', { name: /download markdown/i })).toBeInTheDocument()
    expect(screen.getByText('Dedicated planner module')).toBeInTheDocument()
    expect(screen.getAllByText('src/planner.py')).toHaveLength(3)
  })
})
