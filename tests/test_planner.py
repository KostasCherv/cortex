import json
import re
from unittest.mock import MagicMock, patch

import pytest

from src.planner import PlannerValidationError, SoftwareDevPlanResponse, generate_software_dev_plan


class _LLMResult:
    def __init__(self, content: str) -> None:
        self.content = content


def _json_result(payload: object) -> _LLMResult:
    return _LLMResult(json.dumps(payload))


@pytest.mark.asyncio
async def test_generate_software_dev_plan_returns_structured_response():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        _json_result(
            {
                "problem_statement": "Users need implementation plans for feature requests.",
                "desired_outcome": "Return a reviewed downloadable plan.",
                "constraints": ["Keep v1 synchronous."],
                "assumptions": ["The current repo layout remains stable."],
                "open_questions": ["Whether to persist plan history in a later version."],
            }
        ),
        _json_result(
            {
                "summary": "Existing backend endpoint and frontend shell patterns are the main extension points.",
                "relevant_files": [
                    {"path": "src/api/endpoints.py", "reason": "Authenticated API routes live here."},
                    {"path": "ui/src/components/shell/AppShell.tsx", "reason": "Top-level view routing lives here."},
                ],
                "existing_patterns": ["Prompt-template driven draft generation.", "Dedicated page routing through AppShell."],
                "constraints": ["Need repo-grounded file references."],
                "unknowns": ["No persisted plan-history model exists yet."],
            }
        ),
        _json_result(
            {
                "approaches": [
                    {
                        "name": "Dedicated planner module",
                        "summary": "Add a new planner backend module and dedicated UI page.",
                        "tradeoffs": ["More code than reusing RAG directly, but clearer ownership."],
                        "file_impact": ["src/planner.py", "ui/src/pages/SoftwarePlannerPage.tsx"],
                    },
                    {
                        "name": "Extend RAG module",
                        "summary": "Place planner generation inside the existing RAG helpers.",
                        "tradeoffs": ["Smaller diff, but muddier domain boundaries."],
                        "file_impact": ["src/rag.py", "src/api/endpoints.py"],
                    },
                ],
                "recommended_approach": "Dedicated planner module",
                "rationale": "Keeps software-planning orchestration separate from RAG chat concerns.",
                "out_of_scope": ["Automated code execution from the generated plan."],
            }
        ),
        _json_result(
            {
                "title": "Software Development Planner",
                "summary": "Adds a staged implementation-planning workflow with a downloadable markdown artifact.",
                "goal": "Turn a feature request into a repo-grounded implementation plan.",
                "repo_fit": "Fits the existing API/client/view split and reuses the prompt-template workflow pattern.",
                "architecture": "Use a dedicated planner service module, one authenticated endpoint, and a dedicated planner page in the shell.",
                "recommended_approach": "Implement a staged prompt pipeline and dedicated planner page.",
                "file_map": [
                    {"path": "src/planner.py", "reason": "Contains staged planning orchestration and markdown rendering."},
                    {"path": "src/api/endpoints.py", "reason": "Exposes the authenticated planner endpoint."},
                    {"path": "ui/src/pages/SoftwarePlannerPage.tsx", "reason": "Hosts the planner request/result UX."},
                ],
                "data_api_ui_impacts": [
                    "Adds a synchronous planner API route.",
                    "Adds a new shell view and markdown download action.",
                ],
                "phases": [
                    {
                        "id": "phase-1",
                        "title": "Backend planner contract",
                        "objective": "Add planner models, orchestration, and API wiring.",
                        "files": ["src/planner.py", "src/api/endpoints.py"],
                        "deliverables": ["Planner service", "Authenticated endpoint"],
                        "verification": ["pytest tests/test_planner.py tests/test_api.py -q"],
                    },
                    {
                        "id": "phase-2",
                        "title": "Frontend planner UX",
                        "objective": "Add a planner page, API client call, and markdown download affordance.",
                        "files": ["ui/src/pages/SoftwarePlannerPage.tsx", "ui/src/api/client.ts", "ui/src/components/shell/AppShell.tsx"],
                        "deliverables": ["Planner page", "Planner navigation", "Download button"],
                        "verification": ["npm run build"],
                    },
                ],
                "validation": ["pytest tests/test_planner.py tests/test_api.py -q", "npm run build"],
                "risks": ["Structured LLM output may require repair retries."],
                "assumptions": ["A synchronous response is acceptable for v1."],
                "open_questions": ["Whether to save plan history in a later iteration."],
                "out_of_scope": ["Executing the implementation plan automatically."],
            }
        ),
        _json_result(
            {
                "approved": True,
                "reviewer_notes": ["Plan is grounded in repo files and includes validation."],
                "revised_plan": {
                    "title": "Software Development Planner",
                    "summary": "Adds a staged implementation-planning workflow with a downloadable markdown artifact.",
                    "goal": "Turn a feature request into a repo-grounded implementation plan.",
                    "repo_fit": "Fits the existing API/client/view split and reuses the prompt-template workflow pattern.",
                    "architecture": "Use a dedicated planner service module, one authenticated endpoint, and a dedicated planner page in the shell.",
                    "recommended_approach": "Implement a staged prompt pipeline and dedicated planner page.",
                    "file_map": [
                        {"path": "src/planner.py", "reason": "Contains staged planning orchestration and markdown rendering."},
                        {"path": "src/api/endpoints.py", "reason": "Exposes the authenticated planner endpoint."},
                        {"path": "ui/src/pages/SoftwarePlannerPage.tsx", "reason": "Hosts the planner request/result UX."},
                    ],
                    "data_api_ui_impacts": [
                        "Adds a synchronous planner API route.",
                        "Adds a new shell view and markdown download action.",
                    ],
                    "phases": [
                        {
                            "id": "phase-1",
                            "title": "Backend planner contract",
                            "objective": "Add planner models, orchestration, and API wiring.",
                            "files": ["src/planner.py", "src/api/endpoints.py"],
                            "deliverables": ["Planner service", "Authenticated endpoint"],
                            "verification": ["pytest tests/test_planner.py tests/test_api.py -q"],
                        },
                        {
                            "id": "phase-2",
                            "title": "Frontend planner UX",
                            "objective": "Add a planner page, API client call, and markdown download affordance.",
                            "files": ["ui/src/pages/SoftwarePlannerPage.tsx", "ui/src/api/client.ts", "ui/src/components/shell/AppShell.tsx"],
                            "deliverables": ["Planner page", "Planner navigation", "Download button"],
                            "verification": ["npm run build"],
                        },
                    ],
                    "validation": ["pytest tests/test_planner.py tests/test_api.py -q", "npm run build"],
                    "risks": ["Structured LLM output may require repair retries."],
                    "assumptions": ["A synchronous response is acceptable for v1."],
                    "open_questions": ["Whether to save plan history in a later iteration."],
                    "out_of_scope": ["Executing the implementation plan automatically."],
                },
            }
        ),
    ]

    with patch("src.llm.factory.get_llm", return_value=mock_llm):
        result = await generate_software_dev_plan("Build a software-dev implementation planner for this repo.")

    assert isinstance(result, SoftwareDevPlanResponse)
    assert result.plan.title == "Software Development Planner"
    assert "## Implementation phases" in result.markdown
    assert re.match(r"\d{4}-\d{2}-\d{2}-software-development-planner-implementation-plan\.md", result.suggested_filename)
    assert mock_llm.invoke.call_count == 5


@pytest.mark.asyncio
async def test_generate_software_dev_plan_raises_validation_error_on_invalid_structured_output():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        _json_result(
            {
                "problem_statement": "Users need implementation plans for feature requests.",
                "desired_outcome": "Return a reviewed downloadable plan.",
                "constraints": ["Keep v1 synchronous."],
                "assumptions": ["The current repo layout remains stable."],
                "open_questions": [],
            }
        ),
        _json_result(
            {
                "summary": "Existing backend endpoint and frontend shell patterns are the main extension points.",
                "relevant_files": [{"path": "src/api/endpoints.py", "reason": "Authenticated API routes live here."}],
                "existing_patterns": ["Prompt-template driven draft generation."],
                "constraints": ["Need repo-grounded file references."],
                "unknowns": [],
            }
        ),
        _json_result(
            {
                "approaches": [
                    {
                        "name": "Dedicated planner module",
                        "summary": "Add a new planner backend module and dedicated UI page.",
                        "tradeoffs": ["More code than reusing RAG directly, but clearer ownership."],
                        "file_impact": ["src/planner.py", "ui/src/pages/SoftwarePlannerPage.tsx"],
                    }
                ],
                "recommended_approach": "Dedicated planner module",
                "rationale": "Keeps software-planning orchestration separate from RAG chat concerns.",
                "out_of_scope": ["Automated code execution from the generated plan."],
            }
        ),
        _LLMResult("not valid json"),
        _LLMResult("still not valid json"),
    ]

    with patch("src.llm.factory.get_llm", return_value=mock_llm):
        with pytest.raises(PlannerValidationError) as exc_info:
            await generate_software_dev_plan("Build a software-dev implementation planner for this repo.")

    assert exc_info.value.code == "planner_generation_failed"
