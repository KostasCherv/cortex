import json
from unittest.mock import MagicMock, patch

import pytest

from src.planner import PlannerValidationError, PRDPlanResponse, generate_prd


class _LLMResult:
    def __init__(self, content: str) -> None:
        self.content = content


def _json_result(payload: object) -> _LLMResult:
    return _LLMResult(json.dumps(payload))


def _minimal_prd_plan() -> dict:
    return {
        "title": "Mobile Onboarding PRD",
        "executive_summary": "Streamline user onboarding.",
        "problem_statement": "Users drop off during onboarding.",
        "goals": ["Reduce drop-off by 30%", "Improve activation", "Increase retention"],
        "non_goals": ["Full app redesign"],
        "target_users": ["New mobile users", "Enterprise admins", "Power users"],
        "user_stories": [
            "As a new user, I want to complete onboarding quickly so that I can start using the app."
        ],
        "requirements": [
            {"id": "REQ-001", "description": "3-step wizard", "priority": "Must Have", "rationale": "Core flow"},
            {"id": "REQ-002", "description": "Progress bar", "priority": "Should Have", "rationale": "UX clarity"},
        ],
        "success_metrics": ["30% drop-off reduction", "NPS +10", "Activation > 60%"],
        "milestones": [
            {"id": "M1", "title": "MVP", "description": "Basic onboarding", "deliverables": ["Wizard", "API"]}
        ],
        "out_of_scope": ["Localization"],
        "risks": ["Scope creep"],
        "assumptions": ["Users on latest version"],
        "open_questions": ["Should onboarding be skippable?"],
    }


@pytest.mark.asyncio
async def test_generate_prd_returns_structured_response():
    mock_llm = MagicMock()
    prd_plan = _minimal_prd_plan()
    mock_llm.invoke.side_effect = [
        # intake
        _json_result({
            "problem_statement": "Users drop off during onboarding.",
            "desired_outcome": "Increase activation rate.",
            "constraints": ["Must ship in Q3"],
            "assumptions": ["Users on latest version"],
            "open_questions": [],
        }),
        # synthesis
        _json_result(prd_plan),
        # review
        _json_result({
            "approved": True,
            "reviewer_notes": ["Plan looks solid."],
            "revised_plan": prd_plan,
        }),
    ]

    with patch("src.llm.factory.get_llm", return_value=mock_llm):
        result = await generate_prd("Build a mobile onboarding flow.")

    assert isinstance(result, PRDPlanResponse)
    assert result.plan.title == "Mobile Onboarding PRD"
    assert "## Goals" in result.markdown
    assert result.suggested_filename.endswith("-prd.md")
    assert mock_llm.invoke.call_count == 3


@pytest.mark.asyncio
async def test_generate_prd_raises_validation_error_on_invalid_structured_output():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        # intake succeeds
        _json_result({
            "problem_statement": "Users drop off during onboarding.",
            "desired_outcome": "Increase activation rate.",
            "constraints": [],
            "assumptions": [],
            "open_questions": [],
        }),
        # synthesis returns invalid JSON (both attempts)
        _LLMResult("not valid json"),
        _LLMResult("still not valid json"),
    ]

    with patch("src.llm.factory.get_llm", return_value=mock_llm):
        with pytest.raises(PlannerValidationError) as exc_info:
            await generate_prd("Build a mobile onboarding flow.")

    assert exc_info.value.code == "planner_generation_failed"
