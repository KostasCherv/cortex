"""PRD (Product Requirements Document) planner service."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Mapping, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.db.supabase_store import SupabaseSessionStore
from src.errors import StructuredOutputError
from src.llm.output_parsers import build_validation_retry_prompt, parse_model_json
from src.prompts.registry import prompt_registry

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)
_PLANS_LIST_LIMIT = 20
_PROMPT_PREVIEW_LIMIT = 160

_store: SupabaseSessionStore | None = None


class PlannerValidationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PlanningBrief(BaseModel):
    problem_statement: str
    desired_outcome: str
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("problem_statement", "desired_outcome")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class PRDRequirement(BaseModel):
    id: str
    description: str
    priority: str  # Must Have / Should Have / Could Have / Won't Have
    rationale: str

    @field_validator("id", "description", "priority", "rationale")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class PRDMilestone(BaseModel):
    id: str
    title: str
    description: str
    deliverables: list[str] = Field(default_factory=list)

    @field_validator("id", "title", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class PRDPlan(BaseModel):
    title: str
    executive_summary: str
    problem_statement: str
    goals: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    target_users: list[str] = Field(default_factory=list)
    user_stories: list[str] = Field(default_factory=list)
    requirements: list[PRDRequirement] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    milestones: list[PRDMilestone] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("title", "executive_summary", "problem_statement")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)

    @model_validator(mode="after")
    def validate_lists(self) -> "PRDPlan":
        if not self.goals:
            raise ValueError("goals must not be empty")
        if not self.target_users:
            raise ValueError("target_users must not be empty")
        if not self.user_stories:
            raise ValueError("user_stories must not be empty")
        if not self.requirements:
            raise ValueError("requirements must not be empty")
        if not self.success_metrics:
            raise ValueError("success_metrics must not be empty")
        if not self.milestones:
            raise ValueError("milestones must not be empty")
        return self

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# {self.title}",
            "",
            f"{self.executive_summary}",
            "",
            "## Problem Statement",
            "",
            self.problem_statement,
            "",
            "## Goals",
            "",
        ]
        for goal in self.goals:
            lines.append(f"- {goal}")

        lines.extend(["", "## Non-Goals", ""])
        for item in self.non_goals:
            lines.append(f"- {item}")

        lines.extend(["", "## Target Users", ""])
        for persona in self.target_users:
            lines.append(f"- {persona}")

        lines.extend(["", "## User Stories", ""])
        for story in self.user_stories:
            lines.append(f"- {story}")

        lines.extend(["", "## Requirements", ""])
        must_haves = [r for r in self.requirements if r.priority == "Must Have"]
        should_haves = [r for r in self.requirements if r.priority == "Should Have"]
        could_haves = [r for r in self.requirements if r.priority == "Could Have"]
        wont_haves = [r for r in self.requirements if r.priority == "Won't Have"]

        for group_label, group in [
            ("Must Have", must_haves),
            ("Should Have", should_haves),
            ("Could Have", could_haves),
            ("Won't Have", wont_haves),
        ]:
            if group:
                lines.extend([f"### {group_label}", ""])
                for req in group:
                    lines.append(f"- **{req.id}**: {req.description}")
                    lines.append(f"  - *Rationale*: {req.rationale}")
                lines.append("")

        lines.extend(["## Success Metrics", ""])
        for metric in self.success_metrics:
            lines.append(f"- {metric}")

        lines.extend(["", "## Milestones", ""])
        for milestone in self.milestones:
            lines.extend([
                f"### {milestone.id}: {milestone.title}",
                "",
                milestone.description,
                "",
                "Deliverables:",
            ])
            for deliverable in milestone.deliverables:
                lines.append(f"- {deliverable}")
            lines.append("")

        lines.extend(["## Out of Scope", ""])
        for item in self.out_of_scope:
            lines.append(f"- {item}")

        lines.extend(["", "## Risks", ""])
        for risk in self.risks:
            lines.append(f"- {risk}")

        lines.extend(["", "## Assumptions", ""])
        for assumption in self.assumptions:
            lines.append(f"- {assumption}")

        if self.open_questions:
            lines.extend(["", "## Open Questions", ""])
            for question in self.open_questions:
                lines.append(f"- {question}")

        return "\n".join(lines).strip() + "\n"


class PRDPlanReview(BaseModel):
    approved: bool
    reviewer_notes: list[str] = Field(default_factory=list)
    revised_plan: PRDPlan


class PRDPlanResponse(BaseModel):
    plan: PRDPlan
    markdown: str
    suggested_filename: str
    planning_brief: PlanningBrief


class SavedPRDSummary(BaseModel):
    plan_id: str
    title: str
    summary: str
    prompt_preview: str
    created_at: str
    updated_at: str


class SavedPRD(PRDPlanResponse):
    plan_id: str
    prompt: str
    prompt_preview: str
    created_at: str
    updated_at: str


class SavedPRDListResponse(BaseModel):
    plans: list[SavedPRDSummary] = Field(default_factory=list)


def _require_text(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError("field must not be blank")
    return cleaned


def _workspace_id_for_user(user_id: str) -> str:
    return user_id


def _get_store() -> SupabaseSessionStore:
    global _store
    if _store is None:
        _store = SupabaseSessionStore()
    return _store


def _prompt_preview(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    if len(cleaned) <= _PROMPT_PREVIEW_LIMIT:
        return cleaned
    return cleaned[: _PROMPT_PREVIEW_LIMIT - 1].rstrip() + "…"


def _saved_prd_summary_from_row(row: dict[str, object]) -> SavedPRDSummary:
    return SavedPRDSummary(
        plan_id=str(row["id"]),
        title=str(row.get("title") or ""),
        summary=str(row.get("summary") or ""),
        prompt_preview=str(row.get("prompt_preview") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


def _saved_prd_from_row(row: dict[str, object]) -> SavedPRD:
    try:
        plan = PRDPlan.model_validate(row.get("plan_json") or {})
    except (ValidationError, Exception):
        raise PlannerValidationError("prd_load_failed", "Stored PRD has an incompatible format.")

    brief_raw = row.get("planning_brief_json") or {}
    try:
        planning_brief = PlanningBrief.model_validate(brief_raw)
    except (ValidationError, Exception):
        planning_brief = PlanningBrief(problem_statement="—", desired_outcome="—")

    response = PRDPlanResponse(
        plan=plan,
        markdown=str(row.get("markdown") or ""),
        suggested_filename=str(row.get("suggested_filename") or ""),
        planning_brief=planning_brief,
    )
    return SavedPRD(
        plan_id=str(row["id"]),
        prompt=str(row.get("prompt") or ""),
        prompt_preview=str(row.get("prompt_preview") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        **response.model_dump(),
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "prd"


def _llm_result_to_text(result: object) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2)


def _invoke_structured_stage(
    *,
    prompt_name: str,
    context: Mapping[str, object],
    model: type[MODEL_T],
    temperature: float,
) -> MODEL_T:
    from src.llm.factory import get_llm

    prompt_text, _ = prompt_registry.render(prompt_name, dict(context))
    llm = get_llm(temperature=temperature)
    try:
        result = llm.invoke(prompt_text)
    except (ValidationError, StructuredOutputError):
        raise
    except Exception as api_exc:
        raise PlannerValidationError(
            "llm_api_error",
            f"LLM API error in stage '{prompt_name}': {api_exc}",
        ) from api_exc
    raw_text = _llm_result_to_text(result)

    try:
        return parse_model_json(raw_text, model=model)
    except StructuredOutputError as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text=_schema_text(model),
            invalid_response=raw_text,
            validation_error=exc,
        )
        try:
            repair_result = llm.invoke(repair_prompt)
        except Exception as api_exc:
            raise PlannerValidationError(
                "llm_api_error",
                f"LLM API error during repair in stage '{prompt_name}': {api_exc}",
            ) from api_exc
        repair_text = _llm_result_to_text(repair_result)
        try:
            return parse_model_json(repair_text, model=model)
        except (StructuredOutputError, ValidationError) as repair_exc:
            raise PlannerValidationError(
                "planner_generation_failed",
                f"Planner stage '{prompt_name}' returned invalid structured output.",
            ) from repair_exc
    except ValidationError as exc:
        raise PlannerValidationError(
            "planner_generation_failed",
            f"Planner stage '{prompt_name}' failed validation.",
        ) from exc


def _generate_prd_sync(user_prompt: str) -> PRDPlanResponse:
    normalized_prompt = user_prompt.strip()
    if not normalized_prompt:
        raise PlannerValidationError("planner_prompt_required", "A planning request is required.")

    shared_context: dict[str, object] = {"user_prompt": normalized_prompt}

    planning_brief = _invoke_structured_stage(
        prompt_name="prd_intake",
        context=shared_context,
        model=PlanningBrief,
        temperature=0.1,
    )
    draft_plan = _invoke_structured_stage(
        prompt_name="prd_synthesis",
        context={
            **shared_context,
            "planning_brief_json": planning_brief.model_dump_json(indent=2),
        },
        model=PRDPlan,
        temperature=0.2,
    )
    review = _invoke_structured_stage(
        prompt_name="prd_review",
        context={
            **shared_context,
            "planning_brief_json": planning_brief.model_dump_json(indent=2),
            "draft_plan_json": draft_plan.model_dump_json(indent=2),
        },
        model=PRDPlanReview,
        temperature=0.1,
    )

    final_plan = review.revised_plan
    markdown = final_plan.to_markdown()
    suggested_filename = (
        f"{datetime.now(UTC).date().isoformat()}-{_slugify(final_plan.title)}-prd.md"
    )
    return PRDPlanResponse(
        plan=final_plan,
        markdown=markdown,
        suggested_filename=suggested_filename,
        planning_brief=planning_brief,
    )


async def generate_prd(user_prompt: str) -> PRDPlanResponse:
    return await asyncio.to_thread(_generate_prd_sync, user_prompt)


async def save_prd(
    user_id: str,
    prompt: str,
    response: PRDPlanResponse,
) -> SavedPRD:
    now = datetime.now(UTC).isoformat()
    plan_id = str(uuid.uuid4())
    normalized_prompt = prompt.strip()
    saved_row = {
        "id": plan_id,
        "owner_id": user_id,
        "workspace_id": _workspace_id_for_user(user_id),
        "prompt": normalized_prompt,
        "prompt_preview": _prompt_preview(normalized_prompt),
        "title": response.plan.title,
        "summary": response.plan.executive_summary,
        "suggested_filename": response.suggested_filename,
        "markdown": response.markdown,
        "plan_json": response.plan.model_dump(mode="json"),
        "planning_brief_json": response.planning_brief.model_dump(mode="json"),
        "repo_analysis_json": {},
        "planning_options_json": {},
        "created_at": now,
        "updated_at": now,
    }
    await _get_store().create_software_dev_plan(saved_row)
    return _saved_prd_from_row(saved_row)


async def list_saved_prds(user_id: str) -> list[SavedPRDSummary]:
    rows = await _get_store().list_software_dev_plans(
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
        limit=_PLANS_LIST_LIMIT,
    )
    return [_saved_prd_summary_from_row(row) for row in rows]


async def get_saved_prd(user_id: str, plan_id: str) -> SavedPRD | None:
    row = await _get_store().get_software_dev_plan(
        plan_id=plan_id,
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )
    if row is None:
        return None
    try:
        return _saved_prd_from_row(row)
    except PlannerValidationError:
        return None


async def delete_saved_prd(user_id: str, plan_id: str) -> bool:
    return await _get_store().delete_software_dev_plan(
        plan_id=plan_id,
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )
