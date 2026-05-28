"""Software-development implementation planner service."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.errors import StructuredOutputError
from src.llm.output_parsers import build_validation_retry_prompt, parse_model_json
from src.prompts.registry import prompt_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REPO_FILES = [
    "src/api/endpoints.py",
    "src/planner.py",
    "src/prompts/registry.py",
    "src/llm/output_parsers.py",
    "src/rag.py",
    "ui/src/api/client.ts",
    "ui/src/components/shell/AppShell.tsx",
    "ui/src/components/shell/AgentRail.tsx",
    "ui/src/components/research/ReportViewer.tsx",
    "tests/test_api.py",
    "tests/test_rag.py",
]
_EXCLUDED_SEGMENTS = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__"}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "build",
    "for",
    "from",
    "into",
    "implementation",
    "need",
    "plan",
    "planner",
    "that",
    "the",
    "this",
    "with",
}

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


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


class RepoRelevantFile(BaseModel):
    path: str
    reason: str

    @field_validator("path", "reason")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class RepoAnalysis(BaseModel):
    summary: str
    relevant_files: list[RepoRelevantFile] = Field(default_factory=list)
    existing_patterns: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class PlanningApproach(BaseModel):
    name: str
    summary: str
    tradeoffs: list[str] = Field(default_factory=list)
    file_impact: list[str] = Field(default_factory=list)

    @field_validator("name", "summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class PlanningOptions(BaseModel):
    approaches: list[PlanningApproach] = Field(default_factory=list)
    recommended_approach: str
    rationale: str
    out_of_scope: list[str] = Field(default_factory=list)

    @field_validator("recommended_approach", "rationale")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)

    @model_validator(mode="after")
    def validate_approaches(self) -> "PlanningOptions":
        if not self.approaches:
            raise ValueError("approaches must not be empty")
        return self


class SoftwareDevPlanFile(BaseModel):
    path: str
    reason: str

    @field_validator("path", "reason")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class SoftwareDevPlanPhase(BaseModel):
    id: str
    title: str
    objective: str
    files: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)

    @field_validator("id", "title", "objective")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class SoftwareDevPlan(BaseModel):
    title: str
    summary: str
    goal: str
    repo_fit: str
    architecture: str
    recommended_approach: str
    file_map: list[SoftwareDevPlanFile] = Field(default_factory=list)
    data_api_ui_impacts: list[str] = Field(default_factory=list)
    phases: list[SoftwareDevPlanPhase] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

    @field_validator("title", "summary", "goal", "repo_fit", "architecture", "recommended_approach")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)

    @model_validator(mode="after")
    def validate_lists(self) -> "SoftwareDevPlan":
        if not self.file_map:
            raise ValueError("file_map must not be empty")
        if not self.phases:
            raise ValueError("phases must not be empty")
        if not self.validation:
            raise ValueError("validation must not be empty")
        if not self.risks:
            raise ValueError("risks must not be empty")
        if not self.assumptions:
            raise ValueError("assumptions must not be empty")
        if not self.out_of_scope:
            raise ValueError("out_of_scope must not be empty")
        return self

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# {self.title}",
            "",
            f"{self.summary}",
            "",
            "## Goal",
            "",
            self.goal,
            "",
            "## Why this approach fits the current repo",
            "",
            self.repo_fit,
            "",
            "## Architecture summary",
            "",
            self.architecture,
            "",
            "## Recommended approach",
            "",
            self.recommended_approach,
            "",
            "## File/component map",
            "",
        ]
        for item in self.file_map:
            lines.append(f"- `{item.path}` — {item.reason}")
        lines.extend(["", "## Data/API/UI impacts", ""])
        for impact in self.data_api_ui_impacts:
            lines.append(f"- {impact}")
        lines.extend(["", "## Implementation phases", ""])
        for phase in self.phases:
            lines.extend(
                [
                    f"### {phase.id}: {phase.title}",
                    "",
                    phase.objective,
                    "",
                    "Files:",
                ]
            )
            for path in phase.files:
                lines.append(f"- `{path}`")
            lines.append("")
            lines.append("Deliverables:")
            for item in phase.deliverables:
                lines.append(f"- {item}")
            lines.append("")
            lines.append("Verification:")
            for item in phase.verification:
                lines.append(f"- {item}")
            lines.append("")
        lines.extend(["## Validation plan", ""])
        for item in self.validation:
            lines.append(f"- {item}")
        lines.extend(["", "## Risks", ""])
        for item in self.risks:
            lines.append(f"- {item}")
        lines.extend(["", "## Open questions / assumptions", ""])
        for item in self.assumptions:
            lines.append(f"- Assumption: {item}")
        for item in self.open_questions:
            lines.append(f"- Open question: {item}")
        lines.extend(["", "## Out of scope", ""])
        for item in self.out_of_scope:
            lines.append(f"- {item}")
        return "\n".join(lines).strip() + "\n"


class SoftwareDevPlanReview(BaseModel):
    approved: bool
    reviewer_notes: list[str] = Field(default_factory=list)
    revised_plan: SoftwareDevPlan


class SoftwareDevPlanResponse(BaseModel):
    plan: SoftwareDevPlan
    markdown: str
    suggested_filename: str
    planning_brief: PlanningBrief
    repo_analysis: RepoAnalysis
    planning_options: PlanningOptions


@dataclass
class RepoContext:
    inventory: list[str]
    highlighted_files: list[str]

    def render(self) -> str:
        lines = ["Repository inventory:"]
        lines.extend(f"- {item}" for item in self.inventory)
        lines.append("")
        lines.append("Highlighted files:")
        lines.extend(f"- {item}" for item in self.highlighted_files)
        return "\n".join(lines)


def _require_text(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError("field must not be blank")
    return cleaned


def _normalize_keywords(prompt: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_./-]+", prompt.lower())
    keywords: list[str] = []
    for word in words:
        token = word.strip("._-/")
        if len(token) < 4 or token in _STOP_WORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:8]


def _iter_repo_files() -> list[Path]:
    candidates: list[Path] = []
    for root_name in ("src", "ui/src", "tests", "docs", ".archon"):
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(segment in _EXCLUDED_SEGMENTS for segment in path.parts):
                continue
            if path.suffix.lower() not in {".py", ".ts", ".tsx", ".md", ".yaml", ".yml", ".json"}:
                continue
            candidates.append(path)
    return candidates


def _build_repo_context(user_prompt: str) -> RepoContext:
    keywords = _normalize_keywords(user_prompt)
    repo_files = _iter_repo_files()

    inventory_counts: dict[str, int] = {}
    for path in repo_files:
        rel = path.relative_to(PROJECT_ROOT)
        top = rel.parts[0]
        inventory_counts[top] = inventory_counts.get(top, 0) + 1

    inventory = [f"{name}/ ({count} files)" for name, count in sorted(inventory_counts.items())]

    highlighted: list[str] = []
    for rel_path in _DEFAULT_REPO_FILES:
        candidate = PROJECT_ROOT / rel_path
        if candidate.exists():
            highlighted.append(rel_path)

    ranked: list[tuple[int, str]] = []
    for path in repo_files:
        rel = str(path.relative_to(PROJECT_ROOT))
        rel_lower = rel.lower()
        score = sum(2 for keyword in keywords if keyword in rel_lower)
        if score:
            ranked.append((score, rel))

    for _, rel in sorted(ranked, key=lambda item: (-item[0], item[1]))[:12]:
        if rel not in highlighted:
            highlighted.append(rel)

    return RepoContext(inventory=inventory[:10], highlighted_files=highlighted[:16])


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "implementation-plan"


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
    result = llm.invoke(prompt_text)
    raw_text = _llm_result_to_text(result)

    try:
        return parse_model_json(raw_text, model=model)
    except StructuredOutputError as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text=_schema_text(model),
            invalid_response=raw_text,
            validation_error=exc,
        )
        repair_result = llm.invoke(repair_prompt)
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


def _generate_software_dev_plan_sync(user_prompt: str) -> SoftwareDevPlanResponse:
    normalized_prompt = user_prompt.strip()
    if not normalized_prompt:
        raise PlannerValidationError("planner_prompt_required", "A planning request is required.")

    repo_context = _build_repo_context(normalized_prompt)
    shared_context = {
        "user_prompt": normalized_prompt,
        "repo_context": repo_context.render(),
    }

    planning_brief = _invoke_structured_stage(
        prompt_name="software_dev_plan_intake",
        context=shared_context,
        model=PlanningBrief,
        temperature=0.1,
    )
    repo_analysis = _invoke_structured_stage(
        prompt_name="software_dev_plan_repo_analysis",
        context={
            **shared_context,
            "planning_brief_json": planning_brief.model_dump_json(indent=2),
        },
        model=RepoAnalysis,
        temperature=0.1,
    )
    planning_options = _invoke_structured_stage(
        prompt_name="software_dev_plan_options",
        context={
            **shared_context,
            "planning_brief_json": planning_brief.model_dump_json(indent=2),
            "repo_analysis_json": repo_analysis.model_dump_json(indent=2),
        },
        model=PlanningOptions,
        temperature=0.2,
    )
    draft_plan = _invoke_structured_stage(
        prompt_name="software_dev_plan_synthesis",
        context={
            **shared_context,
            "planning_brief_json": planning_brief.model_dump_json(indent=2),
            "repo_analysis_json": repo_analysis.model_dump_json(indent=2),
            "planning_options_json": planning_options.model_dump_json(indent=2),
        },
        model=SoftwareDevPlan,
        temperature=0.15,
    )
    review = _invoke_structured_stage(
        prompt_name="software_dev_plan_review",
        context={
            **shared_context,
            "planning_brief_json": planning_brief.model_dump_json(indent=2),
            "repo_analysis_json": repo_analysis.model_dump_json(indent=2),
            "planning_options_json": planning_options.model_dump_json(indent=2),
            "draft_plan_json": draft_plan.model_dump_json(indent=2),
        },
        model=SoftwareDevPlanReview,
        temperature=0.1,
    )

    final_plan = review.revised_plan
    markdown = final_plan.to_markdown()
    suggested_filename = (
        f"{datetime.now(UTC).date().isoformat()}-{_slugify(final_plan.title)}-implementation-plan.md"
    )
    return SoftwareDevPlanResponse(
        plan=final_plan,
        markdown=markdown,
        suggested_filename=suggested_filename,
        planning_brief=planning_brief,
        repo_analysis=repo_analysis,
        planning_options=planning_options,
    )


async def generate_software_dev_plan(user_prompt: str) -> SoftwareDevPlanResponse:
    return await asyncio.to_thread(_generate_software_dev_plan_sync, user_prompt)
