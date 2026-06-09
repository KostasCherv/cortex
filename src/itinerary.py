"""Interactive itinerary planner domain and orchestration helpers."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.db.provider import get_session_store
from src.errors import StructuredOutputError
from src.llm.output_parsers import build_validation_retry_prompt, parse_model_json
from src.prompts.registry import prompt_registry
from src.tools.fetcher import fetch_url_content
from src.tools.web_search import get_web_search_tool
from src import outbox
from src.user_memory import enqueue_memory_refresh, get_user_memory_prompt_block

_ITINERARY_LIST_LIMIT = 20
_MAX_CONTEXT_RESULTS = 3
_MAX_FETCHED_PAGES = 2


class ItineraryPlannerValidationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _workspace_id_for_user(user_id: str) -> str:
    return user_id


def _require_text(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError("field must not be blank")
    return cleaned


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        cleaned = _normalize_optional_text(raw)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


class PlannerTravelRequirements(BaseModel):
    destination: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    trip_length_days: int | None = Field(default=None, ge=1, le=60)
    traveler_count: int | None = Field(default=None, ge=1, le=20)
    party_type: str | None = None
    budget_band: str | None = None
    interests: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    pace: str | None = None

    @field_validator("destination", "start_date", "end_date", "party_type", "budget_band", "pace")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value)

    @field_validator("interests", "constraints", mode="before")
    @classmethod
    def normalize_list_text(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return _dedupe_preserve(parts)
        if isinstance(value, list):
            return _dedupe_preserve([str(item) for item in value])
        return []

    def merge(self, update: "PlannerTravelRequirements") -> "PlannerTravelRequirements":
        merged = self.model_copy(deep=True)
        for field_name in (
            "destination",
            "start_date",
            "end_date",
            "trip_length_days",
            "traveler_count",
            "party_type",
            "budget_band",
            "pace",
        ):
            value = getattr(update, field_name)
            if value is not None:
                setattr(merged, field_name, value)
        if update.interests:
            merged.interests = _dedupe_preserve([*merged.interests, *update.interests])
        if update.constraints:
            merged.constraints = _dedupe_preserve([*merged.constraints, *update.constraints])
        return merged

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.destination:
            missing.append("destination")
        if not self.start_date or not self.end_date:
            missing.append("dates")
        if self.traveler_count is None:
            missing.append("traveler_count")
        if not self.party_type:
            missing.append("party_type")
        if not self.budget_band:
            missing.append("budget_band")
        if not self.interests:
            missing.append("interests")
        if not self.pace and not self.constraints:
            missing.append("pace_or_constraints")
        return missing


class PlannerTravelRequirementsUpdate(PlannerTravelRequirements):
    """Partial update model for extraction from chat."""


class ItineraryDay(BaseModel):
    day_number: int = Field(ge=1)
    title: str
    morning: list[str] = Field(default_factory=list)
    afternoon: list[str] = Field(default_factory=list)
    evening: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _require_text(value)

    @field_validator("morning", "afternoon", "evening", "notes", mode="before")
    @classmethod
    def normalize_text_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return _dedupe_preserve([str(item) for item in value])
        if isinstance(value, str):
            return _dedupe_preserve([value])
        return []


class RecommendedArea(BaseModel):
    name: str
    why: str
    vibe: str

    @field_validator("name", "why", "vibe")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)


class GeneratedItinerary(BaseModel):
    title: str
    summary: str
    destination: str
    budget_band: str
    days: list[ItineraryDay] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    recommended_areas: list[RecommendedArea] = Field(default_factory=list)
    getting_there: list[str] = Field(default_factory=list)
    getting_around: list[str] = Field(default_factory=list)
    must_do_highlights: list[str] = Field(default_factory=list)
    booking_advice: list[str] = Field(default_factory=list)
    revision_summary: str | None = None

    @field_validator("title", "summary", "destination", "budget_band")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_text(value)

    @field_validator("tips", "getting_there", "getting_around", "must_do_highlights", "booking_advice", mode="before")
    @classmethod
    def normalize_tips(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return _dedupe_preserve([str(item) for item in value])
        if isinstance(value, str):
            return _dedupe_preserve([value])
        return []


class ItinerarySessionMessage(BaseModel):
    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _require_text(value)


class ItineraryVersion(BaseModel):
    version_id: str
    session_id: str
    version_number: int = Field(ge=1)
    revision_summary: str
    markdown: str
    itinerary: GeneratedItinerary
    created_at: str

    @field_validator("revision_summary", "markdown")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_text(value)

    @property
    def structured_itinerary(self) -> GeneratedItinerary:
        return self.itinerary


class ItinerarySessionSummary(BaseModel):
    session_id: str
    owner_id: str
    workspace_id: str
    title: str
    status: str
    current_version_id: str | None = None
    prompt_preview: str = ""
    last_message_preview: str = ""
    created_at: str
    updated_at: str

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> str:
        return _normalize_optional_text(str(value) if value is not None else "") or "New itinerary"


class ItinerarySessionDetail(ItinerarySessionSummary):
    requirements: PlannerTravelRequirements = Field(default_factory=PlannerTravelRequirements)
    messages: list[ItinerarySessionMessage] = Field(default_factory=list)
    versions: list[ItineraryVersion] = Field(default_factory=list)
    current_version: ItineraryVersion | None = None


class ItinerarySessionListResponse(BaseModel):
    sessions: list[ItinerarySessionSummary] = Field(default_factory=list)


class ItineraryPlannerResponse(BaseModel):
    session: ItinerarySessionDetail
    assistant_message: ItinerarySessionMessage
    current_itinerary: GeneratedItinerary | None = None
    new_version: ItineraryVersion | None = None
    created_new_version: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class _ItineraryMessageMetadata(BaseModel):
    action: Literal["collect_requirements", "generate_itinerary", "revise_itinerary"]
    missing_fields: list[str] = Field(default_factory=list)
    revision_summary: str | None = None


def _prompt_preview(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= 160:
        return cleaned
    return cleaned[:159].rstrip() + "…"


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _suggest_session_title(
    requirements: PlannerTravelRequirements,
    *,
    fallback_text: str,
    itinerary: GeneratedItinerary | None = None,
) -> str:
    if itinerary and itinerary.title:
        return itinerary.title[:120]
    if requirements.destination:
        if requirements.trip_length_days:
            return f"{requirements.destination} {requirements.trip_length_days}-day plan"[:120]
        return f"{requirements.destination} itinerary"[:120]
    preview = _prompt_preview(fallback_text)
    return preview or "New itinerary"


def _llm_result_to_text(result: object) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2)


def _render_itinerary_markdown(itinerary: GeneratedItinerary) -> str:
    lines = [f"# {itinerary.title}", "", itinerary.summary, "", "## Overview", ""]
    lines.append(f"- Destination: {itinerary.destination}")
    lines.append(f"- Budget: {itinerary.budget_band}")
    if itinerary.recommended_areas:
        lines.extend(["", "## Recommended Areas", ""])
        for area in itinerary.recommended_areas:
            lines.append(f"- {area.name}: {area.why} ({area.vibe})")
    if itinerary.getting_there:
        lines.extend(["", "## Getting There", ""])
        for item in itinerary.getting_there:
            lines.append(f"- {item}")
    if itinerary.getting_around:
        lines.extend(["", "## Getting Around", ""])
        for item in itinerary.getting_around:
            lines.append(f"- {item}")
    lines.extend(["", "## Daily Plan", ""])
    for day in itinerary.days:
        lines.append(f"### Day {day.day_number}: {day.title}")
        lines.append("")
        if day.morning:
            lines.append("- Morning: " + "; ".join(day.morning))
        if day.afternoon:
            lines.append("- Afternoon: " + "; ".join(day.afternoon))
        if day.evening:
            lines.append("- Evening: " + "; ".join(day.evening))
        if day.notes:
            lines.append("- Notes: " + "; ".join(day.notes))
        lines.append("")
    if itinerary.tips:
        lines.extend(["## Tips", ""])
        for tip in itinerary.tips:
            lines.append(f"- {tip}")
    if itinerary.must_do_highlights:
        lines.extend(["", "## Must-Do Highlights", ""])
        for item in itinerary.must_do_highlights:
            lines.append(f"- {item}")
    if itinerary.booking_advice:
        lines.extend(["", "## Booking Advice", ""])
        for item in itinerary.booking_advice:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def _session_summary_from_row(row: dict[str, Any]) -> ItinerarySessionSummary:
    return ItinerarySessionSummary(
        session_id=str(row["id"]),
        owner_id=str(row.get("owner_id") or ""),
        workspace_id=str(row.get("workspace_id") or ""),
        title=str(row.get("title") or "New itinerary"),
        status=str(row.get("status") or "collecting_requirements"),
        current_version_id=str(row["current_version_id"]) if row.get("current_version_id") else None,
        prompt_preview=str(row.get("prompt_preview") or ""),
        last_message_preview=str(row.get("last_message_preview") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


def _version_from_row(row: dict[str, Any]) -> ItineraryVersion:
    return ItineraryVersion(
        version_id=str(row["id"]),
        session_id=str(row.get("session_id") or ""),
        version_number=int(row.get("version_number") or 1),
        revision_summary=str(row.get("revision_summary") or ""),
        markdown=str(row.get("markdown") or ""),
        itinerary=GeneratedItinerary.model_validate(row.get("itinerary_json") or {}),
        created_at=str(row.get("created_at") or ""),
    )


def _message_from_row(row: dict[str, Any]) -> ItinerarySessionMessage:
    return ItinerarySessionMessage(
        message_id=str(row["id"]),
        session_id=str(row.get("session_id") or ""),
        role=str(row.get("role") or "assistant"),  # type: ignore[arg-type]
        content=str(row.get("content") or ""),
        metadata=row.get("metadata_json") or {},
        created_at=str(row.get("created_at") or ""),
    )


def _session_detail_from_parts(
    row: dict[str, Any],
    *,
    messages: list[ItinerarySessionMessage],
    versions: list[ItineraryVersion],
) -> ItinerarySessionDetail:
    summary = _session_summary_from_row(row)
    current_version_id = summary.current_version_id
    current_version = next((item for item in versions if item.version_id == current_version_id), None)
    return ItinerarySessionDetail(
        **summary.model_dump(),
        requirements=PlannerTravelRequirements.model_validate(row.get("requirements_json") or {}),
        messages=messages,
        versions=versions,
        current_version=current_version or (versions[-1] if versions else None),
    )


async def create_itinerary_session(user_id: str) -> ItinerarySessionSummary:
    now = datetime.now(UTC).isoformat()
    row = {
        "id": str(uuid.uuid4()),
        "owner_id": user_id,
        "workspace_id": _workspace_id_for_user(user_id),
        "title": "New itinerary",
        "status": "collecting_requirements",
        "requirements_json": PlannerTravelRequirements().model_dump(mode="json"),
        "current_version_id": None,
        "prompt_preview": "",
        "last_message_preview": "",
        "created_at": now,
        "updated_at": now,
    }
    await get_session_store().create_itinerary_session(row)
    return _session_summary_from_row(row)


async def list_itinerary_sessions(user_id: str) -> ItinerarySessionListResponse:
    rows = await get_session_store().list_itinerary_sessions(
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
        limit=_ITINERARY_LIST_LIMIT,
    )
    return ItinerarySessionListResponse(sessions=[_session_summary_from_row(row) for row in rows])


async def get_itinerary_session_detail(session_id: str, user_id: str) -> ItinerarySessionDetail | None:
    row = await get_session_store().get_itinerary_session(
        session_id=session_id,
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )
    if row is None:
        return None
    messages = [_message_from_row(item) for item in await get_session_store().list_itinerary_messages(session_id=session_id, owner_id=user_id)]
    versions = [_version_from_row(item) for item in await get_session_store().list_itinerary_versions(session_id=session_id, owner_id=user_id)]
    return _session_detail_from_parts(row, messages=messages, versions=versions)


async def rename_itinerary_session(session_id: str, user_id: str, title: str) -> bool:
    normalized_title = _require_text(title)[:120]
    return await get_session_store().update_itinerary_session(
        session_id=session_id,
        owner_id=user_id,
        patch={"title": normalized_title, "updated_at": datetime.now(UTC).isoformat()},
    )


async def delete_itinerary_session(session_id: str, user_id: str) -> bool:
    return await get_session_store().delete_itinerary_session(
        session_id=session_id,
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )


async def append_itinerary_message(
    *,
    user_id: str,
    session_id: str,
    role: Literal["user", "assistant"],
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ItinerarySessionMessage:
    message = ItinerarySessionMessage(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        metadata=metadata or {},
        created_at=datetime.now(UTC).isoformat(),
    )
    await get_session_store().create_itinerary_message(
        {
            "id": message.message_id,
            "session_id": session_id,
            "owner_id": user_id,
            "role": role,
            "content": content,
            "metadata_json": message.metadata,
            "created_at": message.created_at,
        }
    )
    return message


async def update_itinerary_session(*, user_id: str, session_id: str, patch: dict[str, Any]) -> bool:
    return await get_session_store().update_itinerary_session(
        session_id=session_id,
        owner_id=user_id,
        patch=patch,
    )


async def create_itinerary_version(
    *,
    user_id: str,
    session_id: str,
    itinerary: GeneratedItinerary,
    version_number: int,
    revision_summary: str,
) -> ItineraryVersion:
    version = ItineraryVersion(
        version_id=str(uuid.uuid4()),
        session_id=session_id,
        version_number=version_number,
        revision_summary=revision_summary,
        markdown=_render_itinerary_markdown(itinerary),
        itinerary=itinerary,
        created_at=datetime.now(UTC).isoformat(),
    )
    await get_session_store().create_itinerary_version(
        {
            "id": version.version_id,
            "session_id": session_id,
            "owner_id": user_id,
            "version_number": version_number,
            "revision_summary": revision_summary,
            "markdown": version.markdown,
            "itinerary_json": itinerary.model_dump(mode="json"),
            "created_at": version.created_at,
        }
    )
    return version


async def search_destination_context(destination: str) -> str:
    query = f"best current travel advice and neighborhoods for {destination}"
    tool = get_web_search_tool()
    results = await asyncio.to_thread(tool.search, query, _MAX_CONTEXT_RESULTS)
    snippets: list[str] = []
    for row in results[:_MAX_CONTEXT_RESULTS]:
        title = str(row.get("title") or row.get("name") or "Untitled source")
        url = str(row.get("url") or "")
        content = str(row.get("content") or row.get("raw_content") or "").strip()
        if not content and url and len(snippets) < _MAX_FETCHED_PAGES:
            try:
                content = await fetch_url_content(url)
            except Exception:
                content = ""
        if content:
            snippets.append(f"Source: {title}\nURL: {url}\nNotes: {content[:1800]}")
    return "\n\n".join(snippets)


def _invoke_structured_stage(
    *,
    prompt_name: str,
    context: dict[str, object],
    model: type[BaseModel],
    failure_code: str,
    failure_message: str,
    temperature: float = 0.2,
) -> BaseModel:
    from src.llm.factory import get_llm

    prompt_text, _ = prompt_registry.render(prompt_name, context)
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
            raise ItineraryPlannerValidationError(failure_code, failure_message) from repair_exc
    except ValidationError as exc:
        raise ItineraryPlannerValidationError(failure_code, failure_message) from exc


async def _extract_requirements(
    *,
    current_requirements: PlannerTravelRequirements,
    conversation_text: str,
    message: str,
    user_id: str,
) -> PlannerTravelRequirements:
    user_memory_context = await get_user_memory_prompt_block(user_id, message)
    parsed = await asyncio.to_thread(
        _invoke_structured_stage,
        prompt_name="itinerary_requirements_extract",
        context={
            "current_requirements_json": current_requirements.model_dump_json(indent=2),
            "conversation_text": conversation_text,
            "user_message": message,
            "user_memory_context": user_memory_context,
            "requirements_schema_json": json.dumps(
                PlannerTravelRequirementsUpdate.model_json_schema(), indent=2
            ),
        },
        model=PlannerTravelRequirementsUpdate,
        failure_code="itinerary_generation_failed",
        failure_message="Travel requirement extraction failed.",
        temperature=0.1,
    )
    return current_requirements.merge(parsed)  # type: ignore[arg-type]


async def _generate_followup_question(
    *,
    requirements: PlannerTravelRequirements,
    missing_fields: list[str],
    conversation_text: str,
    user_id: str,
) -> str:
    from src.llm.factory import get_llm

    user_memory_context = await get_user_memory_prompt_block(
        user_id,
        f"{conversation_text}\nMissing fields: {', '.join(missing_fields)}",
    )
    prompt_text, _ = prompt_registry.render(
        "itinerary_followup_question",
        {
            "requirements_json": requirements.model_dump_json(indent=2),
            "missing_fields_json": json.dumps(missing_fields, indent=2),
            "conversation_text": conversation_text,
            "user_memory_context": user_memory_context,
        },
    )
    llm = get_llm(temperature=0.2)
    result = await asyncio.to_thread(llm.invoke, prompt_text)
    text = _llm_result_to_text(result).strip()
    return text or "What dates are you considering for this trip?"


async def _generate_itinerary(
    *,
    requirements: PlannerTravelRequirements,
    conversation_text: str,
    failure_code: str,
    failure_message: str,
    user_id: str,
) -> GeneratedItinerary:
    grounding_context = await search_destination_context(requirements.destination or "")
    user_memory_context = await get_user_memory_prompt_block(
        user_id,
        f"{requirements.destination or ''} {conversation_text}",
    )
    parsed = await asyncio.to_thread(
        _invoke_structured_stage,
        prompt_name="itinerary_generate",
        context={
            "requirements_json": requirements.model_dump_json(indent=2),
            "conversation_text": conversation_text,
            "user_memory_context": user_memory_context,
            "grounding_context": grounding_context,
            "itinerary_schema_json": json.dumps(GeneratedItinerary.model_json_schema(), indent=2),
        },
        model=GeneratedItinerary,
        failure_code=failure_code,
        failure_message=failure_message,
        temperature=0.25,
    )
    return parsed  # type: ignore[return-value]


async def _revise_itinerary(
    *,
    requirements: PlannerTravelRequirements,
    current_itinerary: GeneratedItinerary,
    conversation_text: str,
    user_message: str,
    user_id: str,
) -> GeneratedItinerary:
    grounding_context = await search_destination_context(requirements.destination or current_itinerary.destination)
    user_memory_context = await get_user_memory_prompt_block(user_id, user_message)
    parsed = await asyncio.to_thread(
        _invoke_structured_stage,
        prompt_name="itinerary_revise",
        context={
            "requirements_json": requirements.model_dump_json(indent=2),
            "current_itinerary_json": current_itinerary.model_dump_json(indent=2),
            "conversation_text": conversation_text,
            "user_message": user_message,
            "user_memory_context": user_memory_context,
            "grounding_context": grounding_context,
            "itinerary_schema_json": json.dumps(GeneratedItinerary.model_json_schema(), indent=2),
        },
        model=GeneratedItinerary,
        failure_code="itinerary_generation_failed",
        failure_message="Revised itinerary output could not be validated.",
        temperature=0.2,
    )
    return parsed  # type: ignore[return-value]


def _conversation_text(session: ItinerarySessionDetail) -> str:
    lines = []
    for message in session.messages:
        lines.append(f"{message.role.upper()}: {message.content}")
    return "\n".join(lines)


def _detail_with_updates(
    session: ItinerarySessionDetail,
    *,
    requirements: PlannerTravelRequirements | None = None,
    title: str | None = None,
    status: str | None = None,
    current_version_id: str | None = None,
    messages: list[ItinerarySessionMessage] | None = None,
    versions: list[ItineraryVersion] | None = None,
) -> ItinerarySessionDetail:
    next_versions = versions if versions is not None else session.versions
    next_current_version_id = current_version_id if current_version_id is not None else session.current_version_id
    next_current_version = next(
        (item for item in next_versions if item.version_id == next_current_version_id),
        None,
    )
    return session.model_copy(
        update={
            "requirements": requirements or session.requirements,
            "title": title or session.title,
            "status": status or session.status,
            "current_version_id": next_current_version_id,
            "messages": messages if messages is not None else session.messages,
            "versions": next_versions,
            "current_version": next_current_version or (next_versions[-1] if next_versions else None),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )


async def process_itinerary_message(
    *,
    session: ItinerarySessionDetail,
    user_id: str,
    message: str,
) -> ItineraryPlannerResponse:
    normalized_message = _require_text(message)
    user_message = await append_itinerary_message(
        user_id=user_id,
        session_id=session.session_id,
        role="user",
        content=normalized_message,
        metadata={},
    )
    working_session = _detail_with_updates(
        session,
        messages=[*session.messages, user_message],
    )
    conversation_text = _conversation_text(working_session)

    if working_session.current_version is not None:
        revised = await _revise_itinerary(
            requirements=working_session.requirements,
            current_itinerary=working_session.current_version.structured_itinerary,
            conversation_text=conversation_text,
            user_message=normalized_message,
            user_id=user_id,
        )
        revision_summary = revised.revision_summary or "Updated the itinerary based on your latest request."
        new_version = await create_itinerary_version(
            user_id=user_id,
            session_id=working_session.session_id,
            itinerary=revised,
            version_number=len(working_session.versions) + 1,
            revision_summary=revision_summary,
        )
        assistant_message = await append_itinerary_message(
            user_id=user_id,
            session_id=working_session.session_id,
            role="assistant",
            content=f"I updated your itinerary. {revision_summary}",
            metadata=_ItineraryMessageMetadata(
                action="revise_itinerary",
                missing_fields=[],
                revision_summary=revision_summary,
            ).model_dump(mode="json"),
        )
        await enqueue_memory_refresh(
            user_id=user_id,
            source_mode="itinerary",
            source_session_id=working_session.session_id,
            user_message=normalized_message,
            assistant_message=assistant_message.content,
            source_user_message_id=user_message.message_id,
            source_assistant_message_id=assistant_message.message_id,
        )
        asyncio.create_task(outbox.dispatch_outbox_events(limit=10))
        title = _suggest_session_title(
            working_session.requirements,
            fallback_text=normalized_message,
            itinerary=revised,
        )
        await update_itinerary_session(
            user_id=user_id,
            session_id=working_session.session_id,
            patch={
                "title": title,
                "status": "generated",
                "current_version_id": new_version.version_id,
                "last_message_preview": _prompt_preview(normalized_message),
                "updated_at": assistant_message.created_at,
            },
        )
        updated_session = _detail_with_updates(
            working_session,
            title=title,
            status="generated",
            current_version_id=new_version.version_id,
            messages=[*working_session.messages, assistant_message],
            versions=[*working_session.versions, new_version],
        )
        return ItineraryPlannerResponse(
            session=updated_session,
            assistant_message=assistant_message,
            current_itinerary=revised,
            new_version=new_version,
            created_new_version=True,
            missing_fields=[],
        )

    updated_requirements = await _extract_requirements(
        current_requirements=working_session.requirements,
        conversation_text=conversation_text,
        message=normalized_message,
        user_id=user_id,
    )
    missing_fields = updated_requirements.missing_fields()

    if missing_fields:
        followup = await _generate_followup_question(
            requirements=updated_requirements,
            missing_fields=missing_fields,
            conversation_text=conversation_text,
            user_id=user_id,
        )
        assistant_message = await append_itinerary_message(
            user_id=user_id,
            session_id=working_session.session_id,
            role="assistant",
            content=followup,
            metadata=_ItineraryMessageMetadata(
                action="collect_requirements",
                missing_fields=missing_fields,
            ).model_dump(mode="json"),
        )
        await enqueue_memory_refresh(
            user_id=user_id,
            source_mode="itinerary",
            source_session_id=working_session.session_id,
            user_message=normalized_message,
            assistant_message=assistant_message.content,
            source_user_message_id=user_message.message_id,
            source_assistant_message_id=assistant_message.message_id,
        )
        asyncio.create_task(outbox.dispatch_outbox_events(limit=10))
        title = _suggest_session_title(updated_requirements, fallback_text=normalized_message)
        await update_itinerary_session(
            user_id=user_id,
            session_id=working_session.session_id,
            patch={
                "title": title,
                "status": "collecting_requirements",
                "requirements_json": updated_requirements.model_dump(mode="json"),
                "prompt_preview": (
                    session.prompt_preview or _prompt_preview(normalized_message)
                ),
                "last_message_preview": _prompt_preview(normalized_message),
                "updated_at": assistant_message.created_at,
            },
        )
        updated_session = _detail_with_updates(
            working_session,
            requirements=updated_requirements,
            title=title,
            status="collecting_requirements",
            messages=[*working_session.messages, assistant_message],
        )
        return ItineraryPlannerResponse(
            session=updated_session,
            assistant_message=assistant_message,
            current_itinerary=None,
            new_version=None,
            created_new_version=False,
            missing_fields=missing_fields,
        )

    generated = await _generate_itinerary(
        requirements=updated_requirements,
        conversation_text=conversation_text,
        failure_code="itinerary_generation_failed",
        failure_message="Generated itinerary output could not be validated.",
        user_id=user_id,
    )
    revision_summary = generated.revision_summary or "Generated the first itinerary draft."
    new_version = await create_itinerary_version(
        user_id=user_id,
        session_id=working_session.session_id,
        itinerary=generated,
        version_number=len(working_session.versions) + 1,
        revision_summary=revision_summary,
    )
    assistant_message = await append_itinerary_message(
        user_id=user_id,
        session_id=working_session.session_id,
        role="assistant",
        content=f"I generated your itinerary. {generated.summary}",
        metadata=_ItineraryMessageMetadata(
            action="generate_itinerary",
            missing_fields=[],
            revision_summary=revision_summary,
        ).model_dump(mode="json"),
    )
    await enqueue_memory_refresh(
        user_id=user_id,
        source_mode="itinerary",
        source_session_id=working_session.session_id,
        user_message=normalized_message,
        assistant_message=assistant_message.content,
        source_user_message_id=user_message.message_id,
        source_assistant_message_id=assistant_message.message_id,
    )
    asyncio.create_task(outbox.dispatch_outbox_events(limit=10))
    title = _suggest_session_title(updated_requirements, fallback_text=normalized_message, itinerary=generated)
    await update_itinerary_session(
        user_id=user_id,
        session_id=working_session.session_id,
        patch={
            "title": title,
            "status": "generated",
            "requirements_json": updated_requirements.model_dump(mode="json"),
            "current_version_id": new_version.version_id,
            "prompt_preview": session.prompt_preview or _prompt_preview(normalized_message),
            "last_message_preview": _prompt_preview(normalized_message),
            "updated_at": assistant_message.created_at,
        },
    )
    updated_session = _detail_with_updates(
        working_session,
        requirements=updated_requirements,
        title=title,
        status="generated",
        current_version_id=new_version.version_id,
        messages=[*working_session.messages, assistant_message],
        versions=[*working_session.versions, new_version],
    )
    return ItineraryPlannerResponse(
        session=updated_session,
        assistant_message=assistant_message,
        current_itinerary=generated,
        new_version=new_version,
        created_new_version=True,
        missing_fields=[],
    )


async def process_itinerary_session_message(
    session_id: str,
    user_id: str,
    message: str,
) -> ItineraryPlannerResponse:
    session = await get_itinerary_session_detail(session_id, user_id)
    if session is None:
        raise ItineraryPlannerValidationError(
            "itinerary_session_not_found",
            f"Itinerary session '{session_id}' not found.",
        )
    return await process_itinerary_message(session=session, user_id=user_id, message=message)
