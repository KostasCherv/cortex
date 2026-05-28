"""Pydantic models and helpers for structured LLM output."""

import re
from typing import Any, List, Literal, NoReturn, TypeVar, cast

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator, model_validator

from src.errors import StructuredOutputParseError, StructuredOutputValidationError

T = TypeVar("T")
MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


def _trim_required_text(value: str, *, field_name: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{field_name} must not be blank")
    return trimmed


def _trim_optional_text(value: str) -> str:
    return value.strip()


class ResearchSource(BaseModel):
    """A citation source used in the research."""
    title: str = Field(description="The title of the website or article")
    url: str = Field(description="The source URL")
    summary: str = Field(description="A brief summary of what this source contributed")


class ResearchReport(BaseModel):
    """Structured research report with key sections."""
    title: str = Field(description="A descriptive title for the research")
    executive_summary: str = Field(description="A high-level summary of the findings")
    key_findings: List[str] = Field(description="List of core insights discovered")
    conclusion: str = Field(description="Closing summary and final thoughts")
    sources: List[ResearchSource] = Field(default_factory=list, description="List of sources cited")

    def to_markdown(self) -> str:
        """Convert the structured report to a polished markdown string."""
        md = f"# {self.title}\n\n"
        md += "## Executive Summary\n\n"
        md += f"{self.executive_summary}\n\n"
        md += "## Key Findings\n\n"
        for finding in self.key_findings:
            md += f"- {finding}\n"
        md += "\n## Conclusion\n\n"
        md += f"{self.conclusion}\n\n"
        
        if self.sources:
            md += "## References\n\n"
            for src in self.sources:
                md += f"- [{src.title}]({src.url})\n"

        return md


class ResearchSummary(BaseModel):
    """Validated source summary returned by the summarization LLM."""

    url: str = Field(description="The source URL")
    title: str = Field(default="", description="The source title")
    summary: str = Field(description="A concise summary grounded in the source")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _trim_required_text(value, field_name="url")

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return _trim_optional_text(value)[:200]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _trim_required_text(value, field_name="summary")


class ResearchSummaryEnvelope(BaseModel):
    """Wrapper for summarize outputs that return an object with a summaries field."""

    summaries: list[ResearchSummary]


class ChatActionDecisionPayload(BaseModel):
    """Validated router decision for chat follow-up handling."""

    action: Literal[
        "answer_direct",
        "answer_from_rag",
        "web_search",
        "asset_price",
        "search_finance_tools",
        "fetch_url",
        "ask_clarifying",
    ]
    reason: str
    query: str = ""
    url: str = ""
    symbols: list[str] = Field(default_factory=list)
    currency: str = ""

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _trim_required_text(value, field_name="reason")

    @field_validator("query", "url", "currency")
    @classmethod
    def normalize_optional_text(cls, value: str) -> str:
        return _trim_optional_text(value)

    @field_validator("symbols", mode="before")
    @classmethod
    def normalize_symbols(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return value

    @model_validator(mode="after")
    def validate_action_requirements(self) -> "ChatActionDecisionPayload":
        if self.action == "web_search" and not self.query:
            raise ValueError("query is required when action is web_search")
        if self.action == "search_finance_tools" and not self.query:
            raise ValueError("query is required when action is search_finance_tools")
        if self.action == "asset_price" and not self.symbols:
            raise ValueError("symbols are required when action is asset_price")
        if self.action == "fetch_url" and not self.url:
            raise ValueError("url is required when action is fetch_url")
        return self


class FinanceToolSelectionPayload(BaseModel):
    """Validated tool-selection response for progressive finance tool discovery."""

    tool_name: str
    reason: str

    @field_validator("tool_name", "reason")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _trim_required_text(value, field_name=str(info.field_name))


class FinanceToolCallPlanPayload(BaseModel):
    """Validated plan for whether and how to call a discovered finance tool."""

    should_call: bool
    reason: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    clarifying_question: str = ""

    @field_validator("reason", "clarifying_question")
    @classmethod
    def normalize_text(cls, value: str, info) -> str:
        if info.field_name == "reason":
            return _trim_required_text(value, field_name="reason")
        return _trim_optional_text(value)

    @model_validator(mode="after")
    def validate_callability(self) -> "FinanceToolCallPlanPayload":
        if not self.should_call and not self.clarifying_question:
            raise ValueError(
                "clarifying_question is required when should_call is false"
            )
        return self


class ExtractedEntity(BaseModel):
    """Validated entity extracted from graph-ingestion text."""

    name: str = Field(description="Entity display name")
    entity_type: str = Field(default="Unknown", description="Entity type label")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _trim_required_text(value, field_name="name")

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, value: str) -> str:
        trimmed = _trim_optional_text(value or "Unknown")
        return (trimmed or "Unknown")[:80]


class ExtractedRelation(BaseModel):
    """Validated relation extracted from graph-ingestion text."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(default="RELATED", description="Relationship type")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("source", "target")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _trim_required_text(value, field_name="relation endpoint")

    @field_validator("type")
    @classmethod
    def normalize_relation_type(cls, value: str) -> str:
        trimmed = _trim_optional_text(value or "RELATED")
        return (trimmed or "RELATED")[:80]


class EntityRelationExtractionEnvelope(BaseModel):
    """Validated entity and relation extraction payload."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class ClarificationDecision(BaseModel):
    """Router decision produced by the clarification node."""

    ready: bool
    question: str | None = None
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _trim_required_text(value, field_name="reason")

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    @model_validator(mode="after")
    def validate_question_required_when_not_ready(self) -> "ClarificationDecision":
        if not self.ready and not self.question:
            raise ValueError("question is required when ready is false")
        return self


CHAT_ACTION_DECISION_ADAPTER = TypeAdapter(ChatActionDecisionPayload)
RESEARCH_SUMMARY_LIST_ADAPTER = TypeAdapter(list[ResearchSummary])
FINANCE_TOOL_SELECTION_ADAPTER = TypeAdapter(FinanceToolSelectionPayload)
FINANCE_TOOL_CALL_PLAN_ADAPTER = TypeAdapter(FinanceToolCallPlanPayload)


def extract_json_candidate(text: str) -> str:
    """Normalize common LLM wrappers and return the most likely JSON substring."""
    candidate = text.strip()
    if not candidate:
        return ""

    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    starts = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx != -1]
    if not starts:
        return candidate

    start = min(starts)
    opening = candidate[start]
    closing = "}" if opening == "{" else "]"
    end = candidate.rfind(closing)
    if end != -1 and end > start:
        return candidate[start : end + 1].strip()
    return candidate[start:].strip()


def _raise_structured_output_error(exc: ValidationError) -> NoReturn:
    details = cast(list[dict[str, Any]], exc.errors())
    if any(error.get("type") == "json_invalid" for error in exc.errors()):
        raise StructuredOutputParseError(
            f"Could not parse structured output JSON: {exc}",
            details=details,
        ) from exc
    raise StructuredOutputValidationError(
        f"Structured output did not match schema: {exc}",
        details=details,
    ) from exc


def format_validation_error_details(exc: Exception, *, max_items: int = 5) -> str:
    """Convert structured output failures into compact retry-friendly feedback."""
    details = getattr(exc, "details", None) or []
    if not details:
        return f"- root (error): {str(exc).strip()}"

    lines: list[str] = []
    for item in details[:max_items]:
        loc = item.get("loc") or ()
        path = ".".join(str(part) for part in loc) if loc else "root"
        error_type = str(item.get("type") or "error")
        message = str(item.get("msg") or "Validation failed")
        lines.append(f"- {path} ({error_type}): {message}")

    if len(details) > max_items:
        lines.append(f"- root (truncated): {len(details) - max_items} additional validation issues omitted")
    return "\n".join(lines)


def build_validation_retry_prompt(
    *,
    schema_text: str,
    invalid_response: str,
    validation_error: Exception,
) -> str:
    """Build a consistent repair prompt after parse or validation failure."""
    return (
        "Validation failed. Return valid JSON only that matches this schema exactly.\n\n"
        f"Schema:\n{schema_text}\n\n"
        "Validation errors:\n"
        f"{format_validation_error_details(validation_error)}\n\n"
        "Do not add markdown fences or explanations.\n\n"
        "Previous response:\n"
        f"{invalid_response}"
    )


def parse_model_json(
    text: str,
    *,
    model: type[MODEL_T],
) -> MODEL_T:
    """Parse JSON-like LLM output directly into a Pydantic model."""
    candidate = extract_json_candidate(text)

    try:
        return model.model_validate_json(candidate)
    except ValidationError as exc:
        _raise_structured_output_error(exc)


def parse_type_json(
    text: str,
    *,
    adapter: TypeAdapter[Any],
) -> Any:
    """Parse JSON-like LLM output directly into an arbitrary typed value."""
    candidate = extract_json_candidate(text)
    try:
        return adapter.validate_json(candidate)
    except ValidationError as exc:
        _raise_structured_output_error(exc)


def parse_chat_action_json(text: str) -> ChatActionDecisionPayload:
    """Parse a structured chat router response into the expected model."""
    return parse_model_json(text, model=ChatActionDecisionPayload)


def parse_finance_tool_selection_json(text: str) -> FinanceToolSelectionPayload:
    """Parse a finance tool-selection response."""
    return parse_model_json(text, model=FinanceToolSelectionPayload)


def parse_finance_tool_call_plan_json(text: str) -> FinanceToolCallPlanPayload:
    """Parse a finance tool call-plan response."""
    return parse_model_json(text, model=FinanceToolCallPlanPayload)


def parse_research_summaries_json(text: str) -> list[ResearchSummary]:
    """Parse summarize output whether it is a top-level list or an envelope object."""
    candidate = extract_json_candidate(text)
    if candidate.lstrip().startswith("{"):
        return parse_model_json(candidate, model=ResearchSummaryEnvelope).summaries
    return parse_type_json(candidate, adapter=RESEARCH_SUMMARY_LIST_ADAPTER)


def parse_entity_relation_extraction_json(text: str) -> EntityRelationExtractionEnvelope:
    """Parse graph extraction output into validated nested entity and relation models."""
    return parse_model_json(text, model=EntityRelationExtractionEnvelope)


def parse_clarification_decision_json(text: str) -> ClarificationDecision:
    """Parse the clarification node's router decision into the expected model."""
    return parse_model_json(text, model=ClarificationDecision)
