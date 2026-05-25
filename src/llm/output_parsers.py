"""Pydantic models and helpers for structured LLM output."""

import re
from typing import List, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from src.errors import StructuredOutputParseError, StructuredOutputValidationError

T = TypeVar("T")


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


class ResearchSummaryEnvelope(BaseModel):
    """Wrapper for summarize outputs that return an object with a summaries field."""

    summaries: list[ResearchSummary]


class ChatActionDecisionPayload(BaseModel):
    """Validated router decision for chat follow-up handling."""

    action: Literal[
        "answer_direct",
        "answer_from_rag",
        "web_search",
        "fetch_url",
        "ask_clarifying",
    ]
    reason: str
    query: str = ""
    url: str = ""


CHAT_ACTION_DECISION_ADAPTER = TypeAdapter(ChatActionDecisionPayload)
RESEARCH_SUMMARY_LIST_ADAPTER = TypeAdapter(list[ResearchSummary])
MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


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


def _raise_structured_output_error(exc: ValidationError) -> None:
    if any(error.get("type") == "json_invalid" for error in exc.errors()):
        raise StructuredOutputParseError(f"Could not parse structured output JSON: {exc}") from exc
    raise StructuredOutputValidationError(f"Structured output did not match schema: {exc}") from exc


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
    adapter: TypeAdapter[T],
) -> T:
    """Parse JSON-like LLM output directly into an arbitrary typed value."""
    candidate = extract_json_candidate(text)
    try:
        return adapter.validate_json(candidate)
    except ValidationError as exc:
        _raise_structured_output_error(exc)


def parse_chat_action_json(text: str) -> ChatActionDecisionPayload:
    """Parse a structured chat router response into the expected model."""
    return parse_model_json(text, model=ChatActionDecisionPayload)


def parse_research_summaries_json(text: str) -> list[ResearchSummary]:
    """Parse summarize output whether it is a top-level list or an envelope object."""
    candidate = extract_json_candidate(text)
    if candidate.lstrip().startswith("{"):
        return parse_model_json(candidate, model=ResearchSummaryEnvelope).summaries
    return parse_type_json(candidate, adapter=RESEARCH_SUMMARY_LIST_ADAPTER)
