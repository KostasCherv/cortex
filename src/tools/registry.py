"""Registry for built-in reference tools used in RAG chat."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

@dataclass(frozen=True)
class ToolSpec:
    id: str
    label: str
    description: str
    factory: Callable[[], BaseTool] | None = None
    default_enabled: bool = False
    requires_key: str | None = None
    mcp: bool = False


def _load_reference_specs() -> list[ToolSpec]:
    from src.tools.reference import make_open_library_tool, make_wikipedia_tool

    return [
        ToolSpec(
            id="wikipedia",
            label="Wikipedia",
            description="encyclopedic background on people, places, concepts, and historical topics",
            factory=make_wikipedia_tool,
            default_enabled=True,
        ),
        ToolSpec(
            id="arxiv",
            label="arXiv",
            description=(
                "search_papers, download_paper, read_paper, list_papers, get_abstract "
                "for academic preprints and research papers"
            ),
            mcp=True,
            default_enabled=False,
        ),
        ToolSpec(
            id="open_library",
            label="Open Library",
            description="book metadata, authors, and publication details from the Open Library catalog",
            factory=make_open_library_tool,
            default_enabled=False,
        ),
    ]


REFERENCE_TOOL_SPECS: list[ToolSpec] = _load_reference_specs()
REFERENCE_TOOL_IDS: frozenset[str] = frozenset(spec.id for spec in REFERENCE_TOOL_SPECS)


def default_reference_tool_flags() -> dict[str, bool]:
    return {spec.id: spec.default_enabled for spec in REFERENCE_TOOL_SPECS}


def reference_tool_prompt_lines() -> list[str]:
    return [f"- {spec.id}: {spec.description}" for spec in REFERENCE_TOOL_SPECS]


def reference_flags_from_tools(tools: BaseModel) -> dict[str, bool]:
    return {spec.id: bool(getattr(tools, spec.id)) for spec in REFERENCE_TOOL_SPECS}


def is_arxiv_mcp_enabled(reference_flags: dict[str, bool] | None = None) -> bool:
    flags = reference_flags or default_reference_tool_flags()
    return bool(flags.get("arxiv", False))


def create_rag_chat_tools_model() -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {
        "web_search": (bool, Field(default=True)),
        "composio": (bool, Field(default=False)),
    }
    for spec in REFERENCE_TOOL_SPECS:
        fields[spec.id] = (bool, Field(default=spec.default_enabled))

    create_dynamic_model: Any = create_model
    return create_dynamic_model(
        "RagChatTools",
        __config__=ConfigDict(frozen=True, extra="forbid"),
        **fields,
    )


def build_reference_tools(*, reference_flags: dict[str, bool] | None = None) -> list[BaseTool]:
    from src.config import settings

    flags = reference_flags or default_reference_tool_flags()
    tools: list[BaseTool] = []
    for spec in REFERENCE_TOOL_SPECS:
        if spec.mcp or spec.factory is None:
            continue
        if not flags.get(spec.id, spec.default_enabled):
            continue
        if spec.requires_key and not getattr(settings, spec.requires_key, None):
            continue
        tools.append(spec.factory())
    return tools
