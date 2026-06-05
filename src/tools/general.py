"""General-purpose LangChain tools for RAG agent chat."""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from langchain_tavily import TavilyExtract
from pydantic import BaseModel, Field

from src.config import settings
from src.tools.registry import build_reference_tools
from src.tools.search import perform_search_cached

GENERAL_WEB_TOOL_NAMES = frozenset({"tavily_search", "tavily_extract"})


class _TavilySearchInput(BaseModel):
    query: str = Field(description="Search query to look up on the web")


def _format_search_results(results: list[dict]) -> str:
    lines = []
    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = (result.get("content") or "")[:400]
        lines.append(f"[{title}]({url})\n{content}")
    return "\n\n".join(lines) if lines else "No results found."


async def _cached_tavily_search(query: str) -> str:
    """Search via the cached, retried Tavily wrapper used elsewhere in the app."""
    results = await perform_search_cached(
        query,
        max_results=settings.max_search_results,
    )
    return _format_search_results(results)


def _make_cached_tavily_search_tool() -> BaseTool:
    return StructuredTool.from_function(
        coroutine=_cached_tavily_search,
        name="tavily_search",
        description=(
            "Search the web for up-to-date information. "
            "Use when the answer requires current data, prices, news, or facts beyond the documents."
        ),
        args_schema=_TavilySearchInput,
    )


def should_mark_web_used(tool_name: str, raw_result: object) -> bool:
    """Return True only when a general web tool returned usable external data."""
    if tool_name not in GENERAL_WEB_TOOL_NAMES:
        return False

    if isinstance(raw_result, dict):
        if raw_result.get("error") is not None:
            return False
        if tool_name == "tavily_extract":
            return bool(raw_result.get("results"))
        return True

    if isinstance(raw_result, str):
        return bool(raw_result.strip())

    return bool(raw_result)


def build_general_tools(*, allow_web: bool) -> list[BaseTool]:
    """Return Tavily search + URL extract tools when web access is enabled."""
    if not settings.tavily_api_key or not allow_web:
        return []

    tavily_kwargs = {"tavily_api_key": settings.tavily_api_key}
    return [
        _make_cached_tavily_search_tool(),
        TavilyExtract(**tavily_kwargs),
    ]


def build_agent_tools(
    *,
    allow_web: bool,
    reference_flags: dict[str, bool] | None = None,
) -> list[BaseTool]:
    """Combine Tavily web tools with enabled reference tools from the registry."""
    return build_general_tools(allow_web=allow_web) + build_reference_tools(
        reference_flags=reference_flags
    )
