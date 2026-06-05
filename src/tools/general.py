"""General-purpose LangChain tools for RAG agent chat."""

from __future__ import annotations

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from langchain_tavily import TavilyExtract
from pydantic import BaseModel, Field

from src.config import settings
from src.tools.search import perform_search_cached

GENERAL_WEB_TOOL_NAMES = frozenset({"tavily_search", "tavily_extract"})
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_WIKIPEDIA_USER_AGENT = "ResearchAgent/1.0 (rag-chat; +https://www.mediawiki.org/wiki/API:Main_page)"
_WIKIPEDIA_MAX_QUERY_LENGTH = 300
_WIKIPEDIA_TOP_K = 5
_WIKIPEDIA_EXTRACT_MAX_CHARS = 1000


class _TavilySearchInput(BaseModel):
    query: str = Field(description="Search query to look up on the web")


class _WikipediaInput(BaseModel):
    query: str = Field(description="Topic to look up on Wikipedia")


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


async def _wikipedia_lookup(query: str) -> str:
    """Search Wikipedia via the MediaWiki API with robust error handling."""
    normalized_query = query.strip()[:_WIKIPEDIA_MAX_QUERY_LENGTH]
    if not normalized_query:
        return "No Wikipedia query provided."

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": normalized_query,
        "gsrlimit": _WIKIPEDIA_TOP_K,
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "format": "json",
    }
    headers = {"User-Agent": _WIKIPEDIA_USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(_WIKIPEDIA_API, params=params, headers=headers)
            response.raise_for_status()
            if not response.text.strip():
                return "Wikipedia returned an empty response. Try again or use web search."
            try:
                data = response.json()
            except ValueError:
                return "Wikipedia returned an unexpected response. Try again or use web search."
    except httpx.HTTPError as exc:
        return f"Wikipedia lookup failed: {exc}"

    summaries: list[str] = []
    for page in data.get("query", {}).get("pages", {}).values():
        if page.get("missing") is not None:
            continue
        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        summaries.append(
            f"Page: {title}\nSummary: {extract[:_WIKIPEDIA_EXTRACT_MAX_CHARS]}"
        )

    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


def _make_wikipedia_tool() -> BaseTool:
    return StructuredTool.from_function(
        coroutine=_wikipedia_lookup,
        name="wikipedia",
        description=(
            "Look up encyclopedic information on Wikipedia. "
            "Use for people, places, concepts, companies, and historical topics."
        ),
        args_schema=_WikipediaInput,
    )


def build_reference_tools(*, allow_wikipedia: bool = True) -> list[BaseTool]:
    """Return Wikipedia tool when enabled."""
    if not allow_wikipedia:
        return []

    return [_make_wikipedia_tool()]


def build_agent_tools(*, allow_web: bool, allow_wikipedia: bool = True) -> list[BaseTool]:
    """Combine Tavily web tools with reference tools."""
    return build_general_tools(allow_web=allow_web) + build_reference_tools(
        allow_wikipedia=allow_wikipedia
    )
