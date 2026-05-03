"""Pluggable web-search tool interface and provider registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.config import settings
from src.errors import ConfigurationError, SearchError
from src.tools.search import perform_search


class WebSearchTool(Protocol):
    """Provider-agnostic web search contract."""

    provider_name: str

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """Return normalized search results for ``query``."""


@dataclass(slots=True)
class TavilyWebSearchTool:
    """Tavily-backed implementation of ``WebSearchTool``."""

    provider_name: str = "tavily"

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        return perform_search(query=query, max_results=max_results)


def get_web_search_tool(provider: str | None = None) -> WebSearchTool:
    """Return the configured web-search tool adapter."""
    active = (provider or settings.web_search_provider).strip().lower()
    if active == "tavily":
        return TavilyWebSearchTool()
    raise ConfigurationError(
        f"Unknown WEB_SEARCH_PROVIDER '{active}'. Supported providers: tavily."
    )


def validate_web_search_provider_health() -> None:
    """Fail fast when the configured web-search provider is unavailable."""
    tool = get_web_search_tool()
    try:
        # Small deterministic probe used only for startup readiness.
        tool.search("health check", max_results=1)
    except SearchError as exc:
        raise ConfigurationError(
            "Configured web search provider is unavailable. "
            "The server requires a working web search tool at startup."
        ) from exc

