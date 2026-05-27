"""Shared Alpha Vantage MCP client with cached tool discovery."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.config import settings
from src.errors import AssetPriceError, McpClientError

logger = logging.getLogger(__name__)

_WRAPPER_TOOL_LIST = "TOOL_LIST"
_WRAPPER_TOOL_GET = "TOOL_GET"
_WRAPPER_TOOL_CALL = "TOOL_CALL"
_REQUIRED_WRAPPER_TOOLS = frozenset({_WRAPPER_TOOL_LIST, _WRAPPER_TOOL_GET, _WRAPPER_TOOL_CALL})


@dataclass(slots=True, frozen=True)
class AlphaVantageMcpToolSummary:
    """Compact catalog entry returned by the remote MCP server."""

    name: str
    description: str = ""


@dataclass(slots=True, frozen=True)
class AlphaVantageMcpToolDefinition:
    """Detailed tool definition fetched lazily via ``TOOL_GET``."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AlphaVantageMcpToolMatch:
    """Ranked match returned by host-side catalog search."""

    name: str
    description: str
    score: float
    why: str


def alpha_vantage_mcp_url() -> str:
    """Resolve the configured Alpha Vantage remote MCP URL."""
    direct_url = (settings.alpha_vantage_mcp_url or "").strip()
    if direct_url:
        return direct_url
    api_key = (settings.alpha_vantage_api_key or "").strip()
    if api_key:
        return f"https://mcp.alphavantage.co/mcp?apikey={api_key}"
    raise AssetPriceError("Alpha Vantage MCP URL or API key is not configured.")


def _extract_payload(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent

    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if not text:
            continue
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text
    return {}


class AlphaVantageMcpClient:
    """Discovers and calls Alpha Vantage MCP tools while caching catalog state in memory.

    The client keeps a lightweight in-memory catalog of available tool names/descriptions and
    lazily caches full tool schemas on first use. This follows the MCP guidance to separate
    discovery from schema loading while still avoiding repeated round-trips for stable
    definitions.
    """

    def __init__(self, url: str, refresh_interval_seconds: float = 3600) -> None:
        self._url = url
        self._refresh_interval_seconds = max(0.01, float(refresh_interval_seconds))
        self._state_lock = RLock()
        self._catalog: dict[str, AlphaVantageMcpToolSummary] = {}
        self._definitions: dict[str, AlphaVantageMcpToolDefinition] = {}
        self._wrapper_tools: dict[str, dict[str, Any]] = {}
        self._refresh_task: asyncio.Task[None] | None = None
        self._started = False
        self._last_refresh_at: datetime | None = None
        self._last_refresh_error: str | None = None

    @property
    def url(self) -> str:
        return self._url

    @property
    def refresh_interval_seconds(self) -> float:
        return self._refresh_interval_seconds

    @property
    def last_refresh_at(self) -> datetime | None:
        with self._state_lock:
            return self._last_refresh_at

    @property
    def last_refresh_error(self) -> str | None:
        with self._state_lock:
            return self._last_refresh_error

    def list_available_tools(self) -> list[AlphaVantageMcpToolSummary]:
        with self._state_lock:
            return sorted(self._catalog.values(), key=lambda tool: tool.name)

    def has_tool(self, tool_name: str) -> bool:
        normalized = tool_name.strip().upper()
        with self._state_lock:
            return normalized in self._catalog

    def catalog_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "url": self._url,
                "refresh_interval_seconds": self._refresh_interval_seconds,
                "last_refresh_at": self._last_refresh_at.isoformat() if self._last_refresh_at else None,
                "last_refresh_error": self._last_refresh_error,
                "wrapper_tools": sorted(self._wrapper_tools),
                "tools": [asdict(tool) for tool in sorted(self._catalog.values(), key=lambda item: item.name)],
            }

    async def initialize(self) -> None:
        if self._started:
            return

        await self.refresh_tool_catalog(force=True)

        with self._state_lock:
            if self._started:
                return
            self._refresh_task = asyncio.create_task(
                self._refresh_loop(),
                name="alpha-vantage-mcp-tool-refresh",
            )
            self._started = True

    async def shutdown(self) -> None:
        task: asyncio.Task[None] | None
        with self._state_lock:
            task = self._refresh_task
            self._refresh_task = None
            self._started = False

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def refresh_tool_catalog(self, *, force: bool = False) -> list[AlphaVantageMcpToolSummary]:
        if not force:
            with self._state_lock:
                if self._catalog:
                    return self.list_available_tools()

        try:
            root_tools, catalog = await self._fetch_catalog_snapshot()
        except Exception as exc:
            with self._state_lock:
                self._last_refresh_error = str(exc)
            raise
        catalog_names = {tool.name for tool in catalog}
        with self._state_lock:
            self._wrapper_tools = root_tools
            self._catalog = {tool.name: tool for tool in catalog}
            self._definitions = {
                name: definition
                for name, definition in self._definitions.items()
                if name in catalog_names
            }
            self._last_refresh_at = datetime.now(UTC)
            self._last_refresh_error = None
            return self.list_available_tools()

    async def get_tool_definition(
        self,
        tool_name: str,
        *,
        refresh_if_missing: bool = True,
    ) -> AlphaVantageMcpToolDefinition:
        normalized = tool_name.strip().upper()
        with self._state_lock:
            cached = self._definitions.get(normalized)
        if cached is not None:
            return cached

        if refresh_if_missing and not self.has_tool(normalized):
            await self.refresh_tool_catalog(force=True)

        if not self.has_tool(normalized):
            raise McpClientError(f"Alpha Vantage MCP tool '{normalized}' is not present in the catalog.")

        definition = await self._fetch_tool_definition(normalized)
        with self._state_lock:
            self._definitions[normalized] = definition
        return definition

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        normalized = tool_name.strip().upper()
        if not self.has_tool(normalized):
            await self.refresh_tool_catalog(force=True)
        if not self.has_tool(normalized):
            raise McpClientError(f"Alpha Vantage MCP tool '{normalized}' is not present in the catalog.")

        await self.get_tool_definition(normalized, refresh_if_missing=False)
        try:
            return await self._call_wrapper_tool(
                _WRAPPER_TOOL_CALL,
                {"tool_name": normalized, "arguments": arguments or {}},
            )
        except McpClientError as exc:
            lower_message = str(exc).lower()
            if "not found" in lower_message or "invalid" in lower_message:
                logger.info(
                    "[alpha_vantage_mcp_client] refreshing catalog after tool-call failure for %s: %s",
                    normalized,
                    exc,
                )
                await self.refresh_tool_catalog(force=True)
            raise

    async def _refresh_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._refresh_interval_seconds)
                try:
                    await self.refresh_tool_catalog(force=True)
                except Exception as exc:  # pragma: no cover - defensive log path
                    with self._state_lock:
                        self._last_refresh_error = str(exc)
                    logger.warning(
                        "[alpha_vantage_mcp_client] background catalog refresh failed: %s",
                        exc,
                    )
        except asyncio.CancelledError:
            raise

    async def _fetch_catalog_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, Any]], list[AlphaVantageMcpToolSummary]]:
        async with streamable_http_client(self._url) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                wrapper_tools = await session.list_tools()
                root_tools = {
                    tool.name: {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": getattr(tool, "inputSchema", None),
                    }
                    for tool in wrapper_tools.tools
                }
                missing_wrappers = sorted(_REQUIRED_WRAPPER_TOOLS.difference(root_tools))
                if missing_wrappers:
                    raise McpClientError(
                        "Alpha Vantage MCP server is missing required wrapper tools: "
                        + ", ".join(missing_wrappers)
                    )

                result = await session.call_tool(_WRAPPER_TOOL_LIST, {})

        payload = _extract_payload(result)
        tools = self._normalize_catalog_payload(payload)
        return root_tools, tools

    async def _fetch_tool_definition(self, tool_name: str) -> AlphaVantageMcpToolDefinition:
        payload = await self._call_wrapper_tool(_WRAPPER_TOOL_GET, {"tool_name": tool_name})
        if not isinstance(payload, dict):
            raise McpClientError(
                f"Alpha Vantage MCP returned an unexpected TOOL_GET payload for '{tool_name}'."
            )
        return AlphaVantageMcpToolDefinition(
            name=str(payload.get("name") or tool_name).strip().upper(),
            description=str(payload.get("description") or "").strip(),
            parameters=payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {},
            raw=payload,
        )

    async def _call_wrapper_tool(self, wrapper_tool_name: str, arguments: dict[str, Any]) -> Any:
        async with streamable_http_client(self._url) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(wrapper_tool_name, arguments)

        if getattr(result, "isError", False):
            message = _extract_payload(result)
            raise McpClientError(
                f"Alpha Vantage MCP {wrapper_tool_name} failed: {message if message else 'unknown error'}"
            )

        payload = _extract_payload(result)
        if isinstance(payload, dict):
            message = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
            if message:
                raise McpClientError(str(message).strip())
        return payload

    @staticmethod
    def _normalize_catalog_payload(payload: Any) -> list[AlphaVantageMcpToolSummary]:
        raw_tools = payload.get("tools") if isinstance(payload, dict) else payload
        if not isinstance(raw_tools, list):
            raise McpClientError("Alpha Vantage MCP TOOL_LIST returned an unexpected payload.")

        tools: list[AlphaVantageMcpToolSummary] = []
        for entry in raw_tools:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip().upper()
            if not name:
                continue
            tools.append(
                AlphaVantageMcpToolSummary(
                    name=name,
                    description=str(entry.get("description") or "").strip(),
                )
            )

        if not tools:
            raise McpClientError("Alpha Vantage MCP TOOL_LIST returned an empty tool catalog.")
        return tools


_client_singleton: AlphaVantageMcpClient | None = None


def get_alpha_vantage_mcp_client() -> AlphaVantageMcpClient:
    """Return the process-wide Alpha Vantage MCP client singleton."""
    global _client_singleton
    expected_url = alpha_vantage_mcp_url()
    expected_refresh = settings.alpha_vantage_mcp_tool_refresh_seconds
    if (
        _client_singleton is None
        or _client_singleton.url != expected_url
        or _client_singleton.refresh_interval_seconds != expected_refresh
    ):
        _client_singleton = AlphaVantageMcpClient(
            url=expected_url,
            refresh_interval_seconds=expected_refresh,
        )
    return _client_singleton


async def initialize_alpha_vantage_mcp_client() -> AlphaVantageMcpClient:
    """Discover and cache the Alpha Vantage MCP catalog, then start periodic refreshes."""
    client = get_alpha_vantage_mcp_client()
    await client.initialize()
    return client


async def shutdown_alpha_vantage_mcp_client() -> None:
    """Stop background refreshes for the process-wide Alpha Vantage MCP client."""
    client = _client_singleton
    if client is not None:
        await client.shutdown()


async def list_alpha_vantage_mcp_tools(
    *,
    force_refresh: bool = False,
) -> list[AlphaVantageMcpToolSummary]:
    """Return the cached Alpha Vantage tool catalog, refreshing when requested."""
    client = get_alpha_vantage_mcp_client()
    if force_refresh or not client.list_available_tools():
        await client.refresh_tool_catalog(force=True)
    return client.list_available_tools()


async def get_alpha_vantage_mcp_tool_definition(
    tool_name: str,
) -> AlphaVantageMcpToolDefinition:
    """Return the cached or freshly loaded schema for a single Alpha Vantage tool."""
    return await get_alpha_vantage_mcp_client().get_tool_definition(tool_name)


async def call_alpha_vantage_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Call any Alpha Vantage MCP tool through the cached client."""
    return await get_alpha_vantage_mcp_client().call_tool(tool_name, arguments)


def _tokenize_catalog_text(text: str) -> list[str]:
    return [token for token in re.split(r"[^A-Z0-9]+", text.upper()) if token]


def _score_tool_match(query: str, tool: AlphaVantageMcpToolSummary) -> AlphaVantageMcpToolMatch | None:
    query_tokens = set(_tokenize_catalog_text(query))
    if not query_tokens:
        return None

    name_tokens = set(_tokenize_catalog_text(tool.name))
    description_tokens = set(_tokenize_catalog_text(tool.description))
    combined_tokens = name_tokens | description_tokens

    overlap = query_tokens & combined_tokens
    name_overlap = query_tokens & name_tokens
    score = float(len(overlap)) + (0.75 * len(name_overlap))

    normalized_query = query.upper()
    if "24" in normalized_query or "24-HOUR" in normalized_query or "24 HOUR" in normalized_query:
        if {"INTRADAY", "REALTIME"} & combined_tokens:
            score += 2.5
        if {"DAILY", "WEEKLY", "MONTHLY"} & combined_tokens:
            score += 0.5
    if {"CHANGE", "RETURNS", "RETURN"} & query_tokens and {"INTRADAY", "DAILY", "ANALYTICS"} & combined_tokens:
        score += 1.5
    if {"NEWS", "SENTIMENT"} & query_tokens and "NEWS" in combined_tokens:
        score += 2.0
    if {"EARNINGS", "TRANSCRIPT"} & query_tokens and {"EARNINGS", "TRANSCRIPT"} & combined_tokens:
        score += 2.0
    if {"RSI", "MACD", "SMA", "EMA"} & query_tokens and name_overlap:
        score += 3.0
    if {"BITCOIN", "ETHEREUM", "CRYPTO", "BTC", "ETH"} & query_tokens and {"CRYPTO", "DIGITAL", "CURRENCY"} & combined_tokens:
        score += 1.5
    if {"FOREX", "FX", "USD", "EUR"} & query_tokens and {"FX", "CURRENCY", "FOREX"} & combined_tokens:
        score += 1.5
    if {"PRICE", "QUOTE", "LATEST"} & query_tokens and {"QUOTE", "REALTIME", "INTRADAY"} & combined_tokens:
        score += 1.0

    if score <= 0:
        return None

    why_parts: list[str] = []
    if name_overlap:
        why_parts.append("name_overlap")
    if overlap:
        why_parts.append("description_overlap")
    if "24" in normalized_query and {"INTRADAY", "REALTIME"} & combined_tokens:
        why_parts.append("intraday_bonus")
    if not why_parts:
        why_parts.append("semantic_overlap")

    return AlphaVantageMcpToolMatch(
        name=tool.name,
        description=tool.description,
        score=score,
        why="_".join(why_parts),
    )


async def search_alpha_vantage_mcp_tools(
    query: str,
    *,
    limit: int = 5,
) -> list[AlphaVantageMcpToolMatch]:
    """Search the cached Alpha Vantage tool catalog and return a ranked shortlist."""
    catalog = await list_alpha_vantage_mcp_tools()
    matches = [
        match
        for tool in catalog
        if (match := _score_tool_match(query, tool)) is not None
    ]
    matches.sort(key=lambda item: (-item.score, item.name))
    return matches[: max(1, limit)]
