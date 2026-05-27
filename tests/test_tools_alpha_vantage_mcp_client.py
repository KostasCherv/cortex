"""Tests for src/tools/alpha_vantage_mcp_client.py."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.alpha_vantage_mcp_client import (
    AlphaVantageMcpClient,
    AlphaVantageMcpToolDefinition,
    AlphaVantageMcpToolMatch,
    AlphaVantageMcpToolSummary,
    search_alpha_vantage_mcp_tools,
)


@pytest.mark.asyncio
async def test_refresh_tool_catalog_caches_wrapper_tools_and_catalog():
    client = AlphaVantageMcpClient("https://example.com/mcp", refresh_interval_seconds=3600)
    wrapper_tools = {
        "TOOL_LIST": {"name": "TOOL_LIST", "description": "list"},
        "TOOL_GET": {"name": "TOOL_GET", "description": "get"},
        "TOOL_CALL": {"name": "TOOL_CALL", "description": "call"},
    }
    catalog = [
        AlphaVantageMcpToolSummary(name="GLOBAL_QUOTE", description="Latest quote"),
        AlphaVantageMcpToolSummary(name="SYMBOL_SEARCH", description="Search symbols"),
    ]

    with patch.object(
        client,
        "_fetch_catalog_snapshot",
        new=AsyncMock(return_value=(wrapper_tools, catalog)),
    ):
        tools = await client.refresh_tool_catalog(force=True)

    assert [tool.name for tool in tools] == ["GLOBAL_QUOTE", "SYMBOL_SEARCH"]
    snapshot = client.catalog_snapshot()
    assert snapshot["wrapper_tools"] == ["TOOL_CALL", "TOOL_GET", "TOOL_LIST"]
    assert [tool["name"] for tool in snapshot["tools"]] == ["GLOBAL_QUOTE", "SYMBOL_SEARCH"]
    assert snapshot["last_refresh_at"] is not None


@pytest.mark.asyncio
async def test_get_tool_definition_caches_tool_schema_after_first_fetch():
    client = AlphaVantageMcpClient("https://example.com/mcp", refresh_interval_seconds=3600)
    client._catalog = {
        "GLOBAL_QUOTE": AlphaVantageMcpToolSummary(name="GLOBAL_QUOTE", description="Latest quote")
    }
    definition = AlphaVantageMcpToolDefinition(
        name="GLOBAL_QUOTE",
        description="Latest quote",
        parameters={"type": "object"},
        raw={"name": "GLOBAL_QUOTE"},
    )

    with patch.object(client, "_fetch_tool_definition", new=AsyncMock(return_value=definition)) as fetch:
        first = await client.get_tool_definition("global_quote")
        second = await client.get_tool_definition("GLOBAL_QUOTE")

    assert first == definition
    assert second == definition
    fetch.assert_awaited_once_with("GLOBAL_QUOTE")


@pytest.mark.asyncio
async def test_initialize_starts_periodic_refresh_and_shutdown_stops_it():
    client = AlphaVantageMcpClient("https://example.com/mcp", refresh_interval_seconds=0.01)
    wrapper_tools = {
        "TOOL_LIST": {"name": "TOOL_LIST", "description": "list"},
        "TOOL_GET": {"name": "TOOL_GET", "description": "get"},
        "TOOL_CALL": {"name": "TOOL_CALL", "description": "call"},
    }
    catalog = [AlphaVantageMcpToolSummary(name="GLOBAL_QUOTE", description="Latest quote")]
    refresh_seen = asyncio.Event()
    call_count = 0

    async def fake_fetch_catalog_snapshot():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            refresh_seen.set()
        return wrapper_tools, catalog

    with patch.object(client, "_fetch_catalog_snapshot", new=AsyncMock(side_effect=fake_fetch_catalog_snapshot)):
        await client.initialize()
        await asyncio.wait_for(refresh_seen.wait(), timeout=0.5)
        await client.shutdown()

    assert call_count >= 2
    assert client._refresh_task is None


@pytest.mark.asyncio
async def test_call_tool_uses_cached_catalog_and_schema():
    client = AlphaVantageMcpClient("https://example.com/mcp", refresh_interval_seconds=3600)
    client._catalog = {
        "GLOBAL_QUOTE": AlphaVantageMcpToolSummary(name="GLOBAL_QUOTE", description="Latest quote")
    }
    client._definitions = {
        "GLOBAL_QUOTE": AlphaVantageMcpToolDefinition(
            name="GLOBAL_QUOTE",
            description="Latest quote",
            parameters={"type": "object"},
            raw={"name": "GLOBAL_QUOTE"},
        )
    }
    payload = {"Global Quote": {"01. symbol": "AAPL", "05. price": "190.12"}}

    with patch.object(client, "_call_wrapper_tool", new=AsyncMock(return_value=payload)) as call_wrapper:
        result = await client.call_tool("GLOBAL_QUOTE", {"symbol": "AAPL"})

    assert result == payload
    call_wrapper.assert_awaited_once_with(
        "TOOL_CALL",
        {"tool_name": "GLOBAL_QUOTE", "arguments": {"symbol": "AAPL"}},
    )


@pytest.mark.asyncio
async def test_search_alpha_vantage_mcp_tools_returns_ranked_matches():
    catalog = [
        AlphaVantageMcpToolSummary(
            name="CRYPTO_INTRADAY",
            description="Returns intraday time series of the cryptocurrency specified, updated realtime.",
        ),
        AlphaVantageMcpToolSummary(
            name="GLOBAL_QUOTE",
            description="Returns the latest price and volume information for a ticker.",
        ),
    ]

    with patch(
        "src.tools.alpha_vantage_mcp_client.list_alpha_vantage_mcp_tools",
        new=AsyncMock(return_value=catalog),
    ):
        matches = await search_alpha_vantage_mcp_tools("What is the 24-hour change for BTC?")

    assert matches
    assert isinstance(matches[0], AlphaVantageMcpToolMatch)
    assert matches[0].name == "CRYPTO_INTRADAY"
