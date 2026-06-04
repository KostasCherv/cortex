"""Tests for src/tools/composio_toolset.py."""

import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.errors import ComposioError
from src.tools.composio_toolset import ComposioToolsetManager


def _connected_toolkit(slug: str) -> SimpleNamespace:
    return SimpleNamespace(slug=slug, name=slug)


@pytest.mark.asyncio
async def test_initialize_collects_connected_app_names():
    manager = ComposioToolsetManager(api_key="test-key")

    with patch.object(
        manager,
        "_get_connected_slugs_for_user",
        return_value=["alpha_vantage", "firecrawl"],
    ):
        await manager.initialize()

    assert manager.get_connected_app_names() == ["alpha_vantage", "firecrawl"]


def test_normalize_connected_slugs_filters_duplicates_and_allowlist():
    manager = ComposioToolsetManager(api_key="test-key")

    with patch("src.tools.composio_toolset.settings.composio_apps", ["firecrawl"]):
        slugs = manager._normalize_connected_slugs(
            [
                _connected_toolkit("alpha_vantage"),
                _connected_toolkit("firecrawl"),
                _connected_toolkit("firecrawl"),
            ]
        )

    assert slugs == ["firecrawl"]


@pytest.mark.asyncio
async def test_initialize_raises_composio_error_on_failure():
    manager = ComposioToolsetManager(api_key="test-key")

    with patch.object(
        manager,
        "_get_connected_slugs_for_user",
        side_effect=RuntimeError("connection refused"),
    ):
        with pytest.raises(ComposioError):
            await manager.initialize()


@pytest.mark.asyncio
async def test_router_tools_context_returns_meta_tools():
    manager = ComposioToolsetManager(api_key="test-key")
    search_tool = MagicMock()
    search_tool.name = "COMPOSIO_SEARCH_TOOLS"
    execute_tool = MagicMock()
    execute_tool.name = "COMPOSIO_MULTI_EXECUTE_TOOL"
    meta_tools = [search_tool, execute_tool]

    mock_session = MagicMock()
    mock_session.session_id = "sess-abc"
    mock_session.tools.return_value = meta_tools
    mock_client = MagicMock()
    mock_client.create.return_value = mock_session

    with (
        patch.object(
            manager,
            "_get_connected_slugs_for_user",
            return_value=["alpha_vantage"],
        ),
        patch.object(manager, "_build_client", return_value=mock_client),
    ):
        async with manager.router_tools_context("user-123") as tools:
            assert tools == meta_tools

    mock_client.create.assert_called_once_with(
        user_id="user-123",
        toolkits={"enable": ["alpha_vantage"]},
    )
    mock_session.tools.assert_called_once()
    assert not mock_client.tools.get.called


@pytest.mark.asyncio
async def test_router_tools_context_reuses_cache_within_ttl():
    manager = ComposioToolsetManager(api_key="test-key")
    meta_tools = [MagicMock(name="COMPOSIO_SEARCH_TOOLS")]

    mock_session = MagicMock()
    mock_session.session_id = "sess-1"
    mock_session.tools.return_value = meta_tools
    mock_client = MagicMock()
    mock_client.create.return_value = mock_session

    slugs = ["firecrawl"]
    with (
        patch.object(manager, "_get_connected_slugs_for_user", return_value=slugs),
        patch.object(manager, "_build_client", return_value=mock_client),
        patch("src.tools.composio_toolset.settings.composio_tool_refresh_seconds", 3600),
    ):
        manager._get_router_tools("user-123", slugs)
        manager._get_router_tools("user-123", slugs)

    assert mock_client.create.call_count == 1


def test_router_cache_invalidates_when_slugs_change():
    manager = ComposioToolsetManager(api_key="test-key")
    manager._router_cache["user-123"] = manager._router_cache.get("user-123") or type(
        "E", (), {}
    )()
    from src.tools.composio_toolset import _RouterSessionCache

    manager._router_cache["user-123"] = _RouterSessionCache(
        session_id="sess-1",
        tools=[MagicMock()],
        slugs=["github"],
        fetched_at=time.monotonic(),
    )

    entry = manager._router_cache["user-123"]
    assert manager._cache_is_valid(entry, ["github"]) is True
    assert manager._cache_is_valid(entry, ["gmail"]) is False


@pytest.mark.asyncio
async def test_mcp_tools_context_is_alias_for_router():
    manager = ComposioToolsetManager(api_key="test-key")

    with patch.object(
        manager,
        "router_tools_context",
    ) as mock_router:
        @asynccontextmanager
        async def fake_router(user_id: str):
            yield ["meta"]

        mock_router.side_effect = fake_router
        async with manager.mcp_tools_context("user-123") as tools:
            assert tools == ["meta"]


@pytest.mark.asyncio
async def test_router_tools_context_returns_empty_list_on_error():
    manager = ComposioToolsetManager(api_key="test-key")

    with patch.object(
        manager,
        "_get_connected_slugs_for_user",
        side_effect=RuntimeError("boom"),
    ):
        async with manager.router_tools_context("user-123") as tools:
            assert tools == []
