"""Tests for src/tools/composio_toolset.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.composio_toolset import (
    ComposioToolsetManager,
    get_composio_toolset_manager,
    initialize_composio_toolset,
    shutdown_composio_toolset,
)


def _make_fake_tool(name: str):
    tool = MagicMock()
    tool.name = name
    tool.description = f"Tool {name}"
    return tool


@pytest.mark.asyncio
async def test_get_tools_returns_cached_tools():
    manager = ComposioToolsetManager(api_key="test-key", refresh_seconds=3600)
    fake_tools = [_make_fake_tool("GITHUB_CREATE_ISSUE"), _make_fake_tool("TAVILY_SEARCH")]
    manager._tools = fake_tools
    assert manager.get_tools() == fake_tools


@pytest.mark.asyncio
async def test_get_connected_app_names_returns_unique_prefixes():
    manager = ComposioToolsetManager(api_key="test-key", refresh_seconds=3600)
    manager._tools = [
        _make_fake_tool("GITHUB_CREATE_ISSUE"),
        _make_fake_tool("GITHUB_LIST_REPOS"),
        _make_fake_tool("TAVILY_SEARCH"),
    ]
    names = manager.get_connected_app_names()
    assert set(names) == {"github", "tavily"}


@pytest.mark.asyncio
async def test_refresh_loads_tools_from_composio():
    manager = ComposioToolsetManager(api_key="test-key", refresh_seconds=3600)
    fake_tools = [_make_fake_tool("GITHUB_CREATE_ISSUE")]

    mock_composio = MagicMock()
    mock_composio.tools.get.return_value = fake_tools

    with patch("src.tools.composio_toolset.Composio", return_value=mock_composio):
        await manager.refresh()

    assert manager.get_tools() == fake_tools


@pytest.mark.asyncio
async def test_refresh_raises_composio_error_on_failure():
    from src.errors import ComposioError

    manager = ComposioToolsetManager(api_key="test-key", refresh_seconds=3600)

    with patch("src.tools.composio_toolset.Composio", side_effect=Exception("connection refused")):
        with pytest.raises(ComposioError):
            await manager.refresh()


@pytest.mark.asyncio
async def test_initialize_starts_refresh_loop_and_shutdown_stops_it():
    manager = ComposioToolsetManager(api_key="test-key", refresh_seconds=0.05)
    fake_tools = [_make_fake_tool("GITHUB_CREATE_ISSUE")]

    mock_composio = MagicMock()
    mock_composio.tools.get.return_value = fake_tools

    with patch("src.tools.composio_toolset.Composio", return_value=mock_composio):
        await manager.initialize()
        assert manager.get_tools() == fake_tools
        await asyncio.sleep(0.1)
        await manager.shutdown()

    assert manager._refresh_task is None or manager._refresh_task.done()


@pytest.mark.asyncio
async def test_get_tools_returns_empty_when_not_initialized():
    manager = ComposioToolsetManager(api_key="test-key", refresh_seconds=3600)
    assert manager.get_tools() == []
