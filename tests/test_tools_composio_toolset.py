"""Tests for src/tools/composio_toolset.py."""

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
async def test_mcp_tools_context_returns_langchain_tools():
    manager = ComposioToolsetManager(api_key="test-key")
    fake_tool = MagicMock()
    fake_tool.name = "ALPHA_VANTAGE_OVERVIEW"
    fake_tools = [fake_tool]
    mock_client = MagicMock()
    mock_client.tools.get.return_value = fake_tools

    with (
        patch.object(
            manager,
            "_get_connected_slugs_for_user",
            return_value=["alpha_vantage"],
        ),
        patch.object(
            manager,
            "_build_client",
            return_value=mock_client,
        ),
    ):
        async with manager.mcp_tools_context("user-123") as tools:
            assert tools == fake_tools

    mock_client.tools.get.assert_called_once_with(
        user_id="user-123",
        toolkits=["alpha_vantage"],
        limit=999,
    )


@pytest.mark.asyncio
async def test_mcp_tools_context_returns_empty_list_on_error():
    manager = ComposioToolsetManager(api_key="test-key")

    with patch.object(
        manager,
        "_get_connected_slugs_for_user",
        side_effect=RuntimeError("boom"),
    ):
        async with manager.mcp_tools_context("user-123") as tools:
            assert tools == []
