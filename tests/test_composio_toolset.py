from types import SimpleNamespace

import pytest

from src.config import settings
from src.tools.composio_toolset import ComposioToolsetManager


@pytest.mark.asyncio
async def test_router_tools_context_uses_initialized_default_user_app_cache(monkeypatch):
    manager = ComposioToolsetManager(api_key="test-key")
    manager._initialized = True
    manager._app_names = ["alpha_vantage", "firecrawl"]

    def fail_refetch(user_id: str) -> list[str]:
        raise AssertionError("should not refetch connected slugs for initialized default user")

    monkeypatch.setattr(manager, "_get_connected_slugs_for_user", fail_refetch)
    monkeypatch.setattr(manager, "_get_router_tools", lambda user_id, slugs: [SimpleNamespace(name="tool")])

    async with manager.router_tools_context(settings.composio_user_id) as tools:
        assert [tool.name for tool in tools] == ["tool"]


@pytest.mark.asyncio
async def test_router_tools_context_still_fetches_for_non_default_user(monkeypatch):
    manager = ComposioToolsetManager(api_key="test-key")
    manager._initialized = True
    manager._app_names = ["alpha_vantage"]

    seen = {"called": False}

    def fake_get_slugs(user_id: str) -> list[str]:
        seen["called"] = True
        assert user_id == "other-user"
        return ["openweather_api"]

    monkeypatch.setattr(manager, "_get_connected_slugs_for_user", fake_get_slugs)
    monkeypatch.setattr(manager, "_get_router_tools", lambda user_id, slugs: [SimpleNamespace(name=slugs[0])])

    async with manager.router_tools_context("other-user") as tools:
        assert [tool.name for tool in tools] == ["openweather_api"]

    assert seen["called"] is True
