"""Process-wide Composio toolset manager with background refresh."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from composio import Composio
from composio_langchain import LangchainProvider

from src.config import settings
from src.errors import ComposioError

logger = logging.getLogger(__name__)


class ComposioToolsetManager:
    """Loads and caches LangChain-compatible Composio tools for the agent loop."""

    def __init__(self, *, api_key: str, refresh_seconds: float = 3600) -> None:
        self._api_key = api_key
        self._refresh_seconds = max(0.01, float(refresh_seconds))
        self._tools: list[Any] = []
        self._refresh_task: asyncio.Task[None] | None = None
        self._started = False

    def get_tools(self) -> list[Any]:
        return list(self._tools)

    def get_connected_app_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for tool in self._tools:
            prefix = tool.name.split("_")[0].lower()
            if prefix not in seen:
                seen.add(prefix)
                names.append(prefix)
        return names

    async def refresh(self) -> None:
        raw = settings.composio_apps
        app_filter: list[str] = (
            [a.strip() for a in raw.split(",") if a.strip()]
            if isinstance(raw, str)
            else list(raw)
        )
        try:
            composio = Composio(provider=LangchainProvider(), api_key=self._api_key)
            if app_filter:
                tools = composio.tools.get(user_id="default", toolkits=app_filter)
            else:
                tools = composio.tools.get(user_id="default")
        except Exception as exc:
            raise ComposioError(f"Composio toolset refresh failed: {exc}") from exc
        self._tools = tools
        logger.info("[composio] Loaded %d tools.", len(self._tools))

    async def initialize(self) -> None:
        if self._started:
            return
        await self.refresh()
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(), name="composio-tool-refresh"
        )
        self._started = True

    async def shutdown(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        self._started = False
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _refresh_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._refresh_seconds)
                try:
                    await self.refresh()
                except ComposioError as exc:
                    logger.warning("[composio] Background refresh failed: %s", exc)
        except asyncio.CancelledError:
            raise


_manager_singleton: ComposioToolsetManager | None = None


def get_composio_toolset_manager() -> ComposioToolsetManager:
    global _manager_singleton
    if (
        _manager_singleton is None
        or _manager_singleton._api_key != settings.composio_api_key
    ):
        _manager_singleton = ComposioToolsetManager(
            api_key=settings.composio_api_key,
            refresh_seconds=settings.composio_tool_refresh_seconds,
        )
    return _manager_singleton


async def initialize_composio_toolset() -> ComposioToolsetManager:
    manager = get_composio_toolset_manager()
    await manager.initialize()
    return manager


async def shutdown_composio_toolset() -> None:
    if _manager_singleton is not None:
        await _manager_singleton.shutdown()
