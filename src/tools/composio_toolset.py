"""Process-wide Composio toolset manager."""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from composio import Composio
from composio_langchain import LangchainProvider

from src.config import settings
from src.errors import ComposioError

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" in ".*" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
    module=r"pydantic\.main",
)


@dataclass
class _RouterSessionCache:
    session_id: str
    tools: list[Any]
    slugs: list[str]
    fetched_at: float


class ComposioToolsetManager:
    """Manages Composio connected apps and vends Tool Router meta tools per user."""

    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key
        self._app_names: list[str] = []
        self._router_cache: dict[str, _RouterSessionCache] = {}

    def _build_client(self) -> Composio:
        return Composio(
            api_key=self._api_key,
            provider=LangchainProvider(),
        )

    def _normalize_connected_slugs(self, items: list[object]) -> list[str]:
        allowlist = {app.lower() for app in settings.composio_apps}
        slugs: list[str] = []

        for item in items:
            slug = (getattr(item, "slug", None) or getattr(item, "name", None) or "").strip()
            if not slug:
                continue
            if allowlist and slug.lower() not in allowlist:
                continue
            slugs.append(slug)

        return sorted(set(slugs))

    def _get_connected_slugs_for_user(self, user_id: str) -> list[str]:
        client = self._build_client()
        session = client.create(user_id=user_id)
        connected = session.toolkits(is_connected=True)
        items = list(getattr(connected, "items", []) or [])
        return self._normalize_connected_slugs(items)

    def _cache_is_valid(self, entry: _RouterSessionCache, slugs: list[str]) -> bool:
        if entry.slugs != slugs:
            return False
        age = time.monotonic() - entry.fetched_at
        return age < settings.composio_tool_refresh_seconds

    def _get_router_tools(self, user_id: str, slugs: list[str]) -> list[Any]:
        cached = self._router_cache.get(user_id)
        if cached is not None and self._cache_is_valid(cached, slugs):
            logger.debug(
                "[composio] Reusing router session %s (%d meta tools) for user %s.",
                cached.session_id,
                len(cached.tools),
                user_id,
            )
            return cached.tools

        client = self._build_client()
        session = client.create(user_id=user_id, toolkits={"enable": slugs})
        tools = list(session.tools())
        self._router_cache[user_id] = _RouterSessionCache(
            session_id=session.session_id,
            tools=tools,
            slugs=list(slugs),
            fetched_at=time.monotonic(),
        )
        logger.info(
            "[composio] Router session ready for %s: %d meta tools, toolkits=%s.",
            user_id,
            len(tools),
            ", ".join(slugs),
        )
        return tools

    async def initialize(self) -> None:
        try:
            self._app_names = self._get_connected_slugs_for_user(settings.composio_user_id)
            if not self._app_names:
                logger.info("[composio] No active connected toolkits found — tool-calling disabled.")
                return
            logger.info(
                "[composio] Connected toolkits for %s: %s",
                settings.composio_user_id,
                ", ".join(self._app_names),
            )
        except Exception as exc:
            raise ComposioError(f"Composio tool discovery failed: {exc}") from exc

    async def shutdown(self) -> None:
        self._app_names = []
        self._router_cache.clear()

    @asynccontextmanager
    async def router_tools_context(
        self, user_id: str
    ) -> AsyncGenerator[list[Any], None]:
        """Yield Tool Router meta tools for *user_id* (search/execute flow)."""
        if not settings.composio_enabled:
            yield []
            return
        try:
            slugs = await asyncio.to_thread(self._get_connected_slugs_for_user, user_id)
            if not slugs:
                yield []
                return

            tools = await asyncio.to_thread(self._get_router_tools, user_id, slugs)
            yield tools
        except Exception as exc:
            logger.warning("[composio] Router tool loading error: %s", exc)
            yield []

    @asynccontextmanager
    async def mcp_tools_context(
        self, user_id: str
    ) -> AsyncGenerator[list[Any], None]:
        """Deprecated alias for :meth:`router_tools_context`."""
        async with self.router_tools_context(user_id) as tools:
            yield tools

    def get_connected_app_names(self) -> list[str]:
        return list(self._app_names)


_manager_singleton: ComposioToolsetManager | None = None


def get_composio_toolset_manager() -> ComposioToolsetManager:
    global _manager_singleton
    if (
        _manager_singleton is None
        or _manager_singleton._api_key != settings.composio_api_key
    ):
        _manager_singleton = ComposioToolsetManager(api_key=settings.composio_api_key)
    return _manager_singleton


async def initialize_composio_toolset() -> ComposioToolsetManager:
    manager = get_composio_toolset_manager()
    await manager.initialize()
    return manager


async def shutdown_composio_toolset() -> None:
    if _manager_singleton is not None:
        await _manager_singleton.shutdown()
