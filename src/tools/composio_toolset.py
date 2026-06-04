"""Process-wide Composio toolset manager."""

from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
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


class ComposioToolsetManager:
    """Manages Composio connected apps and vends LangChain tools per user."""

    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key
        self._app_names: list[str] = []

    def _build_client(self, *, wrap_for_langchain: bool = False) -> Composio:
        if wrap_for_langchain:
            return Composio(
                api_key=self._api_key,
                provider=LangchainProvider(),
            )
        return Composio(api_key=self._api_key)

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

    @asynccontextmanager
    async def mcp_tools_context(
        self, user_id: str
    ) -> AsyncGenerator[list[Any], None]:
        """Yield LangChain tools for *user_id* based on connected Composio apps."""
        try:
            slugs = self._get_connected_slugs_for_user(user_id)
            if not slugs:
                yield []
                return

            client = self._build_client(wrap_for_langchain=True)
            tools: list[Any] = []
            seen_tool_names: set[str] = set()
            for slug in slugs:
                toolkit_tools = client.tools.get(user_id=user_id, toolkits=[slug], limit=999)
                for tool in toolkit_tools:
                    tool_name = getattr(tool, "name", None)
                    if not isinstance(tool_name, str) or tool_name in seen_tool_names:
                        continue
                    seen_tool_names.add(tool_name)
                    tools.append(tool)
            logger.info("[composio] Loaded %d tools for user %s.", len(tools), user_id)
            yield tools
        except Exception as exc:
            logger.warning("[composio] Tool loading error: %s", exc)
            yield []

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
