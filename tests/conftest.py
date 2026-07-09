import os
from unittest.mock import AsyncMock, patch

import pytest

# Must be set before any src imports so inngest.Inngest initialises in dev mode.
os.environ.setdefault("INNGEST_DEV", "1")


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset provider singletons between tests to prevent state leaks."""
    yield
    from src.db import provider
    provider._session_store = None
    provider._storage_adapter = None


@pytest.fixture(autouse=True)
def mock_list_ready_session_attachment_resource_ids():
    with patch(
        "src.api.rag_chat_helpers.list_ready_rag_chat_session_attachment_resource_ids",
        new=AsyncMock(return_value=[]),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_enqueue_memory_refresh():
    # ponytail: several endpoint tests don't mock this fire-and-forget outbox
    # write and previously only passed because a local .env pointed at a real
    # Supabase project. Default it to a no-op here; tests that care about the
    # call still patch/assert it explicitly (see test_api.py:1363).
    with (
        patch(
            "src.api.routers.sessions.enqueue_memory_refresh",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "src.api.routers.rag_agents.enqueue_memory_refresh",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "src.api.routers.rag_chat.enqueue_memory_refresh",
            new=AsyncMock(return_value=False),
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_list_session_attachments():
    with (
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.routers.rag_agents.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.routers.rag_chat.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
    ):
        yield
