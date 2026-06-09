from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_get_user_memory_prompt_block_returns_empty_when_memory_missing():
    from src.user_memory import get_user_memory_prompt_block

    store = AsyncMock()
    store.get_user_memory.return_value = None

    with patch("src.user_memory.get_session_store", return_value=store):
        block = await get_user_memory_prompt_block("user-1", "Anything")

    assert block == ""


@pytest.mark.asyncio
async def test_get_user_memory_prompt_block_renders_saved_memory():
    from src.user_memory import get_user_memory_prompt_block

    store = AsyncMock()
    store.get_user_memory.return_value = {
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "content": "Prefers concise answers.\nWorks in fintech.",
        "updated_at": "2026-06-04T10:00:00+00:00",
        "last_refreshed_at": "2026-06-04T10:05:00+00:00",
    }

    with patch("src.user_memory.get_session_store", return_value=store):
        block = await get_user_memory_prompt_block("user-1", "Need product advice")

    assert "Prefers concise answers." in block
    assert "Works in fintech." in block


@pytest.mark.asyncio
async def test_update_user_memory_persists_single_document():
    from src.user_memory import update_user_memory

    store = AsyncMock()
    store.get_user_memory.return_value = {
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "content": "Existing memory.",
        "updated_at": "2026-06-04T10:00:00+00:00",
        "last_refreshed_at": "2026-06-04T10:05:00+00:00",
    }

    with patch("src.user_memory.get_session_store", return_value=store):
        result = await update_user_memory("user-1", "  Updated memory.  ")

    assert result["content"] == "Updated memory."
    payload = store.upsert_user_memory.await_args.kwargs["payload"]
    assert payload["content"] == "Updated memory."
    assert payload["last_refreshed_at"] == "2026-06-04T10:05:00+00:00"


@pytest.mark.asyncio
async def test_delete_user_memory_deletes_row():
    from src.user_memory import delete_user_memory

    store = AsyncMock()
    store.delete_user_memory.return_value = True

    with patch("src.user_memory.get_session_store", return_value=store):
        result = await delete_user_memory("user-1")

    assert result == {"deleted": True}
    store.delete_user_memory.assert_awaited_once_with(owner_id="user-1", workspace_id="user-1")


@pytest.mark.asyncio
async def test_enqueue_memory_refresh_enqueues_event_for_non_empty_exchange():
    from src.user_memory import enqueue_memory_refresh

    with patch("src.user_memory.outbox.enqueue_event", new=AsyncMock()) as enqueue_event:
        queued = await enqueue_memory_refresh(
            user_id="user-1",
            source_mode="workspace_chat",
            source_session_id="sess-1",
            user_message="I prefer concise answers.",
            assistant_message="I'll keep future answers tight.",
            source_user_message_id="user-msg-1",
            source_assistant_message_id="assistant-msg-1",
        )

    assert queued is True
    enqueue_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_user_memory_claims_event_before_upserting_memory():
    from src.user_memory import refresh_user_memory

    store = AsyncMock()
    store.claim_user_memory_refresh_event.return_value = True
    store.get_user_memory.return_value = None

    with patch("src.user_memory.get_session_store", return_value=store):
        result = await refresh_user_memory(
            user_id="user-1",
            source_mode="workspace_chat",
            source_session_id="sess-1",
            user_message="I prefer concise answers and I work in fintech.",
            assistant_message="Understood. I'll keep replies concise and fintech-aware.",
            event_key="evt-memory-1",
            source_user_message_id="user-msg-1",
            source_assistant_message_id="assistant-msg-1",
        )

    assert result == "updated"
    store.claim_user_memory_refresh_event.assert_awaited_once()
    payload = store.upsert_user_memory.await_args.kwargs["payload"]
    assert "Prefers concise answers." in payload["content"]
    assert "I work in fintech." in payload["content"]


@pytest.mark.asyncio
async def test_refresh_user_memory_skips_duplicate_events():
    from src.user_memory import refresh_user_memory

    store = AsyncMock()
    store.claim_user_memory_refresh_event.return_value = False

    with patch("src.user_memory.get_session_store", return_value=store):
        result = await refresh_user_memory(
            user_id="user-1",
            source_mode="workspace_chat",
            source_session_id="sess-1",
            user_message="I prefer concise answers.",
            assistant_message="I'll keep it brief.",
            event_key="evt-memory-2",
            source_user_message_id="user-msg-1",
            source_assistant_message_id="assistant-msg-1",
        )

    assert result == "skipped"
    store.get_user_memory.assert_not_called()
    store.upsert_user_memory.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_user_memory_merges_new_candidates_into_existing_content():
    from src.user_memory import refresh_user_memory

    store = AsyncMock()
    store.claim_user_memory_refresh_event.return_value = True
    store.get_user_memory.return_value = {
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "content": "Prefers concise answers.",
        "updated_at": "2026-06-04T10:00:00+00:00",
        "last_refreshed_at": "2026-06-04T10:05:00+00:00",
    }

    with patch("src.user_memory.get_session_store", return_value=store):
        result = await refresh_user_memory(
            user_id="user-1",
            source_mode="workspace_chat",
            source_session_id="sess-1",
            user_message="I work in fintech.",
            assistant_message="Noted.",
            event_key="evt-memory-3",
            source_user_message_id="user-msg-1",
            source_assistant_message_id="assistant-msg-1",
        )

    assert result == "updated"
    payload = store.upsert_user_memory.await_args.kwargs["payload"]
    assert payload["content"] == "Prefers concise answers.\nI work in fintech."


@pytest.mark.asyncio
async def test_refresh_user_memory_returns_error_for_non_retryable_store_failure():
    from src.user_memory import refresh_user_memory

    store = AsyncMock()
    store.claim_user_memory_refresh_event.return_value = True
    store.get_user_memory.return_value = None
    store.upsert_user_memory.side_effect = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("POST", "https://example.com/rest/v1/user_memory"),
        response=httpx.Response(
            400,
            json={"code": "PGRST204", "message": "schema cache mismatch"},
            request=httpx.Request("POST", "https://example.com/rest/v1/user_memory"),
        ),
    )

    with (
        patch("src.user_memory.get_session_store", return_value=store),
        patch("src.user_memory.logger") as mock_logger,
    ):
        result = await refresh_user_memory(
            user_id="user-1",
            source_mode="workspace_chat",
            source_session_id="sess-1",
            user_message="I prefer concise answers.",
            assistant_message="I'll keep it brief.",
            event_key="evt-memory-4",
            source_user_message_id="user-msg-1",
            source_assistant_message_id="assistant-msg-1",
        )

    assert result == "error"
    assert isinstance(mock_logger.warning, MagicMock)
    mock_logger.warning.assert_called_once()
