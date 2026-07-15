from unittest.mock import AsyncMock, patch

from src.outbox import OutboxEvent, dispatch_outbox_events, enqueue_event

_PENDING_ROW = {
    "id": "evt-1",
    "event_name": "rag/ingestion.requested",
    "payload": {"job_id": "j-1", "resource_id": "r-1", "owner_id": "u-1", "workspace_id": "u-1"},
    "status": "pending",
    "attempts": 0,
    "last_error": None,
    "next_attempt_at": "2026-01-01T00:00:00+00:00",
    "created_at": "2026-01-01T00:00:00+00:00",
    "sent_at": None,
}


def _make_store(rows=None, claim_result=True):
    mock_store = AsyncMock()
    mock_store.fetch_pending_outbox_events.return_value = rows if rows is not None else [dict(_PENDING_ROW)]
    mock_store.claim_outbox_event.return_value = claim_result
    return mock_store


async def test_enqueue_event_inserts_outbox_row():
    mock_store = AsyncMock()

    with patch("src.outbox.get_session_store", return_value=mock_store):
        event = await enqueue_event("rag/ingestion.requested", {"job_id": "j-1"})

    assert mock_store.insert_outbox_event.await_count == 1
    inserted = mock_store.insert_outbox_event.await_args[0][0]
    assert inserted["event_name"] == "rag/ingestion.requested"
    assert inserted["payload"]["job_id"] == "j-1"
    assert isinstance(event, OutboxEvent)
    assert event.event_name == "rag/ingestion.requested"


async def test_dispatch_marks_sent_on_success():
    mock_store = _make_store()
    mock_inngest = AsyncMock()
    mock_inngest.send = AsyncMock(return_value=["event-id"])

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", mock_inngest),
    ):
        count = await dispatch_outbox_events()

    assert count == 1
    mock_store.claim_outbox_event.assert_awaited_once_with("evt-1")
    update_patch = mock_store.update_outbox_event.await_args[0][1]
    assert update_patch["status"] == "sent"
    assert "sent_at" in update_patch


async def test_dispatch_skips_when_claim_fails():
    mock_store = _make_store(claim_result=False)
    mock_inngest = AsyncMock()

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", mock_inngest),
    ):
        count = await dispatch_outbox_events()

    assert count == 0
    mock_inngest.send.assert_not_awaited()
    mock_store.update_outbox_event.assert_not_awaited()


async def test_dispatch_schedules_retry_on_transient_failure():
    mock_store = _make_store()
    mock_inngest = AsyncMock()
    mock_inngest.send = AsyncMock(side_effect=RuntimeError("network error"))

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", mock_inngest),
        patch("src.outbox.capture_handled_exception") as mock_capture,
    ):
        count = await dispatch_outbox_events()

    assert count == 0
    update_patch = mock_store.update_outbox_event.await_args[0][1]
    assert update_patch["status"] == "pending"
    assert update_patch["attempts"] == 1
    assert "next_attempt_at" in update_patch
    assert update_patch["last_error"] == "network error"
    mock_capture.assert_not_called()


async def test_dispatch_permanently_fails_after_max_attempts():
    row = {**_PENDING_ROW, "attempts": 4}  # next attempt will reach _MAX_ATTEMPTS (5)
    mock_store = _make_store(rows=[row])
    mock_inngest = AsyncMock()
    mock_inngest.send = AsyncMock(side_effect=RuntimeError("still failing"))

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", mock_inngest),
        patch("src.outbox.capture_handled_exception") as mock_capture,
    ):
        count = await dispatch_outbox_events()

    assert count == 0
    update_patch = mock_store.update_outbox_event.await_args[0][1]
    assert update_patch["status"] == "failed"
    assert update_patch["attempts"] == 5
    mock_capture.assert_called_once()
    assert mock_capture.call_args.kwargs["operation"] == "outbox.delivery_terminal"
    assert mock_capture.call_args.kwargs["identifiers"] == {
        "event_id": "evt-1",
        "event_name": "rag/ingestion.requested",
    }


async def test_dispatch_reports_pre_fetch_failure():
    mock_store = _make_store(rows=[])
    mock_store.reset_stuck_dispatching_events.side_effect = RuntimeError("database offline")

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", AsyncMock()),
        patch("src.outbox.capture_handled_exception") as mock_capture,
    ):
        count = await dispatch_outbox_events()

    assert count == 0
    mock_capture.assert_called_once()
    assert mock_capture.call_args.kwargs["operation"] == "outbox.pre_dispatch_fetch"


async def test_dispatch_processes_multiple_events():
    rows = [
        {**_PENDING_ROW, "id": "evt-1"},
        {**_PENDING_ROW, "id": "evt-2"},
        {**_PENDING_ROW, "id": "evt-3"},
    ]
    mock_store = _make_store(rows=rows)
    mock_inngest = AsyncMock()
    mock_inngest.send = AsyncMock(return_value=["event-id"])

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", mock_inngest),
    ):
        count = await dispatch_outbox_events()

    assert count == 3
    assert mock_store.claim_outbox_event.await_count == 3
    assert mock_store.update_outbox_event.await_count == 3


async def test_dispatch_resets_stuck_dispatching_before_fetch():
    mock_store = _make_store(rows=[])

    with (
        patch("src.outbox.get_session_store", return_value=mock_store),
        patch("src.inngest_client.inngest_client", AsyncMock()),
    ):
        await dispatch_outbox_events()

    mock_store.reset_stuck_dispatching_events.assert_awaited_once()
