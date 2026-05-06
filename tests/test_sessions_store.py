from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.db.supabase_store import SupabaseSessionStore
from src.sessions import ConversationTurn, SessionRun, append_run, append_turn, create_session, get_session


async def test_sessions_module_delegates_create_to_store():
    mock_store = AsyncMock()
    mock_store.create_session.return_value = "session-object"
    with patch("src.sessions._get_store", return_value=mock_store):
        created = await create_session("user-1")
    assert created == "session-object"
    mock_store.create_session.assert_awaited_once_with(user_id="user-1", title="New session")


async def test_sessions_module_delegates_get_to_store():
    mock_store = AsyncMock()
    mock_store.get_session.return_value = None
    with patch("src.sessions._get_store", return_value=mock_store):
        session = await get_session("session-1", "user-1")
    assert session is None
    mock_store.get_session.assert_awaited_once_with(session_id="session-1", user_id="user-1")


async def test_sessions_module_delegates_append_operations():
    mock_store = AsyncMock()
    run = SessionRun(run_id="r1", query="q")
    turn = ConversationTurn(role="user", content="c")
    with patch("src.sessions._get_store", return_value=mock_store):
        await append_run("user-1", "session-1", run)
        await append_turn("user-1", "session-1", turn)
    mock_store.append_run.assert_awaited_once_with(user_id="user-1", session_id="session-1", run=run)
    mock_store.append_turn.assert_awaited_once_with(user_id="user-1", session_id="session-1", turn=turn)


async def test_store_lists_rag_chat_sessions_with_batched_latest_messages():
    store = object.__new__(SupabaseSessionStore)
    sessions_response = MagicMock()
    sessions_response.json.return_value = [
        {
            "id": "chat-1",
            "owner_id": "user-1",
            "workspace_id": "user-1",
            "agent_id": "agent-1",
            "created_at": "2026-04-23T09:00:00+00:00",
        },
        {
            "id": "chat-2",
            "owner_id": "user-1",
            "workspace_id": "user-1",
            "agent_id": "agent-1",
            "created_at": "2026-04-23T10:00:00+00:00",
        },
    ]
    messages_response = MagicMock()
    messages_response.json.return_value = [
        {
            "session_id": "chat-2",
            "content": "Most recent chat",
            "created_at": "2026-04-23T10:05:00+00:00",
        },
        {
            "session_id": "chat-1",
            "content": "Earlier chat",
            "created_at": "2026-04-23T09:05:00+00:00",
        },
    ]
    store._request = AsyncMock(side_effect=[sessions_response, messages_response])  # type: ignore[method-assign]

    summaries = await store.list_rag_chat_sessions(agent_id="agent-1", owner_id="user-1")

    assert store._request.await_count == 2
    messages_call = store._request.await_args_list[1]
    assert messages_call.args == ("GET", "rag_chat_messages")
    assert messages_call.kwargs["params"]["session_id"] == "in.(chat-1,chat-2)"
    assert summaries[0]["session_id"] == "chat-2"
    assert summaries[0]["title"] == "New chat"
    assert summaries[0]["last_message_preview"] == "Most recent chat"


def test_session_run_to_dict_includes_langfuse_and_feedback_fields():
    run = SessionRun(
        run_id="r1",
        query="What is LangGraph?",
        langfuse_trace_id="trace-1",
        langfuse_observation_id="obs-1",
        feedback_submitted_at="2026-05-05T10:00:00+00:00",
        feedback_helpful=True,
    )

    payload = run.to_dict()

    assert payload["langfuse_trace_id"] == "trace-1"
    assert payload["langfuse_observation_id"] == "obs-1"
    assert payload["feedback_submitted_at"] == "2026-05-05T10:00:00+00:00"
    assert payload["feedback_helpful"] is True


async def test_get_session_falls_back_when_extended_run_fields_are_missing():
    store = object.__new__(SupabaseSessionStore)
    store._session_run_extended_fields_supported = None

    session_response = MagicMock()
    session_response.json.return_value = [
        {"id": "session-1", "title": "Title", "created_at": "2026-01-01T00:00:00+00:00"}
    ]
    legacy_runs_response = MagicMock()
    legacy_runs_response.json.return_value = [
        {
            "id": "run-1",
            "query": "q",
            "source_urls": [],
            "report": "",
            "status": "completed",
            "error_details": None,
            "latest_node": None,
            "latest_event_at": None,
            "partial_report": "",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    turns_response = MagicMock()
    turns_response.json.return_value = []

    bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("GET", "https://example.com"),
        response=httpx.Response(
            400,
            json={"message": "column session_runs.langfuse_trace_id does not exist"},
        ),
    )

    store._request = AsyncMock(  # type: ignore[method-assign]
        side_effect=[session_response, bad_request, legacy_runs_response, turns_response]
    )

    session = await store.get_session(session_id="session-1", user_id="user-1")

    assert session is not None
    assert session.get_run("run-1") is not None
    assert store._session_run_extended_fields_supported is False


async def test_create_session_run_retries_without_optional_fields_on_legacy_schema():
    store = object.__new__(SupabaseSessionStore)
    store._session_run_extended_fields_supported = None

    bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("POST", "https://example.com"),
        response=httpx.Response(
            400,
            json={"message": "column session_runs.langfuse_trace_id does not exist"},
        ),
    )
    store._request = AsyncMock(side_effect=[bad_request, MagicMock()])  # type: ignore[method-assign]

    await store.create_session_run(
        user_id="user-1",
        session_id="session-1",
        run=SessionRun(
            run_id="run-1",
            query="What is LangGraph?",
            langfuse_trace_id="trace-1",
            langfuse_observation_id="obs-1",
            feedback_submitted_at="2026-01-01T00:00:00+00:00",
            feedback_helpful=True,
        ),
    )

    retry_payload = store._request.await_args_list[1].kwargs["json_body"]
    assert "langfuse_trace_id" not in retry_payload
    assert "feedback_helpful" not in retry_payload
    assert store._session_run_extended_fields_supported is False


async def test_update_session_run_retries_without_optional_fields_on_legacy_schema():
    store = object.__new__(SupabaseSessionStore)
    store._session_run_extended_fields_supported = None

    bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("PATCH", "https://example.com"),
        response=httpx.Response(
            400,
            json={"message": "column session_runs.feedback_submitted_at does not exist"},
        ),
    )
    success_response = MagicMock()
    success_response.json.return_value = [{"id": "run-1"}]
    store._request = AsyncMock(side_effect=[bad_request, success_response])  # type: ignore[method-assign]

    updated = await store.update_session_run(
        run_id="run-1",
        user_id="user-1",
        session_id="session-1",
        patch={
            "feedback_submitted_at": "2026-01-01T00:00:00+00:00",
            "feedback_helpful": True,
            "status": "completed",
        },
    )

    assert updated is True
    retry_payload = store._request.await_args_list[1].kwargs["json_body"]
    assert retry_payload == {"status": "completed"}


async def test_update_session_run_retries_on_generic_bad_request_when_optional_fields_present():
    store = object.__new__(SupabaseSessionStore)
    store._session_run_extended_fields_supported = None

    bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("PATCH", "https://example.com"),
        response=httpx.Response(
            400,
            json={"code": "PGRST204", "message": "schema cache mismatch"},
        ),
    )
    success_response = MagicMock()
    success_response.json.return_value = [{"id": "run-1"}]
    store._request = AsyncMock(side_effect=[bad_request, success_response])  # type: ignore[method-assign]

    updated = await store.update_session_run(
        run_id="run-1",
        user_id="user-1",
        session_id="session-1",
        patch={
            "langfuse_trace_id": "trace-1",
            "langfuse_observation_id": "obs-1",
            "status": "completed",
        },
    )

    assert updated is True
    retry_payload = store._request.await_args_list[1].kwargs["json_body"]
    assert retry_payload == {"status": "completed"}
    assert store._session_run_extended_fields_supported is False


async def test_update_session_run_retries_when_bad_request_body_is_not_json():
    store = object.__new__(SupabaseSessionStore)
    store._session_run_extended_fields_supported = None

    bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("PATCH", "https://example.com"),
        response=httpx.Response(400, text="bad request"),
    )
    success_response = MagicMock()
    success_response.json.return_value = [{"id": "run-1"}]
    store._request = AsyncMock(side_effect=[bad_request, success_response])  # type: ignore[method-assign]

    updated = await store.update_session_run(
        run_id="run-1",
        user_id="user-1",
        session_id="session-1",
        patch={
            "langfuse_trace_id": "trace-1",
            "status": "completed",
        },
    )

    assert updated is True
    retry_payload = store._request.await_args_list[1].kwargs["json_body"]
    assert retry_payload == {"status": "completed"}


async def test_update_session_run_falls_back_to_base_payload_when_status_shape_is_legacy():
    store = object.__new__(SupabaseSessionStore)
    store._session_run_extended_fields_supported = None

    first_bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("PATCH", "https://example.com"),
        response=httpx.Response(400, text="invalid column"),
    )
    second_bad_request = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("PATCH", "https://example.com"),
        response=httpx.Response(400, text="invalid column"),
    )
    success_response = MagicMock()
    success_response.json.return_value = [{"id": "run-1"}]
    store._request = AsyncMock(  # type: ignore[method-assign]
        side_effect=[first_bad_request, second_bad_request, success_response]
    )

    updated = await store.update_session_run(
        run_id="run-1",
        user_id="user-1",
        session_id="session-1",
        patch={
            "query": "q",
            "source_urls": [],
            "report": "r",
            "status": "completed",
            "error_details": None,
            "langfuse_trace_id": "trace-1",
        },
    )

    assert updated is True
    second_payload = store._request.await_args_list[1].kwargs["json_body"]
    third_payload = store._request.await_args_list[2].kwargs["json_body"]
    assert second_payload == {
        "query": "q",
        "source_urls": [],
        "report": "r",
        "status": "completed",
        "error_details": None,
    }
    assert third_payload == {
        "query": "q",
        "source_urls": [],
        "report": "r",
    }
