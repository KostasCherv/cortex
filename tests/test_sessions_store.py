from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.db.supabase_store import SupabaseSessionStore
from src.sessions import ConversationTurn, SessionRun, append_run, append_turn, create_session, get_session


async def test_sessions_module_delegates_create_to_store():
    mock_store = AsyncMock()
    mock_store.create_session.return_value = "session-object"
    with patch("src.sessions.get_session_store", return_value=mock_store):
        created = await create_session("user-1")
    assert created == "session-object"
    mock_store.create_session.assert_awaited_once_with(user_id="user-1", title="New session")


async def test_sessions_module_delegates_get_to_store():
    mock_store = AsyncMock()
    mock_store.get_session.return_value = None
    with patch("src.sessions.get_session_store", return_value=mock_store):
        session = await get_session("session-1", "user-1")
    assert session is None
    mock_store.get_session.assert_awaited_once_with(session_id="session-1", user_id="user-1")


async def test_sessions_module_delegates_append_operations():
    mock_store = AsyncMock()
    run = SessionRun(run_id="r1", query="q")
    turn = ConversationTurn(role="user", content="c")
    with patch("src.sessions.get_session_store", return_value=mock_store):
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

    with patch("src.db.supabase_store.get_cache", return_value=None):
        summaries = await store.list_rag_chat_sessions(agent_id="agent-1", owner_id="user-1")

    assert store._request.await_count == 2
    messages_call = store._request.await_args_list[1]
    assert messages_call.args == ("GET", "rag_chat_messages")
    assert messages_call.kwargs["params"]["session_id"] == "in.(chat-1,chat-2)"
    assert summaries[0]["session_id"] == "chat-2"
    assert summaries[0]["title"] == "New chat"
    assert summaries[0]["last_message_preview"] == "Most recent chat"


async def test_store_lists_ready_rag_chat_session_attachment_resource_ids():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = [
        {"id": "att-1", "resource_id": "res-a"},
        {"id": "att-2", "resource_id": "res-b"},
    ]
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    ready_ids = await store.list_ready_rag_chat_session_attachment_resource_ids(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
    )

    assert ready_ids == ["res-a", "res-b"]
    store._request.assert_awaited_once()
    request_call = store._request.await_args
    assert request_call.args == ("GET", "rag_chat_session_attachments")
    assert request_call.kwargs["params"] == {
        "select": "id,resource_id",
        "session_id": "eq.chat-1",
        "owner_id": "eq.user-1",
        "agent_id": "eq.agent-1",
        "state": "eq.ready",
        "order": "created_at.asc",
    }


async def test_store_deletes_session_attachments_for_chat_session():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = [
        {"id": "att-1", "resource_id": "res-a", "storage_uri": "attachments/chat-1/att-1.pdf"},
        {"id": "att-2", "resource_id": "res-b", "storage_uri": "attachments/chat-1/att-2.pdf"},
    ]
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    deleted = await store.delete_rag_chat_session_attachments(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
    )

    assert deleted == [
        {
            "attachment_id": "att-1",
            "resource_id": "res-a",
            "storage_uri": "attachments/chat-1/att-1.pdf",
        },
        {
            "attachment_id": "att-2",
            "resource_id": "res-b",
            "storage_uri": "attachments/chat-1/att-2.pdf",
        },
    ]
    store._request.assert_awaited_once()
    request_call = store._request.await_args
    assert request_call.args == ("DELETE", "rag_chat_session_attachments")
    assert request_call.kwargs["params"] == {
        "session_id": "eq.chat-1",
        "owner_id": "eq.user-1",
        "agent_id": "eq.agent-1",
    }
    assert request_call.kwargs["extra_headers"] == {"Prefer": "return=representation"}


async def test_store_creates_rag_chat_session_attachment():
    store = object.__new__(SupabaseSessionStore)
    store._request = AsyncMock()  # type: ignore[method-assign]

    await store.create_rag_chat_session_attachment(
        {
            "attachment_id": "att-1",
            "session_id": "chat-1",
            "agent_id": "agent-1",
            "owner_id": "user-1",
            "workspace_id": "ws-1",
            "resource_id": "res-1",
            "filename": "brief.pdf",
            "mime_type": "application/pdf",
            "byte_size": 123,
            "storage_uri": "supabase://bucket/key",
            "state": "uploaded",
            "error_details": None,
            "created_at": "2026-06-11T10:00:00+00:00",
            "updated_at": "2026-06-11T10:00:00+00:00",
        }
    )

    store._request.assert_awaited_once_with(
        "POST",
        "rag_chat_session_attachments",
        json_body={
            "id": "att-1",
            "session_id": "chat-1",
            "agent_id": "agent-1",
            "owner_id": "user-1",
            "workspace_id": "ws-1",
            "resource_id": "res-1",
            "filename": "brief.pdf",
            "mime_type": "application/pdf",
            "byte_size": 123,
            "storage_uri": "supabase://bucket/key",
            "state": "uploaded",
            "error_details": None,
            "created_at": "2026-06-11T10:00:00+00:00",
            "updated_at": "2026-06-11T10:00:00+00:00",
        },
        extra_headers={"Prefer": "resolution=ignore-duplicates"},
    )


async def test_store_updates_rag_chat_session_attachment():
    store = object.__new__(SupabaseSessionStore)
    store._request = AsyncMock()  # type: ignore[method-assign]

    await store.update_rag_chat_session_attachment(
        attachment_id="att-1",
        session_id="chat-1",
        agent_id="agent-1",
        owner_id="user-1",
        patch={"state": "ready", "error_details": None},
    )

    store._request.assert_awaited_once()
    request_call = store._request.await_args
    assert request_call.args == ("PATCH", "rag_chat_session_attachments")
    assert request_call.kwargs["params"] == {
        "id": "eq.att-1",
        "session_id": "eq.chat-1",
        "agent_id": "eq.agent-1",
        "owner_id": "eq.user-1",
    }
    assert request_call.kwargs["json_body"]["state"] == "ready"
    assert request_call.kwargs["json_body"]["error_details"] is None
    assert "updated_at" in request_call.kwargs["json_body"]


async def test_store_lists_rag_chat_session_attachments():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = [
        {"id": "att-1", "session_id": "chat-1"},
        {"id": "att-2", "session_id": "chat-1"},
    ]
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    rows = await store.list_rag_chat_session_attachments(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
    )

    assert rows == response.json.return_value
    request_call = store._request.await_args
    assert request_call.args == ("GET", "rag_chat_session_attachments")
    assert request_call.kwargs["params"]["session_id"] == "eq.chat-1"
    assert request_call.kwargs["params"]["owner_id"] == "eq.user-1"
    assert request_call.kwargs["params"]["agent_id"] == "eq.agent-1"
    assert request_call.kwargs["params"]["order"] == "created_at.asc"


async def test_store_lists_rag_chat_session_attachments_for_workspace_scope():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = []
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    await store.list_rag_chat_session_attachments(
        session_id="chat-1",
        owner_id="user-1",
        agent_id=None,
    )

    request_call = store._request.await_args
    assert request_call.kwargs["params"]["agent_id"] == "is.null"


async def test_store_deletes_rag_chat_session_attachments_by_ids():
    store = object.__new__(SupabaseSessionStore)
    response = MagicMock()
    response.json.return_value = [
        {"id": "att-1", "resource_id": "res-a", "storage_uri": "supabase://bucket/att-1.pdf"}
    ]
    store._request = AsyncMock(return_value=response)  # type: ignore[method-assign]

    deleted = await store.delete_rag_chat_session_attachments_by_ids(
        attachment_ids=["att-1", "att-2"],
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
    )

    assert deleted == [
        {
            "attachment_id": "att-1",
            "resource_id": "res-a",
            "storage_uri": "supabase://bucket/att-1.pdf",
        }
    ]
    request_call = store._request.await_args
    assert request_call.args == ("DELETE", "rag_chat_session_attachments")
    assert request_call.kwargs["params"] == {
        "id": "in.(att-1,att-2)",
        "session_id": "eq.chat-1",
        "owner_id": "eq.user-1",
        "agent_id": "eq.agent-1",
    }
    assert request_call.kwargs["extra_headers"] == {"Prefer": "return=representation"}


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


async def test_create_resource_job_and_outbox_invalidates_resources_list_cache():
    store = object.__new__(SupabaseSessionStore)
    store._request = AsyncMock()  # type: ignore[method-assign]
    store._invalidate_rag_resources_list_cache = AsyncMock()  # type: ignore[method-assign]

    await store.create_resource_job_and_outbox(
        resource_payload={
            "resource_id": "res-1",
            "owner_id": "user-1",
            "workspace_id": "ws-1",
            "filename": "doc.pdf",
            "state": "uploaded",
        },
        job_payload={"job_id": "job-1", "resource_id": "res-1"},
        outbox_payload={"id": "evt-1", "event_name": "rag/ingestion.requested"},
    )

    store._request.assert_awaited_once_with(
        "POST",
        "rpc/create_resource_job_and_outbox",
        json_body={
            "p_resource": {
                "resource_id": "res-1",
                "owner_id": "user-1",
                "workspace_id": "ws-1",
                "filename": "doc.pdf",
                "state": "uploaded",
            },
            "p_job": {"job_id": "job-1", "resource_id": "res-1"},
            "p_outbox": {"id": "evt-1", "event_name": "rag/ingestion.requested"},
        },
    )
    store._invalidate_rag_resources_list_cache.assert_awaited_once_with("user-1", "ws-1")
