from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.datastructures import UploadFile

from src.rag import (
    CHAT_SCOPE_WORKSPACE,
    RagChatMessage,
    RagValidationError,
    _run_ingestion_job,
    create_resource_and_ingest,
    delete_agent,
    delete_chat_session,
    delete_last_exchange,
    get_chat_session,
    list_chat_sessions,
    retrieve_context_for_query,
    suggest_chat_session_title,
    update_chat_session_title,
)
from src.rag_engine import query_resource_context


async def test_retrieve_context_for_query_returns_empty_without_resource_ids():
    with patch("src.rag.query_resource_context", new=AsyncMock()) as query_mock:
        result = await retrieve_context_for_query(
            user_id="user-1",
            resource_ids=[],
            question="Hello?",
        )

    query_mock.assert_not_awaited()
    assert result.context == ""
    assert result.chunks == []


async def test_create_resource_rejects_unsupported_extension():
    file = UploadFile(filename="notes.exe", file=BytesIO(b"abc"), headers={"content-type": "text/plain"})
    with patch("src.rag._get_store", return_value=AsyncMock()):
        try:
            await create_resource_and_ingest(file, "user-1")
            assert False, "Expected RagValidationError"
        except RagValidationError as exc:
            assert exc.code == "unsupported_type"


async def test_list_chat_sessions_returns_agent_scoped_summaries():
    mock_store = AsyncMock()
    mock_store.list_rag_chat_sessions.return_value = [
        {
            "session_id": "chat-2",
            "agent_id": "agent-1",
            "owner_id": "user-1",
            "created_at": "2026-04-23T09:00:00+00:00",
            "last_message_at": "2026-04-23T09:05:00+00:00",
            "last_message_preview": "Latest answer",
        },
        {
            "session_id": "chat-1",
            "agent_id": "agent-1",
            "owner_id": "user-1",
            "created_at": "2026-04-22T09:00:00+00:00",
            "last_message_at": "2026-04-22T09:01:00+00:00",
            "last_message_preview": "Earlier answer",
        },
    ]

    with patch("src.rag._get_store", return_value=mock_store):
        sessions = await list_chat_sessions(agent_id="agent-1", user_id="user-1")

    mock_store.list_rag_chat_sessions.assert_awaited_once_with(
        agent_id="agent-1",
        owner_id="user-1",
        chat_scope="agent",
    )
    assert sessions[0]["session_id"] == "chat-2"
    assert sessions[0]["last_message_preview"] == "Latest answer"


async def test_get_chat_session_is_scoped_to_owner_and_agent():
    mock_store = AsyncMock()
    mock_store.get_rag_chat_session.return_value = {
        "session_id": "chat-1",
        "agent_id": "agent-1",
        "owner_id": "user-1",
        "created_at": "2026-04-23T09:00:00+00:00",
    }

    with patch("src.rag._get_store", return_value=mock_store):
        session = await get_chat_session(
            session_id="chat-1",
            agent_id="agent-1",
            user_id="user-1",
        )

    mock_store.get_rag_chat_session.assert_awaited_once_with(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
        chat_scope="agent",
    )
    assert session is not None
    assert session["session_id"] == "chat-1"


async def test_update_chat_session_title_delegates_to_store():
    mock_store = AsyncMock()
    mock_store.update_rag_chat_session_title.return_value = True

    with patch("src.rag._get_store", return_value=mock_store):
        updated = await update_chat_session_title(
            session_id="chat-1",
            agent_id="agent-1",
            user_id="user-1",
            title="Updated title",
        )

    assert updated is True
    mock_store.update_rag_chat_session_title.assert_awaited_once_with(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
        title="Updated title",
        chat_scope="agent",
    )


async def test_delete_chat_session_delegates_to_store():
    mock_store = AsyncMock()
    mock_store.delete_rag_chat_session.return_value = True

    with patch("src.rag._get_store", return_value=mock_store):
        deleted = await delete_chat_session(
            session_id="chat-1",
            agent_id="agent-1",
            user_id="user-1",
        )

    assert deleted is True
    mock_store.delete_rag_chat_session.assert_awaited_once_with(
        session_id="chat-1",
        owner_id="user-1",
        agent_id="agent-1",
        chat_scope="agent",
    )


async def test_delete_agent_delegates_to_store():
    mock_store = AsyncMock()
    mock_store.get_rag_agent.return_value = {"agent_id": "agent-1"}
    mock_store.delete_rag_agent.return_value = True

    with patch("src.rag._get_store", return_value=mock_store):
        deleted = await delete_agent(agent_id="agent-1", user_id="user-1")

    assert deleted is True
    mock_store.get_rag_agent.assert_awaited_once()
    mock_store.delete_rag_agent.assert_awaited_once_with(
        agent_id="agent-1",
        owner_id="user-1",
        workspace_id="user-1",
    )


async def test_delete_agent_returns_false_when_missing():
    mock_store = AsyncMock()
    mock_store.get_rag_agent.return_value = None

    with patch("src.rag._get_store", return_value=mock_store):
        deleted = await delete_agent(agent_id="agent-404", user_id="user-1")

    assert deleted is False
    mock_store.delete_rag_agent.assert_not_awaited()


async def test_suggest_chat_session_title_uses_llm_result():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = type("R", (), {"content": "Pricing policy comparison"})()

    with patch("src.llm.factory.get_llm", return_value=mock_llm):
        assert (
            await suggest_chat_session_title("Compare pricing policy docs")
            == "Pricing policy comparison"
        )


async def test_suggest_chat_session_title_fallback_when_llm_fails():
    with patch("src.llm.factory.get_llm", side_effect=RuntimeError("llm down")):
        assert (
            await suggest_chat_session_title("How to reset access token quickly")
            == "How to reset access token quickly"
        )


async def test_query_resource_context_returns_reranked_chunks_and_context():
    raw_result = MagicMock()
    raw_result.chunks = [
        {
            "resource_id": "res-1",
            "chunk_id": "raw-1",
            "text": "Less relevant",
            "source_title": "Doc",
            "source_url": "https://example.com/1",
        },
        {
            "resource_id": "res-1",
            "chunk_id": "raw-2",
            "text": "More relevant",
            "source_title": "Doc",
            "source_url": "https://example.com/2",
        },
    ]
    raw_result.entities = ["entity"]

    reranked = [
        {
            "resource_id": "res-1",
            "chunk_id": "raw-2",
            "text": "More relevant",
            "source_title": "Doc",
            "source_url": "https://example.com/2",
            "rerank_score": 0.92,
        }
    ]

    with (
        patch("src.rag_engine.Neo4jGraphStore") as graph_cls,
        patch("src.rag_engine.rerank_chunks", return_value=reranked) as rerank_mock,
    ):
        graph_cls.return_value.query_context.return_value = raw_result

        result = await query_resource_context(
            store=AsyncMock(),
            resource_ids=["res-1"],
            owner_id="user-1",
            workspace_id="user-1",
            query="What is relevant?",
        )

    rerank_mock.assert_called_once()
    assert [chunk["chunk_id"] for chunk in result.chunks] == ["raw-2"]
    assert result.chunks[0]["rerank_score"] == 0.92
    assert "[source:Doc chunk:raw-2]" in result.context
    assert "Less relevant" not in result.context


def test_rag_chat_message_to_dict_includes_chat_scope():
    msg = RagChatMessage(
        message_id="m1",
        session_id="s1",
        agent_id=None,
        owner_id="u1",
        role="assistant",
        content="hello",
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    payload = msg.to_dict()
    assert payload["chat_scope"] == CHAT_SCOPE_WORKSPACE


async def test_run_ingestion_job_marks_ready_on_success():
    mock_store = AsyncMock()
    mock_store.get_rag_ingestion_job.return_value = {
        "job_id": "job-1",
        "resource_id": "res-1",
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "status": "queued",
        "stage": "queued",
        "retries": 0,
        "max_retries": 1,
        "error_details": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    mock_store.get_rag_resource.return_value = {
        "resource_id": "res-1",
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "filename": "doc.txt",
        "mime_type": "text/plain",
        "byte_size": 12,
        "storage_uri": "/tmp/doc.txt",
        "state": "uploaded",
        "error_details": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }

    mock_storage = AsyncMock()
    mock_storage.create_signed_download_url = AsyncMock(return_value="https://signed-url")

    with (
        patch("src.rag._get_store", return_value=mock_store),
        patch("src.rag._get_storage", return_value=mock_storage),
        patch("src.rag.ingest_resource_from_locator", new=AsyncMock(return_value=3)),
    ):
        await _run_ingestion_job("job-1")

    assert mock_store.update_rag_resource.await_count >= 1
    assert any(
        call.args[1].get("state") == "ready"
        for call in mock_store.update_rag_resource.await_args_list
    )


async def test_run_ingestion_job_marks_failed_after_retries():
    mock_store = AsyncMock()
    mock_store.get_rag_ingestion_job.return_value = {
        "job_id": "job-1",
        "resource_id": "res-1",
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "status": "queued",
        "stage": "queued",
        "retries": 0,
        "max_retries": 1,
        "error_details": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    mock_store.get_rag_resource.return_value = {
        "resource_id": "res-1",
        "owner_id": "user-1",
        "workspace_id": "user-1",
        "filename": "doc.txt",
        "mime_type": "text/plain",
        "byte_size": 12,
        "storage_uri": "/tmp/doc.txt",
        "state": "uploaded",
        "error_details": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }

    mock_storage = AsyncMock()
    mock_storage.create_signed_download_url = AsyncMock(return_value="https://signed-url")

    with (
        patch("src.rag._get_store", return_value=mock_store),
        patch("src.rag._get_storage", return_value=mock_storage),
        patch("src.rag.ingest_resource_from_locator", new=AsyncMock(side_effect=RuntimeError("ingest failed"))),
    ):
        await _run_ingestion_job("job-1")

    assert any(
        call.args[1].get("state") == "failed"
        for call in mock_store.update_rag_resource.await_args_list
    )


async def test_create_resource_writes_outbox_event():
    file = UploadFile(
        filename="notes.txt",
        file=BytesIO(b"hello world"),
        headers={"content-type": "text/plain"},
    )
    mock_store = AsyncMock()
    mock_store.count_rag_resources_in_workspace.return_value = 0
    mock_storage = AsyncMock()
    mock_storage.upload_bytes = AsyncMock(return_value="supabase://rag-resources/user-1/path")

    with (
        patch("src.rag._get_store", return_value=mock_store),
        patch("src.rag._get_storage", return_value=mock_storage),
    ):
        resource, job = await create_resource_and_ingest(file, "user-1")

    assert mock_store.create_resource_job_and_outbox.await_count == 1
    call_kwargs = mock_store.create_resource_job_and_outbox.await_args.kwargs
    outbox = call_kwargs["outbox_payload"]
    assert outbox["event_name"] == "rag/ingestion.requested"
    assert outbox["payload"]["job_id"] == job.job_id
    assert outbox["payload"]["resource_id"] == resource.resource_id
    assert outbox["payload"]["owner_id"] == "user-1"
    assert "workspace_id" in outbox["payload"]
    # resource and job must NOT be written separately — the RPC handles everything
    mock_store.create_rag_resource.assert_not_awaited()
    mock_store.create_rag_ingestion_job.assert_not_awaited()


async def test_delete_last_exchange_uses_store_method():
    mock_store = AsyncMock()
    mock_store.delete_last_user_assistant_pair.return_value = (True, None)

    with patch("src.rag._get_store", return_value=mock_store):
        deleted, err = await delete_last_exchange(session_id="sess-1", user_id="user-1")

    assert deleted is True
    assert err is None
    mock_store.delete_last_user_assistant_pair.assert_awaited_once_with(
        session_id="sess-1", owner_id="user-1"
    )


async def test_delete_last_exchange_propagates_empty_error():
    mock_store = AsyncMock()
    mock_store.delete_last_user_assistant_pair.return_value = (False, "empty")

    with patch("src.rag._get_store", return_value=mock_store):
        deleted, err = await delete_last_exchange(session_id="sess-1", user_id="user-1")

    assert deleted is False
    assert err == "empty"
