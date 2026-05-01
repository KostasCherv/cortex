"""Tests for FastAPI endpoints (src/api/endpoints.py)"""

import asyncio
import json
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.endpoints import app
from src.auth import AuthenticatedUser, get_authenticated_user
from src.rag import RagValidationError
from src.sessions import Session, SessionRun

client = TestClient(app)


def _auth_override() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="test-user", email="test@example.com")


app.dependency_overrides[get_authenticated_user] = _auth_override


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_research_streams_events():
    search_result = [{"url": "https://example.com", "title": "Example", "content": "Test"}]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(
                content='[{"url":"https://example.com","title":"Example","summary":"Summary text."}]'
            ),
            MagicMock(content="# Report\nFinal output."),
        ]
    )

    with (
        patch("src.graph.nodes.perform_search", return_value=search_result),
        patch("src.graph.nodes.fetch_url_content", new=AsyncMock(return_value="Page text")),
        patch("src.graph.nodes.get_llm", return_value=mock_llm),
        patch("src.graph.nodes.VectorStoreManager") as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs.search_reports.return_value = []
        mock_vs.rerank_documents.return_value = [
            {
                "url": "https://example.com",
                "title": "Example",
                "raw_text": "Page text",
                "score": 0.9,
            }
        ]
        mock_vs_cls.return_value = mock_vs
        response = client.post(
            "/research",
            json={"query": "What is LangGraph?", "use_vector_store": False},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE lines
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    node_names = [e["node"] for e in events]
    assert "__end__" in node_names


def test_research_bad_request_returns_422():
    response = client.post("/research", json={})  # missing 'query'
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Session endpoint tests
# ---------------------------------------------------------------------------

def test_create_session_returns_session_id():
    mock_session = Session(
        session_id="session-1",
        title="LangGraph basics",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with patch("src.api.endpoints.create_session", new=AsyncMock(return_value=mock_session)):
        response = client.post("/sessions", json={"query": "What is LangGraph?"})
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-1"
    assert data["title"] == "LangGraph basics"
    assert data["created_at"] == "2026-01-01T00:00:00+00:00"


def test_get_session_returns_session_state():
    mock_session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)):
        get_resp = client.get("/sessions/session-1")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == "session-1"
        assert data["runs"] == []
        assert data["conversation"] == []


def test_list_sessions_returns_summaries():
    with patch(
        "src.api.endpoints.list_sessions",
        new=AsyncMock(
            return_value=[
                {
                    "session_id": "session-1",
                    "title": "LangGraph basics",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "session_id": "session-2",
                    "title": "Agent architecture",
                    "created_at": "2026-01-02T00:00:00+00:00",
                },
            ]
        ),
    ):
        response = client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) == 2
    assert data["sessions"][0]["session_id"] == "session-1"


def test_get_session_returns_404_for_unknown_id():
    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=None)):
        response = client.get("/sessions/does-not-exist")
        assert response.status_code == 404


def test_followup_returns_400_when_no_run_exists():
    mock_session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)):
        followup_resp = client.post(
            "/sessions/session-1/followup",
            json={"question": "What did you find?"},
        )
        assert followup_resp.status_code == 400


def test_followup_returns_404_for_unknown_session():
    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=None)):
        response = client.post(
            "/sessions/no-such-session/followup",
            json={"question": "anything"},
        )
        assert response.status_code == 404


def test_followup_returns_404_for_unknown_run_id():
    mock_session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)):
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "anything", "run_id": "nonexistent-run"},
        )
        assert response.status_code == 404


def test_session_research_queues_background_run():
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="old", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    mock_create_session_run = AsyncMock(return_value=None)
    mock_enqueue_event = AsyncMock(return_value=None)

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.create_session_run", new=mock_create_session_run),
        patch("src.api.endpoints.outbox.enqueue_event", new=mock_enqueue_event),
        patch("src.api.endpoints.outbox.dispatch_outbox_events", new=AsyncMock(return_value=1)),
    ):
        response = client.post(
            "/sessions/session-1/research",
            json={"query": "What is LangGraph?", "use_vector_store": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["run_id"]
    assert mock_create_session_run.await_count == 1
    assert mock_enqueue_event.await_count == 1


def test_execute_research_run_marks_completed_and_records():
    from src.api.endpoints import _execute_research_run

    class FakeGraph:
        async def astream(self, _initial_state):
            yield {
                "report_node": {
                    "report": "Final report",
                    "retrieved_contents": [{"url": "https://example.com", "title": "Example", "content": "Chunk"}],
                    "summaries": [],
                }
            }

    @contextmanager
    def _mock_trace_ctx(**_kwargs):
        yield MagicMock(workflow_id="wf-1")

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    mock_update = AsyncMock(return_value=True)
    mock_manager = MagicMock()

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=session)),
        patch("src.api.endpoints.build_graph", return_value=FakeGraph()),
        patch("src.api.endpoints.update_session_run", new=mock_update),
        patch("src.api.endpoints.VectorStoreManager", return_value=mock_manager),
        patch("src.api.endpoints.start_workflow_run", side_effect=_mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run"),
    ):
        asyncio.run(
            _execute_research_run(
                session_id="session-1",
                run_id="run-1",
                user_id="user-1",
                query="What is LangGraph?",
                use_vector_store=False,
            )
        )

    mock_update.assert_awaited_once_with(
        run_id="run-1",
        user_id="user-1",
        session_id="session-1",
        patch={
            "query": "What is LangGraph?",
            "source_urls": ["https://example.com"],
            "report": "Final report",
            "status": "completed",
            "error_details": None,
        },
    )
    mock_manager.save_source_chunks.assert_called_once()


def test_execute_research_run_marks_failed_on_error():
    from src.api.endpoints import _execute_research_run

    class FailingGraph:
        async def astream(self, _initial_state):
            raise RuntimeError("graph failure")
            yield  # pragma: no cover

    @contextmanager
    def _mock_trace_ctx(**_kwargs):
        yield MagicMock(workflow_id="wf-1")

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    mock_update = AsyncMock(return_value=True)

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=session)),
        patch("src.api.endpoints.build_graph", return_value=FailingGraph()),
        patch("src.api.endpoints.update_session_run", new=mock_update),
        patch("src.api.endpoints.start_workflow_run", side_effect=_mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run"),
    ):
        try:
            asyncio.run(
                _execute_research_run(
                    session_id="session-1",
                    run_id="run-1",
                    user_id="user-1",
                    query="What is LangGraph?",
                    use_vector_store=False,
                )
            )
        except RuntimeError:
            pass

    mock_update.assert_awaited_once_with(
        run_id="run-1",
        user_id="user-1",
        session_id="session-1",
        patch={"status": "failed", "error_details": "graph failure"},
    )


def test_execute_research_runs_can_overlap_in_time():
    from src.api.endpoints import _execute_research_run

    starts: list[tuple[str, float]] = []
    ends: list[tuple[str, float]] = []

    class SlowGraph:
        async def astream(self, initial_state):
            run_id = initial_state["run_id"]
            starts.append((run_id, time.perf_counter()))
            await asyncio.sleep(0.05)
            ends.append((run_id, time.perf_counter()))
            yield {
                "report_node": {
                    "report": f"Final report {run_id}",
                    "retrieved_contents": [],
                    "summaries": [],
                }
            }

    @contextmanager
    def _mock_trace_ctx(**_kwargs):
        yield MagicMock(workflow_id="wf-1")

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=session)),
        patch("src.api.endpoints.build_graph", return_value=SlowGraph()),
        patch("src.api.endpoints._record_session_run", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.start_workflow_run", side_effect=_mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run"),
    ):
        async def _run_pair() -> None:
            await asyncio.gather(
                _execute_research_run(
                    session_id="session-1",
                    run_id="run-a",
                    user_id="user-1",
                    query="What is LangGraph?",
                    use_vector_store=False,
                ),
                _execute_research_run(
                    session_id="session-1",
                    run_id="run-b",
                    user_id="user-1",
                    query="What is LangGraph?",
                    use_vector_store=False,
                ),
            )

        asyncio.run(_run_pair())

    assert len(starts) == 2
    assert len(ends) == 2
    latest_start = max(ts for _, ts in starts)
    earliest_end = min(ts for _, ts in ends)
    assert latest_start < earliest_end


def test_record_session_run_raises_when_finalize_update_fails():
    from src.api.endpoints import _record_session_run

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    with patch("src.api.endpoints.update_session_run", new=AsyncMock(return_value=False)):
        try:
            asyncio.run(
                _record_session_run(
                    session=session,
                    user_id="user-1",
                    run_id="run-1",
                    query="What is LangGraph?",
                    final_state={"report": "Final report", "retrieved_contents": [], "summaries": []},
                )
            )
            assert False, "Expected RuntimeError when run finalization update fails"
        except RuntimeError as exc:
            assert "Could not finalize run 'run-1'" in str(exc)


def test_session_endpoints_require_auth():
    app.dependency_overrides.pop(get_authenticated_user, None)
    try:
        create_resp = client.post("/sessions", json={})
        assert create_resp.status_code == 401
    finally:
        app.dependency_overrides[get_authenticated_user] = _auth_override


def test_startup_validation_does_not_fail_without_supabase_configuration():
    with (
        patch("src.api.endpoints.settings.supabase_url", ""),
        patch("src.api.endpoints.settings.supabase_service_role_key", ""),
        patch("src.api.endpoints.ensure_store_initialized") as mock_init,
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()) as mock_storage_ready,
    ):
        asyncio.run(app.router.on_startup[0]())
        mock_init.assert_not_called()
        mock_storage_ready.assert_not_awaited()


def test_startup_validation_checks_rag_storage_when_supabase_configured():
    with (
        patch("src.api.endpoints.settings.supabase_url", "https://example.supabase.co"),
        patch("src.api.endpoints.settings.supabase_service_role_key", "service-role"),
        patch("src.api.endpoints.ensure_store_initialized") as mock_init,
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()) as mock_storage_ready,
    ):
        asyncio.run(app.router.on_startup[0]())
        mock_init.assert_called_once()
        mock_storage_ready.assert_awaited_once()


# ---------------------------------------------------------------------------
# Follow-up suggestion tests
# ---------------------------------------------------------------------------

def test_generate_suggestions_returns_list():
    """_generate_suggestions parses numbered lines into a list of strings."""
    from src.api.endpoints import _generate_suggestions

    mock_result = MagicMock()
    mock_result.content = "1. What are the limitations?\n2. How does it compare to X?\n3. What are real-world use cases?"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_result)

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        suggestions = asyncio.run(
            _generate_suggestions("What is LangGraph?", "LangGraph is a library...", "topics: graphs, agents")
        )

    assert isinstance(suggestions, list)
    assert len(suggestions) == 3
    assert suggestions[0] == "What are the limitations?"
    assert suggestions[1] == "How does it compare to X?"
    assert suggestions[2] == "What are real-world use cases?"


def test_generate_suggestions_returns_empty_on_error():
    """_generate_suggestions returns [] when the LLM raises an exception."""
    from src.api.endpoints import _generate_suggestions

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        suggestions = asyncio.run(
            _generate_suggestions("What is LangGraph?", "Some answer", "context")
        )

    assert suggestions == []


def test_followup_stream_includes_suggestions_event():
    """The followup SSE stream emits a 'suggestions' event after citations."""
    mock_chunk = MagicMock()
    mock_chunk.content = "Here is the answer."

    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([mock_chunk]))

    mock_suggestions_llm = AsyncMock()
    mock_suggestions_result = MagicMock()
    mock_suggestions_result.content = "1. Question one?\n2. Question two?\n3. Question three?"
    mock_suggestions_llm.ainvoke = AsyncMock(return_value=mock_suggestions_result)

    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    def get_llm_side_effect(temperature=0.2):
        if temperature == 0.7:
            return mock_suggestions_llm
        return mock_llm

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.VectorStoreManager"),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", side_effect=get_llm_side_effect),
    ):
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "What did you find?", "run_id": "run-1"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    event_types = [e["type"] for e in events]
    assert "suggestions" in event_types

    suggestions_event = next(e for e in events if e["type"] == "suggestions")
    assert isinstance(suggestions_event["suggestions"], list)
    assert len(suggestions_event["suggestions"]) > 0

    # suggestions must appear before done
    suggestions_idx = event_types.index("suggestions")
    done_idx = event_types.index("done")
    assert suggestions_idx < done_idx


async def _async_iter_impl(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# RAG endpoint tests
# ---------------------------------------------------------------------------


def test_rag_list_resources_returns_payload():
    mock_resource = MagicMock()
    mock_resource.to_dict.return_value = {"resource_id": "r-1", "state": "ready"}
    with patch(
        "src.api.endpoints.list_rag_resources_records",
        new=AsyncMock(return_value=[mock_resource]),
    ):
        response = client.get("/api/rag/resources")
    assert response.status_code == 200
    payload = response.json()
    assert payload["resources"] == [{"resource_id": "r-1", "state": "ready"}]


def test_rag_upload_maps_validation_errors():
    with patch(
        "src.api.endpoints.create_resource_and_ingest",
        new=AsyncMock(side_effect=RagValidationError("unsupported_type", "Unsupported file type.")),
    ):
        response = client.post(
            "/api/rag/resources/upload",
            files={"file": ("test.exe", b"abc", "application/octet-stream")},
        )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_type"


def test_rag_upload_dispatches_outbox_after_success():
    mock_resource = MagicMock()
    mock_resource.to_dict.return_value = {"resource_id": "r-1", "state": "uploaded"}
    mock_job = MagicMock()
    mock_job.to_dict.return_value = {"job_id": "j-1", "status": "queued"}

    with (
        patch(
            "src.api.endpoints.create_resource_and_ingest",
            new=AsyncMock(return_value=(mock_resource, mock_job)),
        ),
        patch("src.outbox.dispatch_outbox_events", new=AsyncMock(return_value=1)) as mock_dispatch,
    ):
        response = client.post(
            "/api/rag/resources/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 200
    mock_dispatch.assert_awaited_once_with(limit=10)


def test_rag_chat_returns_agent_reply():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."

    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = [
        {
            "source_title": "Doc",
            "source_url": "https://example.com",
            "chunk_id": "chunk-1",
            "text": "Retrieved chunk text.",
        }
    ]

    mock_user_message = MagicMock()
    mock_user_message.to_dict.return_value = {"role": "user"}
    mock_assistant_message = MagicMock()
    mock_assistant_message.to_dict.return_value = {
        "role": "assistant",
        "content": "Answer",
        "citations": [
            {
                "source_title": "Doc",
                "source_url": "https://example.com",
                "chunk_id": "chunk-1",
                "text": "Retrieved chunk text.",
            }
        ],
    }

    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints.RagChatMessage", side_effect=[mock_user_message, mock_assistant_message]),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "chat-1"
    assert payload["reply"]["citations"] == [
        {
            "source_title": "Doc",
            "source_url": "https://example.com",
            "chunk_id": "chunk-1",
            "text": "Retrieved chunk text.",
        }
    ]


def test_rag_chat_stream_returns_rich_citations():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."

    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = [
        {
            "source_title": "Doc",
            "source_url": "https://example.com",
            "chunk_id": "chunk-1",
            "text": "Retrieved chunk text.",
        }
    ]

    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {
            "source_title": "Doc",
            "source_url": "https://example.com",
            "chunk_id": "chunk-1",
            "text": "Retrieved chunk text.",
        }
    ]


def test_rag_chat_sessions_returns_agent_scoped_summaries():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch(
            "src.api.endpoints.list_rag_chat_sessions",
            new=AsyncMock(
                return_value=[
                    {
                        "session_id": "chat-1",
                        "agent_id": "agent-1",
                        "owner_id": "test-user",
                        "title": "Refund policy discussion",
                        "created_at": "2026-04-23T09:00:00+00:00",
                        "last_message_at": "2026-04-23T09:05:00+00:00",
                        "last_message_preview": "What is the refund window?",
                    }
                ]
            ),
        ),
    ):
        response = client.get("/api/rag/agents/agent-1/chat/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["sessions"][0]["session_id"] == "chat-1"
    assert payload["sessions"][0]["title"] == "Refund policy discussion"
    assert payload["sessions"][0]["last_message_preview"] == "What is the refund window?"


def test_rag_chat_sessions_returns_404_for_unknown_agent():
    with patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=None)):
        response = client.get("/api/rag/agents/agent-404/chat/sessions")

    assert response.status_code == 404


def test_rag_chat_session_messages_returns_scoped_messages():
    mock_agent = MagicMock()
    mock_message = MagicMock()
    mock_message.to_dict.return_value = {
        "message_id": "msg-1",
        "session_id": "chat-1",
        "agent_id": "agent-1",
        "owner_id": "test-user",
        "role": "user",
        "content": "Hello",
        "citations": [],
        "created_at": "2026-04-23T09:00:00+00:00",
    }
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(
                return_value={
                    "session_id": "chat-1",
                    "agent_id": "agent-1",
                    "owner_id": "test-user",
                    "created_at": "2026-04-23T09:00:00+00:00",
                }
            ),
        ),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[mock_message])),
    ):
        response = client.get("/api/rag/agents/agent-1/chat/sessions/chat-1/messages")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "chat-1"
    assert payload["messages"] == [mock_message.to_dict.return_value]


def test_rag_chat_session_messages_returns_404_for_unknown_session():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value=None)),
    ):
        response = client.get("/api/rag/agents/agent-1/chat/sessions/chat-404/messages")

    assert response.status_code == 404


def test_rag_chat_session_title_update():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.update_rag_chat_session_title", new=AsyncMock(return_value=True)),
    ):
        response = client.patch(
            "/api/rag/agents/agent-1/chat/sessions/chat-1",
            json={"title": "Policy summary"},
        )

    assert response.status_code == 200
    assert response.json() == {"session_id": "chat-1", "title": "Policy summary"}


def test_rag_chat_session_title_update_rejects_empty_title():
    mock_agent = MagicMock()
    with patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))):
        response = client.patch(
            "/api/rag/agents/agent-1/chat/sessions/chat-1",
            json={"title": "    "},
        )

    assert response.status_code == 400


def test_rag_chat_session_delete():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.delete_rag_chat_session", new=AsyncMock(return_value=True)),
    ):
        response = client.delete("/api/rag/agents/agent-1/chat/sessions/chat-1")

    assert response.status_code == 200
    assert response.json() == {"session_id": "chat-1", "deleted": True}


def _async_iter(items):
    return _async_iter_impl(items)
