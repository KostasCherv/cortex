"""Tests for FastAPI endpoints (src/api/endpoints.py)"""

import asyncio
import json
import time
from datetime import UTC, datetime
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.endpoints import _stream_research, app
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing.application.service import UsageIncrement
from src.billing.domain.errors import QuotaExceededError
from src.billing.domain.models import DailyUsage, Plan, QuotaLimits, UsageSummary, UserSubscription
from src.rag import RagValidationError
from src.sessions import Session, SessionRun
import src.api.endpoints as endpoints

with patch("src.api.endpoints.validate_web_search_provider_health"):
    client = TestClient(app)


class _FakeBillingService:
    def __init__(self) -> None:
        self.raise_quota = False

    async def check_and_consume_usage(self, user_id: str, increment: UsageIncrement):
        if self.raise_quota:
            raise QuotaExceededError(
                plan="free",
                limit_type="questions_daily",
                limit=10,
                used=10,
                resets_at="2026-01-01T00:00:00+00:00",
                message="Daily question limit reached.",
            )
        return None

    async def get_usage_summary(self, user_id: str):
        raise NotImplementedError

    async def start_checkout(self, *, user_id: str, email: str | None):
        return "https://example.com/checkout"

    async def start_portal(self, *, user_id: str):
        return "https://example.com/portal"

    async def handle_webhook(self, payload: bytes, signature: str):
        return None


_fake_billing = _FakeBillingService()
endpoints._billing_service = _fake_billing


def _auth_override() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="test-user", email="test@example.com")


app.dependency_overrides[get_authenticated_user] = _auth_override


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_billing_usage_endpoint():
    summary = UsageSummary(
        plan=Plan.FREE,
        date=datetime(2026, 1, 1, tzinfo=UTC).date(),
        limits=QuotaLimits(research_queries_daily=3, total_questions_daily=10),
        usage=DailyUsage(
            usage_date=datetime(2026, 1, 1, tzinfo=UTC).date(),
            research_queries_count=1,
            total_questions_count=2,
        ),
        resets_at=datetime(2026, 1, 2, tzinfo=UTC),
        subscription=UserSubscription(
            user_id="test-user",
            plan=Plan.FREE,
            status="active",
            current_period_end=datetime(2026, 1, 10, tzinfo=UTC),
            cancel_at_period_end=True,
            cancel_at=datetime(2026, 1, 10, tzinfo=UTC),
            canceled_at=datetime(2026, 1, 5, tzinfo=UTC),
        ),
    )

    with patch.object(_fake_billing, "get_usage_summary", new=AsyncMock(return_value=summary)):
        response = client.get("/api/billing/usage")
    assert response.status_code == 200
    data = response.json()
    assert data["plan"] == "free"
    assert data["limits"]["research_queries_daily"] == 3
    assert data["subscription"]["status"] == "active"
    assert data["subscription"]["cancel_at_period_end"] is True


def test_stream_research_emits_node_completion_from_langgraph_node_metadata():
    """astream_events may use a runnable ``name`` that differs from the graph node key."""

    async def fake_astream_events(initial_state, version="v2"):
        yield {
            "event": "on_chain_end",
            "name": "RunnableSequence",
            "metadata": {"langgraph_node": "report"},
            "data": {"output": {"report": "# From metadata", "error": None}},
        }

    mock_graph = MagicMock()
    mock_graph.astream_events = fake_astream_events

    async def collect():
        lines: list[str] = []
        async for line in _stream_research("q", False):
            lines.append(line)
        return lines

    with patch("src.api.endpoints.build_graph", return_value=mock_graph):
        raw_lines = asyncio.run(collect())

    events = []
    for line in raw_lines:
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    report_events = [e for e in events if e.get("node") == "report"]
    assert len(report_events) == 1
    assert report_events[0]["data"]["report"] == "# From metadata"
    assert any(e.get("node") == "__end__" for e in events)


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
    _fake_billing.raise_quota = False
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


def test_session_research_returns_429_when_quota_exceeded():
    _fake_billing.raise_quota = True
    mock_session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)):
        response = client.post(
            "/sessions/session-1/research",
            json={"query": "What is LangGraph?", "use_vector_store": False},
        )
    assert response.status_code == 429
    payload = response.json()["detail"]
    assert payload["code"] == "quota_exceeded"
    assert payload["limit_type"] == "questions_daily"
    _fake_billing.raise_quota = False


def test_submit_run_feedback_creates_boolean_score_and_marks_run():
    mock_session = Session(
        session_id="session-1",
        runs=[
            SessionRun(
                run_id="run-1",
                query="q",
                source_urls=[],
                report="Final report",
                created_at="2026",
                langfuse_trace_id="trace-1",
                langfuse_observation_id="obs-1",
            )
        ],
        conversation=[],
        created_at="2026",
    )

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.submit_user_feedback_score") as mock_score,
        patch("src.api.endpoints.update_session_run", new=AsyncMock(return_value=True)) as mock_update,
    ):
        response = client.post(
            "/sessions/session-1/runs/run-1/feedback",
            json={"helpful": True, "comment": "Very helpful"},
        )

    assert response.status_code == 200
    mock_score.assert_called_once_with(
        trace_id="trace-1",
        observation_id="obs-1",
        helpful=True,
        comment="Very helpful",
    )
    assert any(
        call.kwargs.get("patch", {}).get("feedback_helpful") is True
        and call.kwargs.get("patch", {}).get("feedback_submitted_at")
        for call in mock_update.await_args_list
    )


def test_submit_run_feedback_backfills_langfuse_linkage_when_missing():
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.create_feedback_anchor_for_run", return_value=("trace-backfill", "obs-backfill")),
        patch("src.api.endpoints.submit_user_feedback_score") as mock_score,
        patch("src.api.endpoints.update_session_run", new=AsyncMock(return_value=True)) as mock_update,
    ):
        response = client.post(
            "/sessions/session-1/runs/run-1/feedback",
            json={"helpful": True},
        )

    assert response.status_code == 200
    mock_score.assert_called_once_with(
        trace_id="trace-backfill",
        observation_id="obs-backfill",
        helpful=True,
        comment=None,
    )
    assert any(
        call.kwargs.get("patch", {}).get("langfuse_trace_id") == "trace-backfill"
        for call in mock_update.await_args_list
    )


def test_submit_run_feedback_rejects_duplicate_submission():
    mock_session = Session(
        session_id="session-1",
        runs=[
            SessionRun(
                run_id="run-1",
                query="q",
                source_urls=[],
                report="",
                created_at="2026",
                langfuse_trace_id="trace-1",
                langfuse_observation_id="obs-1",
                feedback_submitted_at="2026-05-05T10:00:00+00:00",
                feedback_helpful=True,
            )
        ],
        conversation=[],
        created_at="2026",
    )

    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)):
        response = client.post(
            "/sessions/session-1/runs/run-1/feedback",
            json={"helpful": False, "comment": "Needs work"},
        )

    assert response.status_code == 409


def test_submit_run_feedback_rejects_non_completed_run():
    mock_session = Session(
        session_id="session-1",
        runs=[
            SessionRun(
                run_id="run-1",
                query="q",
                source_urls=[],
                report="",
                status="running",
                created_at="2026",
                langfuse_trace_id="trace-1",
                langfuse_observation_id="obs-1",
            )
        ],
        conversation=[],
        created_at="2026",
    )

    with patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)):
        response = client.post(
            "/sessions/session-1/runs/run-1/feedback",
            json={"helpful": True},
        )

    assert response.status_code == 409


def test_submit_run_feedback_returns_404_when_backfill_linkage_persist_fails():
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.create_feedback_anchor_for_run", return_value=("trace-backfill", "obs-backfill")),
        patch("src.api.endpoints.update_session_run", new=AsyncMock(return_value=False)),
        patch("src.api.endpoints.submit_user_feedback_score") as mock_score,
    ):
        response = client.post(
            "/sessions/session-1/runs/run-1/feedback",
            json={"helpful": True},
        )

    assert response.status_code == 404
    mock_score.assert_not_called()


def test_execute_research_run_marks_completed_and_records():
    from src.api.endpoints import _execute_research_run

    class FakeGraph:
        async def astream_events(self, _initial_state, version="v2"):
            yield {"event": "on_chain_start", "metadata": {"langgraph_node": "search_and_memory"}, "data": {}}
            yield {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "report"}, "data": {"chunk": MagicMock(content="Final ")}}
            yield {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "report"}, "data": {"chunk": MagicMock(content="report")}}
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "report"},
                "data": {
                    "output": {
                        "report": "Final report",
                        "retrieved_contents": [{"url": "https://example.com", "title": "Example", "content": "Chunk"}],
                        "summaries": [],
                    }
                },
            }

    @contextmanager
    def _mock_trace_ctx(**_kwargs):
        yield MagicMock(workflow_id="wf-1")

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    mock_update = AsyncMock(return_value=True)
    mock_graph_store = MagicMock()
    mock_graph_store.ingest_document.return_value = 1

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=session)),
        patch("src.api.endpoints.build_graph", return_value=FakeGraph()),
        patch("src.api.endpoints.update_session_run", new=mock_update),
        patch("src.api.endpoints.Neo4jGraphStore", return_value=mock_graph_store),
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

    assert any(
        call.kwargs.get("patch", {}).get("status") == "completed"
        and call.kwargs.get("patch", {}).get("report") == "Final report"
        for call in mock_update.await_args_list
    )
    assert mock_graph_store.ingest_document.call_count >= 1


def test_execute_research_run_marks_failed_on_error():
    from src.api.endpoints import _execute_research_run

    class FailingGraph:
        async def astream_events(self, _initial_state, version="v2"):
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

    assert any(
        call.kwargs.get("patch", {}).get("status") == "failed"
        and call.kwargs.get("patch", {}).get("error_details") == "graph failure"
        for call in mock_update.await_args_list
    )


def test_execute_research_run_ignores_live_state_persist_errors():
    from src.api.endpoints import _execute_research_run

    class FakeGraph:
        async def astream_events(self, _initial_state, version="v2"):
            yield {"event": "on_chain_start", "metadata": {"langgraph_node": "search_and_memory"}, "data": {}}
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "report"},
                "data": {"output": {"report": "Final report", "retrieved_contents": [], "summaries": []}},
            }

    @contextmanager
    def _mock_trace_ctx(**_kwargs):
        yield MagicMock(workflow_id="wf-1")

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")
    calls = {"n": 0}

    async def flaky_update(**kwargs):
        patch = kwargs.get("patch", {})
        if patch.get("status") in {"completed", "failed"}:
            return True
        calls["n"] += 1
        raise RuntimeError("temporary db outage")

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=session)),
        patch("src.api.endpoints.build_graph", return_value=FakeGraph()),
        patch("src.api.endpoints.update_session_run", new=AsyncMock(side_effect=flaky_update)),
        patch("src.api.endpoints._record_session_run", new=AsyncMock(return_value=None)),
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

    assert calls["n"] > 0


def test_execute_research_runs_can_overlap_in_time():
    from src.api.endpoints import _execute_research_run

    starts: list[tuple[str, float]] = []
    ends: list[tuple[str, float]] = []

    class SlowGraph:
        async def astream_events(self, initial_state, version="v2"):
            run_id = initial_state["run_id"]
            starts.append((run_id, time.perf_counter()))
            await asyncio.sleep(0.05)
            ends.append((run_id, time.perf_counter()))
            yield {"event": "on_chain_start", "metadata": {"langgraph_node": "search_and_memory"}, "data": {}}
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "report"},
                "data": {
                    "output": {
                        "report": f"Final report {run_id}",
                        "retrieved_contents": [],
                        "summaries": [],
                    }
                },
            }

    @contextmanager
    def _mock_trace_ctx(**_kwargs):
        yield MagicMock(workflow_id="wf-1")

    session = Session(session_id="session-1", runs=[], conversation=[], created_at="2026")

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=session)),
        patch("src.api.endpoints.build_graph", return_value=SlowGraph()),
        patch("src.api.endpoints.update_session_run", new=AsyncMock(return_value=True)),
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
        patch("src.api.endpoints.validate_web_search_provider_health"),
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
        patch("src.api.endpoints.validate_web_search_provider_health"),
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
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", side_effect=get_llm_side_effect),
    ):
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(
            context="ctx",
            chunks=[{"text": "Evidence", "source_url": "https://a.com", "source_title": "A"}],
            entities=["a"],
        )
        mock_graph_cls.return_value = mock_graph
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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": False})),
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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": False})),
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


def test_rag_chat_stream_includes_suggestions_event_before_done():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."

    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []

    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": False})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._generate_suggestions",
            new=AsyncMock(return_value=["Follow-up one?", "Follow-up two?", "Follow-up three?"]),
        ),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    event_types = [event["type"] for event in events]
    assert "suggestions" in event_types
    suggestions_idx = event_types.index("suggestions")
    done_idx = event_types.index("done")
    assert suggestions_idx < done_idx
    suggestions_event = next(event for event in events if event["type"] == "suggestions")
    assert suggestions_event["suggestions"] == ["Follow-up one?", "Follow-up two?", "Follow-up three?"]


def test_rag_chat_persists_suggestions_on_assistant_message():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."

    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []

    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    append_chat = AsyncMock(return_value=None)

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": False})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=append_chat),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._generate_suggestions",
            new=AsyncMock(return_value=["Next question?", "Another question?"]),
        ),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    assistant_message = append_chat.await_args_list[1].args[0]
    assert assistant_message.role == "assistant"
    assert assistant_message.suggestions == ["Next question?", "Another question?"]


def test_rag_chat_web_toggle_true_calls_web_tool_when_model_decides_needed():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context."
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://web.example", "title": "Web", "content": "Fact"}]

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.update_chat_session_web_search_enabled", new=AsyncMock(return_value=True)),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": True})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._should_use_web_search", new=AsyncMock(return_value=(True, "fresh query"))),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Need latest info", "web_search_enabled": True},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is True
    mock_tool.search.assert_called_once()


def test_rag_chat_web_toggle_true_skips_web_tool_when_model_decides_not_needed():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context is enough."
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.update_chat_session_web_search_enabled", new=AsyncMock(return_value=True)),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": True})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._should_use_web_search", new=AsyncMock(return_value=(False, ""))),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Summarize my docs", "web_search_enabled": True},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is False
    mock_tool.search.assert_not_called()


def test_rag_chat_explicit_url_fetch_forces_web_tool_when_enabled():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context."
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://example.com", "title": "Page", "content": "Body"}]

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.update_chat_session_web_search_enabled", new=AsyncMock(return_value=True)),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": True})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={
                "message": "Please fetch this URL: https://aws.amazon.com/certification/",
                "web_search_enabled": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is True
    assert payload["web_provider"] == "direct_fetch"
    mock_tool.search.assert_not_called()


def test_rag_chat_uses_prior_url_reference_for_direct_fetch():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context."
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    prior_msg = MagicMock()
    prior_msg.role = "user"
    prior_msg.content = "Use this URL next: https://aws.amazon.com/certification/"

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.update_chat_session_web_search_enabled", new=AsyncMock(return_value=True)),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": True})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[prior_msg])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched page content")),
        patch("src.api.endpoints.get_web_search_tool") as mock_get_tool,
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={
                "message": "Fetch the content from the url i provided",
                "web_search_enabled": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is True
    assert payload["web_provider"] == "direct_fetch"
    mock_get_tool.assert_not_called()


def test_rag_chat_repairs_url_access_refusal_when_web_content_exists():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context."
    mock_context.chunks = []
    bad = MagicMock()
    bad.content = (
        "I currently don't have the capability to directly fetch or retrieve "
        "content from external URLs."
    )
    repaired = MagicMock()
    repaired.content = "I fetched the URL content. Here is the summary."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[bad, repaired])

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.update_chat_session_web_search_enabled", new=AsyncMock(return_value=True)),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": True})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched text")),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={
                "message": "fetch this url https://aws.amazon.com/certification/",
                "web_search_enabled": True,
            },
        )

    assert response.status_code == 200
    reply = response.json()["reply"]["content"].lower()
    assert "don't have the capability" not in reply
    assert "cannot access" not in reply


def test_rag_chat_stream_repairs_url_access_refusal_when_web_content_exists():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context."
    mock_context.chunks = []
    bad = MagicMock()
    bad.content = (
        "I currently don't have the capability to directly fetch or retrieve "
        "content from external URLs."
    )
    repaired = MagicMock()
    repaired.content = "I fetched the URL content. Here is the summary."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[bad, repaired])

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.update_chat_session_web_search_enabled", new=AsyncMock(return_value=True)),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"web_search_enabled": True})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched text")),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={
                "message": "fetch this url https://aws.amazon.com/certification/",
                "web_search_enabled": True,
            },
        )

    assert response.status_code == 200
    lines = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [evt["text"] for evt in lines if evt.get("type") == "chunk"]
    final_text = "".join(chunks).lower()
    assert "don't have the capability" not in final_text
    assert "cannot access" not in final_text


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


def test_rag_agent_delete():
    with patch(
        "src.api.endpoints.delete_rag_agent_record",
        new=AsyncMock(return_value=True),
    ):
        response = client.delete("/api/rag/agents/agent-1")

    assert response.status_code == 200
    assert response.json() == {"agent_id": "agent-1", "deleted": True}


def test_rag_agent_delete_not_found():
    with patch(
        "src.api.endpoints.delete_rag_agent_record",
        new=AsyncMock(return_value=False),
    ):
        response = client.delete("/api/rag/agents/agent-404")

    assert response.status_code == 404


def _async_iter(items):
    return _async_iter_impl(items)
