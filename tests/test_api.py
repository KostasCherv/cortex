"""Tests for FastAPI endpoints (src/api/endpoints.py)"""

import asyncio
from io import BytesIO
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import pytest
from starlette.datastructures import Headers, UploadFile as StarletteUploadFile

from src.api.endpoints import app
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing.application.service import UsageIncrement
from src.billing.domain.errors import QuotaExceededError
from src.billing.domain.models import DailyUsage, Plan, QuotaLimits, UsageSummary, UserSubscription
from src.rag import AgentDefinitionDraft, RagValidationError
from src.sessions import ConversationTurn, Session, SessionRun
import src.api.endpoints as endpoints

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


@pytest.fixture(autouse=True)
def override_auth_dependency():
    app.dependency_overrides[get_authenticated_user] = _auth_override
    yield
    app.dependency_overrides.pop(get_authenticated_user, None)


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


def test_get_memory_returns_single_memory_document():
    with patch(
        "src.api.endpoints.get_user_memory",
        new=AsyncMock(
            return_value={
                "content": "Prefers concise answers.\nWorks in fintech.",
                "updated_at": "2026-06-04T10:00:00+00:00",
                "last_refreshed_at": "2026-06-04T10:05:00+00:00",
            }
        ),
    ):
        response = client.get("/api/memory")
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Prefers concise answers.\nWorks in fintech."

def test_put_memory_updates_content():
    with patch(
        "src.api.endpoints.update_user_memory",
        new=AsyncMock(
            return_value={
                "content": "Works in fintech.",
                "updated_at": "2026-06-04T10:00:00+00:00",
                "last_refreshed_at": None,
            }
        ),
    ) as update_memory:
        response = client.put("/api/memory", json={"content": "Works in fintech."})
    assert response.status_code == 200
    assert response.json()["content"] == "Works in fintech."
    update_memory.assert_awaited_once_with("test-user", "Works in fintech.")


def test_put_memory_rejects_blank_content():
    response = client.put("/api/memory", json={"content": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Memory content cannot be empty."


def test_delete_memory_clears_saved_memory():
    with patch(
        "src.api.endpoints.delete_user_memory",
        new=AsyncMock(return_value={"deleted": True}),
    ) as delete_memory:
        response = client.delete("/api/memory")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    delete_memory.assert_awaited_once_with("test-user")


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
            json={"query": "What is LangGraph?"},
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
            json={"query": "What is LangGraph?"},
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

        async def _run_and_drain() -> None:
            # ponytail: _execute_research_run fires Neo4j persistence via a
            # fire-and-forget asyncio.create_task (intentionally decoupled
            # from run completion). asyncio.run() only awaits the main
            # coroutine, so the background task's completion is a race —
            # drain pending tasks here so the assertion below is deterministic.
            await _execute_research_run(
                session_id="session-1",
                run_id="run-1",
                user_id="user-1",
                query="What is LangGraph?",
            )
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending)

        asyncio.run(_run_and_drain())

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
                ),
                _execute_research_run(
                    session_id="session-1",
                    run_id="run-b",
                    user_id="user-1",
                    query="What is LangGraph?",
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
        patch("src.api.endpoints.settings.supabase_secret_key", ""),
        patch("src.api.endpoints.ensure_store_initialized") as mock_init,
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()) as mock_storage_ready,
        patch("src.api.endpoints.ensure_arxiv_mcp_available", new=AsyncMock()),
    ):
        asyncio.run(app.router.on_startup[0]())
        mock_init.assert_not_called()
        mock_storage_ready.assert_not_awaited()


def test_startup_validation_configures_application_logging():
    with (
        patch("src.api.endpoints._configure_application_logging") as mock_configure_logging,
        patch("src.api.endpoints.settings.supabase_url", ""),
        patch("src.api.endpoints.settings.supabase_secret_key", ""),
        patch("src.api.endpoints.ensure_store_initialized"),
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()),
        patch("src.api.endpoints.ensure_arxiv_mcp_available", new=AsyncMock()),
    ):
        asyncio.run(app.router.on_startup[0]())

    mock_configure_logging.assert_called_once()


def test_startup_validation_checks_rag_storage_when_supabase_configured():
    with (
        patch("src.api.endpoints.settings.supabase_url", "https://example.supabase.co"),
        patch("src.api.endpoints.settings.supabase_secret_key", "service-role"),
        patch("src.api.endpoints.ensure_store_initialized") as mock_init,
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()) as mock_storage_ready,
        patch("src.api.endpoints.ensure_arxiv_mcp_available", new=AsyncMock()),
    ):
        asyncio.run(app.router.on_startup[0]())
        mock_init.assert_called_once()
        mock_storage_ready.assert_awaited_once()


def test_startup_validation_fails_when_arxiv_mcp_is_unavailable():
    with (
        patch("src.api.endpoints.settings.supabase_url", ""),
        patch("src.api.endpoints.settings.supabase_secret_key", ""),
        patch("src.api.endpoints.ensure_store_initialized"),
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()),
        patch(
            "src.api.endpoints.ensure_arxiv_mcp_available",
            new=AsyncMock(side_effect=RuntimeError("arxiv-mcp-server missing")),
        ),
    ):
        with pytest.raises(RuntimeError, match="arxiv-mcp-server missing"):
            asyncio.run(app.router.on_startup[0]())



def test_configure_application_logging_sets_src_logger_level():
    root_logger = endpoints.logging.getLogger()
    src_logger = endpoints.logging.getLogger("src")
    old_root_level = root_logger.level
    old_src_level = src_logger.level
    old_src_propagate = src_logger.propagate

    try:
        root_logger.setLevel(endpoints.logging.WARNING)
        src_logger.setLevel(endpoints.logging.NOTSET)
        src_logger.propagate = False

        with patch("src.api.endpoints.settings.app_log_level", "INFO"):
            endpoints._configure_application_logging()

        assert root_logger.level == endpoints.logging.INFO
        assert src_logger.level == endpoints.logging.INFO
        assert src_logger.propagate is True
    finally:
        root_logger.setLevel(old_root_level)
        src_logger.setLevel(old_src_level)
        src_logger.propagate = old_src_propagate


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


def test_generate_suggestions_handles_non_dict_content_blocks():
    """_generate_suggestions tolerates structured content blocks with .text attributes."""
    from src.api.endpoints import _generate_suggestions

    mock_result = MagicMock()
    mock_result.content = [
        SimpleNamespace(text="1. First follow-up?\n"),
        SimpleNamespace(text="2. Second follow-up?\n"),
        SimpleNamespace(text="3. Third follow-up?"),
    ]

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_result)

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        suggestions = asyncio.run(
            _generate_suggestions("What is LangGraph?", "Some answer", "context")
        )

    assert suggestions == [
        "First follow-up?",
        "Second follow-up?",
        "Third follow-up?",
    ]


def test_followup_stream_includes_suggestions_event():
    """The followup SSE stream emits a 'suggestions' event after citations."""
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Here is the answer.", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=["Q1?", "Q2?", "Q3?"])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="ctx", chunks=[], entities=[])
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

    suggestions_idx = event_types.index("suggestions")
    done_idx = event_types.index("done")
    assert suggestions_idx < done_idx


def test_followup_stream_citations_match_reranked_chunks():
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    raw_chunks = [
        {"chunk_id": "raw-1", "text": "Less relevant", "source_url": "https://a.com", "source_title": "A"},
        {"chunk_id": "raw-2", "text": "More relevant", "source_url": "https://b.com", "source_title": "B"},
    ]
    reranked_chunks = [raw_chunks[1]]

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=reranked_chunks),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="ctx", chunks=raw_chunks, entities=["a"])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "What did you find?", "run_id": "run-1"},
        )

    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {"source_title": "B", "source_url": "https://b.com", "chunk_id": "raw-2", "text": "More relevant"}
    ]


def test_followup_prompt_includes_originating_report_context_fields():
    mock_session = Session(
        session_id="session-1",
        runs=[
            SessionRun(
                run_id="run-1",
                query="How is MCP used in agent workflows?",
                source_urls=["https://mcp.example/docs", "https://agents.example/guide"],
                report="MCP helps standardize tool interfaces across agents.",
                created_at="2026",
            )
        ],
        conversation=[],
        created_at="2026",
    )
    captured_messages: list = []

    async def capture_loop(messages, metadata, on_event=None, **kwargs):
        captured_messages.extend(messages)
        return "Answer", False

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", side_effect=capture_loop),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "Can you expand on that?", "run_id": "run-1"},
        )

    assert response.status_code == 200
    system_content = "\n".join(
        m.content for m in captured_messages if getattr(m, "type", "") == "system"
    )
    assert "MCP helps standardize tool interfaces across agents." in system_content
    assert "How is MCP used in agent workflows?" in system_content
    assert "https://mcp.example/docs" in system_content
    assert "https://agents.example/guide" in system_content


def test_followup_report_context_fallback_is_safe_for_missing_run():
    mock_session = Session(
        session_id="session-1",
        runs=[],
        conversation=[],
        created_at="2026",
    )
    captured_messages: list = []

    async def capture_loop(messages, metadata, on_event=None, **kwargs):
        captured_messages.extend(messages)
        return "Answer", False

    with (
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", side_effect=capture_loop),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        chunks = asyncio.run(
            _collect_stream(
                endpoints._stream_followup(
                    session=mock_session,
                    user_id="user-1",
                    question="Can you elaborate?",
                    run_id="missing-run",
                )
            )
        )

    system_content = "\n".join(
        m.content for m in captured_messages if getattr(m, "type", "") == "system"
    )
    assert "No stored report context found for run 'missing-run'" in system_content
    events = [json.loads(line[6:]) for line in chunks if line.startswith("data: ")]
    assert any(event.get("type") == "done" for event in events)


def test_followup_stream_returns_answer_from_agent_loop():
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Online follow-up answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "What is Archon?", "run_id": "run-1"},
        )

    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    chunk_event = next((e for e in events if e.get("type") == "chunk"), None)
    assert chunk_event is not None
    assert chunk_event["text"] == "Online follow-up answer"


def test_followup_stream_calls_agent_loop_with_normalized_message():
    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[
            ConversationTurn(
                role="user",
                content="is archon a good tool to add in my stack and why?",
                run_id="run-1",
            )
        ],
        created_at="2026",
    )
    mock_loop = AsyncMock(return_value=("Follow-up answer", False))

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", mock_loop),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "search online for infos", "run_id": "run-1"},
        )

    assert response.status_code == 200
    mock_loop.assert_awaited_once()


async def _async_iter_impl(items):
    for item in items:
        yield item


async def _collect_stream(stream):
    events: list[str] = []
    async for item in stream:
        events.append(item)
    return events


def _fake_prepared_chat(*, session_id: str = "chat-1", resource_ids: list[str] | None = None):
    return SimpleNamespace(
        agent=MagicMock(system_instructions="Keep it concise."),
        resource_ids=resource_ids or ["res-1"],
        rag_context=MagicMock(context="Relevant context.", chunks=[]),
        chat_session_id=session_id,
        messages=[],
        bind_tools=True,
        tool_skip_reason=None,
        composio_apps=[],
        allow_web_search=True,
        reference_tools={},
    )



def test_rag_chat_calls_agent_loop():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."
    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []
    mock_loop = AsyncMock(return_value=("Answer", False))
    trace_ctx = MagicMock(workflow_id="wf-1")
    end_workflow = MagicMock()

    @contextmanager
    def mock_trace_ctx(**_kwargs):
        yield trace_ctx

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", mock_loop),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
        patch("src.api.endpoints.start_workflow_run", side_effect=mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run", end_workflow),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    mock_loop.assert_awaited_once()
    end_workflow.assert_called_once()
    assert end_workflow.call_args.args[0] is trace_ctx
    assert end_workflow.call_args.kwargs["status"] == "success"
    assert end_workflow.call_args.kwargs["outputs"]["answer"] == "Answer"


def test_rag_chat_stream_calls_agent_loop():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."
    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []
    mock_loop = AsyncMock(return_value=("Answer", False))
    trace_ctx = MagicMock(workflow_id="wf-1")
    end_workflow = MagicMock()

    async def loop_with_trace_snapshot(*args, **kwargs):
        return await mock_loop(*args, **kwargs)

    @contextmanager
    def mock_trace_ctx(**_kwargs):
        yield trace_ctx

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(side_effect=loop_with_trace_snapshot)),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
        patch("src.api.endpoints.start_workflow_run", side_effect=mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run", end_workflow),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    mock_loop.assert_awaited_once()
    end_workflow.assert_called_once()
    assert end_workflow.call_args.args[0] is trace_ctx
    assert end_workflow.call_args.kwargs["status"] == "success"
    assert end_workflow.call_args.kwargs["outputs"]["answer"] == "Answer"


def test_rag_agent_chat_with_upload_rejects_invalid_file_type():
    with (
        patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()),
        patch(
            "src.api.rag_chat_helpers.ensure_agent_chat_session_id",
            new=AsyncMock(return_value="chat-1"),
        ),
        patch(
            "src.api.endpoints.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.prepare_agent_rag_chat",
            new=AsyncMock(return_value=_fake_prepared_chat()),
        ),
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(
                side_effect=RagValidationError("unsupported_type", "Unsupported file type.")
            ),
        ),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            data={"message": "Use this file"},
            files={"files": ("payload.exe", b"binary", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_rag_agent_chat_rejects_malformed_json_with_structured_validation_error():
    response = client.post(
        "/api/rag/agents/agent-1/chat",
        content='{"message": ',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["type"] == "json_invalid"
    assert detail[0]["loc"] == ["body", 12]


def test_rag_agent_chat_rejects_malformed_multipart_tools_with_structured_validation_error():
    response = client.post(
        "/api/rag/agents/agent-1/chat",
        data={"message": "Hello", "tools": '{"web": '},
        files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["type"] == "json_invalid"
    assert detail[0]["loc"] == ["body", "tools"]


def test_rag_agent_chat_rejects_schema_invalid_multipart_tools_with_structured_validation_error():
    response = client.post(
        "/api/rag/agents/agent-1/chat",
        data={"message": "Hello", "tools": '{"unknown_field": true}'},
        files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["type"] == "value_error"
    assert detail[0]["loc"] == ["body", "tools"]


def test_rag_agent_chat_stream_uploads_files_before_running_loop():
    call_order: list[str] = []

    async def record_ingest(**_kwargs):
        call_order.append("ingest")
        return []

    async def record_loop(**_kwargs):
        call_order.append("loop")
        return "Answer", False

    with (
        patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()),
        patch(
            "src.api.rag_chat_helpers.ensure_agent_chat_session_id",
            new=AsyncMock(return_value="chat-1"),
        ),
        patch(
            "src.api.endpoints.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.prepare_agent_rag_chat",
            new=AsyncMock(
                side_effect=[
                    _fake_prepared_chat(resource_ids=["agent-res-1"]),
                    _fake_prepared_chat(resource_ids=["agent-res-1", "session-res-1"]),
                ]
            ),
        ) as mock_prepare,
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(side_effect=record_ingest),
        ) as mock_ingest,
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(side_effect=record_loop)) as mock_loop,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            data={"message": "Use this PDF"},
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    mock_ingest.assert_awaited_once()
    assert mock_prepare.await_count == 1
    mock_loop.assert_awaited_once()
    assert call_order == ["ingest", "loop"]


def test_rag_agent_chat_uploads_files_before_running_loop():
    call_order: list[str] = []

    async def record_ingest(**_kwargs):
        call_order.append("ingest")
        return []

    async def record_loop(**_kwargs):
        call_order.append("loop")
        return "Answer", False

    with (
        patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()),
        patch(
            "src.api.rag_chat_helpers.ensure_agent_chat_session_id",
            new=AsyncMock(return_value="chat-1"),
        ),
        patch(
            "src.api.endpoints.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.prepare_agent_rag_chat",
            new=AsyncMock(
                side_effect=[
                    _fake_prepared_chat(resource_ids=["agent-res-1"]),
                    _fake_prepared_chat(resource_ids=["agent-res-1", "session-res-1"]),
                ]
            ),
        ) as mock_prepare,
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(side_effect=record_ingest),
        ) as mock_ingest,
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.enqueue_memory_refresh", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(side_effect=record_loop)) as mock_loop,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            data={"message": "Use this PDF"},
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    mock_ingest.assert_awaited_once()
    assert mock_prepare.await_count == 1
    mock_loop.assert_awaited_once()
    assert call_order == ["ingest", "loop"]


def test_rag_agent_chat_upload_validation_ends_workflow_before_400():
    trace_ctx = MagicMock(workflow_id="wf-1")
    end_workflow = MagicMock()

    @contextmanager
    def mock_trace_ctx(**_kwargs):
        yield trace_ctx

    with (
        patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()),
        patch(
            "src.api.rag_chat_helpers.ensure_agent_chat_session_id",
            new=AsyncMock(return_value="chat-1"),
        ),
        patch(
            "src.api.endpoints.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.prepare_agent_rag_chat",
            new=AsyncMock(return_value=_fake_prepared_chat(resource_ids=["agent-res-1"])),
        ),
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(side_effect=RagValidationError("unsupported_type", "Unsupported file type.")),
        ),
        patch("src.api.endpoints.start_workflow_run", side_effect=mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run", end_workflow),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            data={"message": "Use this PDF"},
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 400
    end_workflow.assert_called_once()
    assert end_workflow.call_args.args[0] is trace_ctx
    assert end_workflow.call_args.kwargs["status"] == "error"


def test_rag_chat_session_attachments_returns_scoped_attachments():
    mock_agent = MagicMock()
    mock_attachment = MagicMock()
    mock_attachment.to_dict.return_value = {
        "attachment_id": "att-1",
        "session_id": "chat-1",
        "resource_id": "res-1",
        "filename": "brief.pdf",
        "state": "ready",
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
                }
            ),
        ),
        patch(
            "src.api.endpoints.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[mock_attachment]),
        ) as mock_list,
    ):
        response = client.get("/api/rag/agents/agent-1/chat/sessions/chat-1/attachments")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "chat-1",
        "agent_id": "agent-1",
        "attachments": [mock_attachment.to_dict.return_value],
    }
    mock_list.assert_awaited_once_with(
        session_id="chat-1",
        owner_id="test-user",
        agent_id="agent-1",
    )


def test_rag_chat_session_attachments_returns_404_for_unknown_session():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.list_rag_chat_session_attachments", new=AsyncMock()) as mock_list,
    ):
        response = client.get("/api/rag/agents/agent-1/chat/sessions/chat-404/attachments")

    assert response.status_code == 404
    mock_list.assert_not_awaited()


def test_create_rag_agent_chat_session_returns_new_session_id():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch(
            "src.api.endpoints.create_or_get_chat_session",
            new=AsyncMock(return_value="chat-new"),
        ) as mock_create,
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/sessions",
            json={"filename": "brief.pdf"},
        )

    assert response.status_code == 200
    assert response.json() == {"session_id": "chat-new", "agent_id": "agent-1"}
    mock_create.assert_awaited_once_with(
        user_id="test-user",
        agent_id="agent-1",
        session_id=None,
        initial_message="Attached: brief.pdf",
    )


def test_upload_rag_agent_chat_session_attachments_ingests_files():
    mock_agent = MagicMock()
    mock_attachment = MagicMock()
    mock_attachment.to_dict.return_value = {
        "attachment_id": "att-1",
        "filename": "brief.pdf",
        "state": "ready",
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
                }
            ),
        ),
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(return_value=[mock_attachment]),
        ) as mock_ingest,
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/sessions/chat-1/attachments",
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["attachments"] == [mock_attachment.to_dict.return_value]
    mock_ingest.assert_awaited_once()


def test_delete_rag_agent_chat_session_attachment_returns_deleted():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(
                return_value={
                    "session_id": "chat-1",
                    "agent_id": "agent-1",
                    "owner_id": "test-user",
                }
            ),
        ),
        patch(
            "src.api.endpoints.delete_rag_chat_session_attachment",
            new=AsyncMock(return_value=True),
        ) as mock_delete,
    ):
        response = client.delete(
            "/api/rag/agents/agent-1/chat/sessions/chat-1/attachments/att-1",
        )

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    mock_delete.assert_awaited_once_with(
        session_id="chat-1",
        attachment_id="att-1",
        owner_id="test-user",
        agent_id="agent-1",
    )


def test_create_rag_workspace_chat_session_returns_new_session_id():
    with patch(
        "src.api.endpoints.create_or_get_workspace_chat_session",
        new=AsyncMock(return_value="chat-new"),
    ) as mock_create:
        response = client.post(
            "/api/rag/chat/sessions",
            json={"filename": "brief.pdf"},
        )

    assert response.status_code == 200
    assert response.json() == {"session_id": "chat-new", "agent_id": None}
    mock_create.assert_awaited_once_with(
        user_id="test-user",
        session_id=None,
        initial_message="Attached: brief.pdf",
    )


def test_upload_rag_workspace_chat_session_attachments_ingests_files():
    mock_attachment = MagicMock()
    mock_attachment.to_dict.return_value = {
        "attachment_id": "att-1",
        "filename": "brief.pdf",
        "state": "ready",
    }
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(
                return_value={
                    "session_id": "chat-1",
                    "agent_id": None,
                    "owner_id": "test-user",
                }
            ),
        ),
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(return_value=[mock_attachment]),
        ) as mock_ingest,
    ):
        response = client.post(
            "/api/rag/chat/sessions/chat-1/attachments",
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["attachments"] == [mock_attachment.to_dict.return_value]
    mock_ingest.assert_awaited_once_with(
        session_id="chat-1",
        agent_id=None,
        user_id="test-user",
        files=mock_ingest.await_args.kwargs["files"],
    )


def test_delete_rag_workspace_chat_session_attachment_returns_deleted():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(
                return_value={
                    "session_id": "chat-1",
                    "agent_id": None,
                    "owner_id": "test-user",
                }
            ),
        ),
        patch(
            "src.api.endpoints.delete_rag_chat_session_attachment",
            new=AsyncMock(return_value=True),
        ) as mock_delete,
    ):
        response = client.delete(
            "/api/rag/chat/sessions/chat-1/attachments/att-1",
        )

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    mock_delete.assert_awaited_once_with(
        session_id="chat-1",
        attachment_id="att-1",
        owner_id="test-user",
        agent_id=None,
    )


def test_rag_agent_chat_stream_skips_ingest_without_files():
    async def record_loop(**_kwargs):
        return "Answer", False

    with (
        patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()),
        patch(
            "src.api.rag_chat_helpers.prepare_agent_rag_chat",
            new=AsyncMock(return_value=_fake_prepared_chat(resource_ids=["agent-res-1", "session-res-1"])),
        ) as mock_prepare,
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(),
        ) as mock_ingest,
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(side_effect=record_loop)) as mock_loop,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "Summarize the attachment", "session_id": "chat-1"},
        )

    assert response.status_code == 200
    mock_ingest.assert_not_awaited()
    mock_prepare.assert_awaited_once()
    mock_loop.assert_awaited_once()


def test_rag_agent_chat_stream_skips_reingest_for_ready_filenames():
    mock_attachment = MagicMock(filename="brief.pdf", state="ready")
    with (
        patch("src.api.endpoints._consume_usage_or_429", new=AsyncMock()),
        patch(
            "src.api.rag_chat_helpers.ensure_agent_chat_session_id",
            new=AsyncMock(return_value="chat-1"),
        ),
        patch(
            "src.api.endpoints.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[mock_attachment]),
        ),
        patch(
            "src.api.rag_chat_helpers.prepare_agent_rag_chat",
            new=AsyncMock(
                side_effect=[
                    _fake_prepared_chat(resource_ids=["agent-res-1"]),
                    _fake_prepared_chat(resource_ids=["agent-res-1", "session-res-1"]),
                ]
            ),
        ),
        patch(
            "src.api.endpoints.ingest_agent_chat_session_uploads",
            new=AsyncMock(),
        ) as mock_ingest,
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            data={"message": "Use this PDF", "session_id": "chat-1"},
            files={"files": ("brief.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    mock_ingest.assert_not_awaited()


def test_rag_chat_session_delete_cleans_up_attachments_before_session_delete():
    mock_agent = MagicMock()
    call_order: list[str] = []

    async def cleanup(**_kwargs):
        call_order.append("cleanup")
        return None

    async def delete_session(*_args, **_kwargs):
        call_order.append("delete")
        return True

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(
                return_value={
                    "session_id": "chat-1",
                    "agent_id": "agent-1",
                    "owner_id": "test-user",
                }
            ),
        ),
        patch(
            "src.api.endpoints.delete_rag_chat_session_attachments_and_artifacts",
            new=AsyncMock(side_effect=cleanup),
        ) as mock_cleanup,
        patch("src.api.endpoints.delete_rag_chat_session", new=AsyncMock(side_effect=delete_session)),
    ):
        response = client.delete("/api/rag/agents/agent-1/chat/sessions/chat-1")

    assert response.status_code == 200
    assert response.json() == {"session_id": "chat-1", "deleted": True}
    mock_cleanup.assert_awaited_once_with(
        session_id="chat-1",
        owner_id="test-user",
        agent_id="agent-1",
    )
    assert call_order == ["cleanup", "delete"]


def test_rag_chat_session_delete_returns_404_for_unknown_session_before_cleanup():
    mock_agent = MagicMock()
    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.delete_rag_chat_session_attachments_and_artifacts", new=AsyncMock()) as mock_cleanup,
        patch("src.api.endpoints.delete_rag_chat_session", new=AsyncMock()) as mock_delete,
    ):
        response = client.delete("/api/rag/agents/agent-1/chat/sessions/chat-404")

    assert response.status_code == 404
    mock_cleanup.assert_not_awaited()
    mock_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_agent_chat_session_uploads_does_not_create_workspace_rag_resources():
    from src.rag import ingest_agent_chat_session_uploads

    file = StarletteUploadFile(
        file=BytesIO(b"%PDF-1.4 session attachment"),
        filename="brief.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    mock_store = MagicMock()
    mock_store.create_rag_chat_session_attachment = AsyncMock(return_value=None)
    mock_store.update_rag_chat_session_attachment = AsyncMock(return_value=None)
    mock_store.delete_rag_chat_session_attachments_by_ids = AsyncMock(return_value=[])
    mock_storage = MagicMock()
    mock_storage.upload_bytes = AsyncMock(return_value="supabase://bucket/session/brief.pdf")
    mock_storage.create_signed_download_url = AsyncMock(return_value="https://signed.example/brief.pdf")

    with (
        patch("src.rag.create_resource_and_ingest", new=AsyncMock(side_effect=AssertionError("workspace resource path should not be used"))),
        patch("src.rag.get_session_store", return_value=mock_store),
        patch("src.rag.get_storage_adapter", return_value=mock_storage),
        patch("src.rag.ingest_resource_from_bytes", new=AsyncMock(return_value=1)) as mock_ingest,
    ):
        attachments = await ingest_agent_chat_session_uploads(
            session_id="chat-1",
            agent_id="agent-1",
            user_id="test-user",
            files=[file],
        )

    assert len(attachments) == 1
    assert attachments[0].session_id == "chat-1"
    assert attachments[0].state == "ready"
    mock_storage.upload_bytes.assert_awaited_once()
    mock_ingest.assert_awaited_once()
    mock_store.create_rag_chat_session_attachment.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_agent_chat_session_uploads_cleans_up_earlier_files_when_later_file_fails():
    from src.rag import ingest_agent_chat_session_uploads

    files = [
        StarletteUploadFile(
            file=BytesIO(b"%PDF-1.4 first"),
            filename="first.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        ),
        StarletteUploadFile(
            file=BytesIO(b"%PDF-1.4 second"),
            filename="second.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        ),
    ]
    mock_store = MagicMock()
    mock_store.create_rag_chat_session_attachment = AsyncMock(return_value=None)
    mock_store.update_rag_chat_session_attachment = AsyncMock(return_value=None)
    mock_store.delete_rag_chat_session_attachments_by_ids = AsyncMock(
        return_value=[
            {
                "attachment_id": "att-1",
                "resource_id": "res-1",
                "storage_uri": "supabase://bucket/first.pdf",
            },
        ]
    )
    mock_storage = MagicMock()
    mock_storage.upload_bytes = AsyncMock(
        side_effect=[
            "supabase://bucket/first.pdf",
            "supabase://bucket/second.pdf",
        ]
    )
    mock_storage.delete_object = AsyncMock()

    with (
        patch("src.rag.get_session_store", return_value=mock_store),
        patch("src.rag.get_storage_adapter", return_value=mock_storage),
        patch(
            "src.rag.ingest_resource_from_bytes",
            new=AsyncMock(side_effect=[1, RuntimeError("second failed")]),
        ),
        patch("src.rag.delete_resource_artifacts", new=AsyncMock()) as mock_delete_artifacts,
        patch("src.rag.uuid.uuid4", side_effect=["res-1", "att-1", "res-2", "att-2"]),
    ):
        with pytest.raises(RagValidationError) as exc_info:
            await ingest_agent_chat_session_uploads(
                session_id="chat-1",
                agent_id="agent-1",
                user_id="test-user",
                files=files,
            )

    assert exc_info.value.code == "processing_failed"
    # Only the completed first attachment should be rolled back.
    mock_store.delete_rag_chat_session_attachments_by_ids.assert_awaited_once_with(
        attachment_ids=["att-1"],
        session_id="chat-1",
        owner_id="test-user",
        agent_id="agent-1",
    )
    # First attachment artifacts cleaned up via rollback.
    mock_delete_artifacts.assert_any_await(
        store=mock_store,
        resource_id="res-1",
        owner_id="test-user",
        workspace_id="test-user",
    )
    # Both blobs deleted: first.pdf via rollback, second.pdf directly (failed attachment blob cleanup).
    deleted_uris = [call.kwargs["storage_uri"] for call in mock_storage.delete_object.await_args_list]
    assert "supabase://bucket/first.pdf" in deleted_uris
    assert "supabase://bucket/second.pdf" in deleted_uris
    # Failed attachment row preserved with state=failed (not deleted).
    update_calls = mock_store.update_rag_chat_session_attachment.await_args_list
    failed_updates = [c for c in update_calls if c.kwargs.get("patch", {}).get("state") == "failed"]
    assert len(failed_updates) == 1
    assert failed_updates[0].kwargs["attachment_id"] == "att-2"


def test_workspace_rag_chat_calls_agent_loop():
    mock_context = MagicMock()
    mock_context.context = "Workspace context."
    mock_context.chunks = []
    mock_loop = AsyncMock(return_value=("Answer", False))
    trace_ctx = MagicMock(workflow_id="wf-1")
    end_workflow = MagicMock()

    @contextmanager
    def mock_trace_ctx(**_kwargs):
        yield trace_ctx

    with (
        patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", mock_loop),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
        patch("src.api.endpoints.start_workflow_run", side_effect=mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run", end_workflow),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    mock_loop.assert_awaited_once()
    end_workflow.assert_called_once()
    assert end_workflow.call_args.args[0] is trace_ctx
    assert end_workflow.call_args.kwargs["status"] == "success"
    assert end_workflow.call_args.kwargs["outputs"]["answer"] == "Answer"


def test_workspace_rag_chat_stream_calls_agent_loop():
    mock_context = MagicMock()
    mock_context.context = "Workspace context."
    mock_context.chunks = []
    mock_loop = AsyncMock(return_value=("Answer", False))
    trace_ctx = MagicMock(workflow_id="wf-1")
    end_workflow = MagicMock()

    async def loop_with_trace_snapshot(*args, **kwargs):
        return await mock_loop(*args, **kwargs)

    @contextmanager
    def mock_trace_ctx(**_kwargs):
        yield trace_ctx

    with (
        patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(side_effect=loop_with_trace_snapshot)),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
        patch("src.api.endpoints.start_workflow_run", side_effect=mock_trace_ctx),
        patch("src.api.endpoints.end_workflow_run", end_workflow),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    mock_loop.assert_awaited_once()
    end_workflow.assert_called_once()
    assert end_workflow.call_args.args[0] is trace_ctx
    assert end_workflow.call_args.kwargs["status"] == "success"
    assert end_workflow.call_args.kwargs["outputs"]["answer"] == "Answer"


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

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
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

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
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

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch(
            "src.api.endpoints._generate_suggestions",
            new=AsyncMock(return_value=["Follow-up one?", "Follow-up two?", "Follow-up three?"]),
        ),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
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


def test_workspace_rag_chat_stream_applies_fallback_citation_when_chunks_missing():
    mock_context = MagicMock()
    mock_context.context = "Context from workspace docs."
    mock_context.chunks = []

    with (
        patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {
            "source_title": "workspace resources",
            "source_url": None,
            "chunk_id": "workspace-context-fallback",
            "text": "Context from workspace docs.",
        }
    ]


def test_workspace_rag_chat_stream_includes_tool_citations_for_simple_chat():
    from src.api.endpoints import AgentLoopResult

    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []

    with (
        patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch(
            "src.api.endpoints._run_agent_loop",
            new=AsyncMock(
                return_value=AgentLoopResult(
                    answer="Answer from arXiv.",
                    web_used=False,
                    citations=[
                        {
                            "source_title": "Retrieval Paper",
                            "source_url": "https://arxiv.org/abs/2401.12345",
                            "chunk_id": "read_paper:2401.12345:50000",
                            "text": "Section 4 shows the retrieval method in detail.",
                        }
                    ],
                )
            ),
        ),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "Summarize that paper", "session_id": None, "tools": {"arxiv": True}},
        )

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {
            "source_title": "Retrieval Paper",
            "source_url": "https://arxiv.org/abs/2401.12345",
            "chunk_id": "read_paper:2401.12345:50000",
            "text": "Section 4 shows the retrieval method in detail.",
        }
    ]


def test_workspace_rag_chat_allows_no_ready_resources():
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []

    with (
        patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)) as retrieve,
        patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("General answer", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/chat",
            json={"message": "What is Archon?"},
        )

    assert response.status_code == 200
    retrieve.assert_awaited_once()
    assert retrieve.await_args.kwargs["agent_resource_ids"] == []
    assert retrieve.await_args.kwargs["session_attachment_resource_ids"] == []


def test_rag_chat_persists_suggestions_on_assistant_message():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."

    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []

    append_chat = AsyncMock(return_value=None)

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=append_chat),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Answer", False))),
        patch(
            "src.api.endpoints._generate_suggestions",
            new=AsyncMock(return_value=["Next question?", "Another question?"]),
        ),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    assistant_message = append_chat.await_args_list[1].args[0]
    assert assistant_message.role == "assistant"
    assert assistant_message.suggestions == ["Next question?", "Another question?"]


def test_rag_chat_keeps_attached_document_queries_grounded_in_linked_resources():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "Resume context: Led ML projects and built Python data pipelines."
    mock_context.chunks = [
        {
            "source_title": "resume.pdf",
            "source_url": "https://storage.example/resume.pdf",
            "chunk_id": "resume-1",
            "text": "Led ML projects and built Python data pipelines.",
            "rerank_score": 0.92,
        }
    ]

    captured_messages: list = []

    async def capture_loop(*, messages, **kwargs):
        captured_messages.extend(messages)
        return "Your strongest themes are machine learning and Python.", False

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", side_effect=capture_loop),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "What strengths stand out from this document?"},
        )

    assert response.status_code == 200
    all_content = " ".join(m.content for m in captured_messages)
    assert "Resume context: Led ML projects and built Python data pipelines." in all_content


def test_workspace_rag_chat_stream_returns_valid_sse_response():
    mock_context = MagicMock()
    mock_context.context = "Workspace context."
    mock_context.chunks = []

    with (
        patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Workspace answer.", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "what can you tell me?"},
        )

    assert response.status_code == 200
    lines = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    chunk_events = [e for e in lines if e.get("type") == "chunk"]
    assert any("Workspace answer." in e["text"] for e in chunk_events)


def test_rag_chat_stream_agent_loop_returns_chunks():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "SaaS Starter Kit context."
    mock_context.chunks = []

    with (
        patch("src.api.rag_chat_helpers.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat", new=AsyncMock(return_value=mock_context)),
        patch("src.api.rag_chat_helpers.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.rag_chat_helpers.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new=AsyncMock(return_value="")),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints._run_agent_loop", new=AsyncMock(return_value=("Archon is a coding assistant platform.", False))),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr,
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "is this a good tool to add in my stack? https://archon.diy and why?"},
        )

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    chunk_events = [e for e in events if e.get("type") == "chunk"]
    assert any("Archon" in e["text"] for e in chunk_events)


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={"session_id": "chat-1"})),
        patch("src.api.endpoints.delete_rag_chat_session_attachments_and_artifacts", new=AsyncMock()),
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


def test_rag_agent_draft_generation():
    with patch(
        "src.api.endpoints.suggest_rag_agent_definition",
        new=AsyncMock(
            return_value=AgentDefinitionDraft(
                name="Policy Analyst",
                description="Compares policy documents.",
                system_instructions="Use the linked policies before answering.",
            )
        ),
    ):
        response = client.post(
            "/api/rag/agents/draft",
            json={"prompt": "Create a policy comparison agent."},
        )

    assert response.status_code == 200
    assert response.json() == {
        "draft": {
            "name": "Policy Analyst",
            "description": "Compares policy documents.",
            "system_instructions": "Use the linked policies before answering.",
        }
    }


def _async_iter(items):
    return _async_iter_impl(items)


# ── delete last-exchange: workspace ──────────────────────────────────────────


def test_delete_workspace_last_exchange_removes_pair():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(return_value={"session_id": "sess-1"}),
        ),
        patch(
            "src.api.endpoints.delete_last_exchange",
            new=AsyncMock(return_value=(True, None)),
        ),
    ):
        response = client.delete("/api/rag/chat/sessions/sess-1/last-exchange")

    assert response.status_code == 200
    assert response.json() == {"session_id": "sess-1", "deleted": True}


def test_delete_workspace_last_exchange_404_when_session_missing():
    with patch(
        "src.api.endpoints.get_rag_chat_session",
        new=AsyncMock(return_value=None),
    ):
        response = client.delete("/api/rag/chat/sessions/missing/last-exchange")

    assert response.status_code == 404


def test_delete_workspace_last_exchange_404_when_empty():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(return_value={"session_id": "sess-1"}),
        ),
        patch(
            "src.api.endpoints.delete_last_exchange",
            new=AsyncMock(return_value=(False, "empty")),
        ),
    ):
        response = client.delete("/api/rag/chat/sessions/sess-1/last-exchange")

    assert response.status_code == 404


def test_delete_workspace_last_exchange_409_when_pair_invalid():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(return_value={"session_id": "sess-1"}),
        ),
        patch(
            "src.api.endpoints.delete_last_exchange",
            new=AsyncMock(return_value=(False, "not_user_assistant_pair")),
        ),
    ):
        response = client.delete("/api/rag/chat/sessions/sess-1/last-exchange")

    assert response.status_code == 409


def test_delete_workspace_last_exchange_requires_auth():
    app.dependency_overrides.pop(get_authenticated_user, None)
    try:
        response = client.delete("/api/rag/chat/sessions/sess-1/last-exchange")
    finally:
        app.dependency_overrides[get_authenticated_user] = _auth_override

    assert response.status_code == 401


# ── delete last-exchange: agent ──────────────────────────────────────────────


def test_delete_agent_last_exchange_removes_pair():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(return_value={"session_id": "sess-1"}),
        ),
        patch(
            "src.api.endpoints.delete_last_exchange",
            new=AsyncMock(return_value=(True, None)),
        ),
    ):
        response = client.delete(
            "/api/rag/agents/agent-1/chat/sessions/sess-1/last-exchange"
        )

    assert response.status_code == 200
    assert response.json() == {"session_id": "sess-1", "deleted": True}


def test_delete_agent_last_exchange_404_when_session_missing():
    with patch(
        "src.api.endpoints.get_rag_chat_session",
        new=AsyncMock(return_value=None),
    ):
        response = client.delete(
            "/api/rag/agents/agent-1/chat/sessions/missing/last-exchange"
        )

    assert response.status_code == 404


def test_delete_agent_last_exchange_404_when_empty():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(return_value={"session_id": "sess-1"}),
        ),
        patch(
            "src.api.endpoints.delete_last_exchange",
            new=AsyncMock(return_value=(False, "empty")),
        ),
    ):
        response = client.delete(
            "/api/rag/agents/agent-1/chat/sessions/sess-1/last-exchange"
        )

    assert response.status_code == 404


def test_delete_agent_last_exchange_409_when_pair_invalid():
    with (
        patch(
            "src.api.endpoints.get_rag_chat_session",
            new=AsyncMock(return_value={"session_id": "sess-1"}),
        ),
        patch(
            "src.api.endpoints.delete_last_exchange",
            new=AsyncMock(return_value=(False, "not_user_assistant_pair")),
        ),
    ):
        response = client.delete(
            "/api/rag/agents/agent-1/chat/sessions/sess-1/last-exchange"
        )

    assert response.status_code == 409


def test_delete_agent_last_exchange_requires_auth():
    app.dependency_overrides.pop(get_authenticated_user, None)
    try:
        response = client.delete(
            "/api/rag/agents/agent-1/chat/sessions/sess-1/last-exchange"
        )
    finally:
        app.dependency_overrides[get_authenticated_user] = _auth_override

    assert response.status_code == 401


def test_rag_chat_request_default_tools():
    from src.api.endpoints import RagChatRequest
    req = RagChatRequest(message="hello")
    assert req.tools.web_search is True
    assert req.tools.composio is False


def test_rag_chat_tools_explicit():
    from src.api.endpoints import RagChatRequest
    req = RagChatRequest(message="hello", tools={"web_search": False, "composio": True})
    assert req.tools.web_search is False
    assert req.tools.composio is True


def test_run_agent_loop_returns_result_object():
    """_run_agent_loop returns a structured result with answer/web_used/citations."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from langchain_core.messages import HumanMessage

    mock_response = MagicMock()
    mock_response.content = "test answer"
    mock_response.tool_calls = []

    with patch("src.api.endpoints.get_llm") as mock_get_llm, \
         patch("src.api.endpoints.settings") as mock_settings, \
         patch("src.api.endpoints.build_agent_tools", return_value=[]):
        mock_settings.composio_max_agent_turns = 1
        mock_settings.composio_enabled = False
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = llm

        from src.api.endpoints import _run_agent_loop
        result = asyncio.run(_run_agent_loop(
            messages=[HumanMessage(content="hi")],
            metadata={},
            bind_tools=False,
            allow_web_search=False,
        ))
        assert isinstance(result.answer, str)
        assert result.web_used is False
        assert result.citations == []
