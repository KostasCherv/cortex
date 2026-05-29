"""Tests for FastAPI endpoints (src/api/endpoints.py)"""

import asyncio
import json
import time
from datetime import UTC, datetime
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.endpoints import app
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing.application.service import UsageIncrement
from src.billing.domain.errors import QuotaExceededError
from src.billing.domain.models import DailyUsage, Plan, QuotaLimits, UsageSummary, UserSubscription
from src.planner import (
    PlannerValidationError,
    PlanningBrief,
    PRDMilestone,
    PRDPlan,
    PRDPlanResponse,
    PRDRequirement,
    SavedPRD,
    SavedPRDListResponse,
    SavedPRDSummary,
)

from src.rag import AgentDefinitionDraft, RagValidationError
from src.sessions import ConversationTurn, Session, SessionRun
import src.api.endpoints as endpoints

with (
    patch("src.api.endpoints.validate_web_search_provider_health"),
    patch("src.api.endpoints.validate_asset_price_provider_health"),
    patch(
        "src.api.endpoints.initialize_alpha_vantage_mcp_client",
        new=AsyncMock(return_value=MagicMock(list_available_tools=MagicMock(return_value=[]))),
    ),
    patch("src.api.endpoints.shutdown_alpha_vantage_mcp_client", new=AsyncMock()),
):
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
        asyncio.run(
            _execute_research_run(
                session_id="session-1",
                run_id="run-1",
                user_id="user-1",
                query="What is LangGraph?",
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
        patch("src.api.endpoints.validate_web_search_provider_health"),
        patch("src.api.endpoints.validate_asset_price_provider_health"),
        patch(
            "src.api.endpoints.initialize_alpha_vantage_mcp_client",
            new=AsyncMock(return_value=MagicMock(list_available_tools=MagicMock(return_value=[]))),
        ),
        patch("src.api.endpoints.settings.supabase_url", ""),
        patch("src.api.endpoints.settings.supabase_secret_key", ""),
        patch("src.api.endpoints.ensure_store_initialized") as mock_init,
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()) as mock_storage_ready,
    ):
        asyncio.run(app.router.on_startup[0]())
        mock_init.assert_not_called()
        mock_storage_ready.assert_not_awaited()


def test_startup_validation_configures_application_logging():
    with (
        patch("src.api.endpoints.validate_web_search_provider_health"),
        patch("src.api.endpoints.validate_asset_price_provider_health"),
        patch(
            "src.api.endpoints.initialize_alpha_vantage_mcp_client",
            new=AsyncMock(return_value=MagicMock(list_available_tools=MagicMock(return_value=[]))),
        ),
        patch("src.api.endpoints._configure_application_logging") as mock_configure_logging,
        patch("src.api.endpoints.settings.supabase_url", ""),
        patch("src.api.endpoints.settings.supabase_secret_key", ""),
        patch("src.api.endpoints.ensure_store_initialized"),
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()),
    ):
        asyncio.run(app.router.on_startup[0]())

    mock_configure_logging.assert_called_once()


def test_startup_validation_checks_rag_storage_when_supabase_configured():
    with (
        patch("src.api.endpoints.validate_web_search_provider_health"),
        patch("src.api.endpoints.validate_asset_price_provider_health"),
        patch(
            "src.api.endpoints.initialize_alpha_vantage_mcp_client",
            new=AsyncMock(return_value=MagicMock(list_available_tools=MagicMock(return_value=[]))),
        ),
        patch("src.api.endpoints.settings.supabase_url", "https://example.supabase.co"),
        patch("src.api.endpoints.settings.supabase_secret_key", "service-role"),
        patch("src.api.endpoints.ensure_store_initialized") as mock_init,
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()) as mock_storage_ready,
    ):
        asyncio.run(app.router.on_startup[0]())
        mock_init.assert_called_once()
        mock_storage_ready.assert_awaited_once()


def test_startup_validation_bootstraps_alpha_vantage_mcp_catalog_when_enabled():
    mock_client = MagicMock()
    mock_client.list_available_tools.return_value = ["GLOBAL_QUOTE", "SYMBOL_SEARCH"]

    with (
        patch("src.api.endpoints.validate_web_search_provider_health"),
        patch("src.api.endpoints.validate_asset_price_provider_health"),
        patch("src.api.endpoints.initialize_alpha_vantage_mcp_client", new=AsyncMock(return_value=mock_client)) as mock_init_client,
        patch("src.api.endpoints.settings.asset_price_provider", "alphavantage_mcp"),
        patch("src.api.endpoints.settings.supabase_url", ""),
        patch("src.api.endpoints.settings.supabase_secret_key", ""),
        patch("src.api.endpoints.ensure_store_initialized"),
        patch("src.api.endpoints.ensure_rag_storage_ready", new=AsyncMock()),
    ):
        asyncio.run(app.router.on_startup[0]())

    mock_init_client.assert_awaited_once()


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


def test_followup_stream_citations_match_reranked_chunks():
    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))

    mock_session = Session(
        session_id="session-1",
        runs=[SessionRun(run_id="run-1", query="q", source_urls=[], report="", created_at="2026")],
        conversation=[],
        created_at="2026",
    )

    raw_chunks = [
        {
            "chunk_id": "raw-1",
            "text": "Less relevant",
            "source_url": "https://a.com",
            "source_title": "A",
        },
        {
            "chunk_id": "raw-2",
            "text": "More relevant",
            "source_url": "https://b.com",
            "source_title": "B",
        },
    ]
    reranked_chunks = [raw_chunks[1]]

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=reranked_chunks),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(
            context="ctx",
            chunks=raw_chunks,
            entities=["a"],
        )
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
        {
            "source_title": "B",
            "source_url": "https://b.com",
            "chunk_id": "raw-2",
            "text": "More relevant",
        }
    ]


def test_followup_prompt_includes_originating_report_context_fields():
    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))

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

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "Can you expand on that?", "run_id": "run-1"},
        )

    assert response.status_code == 200
    prompt_sent_to_llm = mock_llm.astream.call_args.args[0]
    assert "MCP helps standardize tool interfaces across agents." in prompt_sent_to_llm
    assert "How is MCP used in agent workflows?" in prompt_sent_to_llm
    assert "https://mcp.example/docs" in prompt_sent_to_llm
    assert "https://agents.example/guide" in prompt_sent_to_llm


def test_followup_report_context_fallback_is_safe_for_missing_run():
    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))

    mock_session = Session(
        session_id="session-1",
        runs=[],
        conversation=[],
        created_at="2026",
    )

    with (
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
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

    prompt_sent_to_llm = mock_llm.astream.call_args.args[0]
    assert "No stored report context found for run 'missing-run'" in prompt_sent_to_llm
    events = [json.loads(line[6:]) for line in chunks if line.startswith("data: ")]
    assert any(event.get("type") == "done" for event in events)


def test_followup_stream_empty_context_searches_web_and_emits_web_citations():
    llm_chunk = MagicMock()
    llm_chunk.content = "Online follow-up answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://web.example", "title": "Web", "content": "Fact"}]

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
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="needs_external_info",
                    query="",
                    url="",
                )
            ),
        ),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "What is Archon?", "run_id": "run-1"},
        )

    assert response.status_code == 200
    mock_tool.search.assert_called_once_with("What is Archon?", endpoints.settings.max_search_results)
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {
            "source_title": "Web",
            "source_url": "https://web.example",
            "chunk_id": "tavily-web-1",
            "text": "Fact",
        }
    ]


def test_followup_stream_uses_prior_topic_when_followup_query_is_generic():
    llm_chunk = MagicMock()
    llm_chunk.content = "Online follow-up answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://archon.diy", "title": "Archon", "content": "Fact"}]

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

    with (
        patch("src.api.endpoints.get_session", new=AsyncMock(return_value=mock_session)),
        patch("src.api.endpoints.Neo4jGraphStore") as mock_graph_cls,
        patch("src.api.endpoints.rerank_chunks", return_value=[]),
        patch("src.api.endpoints.append_turn", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="needs_external_info",
                    query="is archon a good tool to add in my stack and why?",
                    url="",
                )
            ),
        ),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph
        response = client.post(
            "/sessions/session-1/followup",
            json={"question": "search online for infos", "run_id": "run-1"},
        )

    assert response.status_code == 200
    mock_tool.search.assert_called_once_with(
        "is archon a good tool to add in my stack and why?",
        endpoints.settings.max_search_results,
    )


async def _async_iter_impl(items):
    for item in items:
        yield item


async def _collect_stream(stream):
    events: list[str] = []
    async for item in stream:
        events.append(item)
    return events


def test_small_talk_messages_are_decided_by_router_model():
    llm_result = MagicMock()
    llm_result.content = json.dumps(
        {
            "action": "answer_direct",
            "reason": "small_talk",
            "query": "",
            "url": "",
        }
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        decision = asyncio.run(
            endpoints._decide_chat_action(
                message="hi",
                rag_context="",
                rag_chunks=[],
                history_block="",
            )
        )

    assert decision.action == "answer_direct"
    assert decision.reason == "small_talk"
    assert decision.query == ""
    assert decision.url == ""
    mock_llm.ainvoke.assert_awaited_once()


def test_chat_action_router_parses_markdown_fenced_json():
    llm_result = MagicMock()
    llm_result.content = (
        "```json\n"
        '{"action":"answer_direct","reason":"small_talk","query":"","url":""}\n'
        "```"
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        decision = asyncio.run(
            endpoints._decide_chat_action(
                message="hi",
                rag_context="",
                rag_chunks=[],
                history_block="",
            )
        )

    assert decision.action == "answer_direct"
    assert decision.reason == "small_talk"


def test_chat_action_router_repairs_schema_invalid_output_once():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content='{"action":"fetch_url","reason":"need content","query":"","url":""}'),
            MagicMock(content='{"action":"fetch_url","reason":"need content","query":"","url":"https://example.com"}'),
        ]
    )

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        decision = asyncio.run(
            endpoints._decide_chat_action(
                message="check https://example.com",
                rag_context="",
                rag_chunks=[],
                history_block="",
            )
        )

    assert mock_llm.ainvoke.await_count == 2
    repair_prompt = mock_llm.ainvoke.await_args_list[1].args[0]
    assert "Validation failed" in repair_prompt
    assert "url" in repair_prompt
    assert decision.action == "fetch_url"
    assert decision.url == "https://example.com"


def test_chat_action_router_falls_back_after_repair_failure():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content='{"action":"web_search","reason":"need fresh info","query":"","url":""}'),
            MagicMock(content='{"action":"web_search","reason":"need fresh info","query":"","url":""}'),
        ]
    )

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        decision = asyncio.run(
            endpoints._decide_chat_action(
                message="latest news",
                rag_context="",
                rag_chunks=[],
                history_block="",
            )
        )

    assert mock_llm.ainvoke.await_count == 2
    assert decision.action == "answer_direct"
    assert decision.reason == "router_parse_failed"


def test_resolve_web_context_does_not_search_when_router_chooses_direct_answer():
    decision = endpoints._ChatActionDecision(
        action="answer_direct",
        reason="small_talk",
        query="",
        url="",
    )

    with patch("src.api.endpoints.get_web_search_tool") as mock_get_tool:
        resolved = asyncio.run(
            endpoints._resolve_web_context(
                normalized_message="hi",
                rag_context="",
                rag_chunks=[],
                history_block="",
                decision=decision,
            )
        )

    assert resolved.used is False
    assert resolved.reason == "not_needed"
    mock_get_tool.assert_not_called()


def test_build_chat_messages_adds_clarifying_instruction_for_router_action():
    messages = endpoints._build_chat_messages(
        system_instructions="None",
        history=[],
        rag_context="",
        web_results=[],
        normalized_message="Can you help with this?",
        router_action="ask_clarifying",
    )

    system_text = "\n".join(
        msg.content for msg in messages if getattr(msg, "type", "") == "system"
    )
    assert "Ask one concise clarifying question" in system_text


def test_rag_chat_passes_router_decision_to_web_context_resolver():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."
    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    decision = endpoints._ChatActionDecision(
        action="answer_from_rag",
        reason="rag_is_sufficient",
        query="",
        url="",
    )
    resolved_web = endpoints._ResolvedWebContext(False, "tavily", [], "not_needed")

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._decide_chat_action", new=AsyncMock(return_value=decision)) as decide_chat_action,
        patch("src.api.endpoints._resolve_web_context", new=AsyncMock(return_value=resolved_web)) as resolve_web_context,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    resolve_web_context.assert_awaited_once()
    assert resolve_web_context.await_args.kwargs["decision"] is decision


def test_rag_chat_stream_passes_router_decision_to_web_context_resolver():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."
    mock_context = MagicMock()
    mock_context.context = "Relevant context."
    mock_context.chunks = []
    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))
    decision = endpoints._ChatActionDecision(
        action="answer_from_rag",
        reason="rag_is_sufficient",
        query="",
        url="",
    )
    resolved_web = endpoints._ResolvedWebContext(False, "tavily", [], "not_needed")

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._decide_chat_action", new=AsyncMock(return_value=decision)) as decide_chat_action,
        patch("src.api.endpoints._resolve_web_context", new=AsyncMock(return_value=resolved_web)) as resolve_web_context,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    resolve_web_context.assert_awaited_once()
    assert resolve_web_context.await_args.kwargs["decision"] is decision


def test_workspace_rag_chat_passes_router_decision_to_web_context_resolver():
    mock_context = MagicMock()
    mock_context.context = "Workspace context."
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    decision = endpoints._ChatActionDecision(
        action="answer_from_rag",
        reason="rag_is_sufficient",
        query="",
        url="",
    )
    resolved_web = endpoints._ResolvedWebContext(False, "tavily", [], "not_needed")

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._decide_chat_action", new=AsyncMock(return_value=decision)) as decide_chat_action,
        patch("src.api.endpoints._resolve_web_context", new=AsyncMock(return_value=resolved_web)) as resolve_web_context,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    resolve_web_context.assert_awaited_once()
    assert resolve_web_context.await_args.kwargs["decision"] is decision


def test_workspace_rag_chat_stream_passes_router_decision_to_web_context_resolver():
    mock_context = MagicMock()
    mock_context.context = "Workspace context."
    mock_context.chunks = []
    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))
    decision = endpoints._ChatActionDecision(
        action="answer_from_rag",
        reason="rag_is_sufficient",
        query="",
        url="",
    )
    resolved_web = endpoints._ResolvedWebContext(False, "tavily", [], "not_needed")

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._decide_chat_action", new=AsyncMock(return_value=decision)) as decide_chat_action,
        patch("src.api.endpoints._resolve_web_context", new=AsyncMock(return_value=resolved_web)) as resolve_web_context,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    resolve_web_context.assert_awaited_once()
    assert resolve_web_context.await_args.kwargs["decision"] is decision


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.RagChatMessage", side_effect=[mock_user_message, mock_assistant_message]),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ),
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


def test_workspace_rag_chat_stream_applies_fallback_citation_when_chunks_missing():
    mock_context = MagicMock()
    mock_context.context = "Context from workspace docs."
    mock_context.chunks = []

    llm_chunk = MagicMock()
    llm_chunk.content = "Answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "Hello", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
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


def test_workspace_rag_chat_stream_searches_web_when_router_requests_it():
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []

    llm_chunk = MagicMock()
    llm_chunk.content = "Online workspace answer"
    mock_llm = MagicMock()
    mock_llm.astream = MagicMock(return_value=_async_iter([llm_chunk]))
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://web.example", "title": "Web", "content": "Fact"}]

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="needs_external_info",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "What is Archon?", "session_id": None},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    mock_tool.search.assert_called_once_with("What is Archon?", endpoints.settings.max_search_results)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    session_event = next(event for event in events if event["type"] == "session")
    assert session_event["web_used"] is True
    assert session_event["web_provider"] == "tavily"
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {
            "source_title": "Web",
            "source_url": "https://web.example",
            "chunk_id": "tavily-web-1",
            "text": "Fact",
        }
    ]


def test_workspace_rag_chat_allows_no_ready_resources():
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "General answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://web.example", "title": "Web", "content": "Fact"}]

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)) as retrieve,
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="needs_external_info",
                    query="",
                    url="",
                )
            ),
        ),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat",
            json={"message": "What is Archon?"},
        )

    assert response.status_code == 200
    retrieve.assert_awaited_once()
    assert retrieve.await_args.kwargs["resource_ids"] == []


def test_workspace_rag_chat_uses_web_decision_query_when_current_message_is_generic():
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Online answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://archon.diy", "title": "Archon", "content": "Fact"}]
    prior_user = MagicMock()
    prior_user.role = "user"
    prior_user.content = "is archon a good tool to add in my stack and why?"
    prior_user.to_dict.return_value = {"role": "user", "content": prior_user.content}

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=["res-1"])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(side_effect=[[prior_user], []])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="needs_external_info",
                    query="is archon a good tool to add in my stack and why?",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat",
            json={"message": "search online for infos"},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    mock_tool.search.assert_called_once_with(
        "is archon a good tool to add in my stack and why?",
        endpoints.settings.max_search_results,
    )


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=append_chat),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
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
    decide_chat_action.assert_awaited_once()
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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="fresh_info_requested",
                    query="fresh query",
                    url="",
                )
            ),
        ),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Need latest info"},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is True
    mock_tool.search.assert_called_once()


def test_rag_chat_empty_context_does_not_auto_search_without_router_web_action():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Archon is a platform."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://web.example", "title": "Web", "content": "Fact"}]

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_direct",
                    reason="insufficient_but_not_external",
                    query="",
                    url="",
                )
            ),
        ),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "What is Archon?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is False
    assert payload["web_provider"] is None
    mock_tool.search.assert_not_called()


def test_rag_chat_greeting_does_not_search_web():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Hi there!"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.search.return_value = []

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_direct",
                    reason="small_talk",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool) as mock_get_tool,
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "hi"},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is False
    decide_chat_action.assert_awaited_once()
    mock_get_tool.assert_not_called()


def test_rag_chat_semantically_wrong_context_searches_web_and_omits_stale_citations():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "SaaS Starter Kit guide for launching subscription templates."
    mock_context.chunks = [
        {
            "source_title": "SaaS_Starter_Kit.pdf",
            "source_url": "https://storage.example/saas.pdf",
            "chunk_id": "saas-1",
            "text": "SaaS Starter Kit guide for launching subscription templates.",
            "rerank_score": 0.05,
        }
    ]
    llm_result = MagicMock()
    llm_result.content = "Use this Archon YAML configuration."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [
        {
            "url": "https://archon.diy/docs/setup",
            "title": "Archon setup",
            "content": "Archon YAML setup details.",
        }
    ]

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="model_decision",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "give me example of a yaml file for the setup of archon in my project"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is True
    assert payload["web_provider"] == "tavily"
    decide_chat_action.assert_awaited_once()
    mock_tool.search.assert_called_once_with(
        "give me example of a yaml file for the setup of archon in my project",
        endpoints.settings.max_search_results,
    )
    assert payload["reply"]["citations"] == [
        {
            "source_title": "Archon setup",
            "source_url": "https://archon.diy/docs/setup",
            "chunk_id": "tavily-web-1",
            "text": "Archon YAML setup details.",
        }
    ]


def test_rag_chat_uses_web_decision_query_when_current_message_is_generic():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "Online answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://archon.diy", "title": "Archon", "content": "Fact"}]
    prior_user = MagicMock()
    prior_user.role = "user"
    prior_user.content = "is archon a good tool to add in my stack and why?"
    prior_user.to_dict.return_value = {"role": "user", "content": prior_user.content}

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(side_effect=[[prior_user], []])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="web_search",
                    reason="needs_external_info",
                    query="is archon a good tool to add in my stack and why?",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "search online for infos"},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    mock_tool.search.assert_called_once_with(
        "is archon a good tool to add in my stack and why?",
        endpoints.settings.max_search_results,
    )


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ),
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "Summarize my docs"},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is False
    mock_tool.search.assert_not_called()


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
    llm_result = MagicMock()
    llm_result.content = "Your strongest themes are machine learning and Python."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    mock_tool = MagicMock()
    mock_tool.provider_name = "tavily"
    mock_tool.search.return_value = [{"url": "https://web.example", "title": "Web", "content": "Fact"}]

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="answer_from_rag",
                    reason="rag_is_sufficient",
                    query="",
                    url="",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "What strengths stand out from this document?"},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is False
    decide_chat_action.assert_awaited_once()
    mock_tool.search.assert_not_called()
    messages = mock_llm.ainvoke.await_args.args[0]
    all_content = " ".join(m.content for m in messages)
    assert "Resume context: Led ML projects and built Python data pipelines." in all_content


def test_rag_context_is_preserved_when_web_search_augments_local_resources():
    resolved_web = endpoints._ResolvedWebContext(
        used=True,
        provider="tavily",
        results=[{"url": "https://example.com", "title": "Example", "content": "Fresh fact"}],
        reason="model_decision",
    )

    assert (
        endpoints._rag_context_for_answer("Resume context here.", resolved_web)
        == "Resume context here."
    )


def test_resolve_web_context_calls_asset_price_tool_for_asset_price_action():
    decision = endpoints._ChatActionDecision(
        action="asset_price",
        reason="current_asset_quote",
        symbols=["BTC-USD", "AAPL"],
        currency="",
    )
    mock_tool = MagicMock()
    mock_tool.provider_name = "yfinance"
    mock_tool.quote.return_value = [
        {"symbol": "BTC-USD", "price": 68000.0, "currency": "USD"},
        {"symbol": "AAPL", "price": 190.0, "currency": "USD"},
    ]

    with patch("src.api.endpoints.get_asset_price_tool", return_value=mock_tool):
        resolved = asyncio.run(
            endpoints._resolve_web_context(
                normalized_message="price of BTC and AAPL",
                rag_context="",
                rag_chunks=[],
                history_block="",
                decision=decision,
            )
        )

    assert resolved.used is True
    assert resolved.provider == "yfinance"
    assert resolved.reason == "asset_price"
    mock_tool.quote.assert_called_once_with(["BTC-USD", "AAPL"], None)


def test_rag_chat_uses_asset_price_tool_when_router_selects_asset_price():
    mock_agent = MagicMock()
    mock_agent.system_instructions = "Keep it concise."
    mock_context = MagicMock()
    mock_context.context = ""
    mock_context.chunks = []
    llm_result = MagicMock()
    llm_result.content = "BTC-USD is at 68000 USD and AAPL is at 190 USD."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)
    decision = endpoints._ChatActionDecision(
        action="asset_price",
        reason="current_asset_quote",
        symbols=["BTC-USD", "AAPL"],
        currency="",
    )
    mock_tool = MagicMock()
    mock_tool.provider_name = "yfinance"
    mock_tool.quote.return_value = [
        {"symbol": "BTC-USD", "name": "Bitcoin USD", "price": 68000.0, "currency": "USD"},
        {"symbol": "AAPL", "name": "Apple Inc.", "price": 190.0, "currency": "USD"},
    ]

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch("src.api.endpoints._decide_chat_action", new=AsyncMock(return_value=decision)),
        patch("src.api.endpoints.get_asset_price_tool", return_value=mock_tool),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "What's the latest price of BTC-USD and AAPL?", "session_id": None},
        )

    assert response.status_code == 200
    assert response.json()["web_used"] is True
    mock_tool.quote.assert_called_once_with(["BTC-USD", "AAPL"], None)
    messages = mock_llm.ainvoke.await_args.args[0]
    all_content = " ".join(m.content for m in messages)
    assert "Bitcoin USD" in all_content
    assert "68000.0" in all_content


def test_resolve_web_context_calls_selected_finance_tool_for_search_finance_tools_action():
    decision = endpoints._ChatActionDecision(
        action="search_finance_tools",
        reason="needs_finance_tool",
        query="What is the 24-hour price change for BTC?",
    )
    tool_match = {
        "name": "CRYPTO_INTRADAY",
        "description": "Intraday crypto time series",
        "score": 9.5,
        "why": "best_fit_for_intraday_change_request",
    }
    tool_definition = MagicMock()
    tool_definition.name = "CRYPTO_INTRADAY"
    tool_definition.description = "Intraday crypto time series"
    tool_definition.parameters = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Crypto symbol such as BTC"},
            "market": {"type": "string", "description": "Market such as USD"},
        },
        "required": ["symbol", "market"],
    }
    tool_payload = {"Time Series Crypto (1min)": {"2026-05-27 21:00:00": {"4. close": "74908.01"}}}

    with (
        patch(
            "src.api.endpoints.search_alpha_vantage_mcp_tools",
            new=AsyncMock(return_value=[tool_match]),
        ),
        patch(
            "src.api.endpoints._select_finance_tool_candidate",
            new=AsyncMock(return_value="CRYPTO_INTRADAY"),
        ),
        patch(
            "src.api.endpoints.get_alpha_vantage_mcp_tool_definition",
            new=AsyncMock(return_value=tool_definition),
        ),
        patch(
            "src.api.endpoints._plan_finance_tool_call",
            new=AsyncMock(
                return_value={
                    "should_call": True,
                    "reason": "enough_information",
                    "arguments": {"symbol": "BTC", "market": "USD"},
                    "clarifying_question": "",
                }
            ),
        ),
        patch(
            "src.api.endpoints.call_alpha_vantage_mcp_tool",
            new=AsyncMock(return_value=tool_payload),
        ) as call_tool,
    ):
        resolved = asyncio.run(
            endpoints._resolve_web_context(
                normalized_message="What is the 24-hour price change for BTC?",
                rag_context="",
                rag_chunks=[],
                history_block="",
                decision=decision,
            )
        )

    assert resolved.used is True
    assert resolved.provider == "alphavantage_mcp"
    assert resolved.reason == "finance_tool_call"
    assert resolved.results[0]["tool_name"] == "CRYPTO_INTRADAY"
    call_tool.assert_awaited_once_with("CRYPTO_INTRADAY", {"symbol": "BTC", "market": "USD"})


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://aws.amazon.com/certification/",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.get_web_search_tool", return_value=mock_tool),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={
                "message": "Please fetch this URL: https://aws.amazon.com/certification/",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is True
    assert payload["web_provider"] == "direct_fetch"
    decide_chat_action.assert_awaited_once()
    mock_tool.search.assert_not_called()


def test_rag_chat_explicit_url_fetch_forces_direct_fetch_when_session_web_disabled():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "Irrelevant SaaS Starter Kit context."
    mock_context.chunks = [
        {
            "source_title": "SaaS_Starter_Kit.pdf",
            "source_url": "https://storage.example/saas.pdf",
            "chunk_id": "saas-1",
            "text": "SaaS Starter Kit Guide",
        }
    ]
    llm_result = MagicMock()
    llm_result.content = "Archon page answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://archon.diy",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Archon page content")),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "check their page https://archon.diy"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is True
    assert payload["web_provider"] == "direct_fetch"
    decide_chat_action.assert_awaited_once()
    assert payload["reply"]["citations"] == [
        {
            "source_title": "https://archon.diy",
            "source_url": "https://archon.diy",
            "chunk_id": "direct_fetch-web-1",
            "text": "Archon page content",
        }
    ]


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[prior_msg])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="referenced_prior_url",
                    query="",
                    url="https://aws.amazon.com/certification/",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched page content")),
        patch("src.api.endpoints.get_web_search_tool") as mock_get_tool,
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={
                "message": "Fetch the content from the url i provided",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["web_used"] is True
    assert payload["web_provider"] == "direct_fetch"
    decide_chat_action.assert_awaited_once()
    mock_get_tool.assert_not_called()


def test_rag_chat_repairs_prior_url_reference_refusal_when_web_content_exists():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "RAG context."
    mock_context.chunks = []
    bad = MagicMock()
    bad.content = "I can't access that link directly."
    repaired = MagicMock()
    repaired.content = "I fetched the link and here is the summary."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[bad, repaired])
    prior_msg = MagicMock()
    prior_msg.role = "user"
    prior_msg.content = "Use this URL next: https://aws.amazon.com/certification/"

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[prior_msg])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="referenced_prior_url",
                    query="",
                    url="https://aws.amazon.com/certification/",
                )
            ),
        ),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched page content")),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={"message": "what does that link say?"},
        )

    assert response.status_code == 200
    reply = response.json()["reply"]["content"].lower()
    assert "can't access" not in reply
    assert "fetched the link" in reply


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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://aws.amazon.com/certification/",
                )
            ),
        ),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched text")),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat",
            json={
                "message": "fetch this url https://aws.amazon.com/certification/",
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
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://aws.amazon.com/certification/",
                )
            ),
        ),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched text")),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={
                "message": "fetch this url https://aws.amazon.com/certification/",
            },
        )

    assert response.status_code == 200
    lines = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [evt["text"] for evt in lines if evt.get("type") == "chunk"]
    final_text = "".join(chunks).lower()
    assert "don't have the capability" not in final_text
    assert "cannot access" not in final_text


def test_workspace_rag_chat_stream_repairs_url_access_refusal_when_web_content_exists():
    mock_context = MagicMock()
    mock_context.context = "Workspace context."
    mock_context.chunks = []
    bad = MagicMock()
    bad.content = "I currently don't have the capability to directly fetch or retrieve content from external URLs."
    repaired = MagicMock()
    repaired.content = "I fetched the URL content. Here is the summary."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[bad, repaired])

    with (
        patch("src.api.endpoints.list_workspace_ready_resource_ids", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_workspace_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://aws.amazon.com/certification/",
                )
            ),
        ),
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Fetched text")),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/chat/stream",
            json={"message": "fetch this url https://aws.amazon.com/certification/"},
        )

    assert response.status_code == 200
    lines = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [evt["text"] for evt in lines if evt.get("type") == "chunk"]
    final_text = "".join(chunks).lower()
    assert "don't have the capability" not in final_text
    assert "cannot access" not in final_text
    assert "fetched the url content" in final_text


def test_rag_chat_stream_direct_fetch_omits_stale_rag_citations_when_web_disabled():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "Irrelevant SaaS Starter Kit context."
    mock_context.chunks = [
        {
            "source_title": "SaaS_Starter_Kit.pdf",
            "source_url": "https://storage.example/saas.pdf",
            "chunk_id": "saas-1",
            "text": "SaaS Starter Kit Guide",
        }
    ]
    llm_result = MagicMock()
    llm_result.content = "Archon page answer"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://archon.diy",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Archon page content")),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "check their page https://archon.diy"},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    citations_event = next(event for event in events if event["type"] == "citations")
    assert citations_event["citations"] == [
        {
            "source_title": "https://archon.diy",
            "source_url": "https://archon.diy",
            "chunk_id": "direct_fetch-web-1",
            "text": "Archon page content",
        }
    ]


def test_rag_chat_stream_direct_fetches_bare_url_questions_and_omits_stale_rag_prompt():
    mock_agent = MagicMock()
    mock_agent.system_instructions = ""
    mock_context = MagicMock()
    mock_context.context = "Irrelevant SaaS Starter Kit context."
    mock_context.chunks = [
        {
            "source_title": "SaaS_Starter_Kit.pdf",
            "source_url": "https://storage.example/saas.pdf",
            "chunk_id": "saas-1",
            "text": "SaaS Starter Kit Guide",
        }
    ]
    llm_result = MagicMock()
    llm_result.content = "Archon is a coding assistant platform."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=llm_result)

    with (
        patch("src.api.endpoints.get_agent_for_chat", new=AsyncMock(return_value=(mock_agent, ["res-1"]))),
        patch("src.api.endpoints.retrieve_context_for_query", new=AsyncMock(return_value=mock_context)),
        patch("src.api.endpoints.create_or_get_chat_session", new=AsyncMock(return_value="chat-1")),
        patch("src.api.endpoints.get_rag_chat_session", new=AsyncMock(return_value={})),
        patch("src.api.endpoints.list_rag_chat_messages", new=AsyncMock(return_value=[])),
        patch("src.api.endpoints.append_chat_message", new=AsyncMock(return_value=None)),
        patch("src.api.endpoints.get_llm", return_value=mock_llm),
        patch(
            "src.api.endpoints._decide_chat_action",
            new=AsyncMock(
                return_value=endpoints._ChatActionDecision(
                    action="fetch_url",
                    reason="explicit_url",
                    query="",
                    url="https://archon.diy",
                )
            ),
        ) as decide_chat_action,
        patch("src.api.endpoints.fetch_url_content", new=AsyncMock(return_value="Archon page content")),
        patch("src.api.endpoints._generate_suggestions", new=AsyncMock(return_value=[])),
    ):
        response = client.post(
            "/api/rag/agents/agent-1/chat/stream",
            json={"message": "is this a good tool to add in my stack? https://archon.diy and why?"},
        )

    assert response.status_code == 200
    decide_chat_action.assert_awaited_once()
    messages = mock_llm.ainvoke.await_args.args[0]
    all_content = " ".join(m.content for m in messages)
    assert "Archon page content" in all_content
    assert "Irrelevant SaaS Starter Kit context" not in all_content
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    session_event = next(event for event in events if event["type"] == "session")
    assert session_event["web_used"] is True
    assert session_event["web_provider"] == "direct_fetch"


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


def _make_prd_plan() -> PRDPlan:
    return PRDPlan(
        title="Mobile Onboarding PRD",
        executive_summary="Streamline user onboarding.",
        problem_statement="Users drop off during onboarding.",
        goals=["Reduce drop-off by 30%", "Improve activation", "Increase retention"],
        non_goals=["Full app redesign"],
        target_users=["New mobile users", "Enterprise admins", "Power users"],
        user_stories=["As a new user, I want to onboard quickly."],
        requirements=[
            PRDRequirement(id="REQ-001", description="3-step wizard", priority="Must Have", rationale="Core flow"),
            PRDRequirement(id="REQ-002", description="Progress bar", priority="Should Have", rationale="UX clarity"),
        ],
        success_metrics=["30% drop-off reduction", "NPS +10", "Activation > 60%"],
        milestones=[PRDMilestone(id="M1", title="MVP", description="Basic onboarding", deliverables=["Wizard"])],
        out_of_scope=["Localization"],
        risks=["Scope creep"],
        assumptions=["Users on latest version"],
        open_questions=["Skip button?"],
    )


def test_prd_plan_generation():
    plan = _make_prd_plan()
    generated_plan = PRDPlanResponse(
        plan=plan,
        markdown="# Mobile Onboarding PRD\n",
        suggested_filename="2026-05-29-mobile-onboarding-prd.md",
        planning_brief=PlanningBrief(
            problem_statement="Users drop off during onboarding.",
            desired_outcome="Increase activation rate.",
            constraints=["Must ship in Q3"],
            assumptions=["Users on latest version"],
            open_questions=[],
        ),
    )
    saved_plan = SavedPRD(
        plan_id="plan-123",
        prompt="Build a mobile onboarding flow.",
        prompt_preview="Build a mobile onboarding flow.",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:00+00:00",
        plan=generated_plan.plan,
        markdown=generated_plan.markdown,
        suggested_filename=generated_plan.suggested_filename,
        planning_brief=generated_plan.planning_brief,
    )

    with (
        patch(
            "src.api.endpoints.generate_prd",
            new=AsyncMock(return_value=generated_plan),
        ) as generate_mock,
        patch(
            "src.api.endpoints.save_prd",
            new=AsyncMock(return_value=saved_plan),
        ) as save_mock,
    ):
        response = client.post(
            "/api/planner/prd",
            json={"prompt": "Build a mobile onboarding flow."},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan_id"] == "plan-123"
    assert payload["plan"]["title"] == "Mobile Onboarding PRD"
    assert payload["suggested_filename"].endswith("-prd.md")
    assert payload["planning_brief"]["problem_statement"] == "Users drop off during onboarding."
    generate_mock.assert_awaited_once_with("Build a mobile onboarding flow.")
    save_mock.assert_awaited_once_with(
        "test-user",
        "Build a mobile onboarding flow.",
        generated_plan,
    )


def test_list_saved_prds():
    summary = SavedPRDSummary(
        plan_id="plan-123",
        title="Mobile Onboarding PRD",
        summary="Streamline user onboarding.",
        prompt_preview="Build a mobile onboarding flow.",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:00+00:00",
    )

    with patch(
        "src.api.endpoints.list_saved_prds",
        new=AsyncMock(return_value=[summary]),
    ) as list_mock:
        response = client.get("/api/planner/prd/plans")

    assert response.status_code == 200
    assert response.json()["plans"][0]["plan_id"] == "plan-123"
    list_mock.assert_awaited_once_with("test-user")


def test_get_saved_prd():
    plan = _make_prd_plan()
    saved_plan = SavedPRD(
        plan_id="plan-123",
        prompt="Build a mobile onboarding flow.",
        prompt_preview="Build a mobile onboarding flow.",
        created_at="2026-05-29T10:00:00+00:00",
        updated_at="2026-05-29T10:00:00+00:00",
        plan=plan,
        markdown="# Mobile Onboarding PRD\n",
        suggested_filename="2026-05-29-mobile-onboarding-prd.md",
        planning_brief=PlanningBrief(
            problem_statement="Users drop off during onboarding.",
            desired_outcome="Increase activation rate.",
            constraints=[],
            assumptions=[],
            open_questions=[],
        ),
    )

    with patch(
        "src.api.endpoints.get_saved_prd",
        new=AsyncMock(return_value=saved_plan),
    ) as get_mock:
        response = client.get("/api/planner/prd/plans/plan-123")

    assert response.status_code == 200
    assert response.json()["plan_id"] == "plan-123"
    get_mock.assert_awaited_once_with("test-user", "plan-123")


def test_get_saved_prd_returns_404_when_missing():
    with patch(
        "src.api.endpoints.get_saved_prd",
        new=AsyncMock(return_value=None),
    ):
        response = client.get("/api/planner/prd/plans/missing-plan")

    assert response.status_code == 404
    assert response.json()["detail"] == "Saved PRD 'missing-plan' not found."


def test_delete_saved_prd_returns_deleted():
    with patch(
        "src.api.endpoints.delete_saved_prd",
        new=AsyncMock(return_value=True),
    ):
        response = client.delete("/api/planner/prd/plans/plan-abc")

    assert response.status_code == 200
    body = response.json()
    assert body["plan_id"] == "plan-abc"
    assert body["deleted"] is True


def test_delete_saved_prd_returns_404_when_missing():
    with patch(
        "src.api.endpoints.delete_saved_prd",
        new=AsyncMock(return_value=False),
    ):
        response = client.delete("/api/planner/prd/plans/no-such-plan")

    assert response.status_code == 404
    assert response.json()["detail"] == "Saved PRD 'no-such-plan' not found."


def test_prd_generation_maps_validation_errors():
    with patch(
        "src.api.endpoints.generate_prd",
        new=AsyncMock(
            side_effect=PlannerValidationError(
                "planner_generation_failed",
                "Planner output could not be validated.",
            )
        ),
    ):
        response = client.post(
            "/api/planner/prd",
            json={"prompt": "Build a mobile onboarding flow."},
        )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "planner_generation_failed"


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
