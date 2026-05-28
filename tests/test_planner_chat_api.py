"""Integration tests for planner chat API endpoints."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("INNGEST_DEV", "1")


# ---------------------------------------------------------------------------
# App / client setup
# ---------------------------------------------------------------------------

with (
    patch("src.api.endpoints.validate_web_search_provider_health"),
    patch("src.api.endpoints.validate_asset_price_provider_health"),
    patch(
        "src.api.endpoints.initialize_alpha_vantage_mcp_client",
        new=AsyncMock(return_value=MagicMock(list_available_tools=MagicMock(return_value=[]))),
    ),
    patch("src.api.endpoints.shutdown_alpha_vantage_mcp_client", new=AsyncMock()),
):
    from fastapi.testclient import TestClient

    from src.api.endpoints import app
    from src.auth import AuthenticatedUser, get_authenticated_user

client = TestClient(app)

_FAKE_USER = AuthenticatedUser(user_id="user-abc", email="test@example.com")


def _override_auth():
    app.dependency_overrides[get_authenticated_user] = lambda: _FAKE_USER


def _clear_auth():
    app.dependency_overrides.pop(get_authenticated_user, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(raw: bytes) -> list[dict]:
    events = []
    for line in raw.decode().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _make_final_plan_mock():
    """Return a mock SoftwareDevPlanResponse that has the attributes planner_chat.py accesses."""
    plan_mock = MagicMock()
    plan_mock.markdown = "# Plan\n\nSome plan content."
    plan_mock.suggested_filename = "plan.md"
    plan_mock.plan.model_dump.return_value = {"phases": []}
    plan_mock.planning_brief.model_dump.return_value = {}
    plan_mock.repo_analysis.model_dump.return_value = {}
    plan_mock.planning_options.model_dump.return_value = {}
    return plan_mock


# ---------------------------------------------------------------------------
# POST /api/planner/chat — new thread (no thread_id)
# ---------------------------------------------------------------------------


class TestPlannerChatNewThread:
    def setup_method(self):
        _override_auth()

    def teardown_method(self):
        _clear_auth()

    def test_new_thread_session_event_returned(self):
        clarification_state = {
            "conversation_history": [],
            "ready_to_generate": False,
            "clarification_question": "What is your target audience?",
            "final_plan": None,
            "error": None,
        }

        mock_state = MagicMock()
        mock_state.values = clarification_state

        with (
            patch("src.api.planner_chat.planner_graph") as mock_graph,
            patch("src.api.planner_chat.save_software_dev_plan", new=AsyncMock()),
        ):
            mock_graph.get_state.return_value = mock_state
            mock_graph.invoke.return_value = None

            resp = client.post("/api/planner/chat", json={"message": "Build a todo app"})

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        session_events = [e for e in events if e["type"] == "session"]
        assert len(session_events) == 1
        assert "thread_id" in session_events[0]

    def test_new_thread_clarification_chunk_and_done(self):
        question = "What database will you use?"
        clarification_state = {
            "conversation_history": [],
            "ready_to_generate": False,
            "clarification_question": question,
            "final_plan": None,
            "error": None,
        }

        mock_state = MagicMock()
        mock_state.values = clarification_state

        with (
            patch("src.api.planner_chat.planner_graph") as mock_graph,
            patch("src.api.planner_chat.save_software_dev_plan", new=AsyncMock()),
        ):
            mock_graph.get_state.return_value = mock_state
            mock_graph.invoke.return_value = None

            resp = client.post("/api/planner/chat", json={"message": "Build a todo app"})

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        chunk_events = [e for e in events if e["type"] == "chunk"]
        done_events = [e for e in events if e["type"] == "done"]
        assert any(question in e["text"] for e in chunk_events)
        assert len(done_events) == 1

    def test_new_thread_final_plan_returns_plan_event(self):
        final_plan = _make_final_plan_mock()
        plan_state = {
            "conversation_history": [],
            "ready_to_generate": True,
            "clarification_question": None,
            "final_plan": final_plan,
            "error": None,
        }

        mock_state = MagicMock()
        mock_state.values = plan_state

        with (
            patch("src.api.planner_chat.planner_graph") as mock_graph,
            patch("src.api.planner_chat.save_software_dev_plan", new=AsyncMock()),
        ):
            mock_graph.get_state.return_value = mock_state
            mock_graph.invoke.return_value = None

            resp = client.post("/api/planner/chat", json={"message": "Build a todo app"})

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        plan_events = [e for e in events if e["type"] == "plan"]
        done_events = [e for e in events if e["type"] == "done"]
        assert len(plan_events) == 1
        assert plan_events[0]["markdown"] == final_plan.markdown
        assert len(done_events) == 1

    def test_graph_error_returns_error_event(self):
        error_state = {
            "conversation_history": [],
            "ready_to_generate": True,
            "clarification_question": None,
            "final_plan": None,
            "error": "some_error_code",
        }

        mock_state = MagicMock()
        mock_state.values = error_state

        with (
            patch("src.api.planner_chat.planner_graph") as mock_graph,
            patch("src.api.planner_chat.save_software_dev_plan", new=AsyncMock()),
        ):
            mock_graph.get_state.return_value = mock_state
            mock_graph.invoke.return_value = None

            resp = client.post("/api/planner/chat", json={"message": "Build something"})

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error"] == "some_error_code"

    def test_graph_invocation_exception_returns_error_event(self):
        with (
            patch("src.api.planner_chat.planner_graph") as mock_graph,
            patch("src.api.planner_chat.save_software_dev_plan", new=AsyncMock()),
        ):
            mock_graph.get_state.return_value = MagicMock(values={})
            mock_graph.invoke.side_effect = RuntimeError("LLM failure")

            resp = client.post("/api/planner/chat", json={"message": "Build something"})

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "Graph execution failed" in error_events[0]["error"]


# ---------------------------------------------------------------------------
# POST /api/planner/chat — existing thread_id
# ---------------------------------------------------------------------------


class TestPlannerChatExistingThread:
    def setup_method(self):
        _override_auth()

    def teardown_method(self):
        _clear_auth()

    def test_unknown_thread_id_returns_404(self):
        resp = client.post(
            "/api/planner/chat",
            json={"message": "Follow up", "thread_id": "nonexistent-thread"},
        )
        assert resp.status_code == 404

    def test_existing_thread_continues_conversation(self):
        from src.planner_graph.thread_store import planner_thread_store

        thread_id = planner_thread_store.create_thread(user_id=_FAKE_USER.user_id)

        clarification_state = {
            "conversation_history": [],
            "ready_to_generate": False,
            "clarification_question": "How many users?",
            "final_plan": None,
            "error": None,
        }
        mock_state = MagicMock()
        mock_state.values = clarification_state

        with (
            patch("src.api.planner_chat.planner_graph") as mock_graph,
            patch("src.api.planner_chat.save_software_dev_plan", new=AsyncMock()),
        ):
            mock_graph.get_state.return_value = mock_state
            mock_graph.invoke.return_value = None

            resp = client.post(
                "/api/planner/chat",
                json={"message": "Follow up", "thread_id": thread_id},
            )

        assert resp.status_code == 200
        events = _parse_sse(resp.content)
        session_events = [e for e in events if e["type"] == "session"]
        assert session_events[0]["thread_id"] == thread_id


# ---------------------------------------------------------------------------
# GET /api/planner/chat/{thread_id}/messages
# ---------------------------------------------------------------------------


class TestGetPlannerChatMessages:
    def setup_method(self):
        _override_auth()

    def teardown_method(self):
        _clear_auth()

    def test_returns_message_history(self):
        from src.planner_graph.thread_store import planner_thread_store

        thread_id = planner_thread_store.create_thread(user_id=_FAKE_USER.user_id)
        planner_thread_store.append_message(
            thread_id,
            {"message_id": "m1", "role": "user", "content": "Hello", "created_at": "2026-01-01T00:00:00Z"},
        )
        planner_thread_store.append_message(
            thread_id,
            {"message_id": "m2", "role": "assistant", "content": "Hi!", "created_at": "2026-01-01T00:00:01Z"},
        )

        resp = client.get(f"/api/planner/chat/{thread_id}/messages")

        assert resp.status_code == 200
        body = resp.json()
        assert body["thread_id"] == thread_id
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][1]["role"] == "assistant"

    def test_unknown_thread_returns_404(self):
        resp = client.get("/api/planner/chat/no-such-thread/messages")
        assert resp.status_code == 404

    def test_wrong_user_returns_404(self):
        from src.planner_graph.thread_store import planner_thread_store

        # Create a thread for a different user
        thread_id = planner_thread_store.create_thread(user_id="other-user-id")
        resp = client.get(f"/api/planner/chat/{thread_id}/messages")
        # Our auth override returns user-abc, so other-user-id's thread should be not found
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/planner/chat/{thread_id}/last
# ---------------------------------------------------------------------------


class TestDeletePlannerChatLastExchange:
    def setup_method(self):
        _override_auth()

    def teardown_method(self):
        _clear_auth()

    def test_deletes_last_exchange(self):
        from src.planner_graph.thread_store import planner_thread_store

        thread_id = planner_thread_store.create_thread(user_id=_FAKE_USER.user_id)
        planner_thread_store.append_message(
            thread_id,
            {"message_id": "m1", "role": "user", "content": "Msg 1", "created_at": "t1"},
        )
        planner_thread_store.append_message(
            thread_id,
            {"message_id": "m2", "role": "assistant", "content": "Reply 1", "created_at": "t2"},
        )
        planner_thread_store.append_message(
            thread_id,
            {"message_id": "m3", "role": "user", "content": "Msg 2", "created_at": "t3"},
        )
        planner_thread_store.append_message(
            thread_id,
            {"message_id": "m4", "role": "assistant", "content": "Reply 2", "created_at": "t4"},
        )

        resp = client.delete(f"/api/planner/chat/{thread_id}/last")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify only 2 messages remain
        get_resp = client.get(f"/api/planner/chat/{thread_id}/messages")
        assert len(get_resp.json()["messages"]) == 2

    def test_unknown_thread_returns_404(self):
        resp = client.delete("/api/planner/chat/nonexistent/last")
        assert resp.status_code == 404
