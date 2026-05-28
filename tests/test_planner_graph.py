"""Tests for the interactive planner LangGraph components."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

from src.errors import StructuredOutputValidationError


# ---------------------------------------------------------------------------
# Phase 1 — ClarificationDecision parser tests
# ---------------------------------------------------------------------------


def test_parse_clarification_decision_ready_true():
    from src.llm.output_parsers import parse_clarification_decision_json

    parsed = parse_clarification_decision_json(
        '{"ready": true, "question": null, "reason": "enough info"}'
    )
    assert parsed.ready is True
    assert parsed.question is None
    assert parsed.reason == "enough info"


def test_parse_clarification_decision_ready_false_with_question():
    from src.llm.output_parsers import parse_clarification_decision_json

    parsed = parse_clarification_decision_json(
        '{"ready": false, "question": "What database will you use?", "reason": "need db info"}'
    )
    assert parsed.ready is False
    assert parsed.question == "What database will you use?"


def test_parse_clarification_decision_ready_false_missing_question_raises():
    from src.llm.output_parsers import parse_clarification_decision_json

    with pytest.raises(StructuredOutputValidationError):
        parse_clarification_decision_json(
            '{"ready": false, "question": null, "reason": "need more info"}'
        )


def test_parse_clarification_decision_ready_false_blank_question_raises():
    from src.llm.output_parsers import parse_clarification_decision_json

    with pytest.raises(StructuredOutputValidationError):
        parse_clarification_decision_json(
            '{"ready": false, "question": "   ", "reason": "need more info"}'
        )


def test_parse_clarification_decision_strips_markdown_fences():
    from src.llm.output_parsers import parse_clarification_decision_json

    parsed = parse_clarification_decision_json(
        '```json\n{"ready": true, "question": null, "reason": "all good"}\n```'
    )
    assert parsed.ready is True


def test_parse_clarification_decision_missing_reason_raises():
    from src.llm.output_parsers import parse_clarification_decision_json

    with pytest.raises((StructuredOutputValidationError, Exception)):
        parse_clarification_decision_json('{"ready": true, "question": null}')


# ---------------------------------------------------------------------------
# Phase 2 — Router edge tests
# ---------------------------------------------------------------------------


def test_route_after_clarification_ready_routes_to_generation():
    from src.planner_graph.edges import route_after_clarification

    state = {"ready_to_generate": True, "turn_count": 2}
    assert route_after_clarification(state) == "generation_node"


def test_route_after_clarification_not_ready_routes_to_end():
    from src.planner_graph.edges import route_after_clarification

    state = {"ready_to_generate": False, "turn_count": 1}
    assert route_after_clarification(state) == END


def test_route_after_clarification_error_routes_to_generation():
    from src.planner_graph.edges import route_after_clarification

    state = {"ready_to_generate": False, "error": "clarification_parse_failed", "turn_count": 1}
    assert route_after_clarification(state) == "generation_node"


def test_route_after_clarification_default_state_routes_to_end():
    from src.planner_graph.edges import route_after_clarification

    assert route_after_clarification({}) == END


# ---------------------------------------------------------------------------
# Phase 3 — Clarification node tests
# ---------------------------------------------------------------------------


def _make_human_message(content: str):
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=content)


def _make_ai_message(content: str):
    from langchain_core.messages import AIMessage
    return AIMessage(content=content)


def _make_llm_mock(json_text: str):
    mock_llm = MagicMock()
    mock_result = MagicMock()
    mock_result.content = json_text
    mock_llm.invoke.return_value = mock_result
    return mock_llm


def test_clarification_node_asks_question_when_not_ready():
    from src.planner_graph.nodes import clarification_node

    state = {
        "conversation_history": [_make_human_message("Add a caching layer")],
        "turn_count": 0,
        "max_clarification_turns": 6,
        "ready_to_generate": False,
    }
    llm_json = '{"ready": false, "question": "What caching backend — Redis or Memcached?", "reason": "need cache type"}'
    with patch("src.planner_graph.nodes.get_llm", return_value=_make_llm_mock(llm_json)):
        result = clarification_node(state)

    assert result["ready_to_generate"] is False
    assert "Redis" in result["clarification_question"]
    assert result["turn_count"] == 1


def test_clarification_node_sets_ready_when_llm_returns_ready():
    from src.planner_graph.nodes import clarification_node

    state = {
        "conversation_history": [_make_human_message("Add a caching layer")],
        "turn_count": 1,
        "max_clarification_turns": 6,
        "ready_to_generate": False,
    }
    llm_json = '{"ready": true, "question": null, "reason": "have enough"}'
    with patch("src.planner_graph.nodes.get_llm", return_value=_make_llm_mock(llm_json)):
        result = clarification_node(state)

    assert result["ready_to_generate"] is True
    assert result["clarification_question"] is None


def test_clarification_node_forces_ready_at_max_turns():
    from src.planner_graph.nodes import clarification_node

    state = {
        "conversation_history": [_make_human_message("Add a caching layer")],
        "turn_count": 6,
        "max_clarification_turns": 6,
        "ready_to_generate": False,
    }
    mock_llm = MagicMock()
    with patch("src.planner_graph.nodes.get_llm", return_value=mock_llm):
        result = clarification_node(state)

    mock_llm.invoke.assert_not_called()
    assert result["ready_to_generate"] is True
    assert result["turn_count"] == 6


def test_clarification_node_increments_turn_count():
    from src.planner_graph.nodes import clarification_node

    state = {
        "conversation_history": [_make_human_message("Hi")],
        "turn_count": 3,
        "max_clarification_turns": 6,
        "ready_to_generate": False,
    }
    llm_json = '{"ready": false, "question": "Any constraints?", "reason": "need constraints"}'
    with patch("src.planner_graph.nodes.get_llm", return_value=_make_llm_mock(llm_json)):
        result = clarification_node(state)

    assert result["turn_count"] == 4


def test_clarification_node_sets_error_on_repeated_parse_failure():
    from src.planner_graph.nodes import clarification_node

    state = {
        "conversation_history": [_make_human_message("Hi")],
        "turn_count": 0,
        "max_clarification_turns": 6,
        "ready_to_generate": False,
    }
    mock_llm = _make_llm_mock("not valid json at all !!!")
    with patch("src.planner_graph.nodes.get_llm", return_value=mock_llm):
        result = clarification_node(state)

    assert result.get("error") is not None
    assert result["ready_to_generate"] is True


def test_generation_node_propagates_error_state_unchanged():
    from src.planner_graph.nodes import generation_node

    state = {
        "conversation_history": [_make_human_message("Hi")],
        "error": "clarification_parse_failed",
        "ready_to_generate": True,
    }
    with patch("src.planner_graph.nodes._generate_software_dev_plan_sync") as mock_gen:
        result = generation_node(state)

    mock_gen.assert_not_called()
    assert result.get("error") == "clarification_parse_failed"


def test_generation_node_calls_planner_and_sets_final_plan():
    from src.planner_graph.nodes import generation_node
    from src.planner import SoftwareDevPlanResponse

    state = {
        "conversation_history": [
            _make_human_message("Add caching"),
            _make_ai_message("What cache backend?"),
            _make_human_message("Redis"),
        ],
        "ready_to_generate": True,
    }
    mock_response = MagicMock(spec=SoftwareDevPlanResponse)
    with patch("src.planner_graph.nodes._generate_software_dev_plan_sync", return_value=mock_response):
        result = generation_node(state)

    assert result["final_plan"] is mock_response


def test_generation_node_sets_error_on_planner_validation_error():
    from src.planner_graph.nodes import generation_node
    from src.planner import PlannerValidationError

    state = {
        "conversation_history": [_make_human_message("Add caching")],
        "ready_to_generate": True,
    }
    with patch(
        "src.planner_graph.nodes._generate_software_dev_plan_sync",
        side_effect=PlannerValidationError("planner_generation_failed", "LLM gave invalid output"),
    ):
        result = generation_node(state)

    assert result.get("error") == "planner_generation_failed"
    assert result.get("final_plan") is None


# ---------------------------------------------------------------------------
# Phase 4 — Thread store tests
# ---------------------------------------------------------------------------


def test_thread_store_create_and_get():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore()
    tid = store.create_thread(user_id="user-1")
    entry = store.get_thread(tid, "user-1")
    assert entry is not None
    assert entry.user_id == "user-1"
    assert entry.thread_id == tid


def test_thread_store_rejects_wrong_user():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore()
    tid = store.create_thread(user_id="user-1")
    assert store.get_thread(tid, "user-2") is None


def test_thread_store_expired_thread_returns_none():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore(ttl=0)
    tid = store.create_thread(user_id="user-1")
    time.sleep(0.01)
    assert store.get_thread(tid, "user-1") is None


def test_thread_store_delete_last_exchange_removes_pair():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore()
    tid = store.create_thread(user_id="u")
    for i in range(4):
        store.append_message(tid, {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)})
    store.delete_last_exchange(tid)
    entry = store.get_thread(tid, "u")
    assert entry is not None
    assert len(entry.messages) == 2


def test_thread_store_delete_last_exchange_noop_when_less_than_two():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore()
    tid = store.create_thread(user_id="u")
    store.append_message(tid, {"role": "user", "content": "hi"})
    store.delete_last_exchange(tid)
    entry = store.get_thread(tid, "u")
    assert entry is not None
    assert len(entry.messages) == 1


def test_thread_store_evicts_lru_at_capacity():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore(max_threads=3)
    t1 = store.create_thread(user_id="u")
    time.sleep(0.01)
    t2 = store.create_thread(user_id="u")
    time.sleep(0.01)
    t3 = store.create_thread(user_id="u")
    time.sleep(0.01)
    t4 = store.create_thread(user_id="u")

    assert store.get_thread(t1, "u") is None
    assert store.get_thread(t2, "u") is not None
    assert store.get_thread(t3, "u") is not None
    assert store.get_thread(t4, "u") is not None


def test_thread_store_evict_expired_returns_count():
    from src.planner_graph.thread_store import PlannerThreadStore

    store = PlannerThreadStore(ttl=0)
    store.create_thread(user_id="u")
    store.create_thread(user_id="u")
    time.sleep(0.01)
    count = store.evict_expired()
    assert count == 2
