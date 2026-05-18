"""Tests for src/graph/state.py"""

from src.graph.state import ResearchState


def test_state_is_typed_dict():
    state: ResearchState = {
        "query": "What is LangGraph?",
        "search_results": [],
        "retrieved_contents": [],
        "reranked_contents": [],
        "rerank_metadata": {},
        "summaries": [],
        "report": "",
        "report_metadata": {},
        "error": None,
    }
    assert state["query"] == "What is LangGraph?"
    assert state["error"] is None


def test_state_partial_is_valid():
    """ResearchState uses total=False so partial dicts are valid."""
    state: ResearchState = {"query": "hello"}
    assert state["query"] == "hello"
    assert "report" not in state


def test_state_error_field():
    state: ResearchState = {"query": "test", "error": "search failed"}
    assert state["error"] == "search failed"
