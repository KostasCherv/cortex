"""Tests for graph edge routing logic (src/graph/edges.py)"""

from src.graph.edges import route_after_search


def test_route_after_search_returns_abort_on_error():
    state = {"query": "test", "error": "something went wrong"}
    assert route_after_search(state) == "abort"


def test_route_after_search_returns_continue_with_results():
    state = {"query": "test", "search_results": [{"url": "https://a.com"}]}
    assert route_after_search(state) == "continue"


def test_route_after_search_returns_empty_with_no_results():
    state = {"query": "test", "search_results": []}
    assert route_after_search(state) == "empty"


def test_route_after_search_returns_empty_when_results_missing():
    state = {"query": "test"}
    assert route_after_search(state) == "empty"
