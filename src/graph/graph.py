"""Compile the LangGraph state machine."""

import logging

from langgraph.graph import StateGraph, END

from src.graph.state import ResearchState
from src.graph.nodes import (
    search_and_memory_node,
    rerank_node,
    summarize_node,
    report_node,
)
from src.graph.edges import route_after_search
from src.observability.langsmith import start_step_span

logger = logging.getLogger(__name__)


def _abort_node(state: ResearchState) -> ResearchState:
    """Terminal node reached on pipeline abort."""
    with start_step_span(
        name="abort_node",
        run_type="tool",
        node_name="abort",
        inputs={"has_error": bool(state.get("error"))},
        tags=["terminal"],
    ):
        logger.error("Research pipeline aborted. Error: %s", state.get("error"))
        return state


def _empty_node(state: ResearchState) -> ResearchState:
    """Terminal node reached when search returned no results."""
    with start_step_span(
        name="empty_node",
        run_type="tool",
        node_name="empty",
        inputs={"query": state.get("query", "")},
        tags=["terminal"],
    ):
        logger.warning("Search returned no results for query: %s", state.get("query"))
        return {**state, "report": "No results found for the given query.", "report_metadata": {}}


def build_graph():
    """Build and compile the research agent graph.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to invoke.
    """
    builder = StateGraph(ResearchState)

    # Register all nodes
    builder.add_node("search_and_memory", search_and_memory_node)
    builder.add_node("rerank",            rerank_node)
    builder.add_node("summarize",         summarize_node)
    builder.add_node("report",            report_node)
    builder.add_node("abort",             _abort_node)
    builder.add_node("empty",             _empty_node)

    # Entry point: search + memory run in parallel inside one node
    builder.set_entry_point("search_and_memory")

    # Route based on error / empty results
    builder.add_conditional_edges(
        "search_and_memory",
        route_after_search,
        {"continue": "rerank", "abort": "abort", "empty": "empty"},
    )

    # Linear tail of the pipeline
    builder.add_edge("rerank",       "summarize")
    builder.add_edge("summarize",    "report")
    builder.add_edge("report",       END)

    # Terminal edges
    builder.add_edge("abort",        END)
    builder.add_edge("empty",        END)

    return builder.compile()
