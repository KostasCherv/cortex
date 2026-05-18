"""Edge (routing) logic for the research graph."""

import logging

from src.graph.state import ResearchState
from src.observability.langsmith import start_step_span

logger = logging.getLogger(__name__)


def route_after_search(state: ResearchState) -> str:
    """Route after search_and_memory_node: abort on error, empty if no results.

    Returns:
        ``"abort"`` on search error, ``"empty"`` if no results, ``"continue"`` otherwise.
    """
    with start_step_span(
        name="edge.route_after_search",
        run_type="tool",
        node_name="search_and_memory",
        inputs={
            "has_error": bool(state.get("error")),
            "result_count": len(state.get("search_results", [])),
        },
        tags=["routing"],
    ):
        if state.get("error"):
            logger.warning("Pipeline aborting due to error: %s", state["error"])
            return "abort"
        return "continue" if state.get("search_results") else "empty"
