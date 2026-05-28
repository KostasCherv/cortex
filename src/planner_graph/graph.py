"""Compiled LangGraph StateGraph for the interactive planner."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.planner_graph.edges import route_after_clarification
from src.planner_graph.state import PlannerState

_checkpointer = MemorySaver()


def build_planner_graph():
    builder = StateGraph(PlannerState)
    builder.set_entry_point("clarification_node")

    # Import nodes here to avoid circular imports at module level
    from src.planner_graph.nodes import clarification_node, generation_node

    builder.add_node("clarification_node", clarification_node)
    builder.add_node("generation_node", generation_node)
    builder.add_conditional_edges(
        "clarification_node",
        route_after_clarification,
        {"generation_node": "generation_node", END: END},
    )
    builder.add_edge("generation_node", END)
    return builder.compile(checkpointer=_checkpointer)


planner_graph = build_planner_graph()
