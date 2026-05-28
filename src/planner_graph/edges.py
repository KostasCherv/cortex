"""Conditional edges for the interactive planner graph."""

from typing import Literal

from langgraph.graph import END


def route_after_clarification(state: dict) -> Literal["generation_node"] | str:
    if state.get("ready_to_generate") or state.get("error"):
        return "generation_node"
    return END
