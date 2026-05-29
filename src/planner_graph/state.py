"""LangGraph state definition for the interactive planner."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

MAX_CLARIFICATION_TURNS: int = 6


class PlannerState(TypedDict, total=False):
    conversation_history: list[BaseMessage]
    ready_to_generate: bool
    clarification_question: str | None
    final_plan: Any  # PRDPlanResponse | None — Any avoids forward-ref resolution by LangGraph
    turn_count: int
    max_clarification_turns: int
    thread_id: str
    user_id: str | None
    error: str | None
