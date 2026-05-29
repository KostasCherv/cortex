"""LangGraph nodes for the interactive planner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage

from src.errors import StructuredOutputError
from src.llm.factory import get_llm
from src.llm.output_parsers import (
    build_validation_retry_prompt,
    parse_clarification_decision_json,
)
from src.planner import (
    PlannerValidationError,
    _generate_prd_sync,
    _llm_result_to_text,
    _schema_text,
)
from src.observability.langsmith import start_step_span
from src.planner_graph.state import MAX_CLARIFICATION_TURNS, PlannerState
from src.prompts.registry import prompt_registry

if TYPE_CHECKING:
    from src.llm.output_parsers import ClarificationDecision

logger = logging.getLogger(__name__)

_CLARIFICATION_DECISION_SCHEMA = None


def _get_clarification_schema() -> str:
    global _CLARIFICATION_DECISION_SCHEMA
    if _CLARIFICATION_DECISION_SCHEMA is None:
        from src.llm.output_parsers import ClarificationDecision
        _CLARIFICATION_DECISION_SCHEMA = _schema_text(ClarificationDecision)
    return _CLARIFICATION_DECISION_SCHEMA


def _serialize_history(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def clarification_node(state: PlannerState) -> PlannerState:
    turn_count = (state.get("turn_count") or 0) + 1
    max_turns = state.get("max_clarification_turns") or MAX_CLARIFICATION_TURNS

    if (turn_count - 1) >= max_turns:
        return {
            "ready_to_generate": True,
            "clarification_question": None,
            "turn_count": turn_count - 1,
        }

    conversation_history = state.get("conversation_history") or []
    history_text = _serialize_history(conversation_history)

    prompt_text, _ = prompt_registry.render(
        "planner_clarification",
        {
            "conversation_history_text": history_text,
            "max_turns": max_turns,
        },
    )

    llm = get_llm(temperature=0.2)
    try:
        with start_step_span(
            name="planner.clarification_node.llm_invoke",
            run_type="llm",
            node_name="clarification_node",
            inputs={"prompt": prompt_text},
            tags=["planner", "clarification"],
        ):
            result = llm.invoke(prompt_text)
    except Exception as exc:
        logger.error("LLM API error in clarification_node: %s", exc)
        return {
            "error": "llm_api_error",
            "ready_to_generate": True,
            "turn_count": turn_count,
        }
    raw_text = _llm_result_to_text(result)

    try:
        decision = parse_clarification_decision_json(raw_text)
    except StructuredOutputError as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text=_get_clarification_schema(),
            invalid_response=raw_text,
            validation_error=exc,
        )
        with start_step_span(
            name="planner.clarification_node.llm_repair",
            run_type="llm",
            node_name="clarification_node",
            inputs={"prompt": repair_prompt},
            tags=["planner", "clarification", "repair"],
        ):
            repair_result = llm.invoke(repair_prompt)
        repair_text = _llm_result_to_text(repair_result)
        try:
            decision = parse_clarification_decision_json(repair_text)
        except (StructuredOutputError, Exception) as repair_exc:
            logger.warning("Clarification node parse failed after repair: %s", repair_exc)
            return {
                "error": "clarification_parse_failed",
                "ready_to_generate": True,
                "turn_count": turn_count,
            }

    if decision.ready:
        return {
            "ready_to_generate": True,
            "clarification_question": None,
            "turn_count": turn_count,
        }
    return {
        "ready_to_generate": False,
        "clarification_question": decision.question,
        "turn_count": turn_count,
    }


def generation_node(state: PlannerState) -> PlannerState:
    if state.get("error"):
        return {"error": state["error"]}

    conversation_history = state.get("conversation_history") or []
    previous_plan = state.get("final_plan")

    if previous_plan is not None:
        # Refinement mode: pass the previous plan + full conversation so the
        # generator knows exactly what to change.
        history_text = _serialize_history(conversation_history)
        consolidated_prompt = (
            "PREVIOUS PLAN (to be refined):\n"
            f"{previous_plan.markdown}\n\n"
            "---\n\n"
            "FULL CONVERSATION (including refinement requests):\n"
            f"{history_text}"
        )
    else:
        # Initial generation: distil all human messages into one requirement block.
        human_messages = [
            msg.content if isinstance(msg.content, str) else str(msg.content)
            for msg in conversation_history
            if msg.type == "human"
        ]
        consolidated_prompt = "\n\n".join(human_messages)

    try:
        with start_step_span(
            name="planner.generation_node.generate_plan",
            run_type="chain",
            node_name="generation_node",
            inputs={"prompt": consolidated_prompt, "is_refinement": previous_plan is not None},
            tags=["planner", "generation"],
        ):
            response = _generate_prd_sync(consolidated_prompt)
    except PlannerValidationError as exc:
        return {
            "error": exc.code,
            "final_plan": None,
        }
    except Exception as exc:
        logger.error("Unexpected error in generation_node: %s", exc)
        return {
            "error": "generation_unexpected_error",
            "final_plan": None,
        }

    return {"final_plan": response}
