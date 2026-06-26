# scripts/finetune/label_inputs.py
"""Label synthetic inputs using Qwen3-30B as teacher; validate with Pydantic."""

from __future__ import annotations

import os

import httpx

from scripts.finetune.router_prompt import ROUTER_SYSTEM_PROMPT, build_training_record, format_user_turn
from scripts.finetune.teacher_client import call_teacher
from src.errors import StructuredOutputParseError, StructuredOutputValidationError
from src.llm.output_parsers import parse_chat_action_json

DEFAULT_TEACHER_MODEL = os.getenv("TEACHER_MODEL", "qwen3:30b")


def label_input(
    *,
    message: str,
    rag_context: str = "",
    history: list[dict[str, str]] | None = None,
    ollama_model: str = DEFAULT_TEACHER_MODEL,
    timeout: float = 120.0,
) -> dict | None:
    """Call teacher at temp=0, validate output, return a training record or None."""
    user_turn = format_user_turn(message=message, rag_context=rag_context, history=history)

    try:
        raw = call_teacher(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_turn},
            ],
            model=ollama_model,
            temperature=0.0,
            timeout=timeout,
        )
    except (httpx.HTTPError, KeyError):
        return None

    try:
        validated = parse_chat_action_json(raw)
    except (StructuredOutputParseError, StructuredOutputValidationError):
        return None

    return build_training_record(
        message=message,
        rag_context=rag_context,
        history=history,
        assistant_json=validated.model_dump_json(),
    )
