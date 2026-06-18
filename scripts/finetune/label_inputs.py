# scripts/finetune/label_inputs.py
"""Label synthetic inputs using Qwen3-30B as teacher; validate with Pydantic."""

from __future__ import annotations

import os

import httpx

from scripts.finetune.router_prompt import ROUTER_SYSTEM_PROMPT, build_training_record, format_user_turn
from src.errors import StructuredOutputParseError, StructuredOutputValidationError
from src.llm.output_parsers import parse_chat_action_json

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
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
