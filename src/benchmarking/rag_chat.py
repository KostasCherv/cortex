"""Helpers for benchmark scripts that touch rag chat internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentLoopBenchmarkResult:
    answer: str
    answer_chars: int
    web_used: bool
    citation_count: int


def coerce_agent_loop_benchmark_result(result: Any) -> AgentLoopBenchmarkResult:
    if isinstance(result, str):
        answer = result
        web_used = False
        citations = []
    else:
        answer = str(getattr(result, "answer", ""))
        web_used = bool(getattr(result, "web_used", False))
        citations = list(getattr(result, "citations", []) or [])

    return AgentLoopBenchmarkResult(
        answer=answer,
        answer_chars=len(answer),
        web_used=web_used,
        citation_count=len(citations),
    )
