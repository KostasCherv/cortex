"""Timing breakdown for RAG chat requests (exposed via X-Rag-Perf when enabled)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class RagChatTimings:
    prepare_ms: float = 0.0
    session_ms: float = 0.0
    agent_loop_ms: float = 0.0
    suggestions_ms: float = 0.0
    persist_ms: float = 0.0
    total_ms: float = 0.0
    tools_bound: bool = False
    tool_skip_reason: str | None = None

    def to_header_value(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))
