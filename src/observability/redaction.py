"""Payload redaction utilities for LangSmith traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SENSITIVE_KEYS = {
    "prompt",
    "raw_text",
    "report",
    "content",
    "memory_context",
    "error",
    "exception",
}

REDACTED = "[REDACTED]"


def _truncate(value: str, limit: int = 200) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...({len(value)} chars)"


def _redact_mapping(data: Mapping[str, object], mode: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in data.items():
        lower_key = key.lower()
        if mode == "metadata_only":
            output[key] = _metadata_shape(value)
            continue
        if lower_key in SENSITIVE_KEYS:
            output[key] = REDACTED
            continue
        output[key] = redact_payload(value, mode=mode)
    return output


def _metadata_shape(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {"type": "dict", "size": len(value)}
    if isinstance(value, str):
        return {"type": "str", "length": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"type": "list", "size": len(value)}
    return {"type": type(value).__name__}


def redact_payload(payload: object, mode: str = "redacted_default") -> object:
    """Redact payload data according to configured mode."""
    if mode == "full_payloads":
        return payload

    if mode == "metadata_only":
        return _metadata_shape(payload)

    if isinstance(payload, Mapping):
        return _redact_mapping(payload, mode=mode)

    if isinstance(payload, str):
        return _truncate(payload)

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [redact_payload(item, mode=mode) for item in payload]

    return payload
