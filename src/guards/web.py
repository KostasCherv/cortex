"""Guards related to web-access and fetched-content behavior."""

from __future__ import annotations


_NO_WEB_ACCESS_PHRASES = (
    "don't have the capability",
    "do not have the capability",
    "can't access external urls",
    "cannot access external urls",
    "can't retrieve content from external urls",
    "cannot retrieve content from external urls",
    "cannot browse",
    "can't browse",
)


def claims_no_web_access(answer: str) -> bool:
    """Detect model refusals that contradict already-fetched web context."""
    lower = answer.lower()
    return any(phrase in lower for phrase in _NO_WEB_ACCESS_PHRASES)
