"""Privacy-safe helpers for reporting handled exceptions to Sentry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sentry_sdk


_SAFE_IDENTIFIER_KEYS = {
    "event_id",
    "event_name",
    "job_id",
    "method",
    "path",
    "run_id",
    "session_id",
}


def capture_handled_exception(
    exc: BaseException,
    *,
    operation: str,
    identifiers: Mapping[str, Any] | None = None,
) -> None:
    """Capture a swallowed exception with deliberately limited metadata.

    Prompts, messages, document contents, tokens, and arbitrary payload values
    are intentionally not accepted. When Sentry is not configured, the SDK is
    a no-op.
    """
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("cortex.operation", operation)
        for key, value in (identifiers or {}).items():
            if key in _SAFE_IDENTIFIER_KEYS and value is not None:
                scope.set_extra(key, str(value))
        sentry_sdk.capture_exception(exc)
