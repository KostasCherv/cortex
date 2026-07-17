from unittest.mock import MagicMock, patch

from src.observability.sentry import capture_handled_exception


def test_capture_handled_exception_only_attaches_allowlisted_identifiers():
    scope = MagicMock()
    scope_context = MagicMock()
    scope_context.__enter__.return_value = scope
    exc = RuntimeError("failed")

    with (
        patch("src.observability.sentry.sentry_sdk.new_scope", return_value=scope_context),
        patch("src.observability.sentry.sentry_sdk.capture_exception") as mock_capture,
    ):
        capture_handled_exception(
            exc,
            operation="sse.run_refresh",
            identifiers={
                "run_id": "run-1",
                "session_id": "session-1",
                "prompt": "must not be sent",
                "authorization": "must not be sent",
            },
        )

    scope.set_tag.assert_called_once_with("cortex.operation", "sse.run_refresh")
    scope.set_extra.assert_any_call("run_id", "run-1")
    scope.set_extra.assert_any_call("session_id", "session-1")
    assert scope.set_extra.call_count == 2
    mock_capture.assert_called_once_with(exc)
