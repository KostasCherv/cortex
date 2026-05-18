"""LangSmith tracing helpers for workflow, step, and child spans."""

from __future__ import annotations

import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4
from typing import Any, Iterator

from src.config import settings
from src.observability.context import (
    build_trace_metadata,
    build_trace_tags,
    is_trace_enabled,
    node_context,
    workflow_context,
)
from src.observability.redaction import redact_payload

try:
    from langsmith import Client
    from langsmith.run_helpers import trace, tracing_context
except Exception:  # pragma: no cover - optional dependency at runtime
    Client = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    tracing_context = None  # type: ignore[assignment]


@dataclass
class WorkflowTraceContext:
    workflow_id: str
    entrypoint: str
    tracing_enabled: bool
    run: Any | None = None
    ended: bool = False


def _sampling_allows_trace() -> bool:
    rate = max(0.0, min(1.0, settings.langsmith_sampling_rate))
    return random.random() <= rate


def _langsmith_ready() -> bool:
    return bool(
        settings.langsmith_tracing
        and bool(settings.langsmith_api_key)
        and trace is not None
        and tracing_context is not None
        and Client is not None
    )


@lru_cache(maxsize=1)
def _get_langsmith_client() -> Any | None:
    if Client is None:
        return None
    if not settings.langsmith_api_key:
        return None
    return Client(
        api_key=settings.langsmith_api_key,
        api_url=settings.langsmith_endpoint,
    )


@contextmanager
def start_workflow_run(
    *,
    entrypoint: str,
    query: str,
) -> Iterator[WorkflowTraceContext]:
    """Start a root workflow run and propagate context."""
    workflow_id = str(uuid4())
    enabled = _langsmith_ready() and _sampling_allows_trace()
    ctx = WorkflowTraceContext(
        workflow_id=workflow_id,
        entrypoint=entrypoint,
        tracing_enabled=enabled,
    )
    redaction_mode = settings.langsmith_redaction_mode
    run_name = "research-workflow"
    run_inputs = redact_payload(
        {"query": query},
        mode=redaction_mode,
    )
    tags = build_trace_tags(["workflow"])
    metadata = build_trace_metadata({"entrypoint": entrypoint})

    with workflow_context(workflow_id, entrypoint, trace_enabled=enabled):
        if not enabled:
            yield ctx
            return

        trace_context = tracing_context(  # type: ignore[misc]
            enabled=True,
            project_name=settings.langsmith_project,
            tags=tags,
            metadata=metadata,
            client=_get_langsmith_client(),
        )
        with trace_context:
            with trace(  # type: ignore[misc]
                run_name,
                run_type="chain",
                inputs=run_inputs,
                tags=tags,
                metadata=metadata,
                client=_get_langsmith_client(),
            ) as run:
                ctx.run = run
                yield ctx


def end_workflow_run(
    ctx: WorkflowTraceContext,
    *,
    status: str,
    outputs: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    """Finalize the root run with redacted outputs."""
    if not ctx.tracing_enabled or ctx.ended or ctx.run is None:
        return

    payload: dict[str, object] = {"status": status}
    if outputs:
        payload["data"] = redact_payload(outputs, mode=settings.langsmith_redaction_mode)
    redacted_error: str | None = None
    if error:
        redacted_error = str(
            redact_payload({"error": error}, mode=settings.langsmith_redaction_mode).get(
                "error", error
            )
        )
    if hasattr(ctx.run, "end"):
        ctx.run.end(outputs=payload, error=redacted_error)
    patch = getattr(ctx.run, "patch", None)
    if callable(patch):
        try:
            patch()
        except Exception:
            pass
    wait = getattr(ctx.run, "wait", None)
    if callable(wait):
        try:
            wait()
        except Exception:
            pass

    client = getattr(ctx.run, "ls_client", None) or _get_langsmith_client()
    run_id = getattr(ctx.run, "id", None)
    if client is not None and run_id is not None:
        try:
            client.update_run(
                run_id=run_id,
                end_time=datetime.now(UTC),
                outputs=payload,
                error=redacted_error,
            )
        except Exception:
            # Best effort update; end() + patch()/wait() above are primary path.
            pass
    flush = getattr(client, "flush", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            # Best effort flush; run has already been ended above.
            pass
    ctx.ended = True


@contextmanager
def start_step_span(
    *,
    name: str,
    run_type: str = "tool",
    node_name: str | None = None,
    inputs: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    tags: list[str] | None = None,
) -> Iterator[None]:
    """Start a child span under the current workflow context."""
    node_label = node_name or name
    redaction_mode = settings.langsmith_redaction_mode
    trace_inputs = redact_payload(inputs or {}, mode=redaction_mode)
    trace_metadata = build_trace_metadata(metadata or {})
    trace_tags = build_trace_tags(tags or [])

    with node_context(node_label):
        if not is_trace_enabled() or trace is None:
            with nullcontext():
                yield
            return

        with trace(  # type: ignore[misc]
            name,
            run_type=run_type,
            inputs=trace_inputs,
            metadata=trace_metadata,
            tags=trace_tags,
            client=_get_langsmith_client(),
        ):
            yield
