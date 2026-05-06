"""LangFuse helpers for generation tracing and feedback scoring."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

from src.config import settings
from src.errors import ConfigurationError

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency at runtime
    Langfuse = None  # type: ignore[assignment]


def _langfuse_ready() -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
        and Langfuse is not None
    )


@lru_cache(maxsize=1)
def get_client() -> Any | None:
    if not _langfuse_ready():
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def require_client() -> Any:
    client = get_client()
    if client is None:
        raise ConfigurationError("LangFuse is not configured. Set LANGFUSE_* environment variables.")
    return client


@dataclass
class ObservedGeneration:
    generation: Any | None = None
    trace_id: str | None = None
    observation_id: str | None = None
    _ended: bool = False

    def mark_output(self, output: str) -> None:
        if self.generation is None:
            return
        try:
            self.generation.update(output=output)
        except Exception:
            pass

    def mark_error(self, exc: Exception) -> None:
        if self.generation is None:
            return
        try:
            self.generation.update(level="ERROR", status_message=str(exc))
        except Exception:
            try:
                self.generation.update(output=str(exc))
            except Exception:
                pass

    def end(self) -> None:
        if self.generation is None or self._ended:
            return
        try:
            self.generation.end()
        finally:
            self._ended = True


def _build_trace_id(client: Any, metadata: dict[str, object]) -> str | None:
    workflow_id = metadata.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id:
        return None
    create_trace_id = getattr(client, "create_trace_id", None)
    if callable(create_trace_id):
        try:
            return create_trace_id(seed=workflow_id)
        except Exception:
            return None
    return None


def create_trace_id_for_workflow(workflow_id: str | None) -> str | None:
    """Build a deterministic LangFuse trace id for a workflow id when client is available."""
    if not workflow_id:
        return None
    client = get_client()
    if client is None:
        return None
    create_trace_id = getattr(client, "create_trace_id", None)
    if not callable(create_trace_id):
        return None
    try:
        return create_trace_id(seed=workflow_id)
    except Exception:
        return None


def _start_generation(
    client: Any,
    *,
    name: str,
    model: str,
    prompt: Any,
    metadata: dict[str, object],
    trace_id: str | None,
) -> Any | None:
    start_generation = getattr(client, "start_generation", None)
    if callable(start_generation):
        try:
            return start_generation(
                name=name,
                model=model,
                input=prompt,
                metadata=metadata,
                trace_id=trace_id,
            )
        except TypeError:
            return start_generation(
                name=name,
                model=model,
                input=prompt,
                metadata=metadata,
            )
        except Exception:
            return None

    start_observation = getattr(client, "start_observation", None)
    if callable(start_observation):
        try:
            kwargs: dict[str, Any] = {
                "name": name,
                "as_type": "generation",
                "model": model,
                "input": prompt,
                "metadata": metadata,
            }
            if trace_id:
                kwargs["trace_context"] = {"trace_id": trace_id}
            return start_observation(**kwargs)
        except Exception:
            return None

    return None


@contextmanager
def observe_llm_generation(
    *,
    step_name: str,
    model: str,
    prompt: str,
    metadata: dict[str, object],
) -> Iterator[ObservedGeneration]:
    client = get_client()
    if client is None:
        yield ObservedGeneration()
        return

    trace_id = _build_trace_id(client, metadata)
    generation = _start_generation(
        client,
        name=f"{step_name}.generation",
        model=model,
        prompt=prompt,
        metadata=metadata,
        trace_id=trace_id,
    )
    if generation is None:
        yield ObservedGeneration()
        return

    observed = ObservedGeneration(
        generation=generation,
        trace_id=getattr(generation, "trace_id", None) or trace_id,
        observation_id=getattr(generation, "id", None),
    )
    try:
        yield observed
    except Exception as exc:
        observed.mark_error(exc)
        observed.end()
        raise
    else:
        observed.end()


def submit_user_feedback_score(
    *,
    trace_id: str,
    observation_id: str | None,
    helpful: bool,
    comment: str | None,
) -> None:
    client = require_client()
    client.create_score(
        name="user_helpful",
        value=1.0 if helpful else 0.0,
        data_type="BOOLEAN",
        trace_id=trace_id,
        observation_id=observation_id,
        comment=comment,
    )
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()


def create_feedback_anchor_for_run(
    *,
    run_id: str,
    session_id: str,
    user_id: str,
    query: str,
    report: str,
) -> tuple[str, str | None]:
    """Create a minimal LangFuse observation so legacy runs can accept feedback."""
    client = require_client()
    trace_id = create_trace_id_for_workflow(run_id) or None
    generation = _start_generation(
        client,
        name="legacy_run.feedback_anchor",
        model="n/a",
        prompt={"query": query},
        metadata={
            "run_id": run_id,
            "session_id": session_id,
            "user_id": user_id,
            "backfilled": True,
        },
        trace_id=trace_id,
    )
    if generation is None:
        raise RuntimeError("Could not create LangFuse feedback anchor generation.")
    if report:
        generation.update(output=report[:4000])
    generation.end()
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()
    return (
        getattr(generation, "trace_id", None) or trace_id or "",
        getattr(generation, "id", None),
    )
