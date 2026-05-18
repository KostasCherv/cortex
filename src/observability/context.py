"""Context propagation for workflow-level tracing metadata."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_workflow_id_var: ContextVar[str] = ContextVar("workflow_id", default="")
_entrypoint_var: ContextVar[str] = ContextVar("entrypoint", default="")
_node_var: ContextVar[str] = ContextVar("node", default="")
_trace_enabled_var: ContextVar[bool] = ContextVar("trace_enabled", default=False)


@contextmanager
def workflow_context(workflow_id: str, entrypoint: str, trace_enabled: bool = False) -> Iterator[None]:
    workflow_token = _workflow_id_var.set(workflow_id)
    entrypoint_token = _entrypoint_var.set(entrypoint)
    trace_enabled_token = _trace_enabled_var.set(trace_enabled)
    try:
        yield
    finally:
        _workflow_id_var.reset(workflow_token)
        _entrypoint_var.reset(entrypoint_token)
        _trace_enabled_var.reset(trace_enabled_token)


@contextmanager
def node_context(node_name: str) -> Iterator[None]:
    token = _node_var.set(node_name)
    try:
        yield
    finally:
        _node_var.reset(token)


def get_workflow_id() -> str:
    return _workflow_id_var.get()


def get_entrypoint() -> str:
    return _entrypoint_var.get()


def get_node() -> str:
    return _node_var.get()


def is_trace_enabled() -> bool:
    return _trace_enabled_var.get()


def build_trace_metadata(extra: dict[str, object] | None = None) -> dict[str, object]:
    data: dict[str, object] = {}
    workflow_id = get_workflow_id()
    entrypoint = get_entrypoint()
    node = get_node()
    if workflow_id:
        data["workflow_id"] = workflow_id
    if entrypoint:
        data["entrypoint"] = entrypoint
    if node:
        data["node"] = node
    if extra:
        data.update(extra)
    return data


def build_trace_tags(extra: list[str] | None = None) -> list[str]:
    tags: list[str] = ["cortex"]
    entrypoint = get_entrypoint()
    node = get_node()
    if entrypoint:
        tags.append(f"entrypoint:{entrypoint}")
    if node:
        tags.append(f"node:{node}")
    if extra:
        tags.extend(extra)
    return tags
