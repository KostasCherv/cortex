"""Inngest client and function definitions for RAG ingestion."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import inngest
import inngest.fast_api


def _apply_bundled_inngest_config() -> None:
    """Expose bundled credentials under the names expected by the Inngest SDK."""
    raw = os.environ.get("INNGEST_CONFIG_JSON", "").strip()
    if not raw:
        return
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("INNGEST_CONFIG_JSON must be valid JSON.") from exc
    if not isinstance(config, dict):
        raise ValueError("INNGEST_CONFIG_JSON must be a JSON object.")
    for name in ("inngest_event_key", "inngest_signing_key"):
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"INNGEST_CONFIG_JSON must include a non-empty {name} value.")
        os.environ[name.upper()] = value


_apply_bundled_inngest_config()

_is_production = os.environ.get("INNGEST_DEV", "").strip() != "1"

inngest_client = inngest.Inngest(
    app_id="cortex",
    is_production=_is_production,
)


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Inngest event field {key!r} must be a non-empty string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"Inngest event field {key!r} must be a string or null")


@inngest_client.create_function(
    fn_id="rag-ingestion",
    trigger=inngest.TriggerEvent(event="rag/ingestion.requested"),
)
async def handle_rag_ingestion(ctx: inngest.Context) -> dict:
    data = ctx.event.data
    job_id = _required_string(data, "job_id")

    from src.db.provider import get_session_store
    from src.rag import _run_ingestion_job

    claimed = await get_session_store().claim_rag_ingestion_job(job_id)
    if not claimed:
        return {"skipped": True, "job_id": job_id}

    await _run_ingestion_job(job_id)
    return {"done": True, "job_id": job_id}


@inngest_client.create_function(
    fn_id="research-run",
    trigger=inngest.TriggerEvent(event="research/run.requested"),
)
async def handle_research_run(ctx: inngest.Context) -> dict:
    from src.api.routers.sessions import _execute_research_run

    data = ctx.event.data
    await _execute_research_run(
        session_id=_required_string(data, "session_id"),
        run_id=_required_string(data, "run_id"),
        user_id=_required_string(data, "user_id"),
        query=_required_string(data, "query"),
    )
    return {"done": True, "run_id": data.get("run_id")}


@inngest_client.create_function(
    fn_id="user-memory-refresh",
    trigger=inngest.TriggerEvent(event="memory/refresh.requested"),
)
async def handle_user_memory_refresh(ctx: inngest.Context) -> dict:
    from src.user_memory import refresh_user_memory

    data = ctx.event.data
    result = await refresh_user_memory(
        user_id=_required_string(data, "user_id"),
        source_mode=_required_string(data, "source_mode"),
        source_session_id=_required_string(data, "source_session_id"),
        user_message=_required_string(data, "user_message"),
        assistant_message=_required_string(data, "assistant_message"),
        event_key=_required_string(data, "event_key"),
        workspace_id=_optional_string(data, "workspace_id"),
        source_user_message_id=_optional_string(data, "source_user_message_id"),
        source_assistant_message_id=_optional_string(data, "source_assistant_message_id"),
    )
    return {"done": True, "result": result, "event_key": data.get("event_key")}


@inngest_client.create_function(
    fn_id="outbox-dispatcher",
    trigger=inngest.TriggerCron(cron="*/2 * * * *"),
)
async def dispatch_outbox_cron(ctx: inngest.Context) -> dict:
    from src.outbox import dispatch_outbox_events

    sent = await dispatch_outbox_events(limit=50)
    return {"dispatched": sent}
