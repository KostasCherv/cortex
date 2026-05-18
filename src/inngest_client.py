"""Inngest client and function definitions for RAG ingestion."""

from __future__ import annotations

import os

import inngest
import inngest.fast_api

_is_production = os.environ.get("INNGEST_DEV", "").strip() != "1"

inngest_client = inngest.Inngest(
    app_id="cortex",
    is_production=_is_production,
)


@inngest_client.create_function(
    fn_id="rag-ingestion",
    trigger=inngest.TriggerEvent(event="rag/ingestion.requested"),
)
async def handle_rag_ingestion(ctx: inngest.Context) -> dict:
    job_id: str = ctx.event.data["job_id"]

    from src.rag import _get_store, _run_ingestion_job

    claimed = await _get_store().claim_rag_ingestion_job(job_id)
    if not claimed:
        return {"skipped": True, "job_id": job_id}

    await _run_ingestion_job(job_id)
    return {"done": True, "job_id": job_id}


@inngest_client.create_function(
    fn_id="research-run",
    trigger=inngest.TriggerEvent(event="research/run.requested"),
)
async def handle_research_run(ctx: inngest.Context) -> dict:
    from src.api.endpoints import _execute_research_run

    await _execute_research_run(**ctx.event.data)
    return {"done": True, "run_id": ctx.event.data.get("run_id")}


@inngest_client.create_function(
    fn_id="outbox-dispatcher",
    trigger=inngest.TriggerCron(cron="*/2 * * * *"),
)
async def dispatch_outbox_cron(ctx: inngest.Context) -> dict:
    from src.outbox import dispatch_outbox_events

    sent = await dispatch_outbox_events(limit=50)
    return {"dispatched": sent}
