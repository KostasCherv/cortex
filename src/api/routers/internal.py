"""Internal-only routes: outbox dispatch and agent-loop benchmarking."""

import logging
import secrets
import time

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from src.api.deps import _run_agent_loop
from src.api.rag_chat_helpers import build_agent_messages
from src.config import settings
from src.tools.composio_toolset import (
    get_composio_toolset_manager,
    initialize_composio_toolset,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class InternalBenchmarkAgentLoopRequest(BaseModel):
    message: str
    bind_tools: bool = True
    allow_web_search: bool = True
    rag_context: str = "Sample document context for benchmarking."
    user_memory_context: str = ""
    system_instructions: str = "Keep answers brief."


@router.post("/internal/dispatch-outbox", tags=["Internal"])
async def dispatch_outbox_endpoint(request: Request):
    """Trigger outbox dispatch manually (Cloud Scheduler fallback). Requires Authorization: Bearer <secret>."""
    from src.outbox import dispatch_outbox_events

    configured_secret = settings.internal_dispatch_secret
    if not configured_secret:
        raise HTTPException(status_code=503, detail="Internal dispatch not configured")

    token = request.headers.get("Authorization", "")
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    provided = token.removeprefix("Bearer ")
    if not secrets.compare_digest(provided, configured_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")

    sent = await dispatch_outbox_events(limit=50)
    return {"dispatched": sent}


@router.post("/internal/benchmark/agent-loop", tags=["Internal"])
async def benchmark_agent_loop_endpoint(
    body: InternalBenchmarkAgentLoopRequest,
    request: Request,
):
    configured_secret = settings.internal_dispatch_secret
    if not configured_secret:
        raise HTTPException(status_code=503, detail="Internal dispatch not configured")

    token = request.headers.get("Authorization", "")
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    provided = token.removeprefix("Bearer ")
    if not secrets.compare_digest(provided, configured_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")

    composio_manager = get_composio_toolset_manager()
    if body.bind_tools and settings.composio_enabled and not composio_manager._initialized:
        try:
            await initialize_composio_toolset()
        except Exception as exc:
            logger.warning(
                "[benchmark] Composio init failed during internal benchmark: %s", exc
            )

    composio_apps = composio_manager.get_connected_app_names()
    messages = build_agent_messages(
        system_instructions=body.system_instructions,
        history=[],
        rag_context=body.rag_context,
        user_memory_context=body.user_memory_context,
        composio_apps=composio_apps,
        normalized_message=body.message.strip(),
        bind_tools=body.bind_tools,
    )

    started = time.perf_counter()
    result = await _run_agent_loop(
        messages=messages,
        metadata={"bench": True, "internal_endpoint": True},
        bind_tools=body.bind_tools,
        allow_web_search=body.allow_web_search,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "answer": result.answer,
        "answer_chars": len(result.answer),
        "web_used": result.web_used,
        "citation_count": len(result.citations),
        "bind_tools": body.bind_tools,
        "wall_ms": round(elapsed_ms, 3),
    }
