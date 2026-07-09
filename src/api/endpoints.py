"""FastAPI application — /health, /research (SSE), and session endpoints."""

import logging

import inngest.fast_api as _inngest_fast_api
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.tools.arxiv_mcp import ensure_arxiv_mcp_available

from src.errors import CortexError
from src.config import settings
from src.cache.client import get_cache
from src.sessions import (
    ensure_store_initialized,
)
from src.tools.composio_toolset import (
    initialize_composio_toolset,
    shutdown_composio_toolset,
)
from src.inngest_client import (
    dispatch_outbox_cron,
    handle_rag_ingestion,
    handle_research_run,
    handle_user_memory_refresh,
    inngest_client,
)
from src.storage import ensure_rag_storage_ready
from src.api.routers.billing import router as billing_router
from src.api.routers.internal import router as internal_router
from src.api.routers.memory import router as memory_router
from src.api.routers.rag_agents import router as rag_agents_router
from src.api.routers.rag_chat import router as rag_chat_router
from src.api.routers.rag_resources import router as rag_resources_router
from src.api.routers.sessions import router as sessions_router

logger = logging.getLogger(__name__)
_LIVE_REPORT_FLUSH_SECONDS = 0.3


def _configure_application_logging() -> None:
    level_name = (settings.app_log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    root_logger.setLevel(level)

    package_logger = logging.getLogger("src")
    package_logger.setLevel(level)
    package_logger.propagate = True

    logger.info("[startup] Application logging configured at level=%s", level_name)


app = FastAPI(
    title="Cortex API",
    description="Multi-step LangGraph research orchestration with SSE streaming.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_inngest_fast_api.serve(
    app,
    inngest_client,
    [
        handle_rag_ingestion,
        handle_research_run,
        handle_user_memory_refresh,
        dispatch_outbox_cron,
    ],
)

app.include_router(internal_router)
app.include_router(billing_router)
app.include_router(memory_router)
app.include_router(sessions_router)
app.include_router(rag_resources_router)
app.include_router(rag_agents_router)
app.include_router(rag_chat_router)


@app.on_event("startup")
async def validate_session_store_configuration() -> None:
    """Validate critical runtime dependencies and session persistence wiring."""
    _configure_application_logging()
    if not settings.cohere_api_key:
        logger.warning(
            "[startup] Cohere reranking is disabled (COHERE_API_KEY not set)."
        )

    # Session persistence is optional for non-session routes.
    has_url = bool(settings.supabase_url)
    has_key = bool(settings.supabase_secret_key)

    if not has_url and not has_key:
        logger.info(
            "[startup] Supabase session persistence is disabled; non-session routes remain available."
        )
    elif not has_url or not has_key:
        logger.warning(
            "[startup] Supabase session persistence is partially configured; "
            "session endpoints may fail until SUPABASE_URL and SUPABASE_SECRET_KEY are both set."
        )
    else:
        ensure_store_initialized()
        try:
            await ensure_rag_storage_ready()
        except Exception as exc:
            logger.warning("[startup] RAG storage readiness check failed: %s", exc)

    cache = get_cache()
    if cache is None:
        logger.info("[startup] Redis caching is disabled (REDIS_URL not set).")
    else:
        reachable = await cache.ping()
        if reachable:
            logger.info("[startup] Redis cache connected.")
        else:
            logger.warning(
                "[startup] Redis is configured but unreachable — caching disabled for this run."
            )

    if settings.composio_enabled:
        try:
            composio_client = await initialize_composio_toolset()
            logger.info(
                "[startup] Composio MCP ready. Connected apps: %s",
                composio_client.get_connected_app_names(),
            )
        except Exception as exc:
            logger.warning(
                "[startup] Composio unavailable; tool-calling disabled for this run: %s",
                exc,
            )

    await ensure_arxiv_mcp_available()
    logger.info(
        "[startup] arxiv-mcp-server ready (storage=%s).",
        settings.arxiv_mcp_storage_path,
    )


@app.on_event("shutdown")
async def shutdown_background_clients() -> None:
    """Stop long-lived background clients gracefully."""
    await shutdown_composio_toolset()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(CortexError)
async def cortex_error_handler(request: Request, exc: CortexError) -> JSONResponse:
    raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Simple liveness probe."""
    return HealthResponse(status="ok", version="0.1.0")


