"""FastAPI application — /health, /research (SSE), and session endpoints."""

import logging
from contextlib import asynccontextmanager

import inngest.fast_api as _inngest_fast_api
import sentry_sdk
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run startup checks, yield control, then shut down background clients."""
    await _run_startup_checks()
    yield
    await _shutdown_background_clients()


if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        # Tracing is already covered by LangFuse/LangSmith; Sentry here is
        # error-capture only.
        traces_sample_rate=0.0,
        send_default_pii=False,
        # Stack-trace locals can hold user research queries, chat messages,
        # and RAG document content — don't ship them to Sentry.
        include_local_variables=False,
    )

app = FastAPI(
    title="Cortex API",
    description="Multi-step LangGraph research orchestration with SSE streaming.",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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


async def _run_startup_checks() -> None:
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


async def _shutdown_background_clients() -> None:
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


@app.get("/health/stream", tags=["Meta"])
async def health_stream():
    """Emit one SSE event so deployments can verify streaming end to end."""

    async def _probe():
        yield 'event: ready\ndata: {"status":"ok"}\n\n'

    return StreamingResponse(
        _probe(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

