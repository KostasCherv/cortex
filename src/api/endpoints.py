"""FastAPI application — /health, /research (SSE), and session endpoints."""

import json
import logging
import re
import uuid
from typing import AsyncGenerator

import inngest.fast_api as _inngest_fast_api
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graph.graph import build_graph
from src.errors import ResearchAgentError
from src.observability import end_workflow_run, start_workflow_run
from src.config import settings
from src.auth import AuthenticatedUser, get_authenticated_user
from src.sessions import (
    Session,
    SessionRun,
    ConversationTurn,
    append_turn,
    create_session_run,
    create_session,
    generate_run_id,
    get_session,
    list_sessions,
    delete_session,
    update_session_run,
    update_session_title,
    ensure_store_initialized,
)
from src.tools.vector_store import VectorStoreManager
from src.llm.factory import get_llm
from src import outbox
from src.rag import (
    RagChatMessage,
    RagValidationError,
    append_chat_message,
    delete_chat_session as delete_rag_chat_session,
    create_agent as create_rag_agent_record,
    create_or_get_chat_session,
    create_resource_and_ingest,
    delete_resource as delete_rag_resource_record,
    get_agent_for_chat,
    get_chat_session as get_rag_chat_session,
    get_resource_status,
    link_resources as link_rag_resources,
    list_agents as list_rag_agents_records,
    list_chat_messages as list_rag_chat_messages,
    list_chat_sessions as list_rag_chat_sessions,
    list_resources as list_rag_resources_records,
    retrieve_context_for_query,
    update_chat_session_title as update_rag_chat_session_title,
    update_agent as update_rag_agent_record,
)
from src.inngest_client import handle_rag_ingestion, handle_research_run, inngest_client
from src.storage import ensure_rag_storage_ready

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Agent API",
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

_inngest_fast_api.serve(app, inngest_client, [handle_rag_ingestion, handle_research_run])


@app.on_event("startup")
async def validate_session_store_configuration() -> None:
    """Warn when session persistence is unavailable, but keep API bootable."""
    has_url = bool(settings.supabase_url)
    has_key = bool(settings.supabase_service_role_key)

    if not has_url and not has_key:
        logger.info(
            "[startup] Supabase session persistence is disabled; non-session routes remain available."
        )
        return

    if not has_url or not has_key:
        logger.warning(
            "[startup] Supabase session persistence is partially configured; "
            "session endpoints may fail until SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are both set."
        )
        return

    ensure_store_initialized()
    try:
        await ensure_rag_storage_ready()
    except Exception as exc:
        logger.warning("[startup] RAG storage readiness check failed: %s", exc)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str
    use_vector_store: bool = False


class FollowupRequest(BaseModel):
    question: str
    run_id: str | None = None


class CreateSessionRequest(BaseModel):
    query: str | None = None


class UpdateSessionTitleRequest(BaseModel):
    title: str


class HealthResponse(BaseModel):
    status: str
    version: str


class RagAgentCreateRequest(BaseModel):
    name: str
    description: str = ""
    system_instructions: str = ""
    linked_resource_ids: list[str] = []


class RagAgentUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    system_instructions: str | None = None
    linked_resource_ids: list[str] | None = None


class RagAgentLinkRequest(BaseModel):
    resource_ids: list[str]


class RagChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ResearchAgentError)
async def research_agent_error_handler(request: Request, exc: ResearchAgentError):
    raise HTTPException(status_code=500, detail=str(exc))


def _raise_rag_validation_error(exc: RagValidationError) -> None:
    status_by_code = {
        "unsupported_type": 400,
        "size_exceeded": 400,
        "workspace_limit_exceeded": 400,
        "agent_resource_limit_exceeded": 400,
        "processing_failed": 409,
        "unauthorized_linkage": 403,
    }
    raise HTTPException(
        status_code=status_by_code.get(exc.code, 400),
        detail={"code": exc.code, "message": str(exc)},
    )


def _build_rag_citations(chunks: list[dict] | None) -> list[dict]:
    """Normalize retrieved RAG chunks into the stable citation API payload."""
    if not chunks:
        return []

    citations: list[dict] = []
    for chunk in chunks:
        citations.append(
            {
                "source_title": chunk.get("source_title") or "resource",
                "source_url": chunk.get("source_url") or "",
                "chunk_id": chunk.get("chunk_id") or "",
                "text": chunk.get("text") or "",
            }
        )
    return citations


# ---------------------------------------------------------------------------
# Shared streaming logic
# ---------------------------------------------------------------------------

async def _stream_research(
    query: str,
    use_vector_store: bool,
    session: Session | None = None,
    run_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Run the research graph and stream node events as SSE.

    When ``session`` and ``run_id`` are provided the final state is recorded
    on the session and source chunks are persisted for follow-up retrieval.
    """
    graph = build_graph()
    initial_state: dict = {
        "query": query,
        "use_vector_store": use_vector_store,
        "error": None,
        "session_id": session.session_id if session else None,
        "run_id": run_id,
        "conversation_history": (
            [t.to_dict() for t in session.conversation] if session else []
        ),
    }

    _GRAPH_NODES = {
        "search_and_memory", "rerank", "summarize", "report",
        "vector_store", "abort", "empty",
    }

    with start_workflow_run(
        entrypoint="api",
        query=query,
        use_vector_store=use_vector_store,
    ) as trace_ctx:
        final_node_state: dict | None = None
        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                event_type = event["event"]
                meta = event.get("metadata", {})
                langgraph_node = meta.get("langgraph_node")

                # Forward LLM token chunks from the report node as they arrive
                if event_type == "on_chat_model_stream" and meta.get("langgraph_node") == "report":
                    chunk = event["data"].get("chunk")
                    if chunk:
                        content = chunk.content if hasattr(chunk, "content") else ""
                        token = ""
                        if isinstance(content, str):
                            token = content
                        elif isinstance(content, list):
                            token = "".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in content
                            )
                        if token:
                            yield f"data: {json.dumps({'workflow_id': trace_ctx.workflow_id, 'node': 'report_stream', 'data': {'chunk': token}})}\n\n"

                # Emit node-completion events (preserves existing SSE format).
                # Use metadata.langgraph_node — the event ``name`` may differ from graph node ids.
                elif event_type == "on_chain_end" and langgraph_node in _GRAPH_NODES:
                    node_state = event["data"].get("output", {})
                    if isinstance(node_state, dict):
                        final_node_state = node_state
                        payload = {
                            "workflow_id": trace_ctx.workflow_id,
                            "node": langgraph_node,
                            "data": {
                                k: v
                                for k, v in node_state.items()
                                if k in {"error", "report"}
                            },
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

            # Persist session state after a successful run
            if session is not None and run_id is not None and final_node_state and user_id:
                await _record_session_run(session, user_id, run_id, query, final_node_state)

            end_workflow_run(
                trace_ctx,
                status="success",
                outputs={
                    "node": "__end__",
                    "has_report": bool(final_node_state and final_node_state.get("report")),
                    "has_error": bool(final_node_state and final_node_state.get("error")),
                },
            )
            yield (
                f"data: {json.dumps({'workflow_id': trace_ctx.workflow_id, 'node': '__end__', 'data': {}})}\n\n"
            )
        except Exception as exc:
            end_workflow_run(trace_ctx, status="error", error=str(exc))
            error_payload = {
                "workflow_id": trace_ctx.workflow_id,
                "node": "__error__",
                "data": {"error": str(exc)},
            }
            yield f"data: {json.dumps(error_payload)}\n\n"


async def _record_session_run(
    session: Session,
    user_id: str,
    run_id: str,
    query: str,
    final_state: dict,
) -> None:
    """Finalize an existing run with metadata and source chunks."""

    retrieved = final_state.get("retrieved_contents") or []
    summaries = final_state.get("summaries") or []
    source_urls = [s.get("url", "") for s in retrieved if s.get("url")]

    finalized = await update_session_run(
        run_id=run_id,
        user_id=user_id,
        session_id=session.session_id,
        patch={
            "query": query,
            "source_urls": source_urls,
            "report": final_state.get("report", ""),
            "status": "completed",
            "error_details": None,
        },
    )
    if not finalized:
        raise RuntimeError(
            f"Could not finalize run '{run_id}' for session '{session.session_id}'."
        )

    # Persist source chunks for follow-up retrieval
    sources_to_chunk = retrieved if retrieved else summaries
    if sources_to_chunk:
        try:
            manager = VectorStoreManager()
            manager.save_source_chunks(
                run_id=run_id,
                session_id=session.session_id,
                sources=sources_to_chunk,
            )
        except Exception as exc:
            logger.warning("[session] could not save source chunks: %s", exc)


async def _execute_research_run(
    session_id: str,
    run_id: str,
    user_id: str,
    query: str,
    use_vector_store: bool,
) -> None:
    """Execute one research run in the background and persist terminal status."""
    logger.info(
        "[run] start run_id=%s session_id=%s user_id=%s use_vector_store=%s",
        run_id,
        session_id,
        user_id,
        use_vector_store,
    )
    session = await get_session(session_id, user_id)
    if session is None:
        logger.warning(
            "[run] abort run_id=%s session_id=%s reason=session-not-found",
            run_id,
            session_id,
        )
        await update_session_run(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
            patch={
                "status": "failed",
                "error_details": f"Session '{session_id}' not found.",
            },
        )
        return

    graph = build_graph()
    initial_state: dict = {
        "query": query,
        "use_vector_store": use_vector_store,
        "error": None,
        "session_id": session.session_id,
        "run_id": run_id,
        "conversation_history": [t.to_dict() for t in session.conversation],
    }

    with start_workflow_run(
        entrypoint="background",
        query=query,
        use_vector_store=use_vector_store,
    ) as trace_ctx:
        final_node_state: dict | None = None
        try:
            async for event in graph.astream(initial_state):
                for _node_name, node_state in event.items():
                    final_node_state = node_state

            if not final_node_state:
                raise RuntimeError("Research run produced no final state.")

            await _record_session_run(session, user_id, run_id, query, final_node_state)
            logger.info("[run] end run_id=%s status=completed", run_id)
            end_workflow_run(
                trace_ctx,
                status="success",
                outputs={
                    "node": "__end__",
                    "has_report": bool(final_node_state.get("report")),
                    "has_error": bool(final_node_state.get("error")),
                },
            )
        except Exception as exc:
            await update_session_run(
                run_id=run_id,
                user_id=user_id,
                session_id=session.session_id,
                patch={
                    "status": "failed",
                    "error_details": str(exc),
                },
            )
            logger.exception("[run] end run_id=%s status=failed error=%s", run_id, exc)
            end_workflow_run(trace_ctx, status="error", error=str(exc))
            raise


async def _generate_suggestions(query: str, answer: str, context: str) -> list[str]:
    """Generate 2-3 follow-up question suggestions based on the Q&A."""
    try:
        llm = get_llm(temperature=0.7)
        prompt = (
            f"Based on this question and answer, generate exactly 3 concise follow-up questions "
            f"a user might ask. Return ONLY a numbered list (1. ... 2. ... 3. ...), no preamble.\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer[:1000]}\n\n"
            f"Context topics: {context[:500]}"
        )
        result = await llm.ainvoke(prompt)
        content = result.content
        if not isinstance(content, str):
            content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )
        lines = content.strip().split("\n")
        suggestions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(\d+[\.\)]\s+|[-*]\s+)", "", line)
            if line:
                suggestions.append(line)
        return suggestions[:3]
    except Exception as exc:
        logger.warning("[suggestions] failed to generate follow-up suggestions: %s", exc)
        return []


async def _stream_followup(
    session: Session,
    user_id: str,
    question: str,
    run_id: str,
) -> AsyncGenerator[str, None]:
    """Retrieve run-scoped sources and stream a cited answer."""
    # Retrieve relevant source chunks
    try:
        manager = VectorStoreManager()
        chunks = manager.search_run_sources(question, run_id=run_id, n_results=5)
    except Exception as exc:
        logger.warning("[followup] source retrieval failed: %s", exc)
        chunks = []

    context_block = "\n\n".join(
        f"[Source: {c['source_title']} ({c['source_url']})]\n{c['text']}"
        for c in chunks
    )

    history_block = "\n".join(
        f"{t.role.upper()}: {t.content}" for t in session.conversation[-6:]
    )

    prompt = (
        f"You are a research assistant answering a follow-up question.\n\n"
        f"Conversation so far:\n{history_block}\n\n"
        f"Retrieved source passages:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely based on the retrieved passages. "
        f"Do NOT append a citations list at the end — citations are handled separately."
    )

    llm = get_llm(temperature=0.2)
    full_answer = ""

    try:
        async for chunk in llm.astream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += token
            yield f"data: {json.dumps({'type': 'chunk', 'text': token})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        return

    # Build citations from retrieved chunks (deduplicate by URL)
    seen: set[str] = set()
    citations: list[dict] = []
    for c in chunks:
        url = c["source_url"]
        if url and url not in seen:
            seen.add(url)
            citations.append({"source_url": url, "source_title": c["source_title"]})

    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

    # Generate suggestions before persisting so they are stored with the turn
    suggestions = await _generate_suggestions(question, full_answer, context_block)

    # Record turns in conversation history
    user_turn = ConversationTurn(role="user", content=question, run_id=run_id)
    assistant_turn = ConversationTurn(
        role="assistant",
        content=full_answer,
        run_id=run_id,
        citations=citations,
        suggestions=suggestions,
    )
    session.conversation.append(user_turn)
    session.conversation.append(assistant_turn)
    await append_turn(user_id=user_id, session_id=session.session_id, turn=user_turn)
    await append_turn(user_id=user_id, session_id=session.session_id, turn=assistant_turn)

    if suggestions:
        yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Simple liveness probe."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/research", tags=["Research"])
async def research(body: ResearchRequest):
    """Run the research pipeline and stream progress via Server-Sent Events."""
    return StreamingResponse(
        _stream_research(body.query, body.use_vector_store),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


def _generate_session_title(query: str | None) -> str:
    """Generate a short session title from the initial query using the LLM."""
    from src.sessions import suggest_session_title

    fallback = suggest_session_title(query)
    if not query or not query.strip():
        return fallback

    prompt = (
        "Create a concise title (max 6 words) for this research session.\n"
        "Return plain text only, no quotes, no punctuation at the end.\n"
        f"Query: {query.strip()}"
    )
    try:
        llm = get_llm(temperature=0.1)
        result = llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        candidate = " ".join(text.strip().split())
        if not candidate:
            return fallback
        words = candidate.split(" ")
        if len(words) > 6:
            candidate = " ".join(words[:6])
        return candidate
    except Exception:
        return fallback

@app.post("/sessions", tags=["Sessions"])
async def create_session_endpoint(
    body: CreateSessionRequest = CreateSessionRequest(),
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Create a new research session."""
    title = _generate_session_title(body.query)
    session = await create_session(current_user.user_id, title=title)
    return {"session_id": session.session_id, "title": session.title, "created_at": session.created_at}


@app.get("/sessions", tags=["Sessions"])
async def list_sessions_endpoint(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """List session summaries for the authenticated user."""
    return {"sessions": await list_sessions(current_user.user_id)}


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session_endpoint(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Return session state including runs and conversation history."""
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.to_dict()


@app.patch("/sessions/{session_id}", tags=["Sessions"])
async def update_session_title_endpoint(
    session_id: str,
    body: UpdateSessionTitleRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Update a session title."""
    title = " ".join(body.title.strip().split())
    if not title:
        raise HTTPException(status_code=400, detail="Session title cannot be empty.")
    if len(title) > 120:
        raise HTTPException(status_code=400, detail="Session title is too long.")

    updated = await update_session_title(
        current_user.user_id,
        session_id=session_id,
        title=title,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, "title": title}


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session_endpoint(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Delete a session owned by the authenticated user."""
    deleted = await delete_session(
        current_user.user_id,
        session_id=session_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, "deleted": True}


@app.post("/sessions/{session_id}/research", tags=["Sessions"])
async def session_research(
    background_tasks: BackgroundTasks,
    session_id: str,
    body: ResearchRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Queue background research within a session and return run metadata."""
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    run_id = generate_run_id()
    pending_run = SessionRun(
        run_id=run_id,
        query=body.query,
        source_urls=[],
        report="",
        status="running",
        error_details=None,
    )
    await create_session_run(
        user_id=current_user.user_id,
        session_id=session.session_id,
        run=pending_run,
    )
    await outbox.enqueue_event(
        "research/run.requested",
        {
            "session_id": session.session_id,
            "run_id": run_id,
            "user_id": current_user.user_id,
            "query": body.query,
            "use_vector_store": body.use_vector_store,
        },
    )
    background_tasks.add_task(outbox.dispatch_outbox_events, limit=10)
    return {"run_id": run_id, "status": "running"}


@app.post("/sessions/{session_id}/followup", tags=["Sessions"])
async def session_followup(
    session_id: str,
    body: FollowupRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Ask a follow-up question grounded to a session's source material."""
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    # Resolve which run to ground the follow-up against
    if body.run_id:
        run = session.get_run(body.run_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{body.run_id}' not found in session '{session_id}'.",
            )
        run_id = body.run_id
    else:
        latest = session.latest_run()
        if latest is None:
            raise HTTPException(
                status_code=400,
                detail="No research runs found in this session. Run /research first.",
            )
        run_id = latest.run_id

    return StreamingResponse(
        _stream_followup(session, current_user.user_id, body.question, run_id=run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# RAG Agent endpoints
# ---------------------------------------------------------------------------


@app.post("/api/rag/resources/upload", tags=["RAG"])
async def rag_upload_resource(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        resource, job = await create_resource_and_ingest(file, current_user.user_id)
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    background_tasks.add_task(outbox.dispatch_outbox_events, limit=10)
    return {
        "resource": resource.to_dict(),
        "job": job.to_dict(),
    }


@app.get("/api/rag/resources", tags=["RAG"])
async def rag_list_resources(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    resources = await list_rag_resources_records(current_user.user_id)
    return {"resources": [r.to_dict() for r in resources]}


@app.delete("/api/rag/resources/{resource_id}", tags=["RAG"])
async def rag_delete_resource(
    resource_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    deleted = await delete_rag_resource_record(resource_id, current_user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Resource '{resource_id}' not found.")
    return {"resource_id": resource_id, "deleted": True}


@app.get("/api/rag/resources/{resource_id}/status", tags=["RAG"])
async def rag_resource_status(
    resource_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    status_payload = await get_resource_status(resource_id, current_user.user_id)
    if not status_payload:
        raise HTTPException(status_code=404, detail=f"Resource '{resource_id}' not found.")
    return status_payload


@app.post("/api/rag/agents", tags=["RAG"])
async def rag_create_agent(
    body: RagAgentCreateRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        agent = await create_rag_agent_record(
            user_id=current_user.user_id,
            name=body.name.strip(),
            description=body.description.strip(),
            system_instructions=body.system_instructions.strip(),
            linked_resource_ids=body.linked_resource_ids,
        )
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    return {"agent": agent.to_dict()}


@app.get("/api/rag/agents", tags=["RAG"])
async def rag_list_agents(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agents = await list_rag_agents_records(current_user.user_id)
    return {"agents": [a.to_dict() for a in agents]}


@app.patch("/api/rag/agents/{agent_id}", tags=["RAG"])
async def rag_update_agent(
    agent_id: str,
    body: RagAgentUpdateRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        updated = await update_rag_agent_record(
            agent_id=agent_id,
            user_id=current_user.user_id,
            name=body.name.strip() if body.name is not None else None,
            description=body.description.strip() if body.description is not None else None,
            system_instructions=(
                body.system_instructions.strip()
                if body.system_instructions is not None
                else None
            ),
            linked_resource_ids=body.linked_resource_ids,
        )
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {"agent": updated.to_dict()}


@app.post("/api/rag/agents/{agent_id}/resources:link", tags=["RAG"])
async def rag_link_resources(
    agent_id: str,
    body: RagAgentLinkRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        agent = await link_rag_resources(
            agent_id=agent_id,
            user_id=current_user.user_id,
            resource_ids=body.resource_ids,
        )
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {"agent": agent.to_dict()}


@app.post("/api/rag/agents/{agent_id}/chat", tags=["RAG"])
async def rag_chat_with_agent(
    agent_id: str,
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    agent, resource_ids = agent_bundle
    if not resource_ids:
        raise HTTPException(
            status_code=409,
            detail={"code": "processing_failed", "message": "Agent has no linked ready resources."},
        )

    rag_context = await retrieve_context_for_query(
        agent_id=agent_id,
        user_id=current_user.user_id,
        resource_ids=resource_ids,
        question=normalized_message,
    )

    chat_session_id = await create_or_get_chat_session(
        user_id=current_user.user_id,
        agent_id=agent_id,
        session_id=body.session_id,
        initial_message=normalized_message,
    )
    history = await list_rag_chat_messages(chat_session_id, current_user.user_id)

    history_block = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in history[-10:]
    )
    prompt = (
        "You are a custom RAG assistant.\n\n"
        f"System instructions:\n{agent.system_instructions or 'None'}\n\n"
        f"Conversation history:\n{history_block or 'None'}\n\n"
        f"Retrieved context:\n{rag_context.context or 'No context returned.'}\n\n"
        f"User question:\n{normalized_message}\n\n"
        "Answer clearly and stay grounded in the retrieved context."
    )
    llm = get_llm(temperature=0.2)
    result = await llm.ainvoke(prompt)
    content = result.content
    if not isinstance(content, str):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    answer = content.strip()

    user_msg = RagChatMessage(
        message_id=str(uuid.uuid4()),
        session_id=chat_session_id,
        agent_id=agent_id,
        owner_id=current_user.user_id,
        role="user",
        content=normalized_message,
    )
    citations = _build_rag_citations(rag_context.chunks)
    assistant_msg = RagChatMessage(
        message_id=str(uuid.uuid4()),
        session_id=chat_session_id,
        agent_id=agent_id,
        owner_id=current_user.user_id,
        role="assistant",
        content=answer,
        citations=citations,
    )
    await append_chat_message(user_msg)
    await append_chat_message(assistant_msg)

    updated_history = await list_rag_chat_messages(chat_session_id, current_user.user_id)
    return {
        "session_id": chat_session_id,
        "agent_id": agent_id,
        "reply": assistant_msg.to_dict(),
        "messages": [m.to_dict() for m in updated_history],
    }


@app.post("/api/rag/agents/{agent_id}/chat/stream", tags=["RAG"])
async def rag_chat_with_agent_stream(
    agent_id: str,
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    agent, resource_ids = agent_bundle
    if not resource_ids:
        raise HTTPException(
            status_code=409,
            detail={"code": "processing_failed", "message": "Agent has no linked ready resources."},
        )

    rag_context = await retrieve_context_for_query(
        agent_id=agent_id,
        user_id=current_user.user_id,
        resource_ids=resource_ids,
        question=normalized_message,
    )

    chat_session_id = await create_or_get_chat_session(
        user_id=current_user.user_id,
        agent_id=agent_id,
        session_id=body.session_id,
        initial_message=normalized_message,
    )
    history = await list_rag_chat_messages(chat_session_id, current_user.user_id)
    history_block = "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-10:])
    prompt = (
        "You are a custom RAG assistant.\n\n"
        f"System instructions:\n{agent.system_instructions or 'None'}\n\n"
        f"Conversation history:\n{history_block or 'None'}\n\n"
        f"Retrieved context:\n{rag_context.context or 'No context returned.'}\n\n"
        f"User question:\n{normalized_message}\n\n"
        "Answer clearly and stay grounded in the retrieved context."
    )

    citations = _build_rag_citations(rag_context.chunks)

    async def _stream_chat() -> AsyncGenerator[str, None]:
        llm = get_llm(temperature=0.2)
        answer_parts: list[str] = []
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': chat_session_id})}\n\n"
            async for chunk in llm.astream(prompt):
                content = chunk.content if hasattr(chunk, "content") else chunk
                token = ""
                if isinstance(content, str):
                    token = content
                elif isinstance(content, list):
                    token = "".join(
                        part if isinstance(part, str) else part.get("text", "")
                        for part in content
                    )
                elif content is not None:
                    token = str(content)

                if token:
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': token})}\n\n"

            answer = "".join(answer_parts).strip()
            user_msg = RagChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=chat_session_id,
                agent_id=agent_id,
                owner_id=current_user.user_id,
                role="user",
                content=normalized_message,
            )
            assistant_msg = RagChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=chat_session_id,
                agent_id=agent_id,
                owner_id=current_user.user_id,
                role="assistant",
                content=answer,
                citations=citations,
            )
            await append_chat_message(user_msg)
            await append_chat_message(assistant_msg)
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        _stream_chat(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/rag/agents/{agent_id}/chat/sessions", tags=["RAG"])
async def list_rag_agent_chat_sessions(
    agent_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    sessions = await list_rag_chat_sessions(agent_id, current_user.user_id)
    return {"sessions": sessions}


@app.get("/api/rag/agents/{agent_id}/chat/sessions/{session_id}/messages", tags=["RAG"])
async def list_rag_agent_chat_session_messages(
    agent_id: str,
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    session = await get_rag_chat_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=current_user.user_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")

    messages = await list_rag_chat_messages(session_id, current_user.user_id)
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "messages": [m.to_dict() for m in messages],
    }


@app.patch("/api/rag/agents/{agent_id}/chat/sessions/{session_id}", tags=["RAG"])
async def update_rag_agent_chat_session_title(
    agent_id: str,
    session_id: str,
    body: UpdateSessionTitleRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    title = " ".join(body.title.strip().split())
    if not title:
        raise HTTPException(status_code=400, detail="Session title cannot be empty.")
    if len(title) > 120:
        raise HTTPException(status_code=400, detail="Session title is too long.")

    updated = await update_rag_chat_session_title(
        session_id=session_id,
        agent_id=agent_id,
        user_id=current_user.user_id,
        title=title,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    return {"session_id": session_id, "title": title}


@app.delete("/api/rag/agents/{agent_id}/chat/sessions/{session_id}", tags=["RAG"])
async def delete_rag_agent_chat_session(
    agent_id: str,
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    deleted = await delete_rag_chat_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=current_user.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    return {"session_id": session_id, "deleted": True}
