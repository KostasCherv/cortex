"""Research session routes: CRUD, background research runs, streaming, feedback, follow-up."""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src import outbox
from src.api.deps import (
    UpdateSessionTitleRequest,
    _build_rag_citations,
    _coerce_agent_loop_result,
    _consume_usage_or_429,
    _generate_suggestions,
    _merge_citations,
    _run_agent_loop,
)
from src.api.rag_chat_helpers import build_agent_messages
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing import UsageIncrement
from src.config import settings
from src.graph.graph import build_graph
from src.llm.factory import get_llm
from src.llm.text_utils import extract_llm_text
from src.observability import end_workflow_run, start_workflow_run
from src.observability.langfuse import (
    create_feedback_anchor_for_run,
    create_trace_id_for_workflow,
    submit_user_feedback_score,
)
from src.sessions import (
    ConversationTurn,
    Session,
    SessionRun,
    append_turn,
    create_session,
    create_session_run,
    delete_session,
    generate_run_id,
    get_session,
    get_session_run,
    list_sessions,
    update_session_run,
    update_session_title,
)
from src.tools.composio_toolset import get_composio_toolset_manager
from src.tools.neo4j_graph_store import Neo4jGraphStore
from src.tools.reranker import rerank_chunks
from src.user_memory import enqueue_memory_refresh, get_user_memory_prompt_block

router = APIRouter()

logger = logging.getLogger(__name__)

_LIVE_REPORT_FLUSH_SECONDS = 0.3


class ResearchRequest(BaseModel):
    query: str


class FollowupRequest(BaseModel):
    question: str
    run_id: str | None = None


class CreateSessionRequest(BaseModel):
    query: str | None = None


class RunFeedbackRequest(BaseModel):
    helpful: bool
    comment: str | None = None


# ---------------------------------------------------------------------------
# Shared streaming logic
# ---------------------------------------------------------------------------


async def _record_session_run(
    session: Session,
    user_id: str,
    run_id: str,
    query: str,
    final_state: dict,
) -> None:
    """Finalize an existing run with metadata."""

    retrieved = final_state.get("retrieved_contents") or []
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
            "langfuse_trace_id": final_state.get("langfuse_trace_id"),
            "langfuse_observation_id": final_state.get("langfuse_observation_id"),
        },
    )
    if not finalized:
        raise RuntimeError(
            f"Could not finalize run '{run_id}' for session '{session.session_id}'."
        )


async def _persist_graph_artifacts_after_run(
    *,
    session_id: str,
    user_id: str,
    run_id: str,
    query: str,
    retrieved: list[dict],
    report_text: str,
) -> None:
    """Best-effort Neo4j persistence after run completion.

    This intentionally runs outside the critical path of run status/tracing finalization.
    """
    graph_store = Neo4jGraphStore()
    workspace_id = user_id
    for idx, source in enumerate(retrieved):
        text = str(source.get("raw_text", "")).strip()
        url = str(source.get("url", "")).strip()
        title = str(source.get("title", "")).strip() or url or f"source-{idx + 1}"
        if not text:
            continue
        document_id = f"run:{run_id}:source:{idx}"
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    graph_store.ingest_document,
                    document_id=document_id,
                    source_type="web_run",
                    owner_id=user_id,
                    workspace_id=workspace_id,
                    title=title,
                    source_url=url,
                    text=text,
                    session_id=session_id,
                    run_id=run_id,
                ),
                timeout=20.0,
            )
        except Exception as exc:
            logger.warning(
                "[session] could not persist web source in graph store: %s", exc
            )

    if report_text:
        report_doc_id = f"run:{run_id}:report"
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    graph_store.ingest_document,
                    document_id=report_doc_id,
                    source_type="report",
                    owner_id=user_id,
                    workspace_id=workspace_id,
                    title=f"Report: {query[:120]}",
                    source_url="",
                    text=report_text,
                    session_id=session_id,
                    run_id=run_id,
                ),
                timeout=20.0,
            )
        except Exception as exc:
            logger.warning("[session] could not persist report in graph store: %s", exc)


async def _execute_research_run(
    session_id: str,
    run_id: str,
    user_id: str,
    query: str,
) -> None:
    """Execute one research run in the background and persist terminal status."""
    logger.info(
        "[run] start run_id=%s session_id=%s user_id=%s",
        run_id,
        session_id,
        user_id,
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
        "error": None,
        "session_id": session.session_id,
        "run_id": run_id,
        "user_id": user_id,
        "conversation_history": [t.to_dict() for t in session.conversation],
        "user_memory_context": await get_user_memory_prompt_block(user_id, query),
    }

    graph_nodes = {
        "search_and_memory",
        "rerank",
        "summarize",
        "report",
        "abort",
        "empty",
    }
    partial_report = ""
    last_report_flush_at = 0.0

    async def _flush_live_state(
        *,
        node: str | None = None,
        force_report_flush: bool = False,
    ) -> None:
        nonlocal last_report_flush_at
        patch: dict[str, str] = {"latest_event_at": datetime.now(UTC).isoformat()}
        if node is not None:
            patch["latest_node"] = node
        if force_report_flush:
            patch["partial_report"] = partial_report
            last_report_flush_at = time.monotonic()
        try:
            await update_session_run(
                run_id=run_id,
                user_id=user_id,
                session_id=session.session_id,
                patch=patch,
            )
        except Exception as exc:
            # Live progress is best-effort and must not abort the research run.
            logger.warning(
                "[run] live-state update failed run_id=%s session_id=%s node=%s error=%s",
                run_id,
                session.session_id,
                node,
                exc,
            )

    with start_workflow_run(
        entrypoint="background",
        query=query,
    ) as trace_ctx:
        final_node_state: dict | None = None
        try:
            await _flush_live_state(node="queued", force_report_flush=True)
            async for event in graph.astream_events(initial_state, version="v2"):
                event_type = event.get("event", "")
                meta = event.get("metadata", {})
                langgraph_node = meta.get("langgraph_node")

                if event_type == "on_chain_start" and langgraph_node in graph_nodes:
                    await _flush_live_state(node=str(langgraph_node))

                elif event_type == "on_chain_end" and langgraph_node in graph_nodes:
                    node_state = event.get("data", {}).get("output", {})
                    if isinstance(node_state, dict):
                        final_node_state = node_state
                    await _flush_live_state(node=str(langgraph_node))

                elif (
                    event_type == "on_chat_model_stream" and langgraph_node == "report"
                ):
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue
                    content = chunk.content if hasattr(chunk, "content") else ""
                    token = (
                        ""  # nosec B105 — local accumulator variable, not a password
                    )
                    if isinstance(content, str):
                        token = content
                    elif isinstance(content, list):
                        token = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    if token:
                        partial_report += token
                        now = time.monotonic()
                        if (now - last_report_flush_at) >= _LIVE_REPORT_FLUSH_SECONDS:
                            await _flush_live_state(force_report_flush=True)

            if not final_node_state:
                raise RuntimeError("Research run produced no final state.")

            if not final_node_state.get("langfuse_trace_id"):
                fallback_trace_id = create_trace_id_for_workflow(trace_ctx.workflow_id)
                if fallback_trace_id:
                    final_node_state["langfuse_trace_id"] = fallback_trace_id

            await _flush_live_state(node="report", force_report_flush=True)
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
            # Neo4j persistence is intentionally decoupled from run completion/tracing.
            asyncio.create_task(
                _persist_graph_artifacts_after_run(
                    session_id=session.session_id,
                    user_id=user_id,
                    run_id=run_id,
                    query=query,
                    retrieved=final_node_state.get("retrieved_contents") or [],
                    report_text=str(final_node_state.get("report", "")).strip(),
                )
            )
        except Exception as exc:
            try:
                await update_session_run(
                    run_id=run_id,
                    user_id=user_id,
                    session_id=session.session_id,
                    patch={
                        "status": "failed",
                        "error_details": str(exc),
                        "latest_node": "abort",
                        "latest_event_at": datetime.now(UTC).isoformat(),
                    },
                )
            except Exception as persist_exc:
                logger.exception(
                    "[run] failed to persist terminal failure run_id=%s error=%s persist_error=%s",
                    run_id,
                    exc,
                    persist_exc,
                )
            logger.exception("[run] end run_id=%s status=failed error=%s", run_id, exc)
            end_workflow_run(trace_ctx, status="error", error=str(exc))
            raise
        finally:
            if not trace_ctx.ended:
                end_workflow_run(
                    trace_ctx,
                    status="error",
                    error="background run exited before workflow trace was explicitly ended",
                )


async def _stream_session_run(
    *,
    session_id: str,
    run_id: str,
    user_id: str,
) -> AsyncGenerator[str, None]:
    last_node: str | None = None
    last_partial_len = 0
    last_event_at: str | None = None
    while True:
        try:
            run = await get_session_run(
                run_id=run_id, user_id=user_id, session_id=session_id
            )
        except Exception as exc:
            error_payload = {
                "type": "error",
                "error": f"Could not refresh run state: {exc}",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            return
        if run is None:
            error_payload = {"type": "error", "error": f"Run '{run_id}' not found."}
            yield f"data: {json.dumps(error_payload)}\n\n"
            return

        if run.latest_node != last_node or run.latest_event_at != last_event_at:
            progress_payload = {
                "type": "progress",
                "node": run.latest_node,
                "status": run.status,
                "updated_at": run.latest_event_at,
            }
            yield f"data: {json.dumps(progress_payload)}\n\n"
            last_node = run.latest_node
            last_event_at = run.latest_event_at

        partial_report = run.partial_report or ""
        if len(partial_report) > last_partial_len:
            chunk = partial_report[last_partial_len:]
            chunk_payload = {"type": "report_chunk", "text": chunk}
            yield f"data: {json.dumps(chunk_payload)}\n\n"
            last_partial_len = len(partial_report)

        if run.status in {"completed", "failed"}:
            terminal_payload = (
                {"type": "done"}
                if run.status == "completed"
                else {"type": "error", "error": run.error_details or "Research failed."}
            )
            yield f"data: {json.dumps(terminal_payload)}\n\n"
            return

        await asyncio.sleep(_LIVE_REPORT_FLUSH_SECONDS)


def _build_followup_report_context(run: SessionRun | None, run_id: str) -> str:
    if run is None:
        return (
            f"No stored report context found for run '{run_id}'. "
            "Use conversation history and retrieved sources."
        )

    sections: list[str] = []

    report_text = (run.report or "").strip()
    if report_text:
        sections.append(f"Report findings:\n{report_text}")

    query_text = (run.query or "").strip()
    if query_text:
        sections.append(f"Original research question:\n{query_text}")

    source_urls = [url.strip() for url in (run.source_urls or []) if str(url).strip()]
    if source_urls:
        bullet_urls = "\n".join(f"- {url}" for url in source_urls)
        sections.append(f"Report source URLs:\n{bullet_urls}")

    if sections:
        return "\n\n".join(sections)

    return (
        f"Run '{run_id}' has no stored report content, question, or source URLs. "
        "Use conversation history and retrieved sources."
    )


async def _stream_followup(
    session: Session,
    user_id: str,
    question: str,
    run_id: str,
) -> AsyncGenerator[str, None]:
    """Retrieve run-scoped sources and stream a cited answer."""
    # Retrieve run-scoped graph chunks with local entity expansion.
    try:
        graph_store = Neo4jGraphStore()
        result = await asyncio.to_thread(
            graph_store.query_context,
            query=question,
            owner_id=user_id,
            workspace_id=user_id,
            run_id=run_id,
            top_k=5,
        )
        chunks = await asyncio.to_thread(
            rerank_chunks,
            query=question,
            chunks=result.chunks,
        )
    except Exception as exc:
        logger.warning("[followup] source retrieval failed: %s", exc)
        chunks = []

    context_block = "\n\n".join(
        f"[Source: {c.get('source_title', '')} ({c.get('source_url', '')})]\n"
        f"{c.get('text', '')}"
        for c in chunks
    )

    report = session.get_run(run_id)
    report_block = _build_followup_report_context(report, run_id)
    user_memory_context = await get_user_memory_prompt_block(user_id, question)

    rag_combined = (
        f"{context_block}\n\n{report_block}" if report_block else context_block
    )
    messages = build_agent_messages(
        system_instructions="You are a research assistant. Answer using the provided context and sources.",
        history=list(session.conversation[-6:]),
        rag_context=rag_combined,
        user_memory_context=user_memory_context,
        composio_apps=get_composio_toolset_manager().get_connected_app_names(),
        normalized_message=question,
    )

    event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def on_event(event: dict) -> None:
        await event_queue.put(event)

    loop_task = asyncio.create_task(
        _run_agent_loop(
            messages=messages,
            metadata={"user_id": user_id, "run_id": run_id},
            on_event=on_event,
            stream_answer_chunks=True,
        )
    )
    while not loop_task.done():
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
            yield f"data: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            pass
    while not event_queue.empty():
        event = event_queue.get_nowait()
        yield f"data: {json.dumps(event)}\n\n"
    try:
        loop_result = _coerce_agent_loop_result(loop_task.result())
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        return

    if loop_result.web_used:
        yield f"data: {json.dumps({'type': 'web_used', 'provider': settings.web_search_provider})}\n\n"
    if not loop_result.streamed_answer:
        yield f"data: {json.dumps({'type': 'chunk', 'text': loop_result.answer})}\n\n"

    citations = _merge_citations(_build_rag_citations(chunks), loop_result.citations)
    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

    # Generate suggestions before persisting so they are stored with the turn
    suggestions = await _generate_suggestions(
        question, loop_result.answer, rag_combined
    )

    # Record turns in conversation history
    user_turn = ConversationTurn(role="user", content=question, run_id=run_id)
    assistant_turn = ConversationTurn(
        role="assistant",
        content=loop_result.answer,
        run_id=run_id,
        citations=citations,
        suggestions=suggestions,
    )
    session.conversation.append(user_turn)
    session.conversation.append(assistant_turn)
    await append_turn(user_id=user_id, session_id=session.session_id, turn=user_turn)
    await append_turn(
        user_id=user_id, session_id=session.session_id, turn=assistant_turn
    )
    await enqueue_memory_refresh(
        user_id=user_id,
        source_mode="research",
        source_session_id=session.session_id,
        user_message=question,
        assistant_message=loop_result.answer,
    )
    asyncio.create_task(outbox.dispatch_outbox_events(limit=10))

    if suggestions:
        yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _generate_session_title(query: str | None) -> str:
    """Generate a short session title from the initial query using the LLM."""
    from src.sessions import suggest_session_title

    fallback = suggest_session_title(query)
    if not query or not query.strip():
        return fallback

    prompt = (
        "Create a concise title (max 4 words) for this research session.\n"
        "Return plain text only, no quotes, no punctuation at the end.\n"
        f"Query: {query.strip()}"
    )
    try:
        llm = get_llm(temperature=0.1)
        result = llm.invoke(prompt)
        text = extract_llm_text(result)
        candidate = " ".join(text.strip().split())
        if not candidate:
            return fallback
        words = candidate.split(" ")
        if len(words) > 6:
            candidate = " ".join(words[:6])
        return candidate
    except Exception:
        return fallback


@router.post("/sessions", tags=["Sessions"])
async def create_session_endpoint(
    body: CreateSessionRequest = CreateSessionRequest(),
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Create a new research session."""
    title = _generate_session_title(body.query)
    session = await create_session(current_user.user_id, title=title)
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
    }


@router.get("/sessions", tags=["Sessions"])
async def list_sessions_endpoint(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """List session summaries for the authenticated user."""
    return {"sessions": await list_sessions(current_user.user_id)}


@router.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session_endpoint(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Return session state including runs and conversation history."""
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )
    return session.to_dict()


@router.patch("/sessions/{session_id}", tags=["Sessions"])
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
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )
    return {"session_id": session_id, "title": title}


@router.delete("/sessions/{session_id}", tags=["Sessions"])
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
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )
    return {"session_id": session_id, "deleted": True}


@router.post("/sessions/{session_id}/research", tags=["Sessions"])
async def session_research(
    background_tasks: BackgroundTasks,
    session_id: str,
    body: ResearchRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Queue background research within a session and return run metadata."""
    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(research_queries=1, total_questions=1),
    )

    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )

    run_id = generate_run_id()
    pending_run = SessionRun(
        run_id=run_id,
        query=body.query,
        source_urls=[],
        report="",
        status="running",
        error_details=None,
        latest_node="queued",
        latest_event_at=datetime.now(UTC).isoformat(),
        partial_report="",
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
        },
    )
    background_tasks.add_task(outbox.dispatch_outbox_events, limit=10)
    return {"run_id": run_id, "status": "running"}


@router.get("/sessions/{session_id}/runs/{run_id}/stream", tags=["Sessions"])
async def stream_session_run(
    session_id: str,
    run_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )
    run = session.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in session '{session_id}'.",
        )
    return StreamingResponse(
        _stream_session_run(
            session_id=session_id,
            run_id=run_id,
            user_id=current_user.user_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/runs/{run_id}/feedback", tags=["Sessions"])
async def submit_run_feedback(
    session_id: str,
    run_id: str,
    body: RunFeedbackRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Record simple user feedback for a completed run in LangFuse."""
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )

    run = session.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in session '{session_id}'.",
        )

    if run.feedback_submitted_at:
        raise HTTPException(
            status_code=409, detail="Feedback has already been submitted for this run."
        )
    if run.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Feedback can only be submitted for completed runs.",
        )
    trace_id = run.langfuse_trace_id
    observation_id = run.langfuse_observation_id
    if not trace_id:
        try:
            trace_id, observation_id = create_feedback_anchor_for_run(
                run_id=run.run_id,
                session_id=session_id,
                user_id=current_user.user_id,
                query=run.query,
                report=run.report,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not link run to LangFuse: {exc}"
            )
        if not trace_id:
            raise HTTPException(
                status_code=502, detail="Could not link run to LangFuse."
            )
        linkage_updated = await update_session_run(
            run_id=run_id,
            user_id=current_user.user_id,
            session_id=session_id,
            patch={
                "langfuse_trace_id": trace_id,
                "langfuse_observation_id": observation_id,
            },
        )
        if not linkage_updated:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{run_id}' not found in session '{session_id}'.",
            )

    comment: str | None = None
    if body.comment is not None:
        trimmed = " ".join(body.comment.strip().split())
        if not trimmed:
            raise HTTPException(
                status_code=400, detail="Feedback comment cannot be empty."
            )
        if len(trimmed) > 500:
            raise HTTPException(status_code=400, detail="Feedback comment is too long.")
        comment = trimmed

    try:
        submit_user_feedback_score(
            trace_id=trace_id,
            observation_id=observation_id,
            helpful=body.helpful,
            comment=comment,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not submit LangFuse feedback: {exc}"
        )

    submitted_at = datetime.now(UTC).isoformat()
    updated = await update_session_run(
        run_id=run_id,
        user_id=current_user.user_id,
        session_id=session_id,
        patch={
            "feedback_submitted_at": submitted_at,
            "feedback_helpful": body.helpful,
        },
    )
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in session '{session_id}'.",
        )

    return {
        "session_id": session_id,
        "run_id": run_id,
        "feedback_submitted_at": submitted_at,
        "feedback_helpful": body.helpful,
    }


@router.post("/sessions/{session_id}/followup", tags=["Sessions"])
async def session_followup(
    session_id: str,
    body: FollowupRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Ask a follow-up question grounded to a session's source material."""
    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(total_questions=1),
    )

    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )

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
