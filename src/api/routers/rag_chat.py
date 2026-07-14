"""Workspace-scoped RAG chat routes: chat, streaming, and chat sessions."""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from src import outbox
from src.api.deps import (
    CreateRagChatSessionRequest,
    RagChatRequest,
    UpdateSessionTitleRequest,
    _build_chat_trace_outputs,
    _coerce_agent_loop_result,
    _consume_usage_or_429,
    _raise_rag_validation_error,
    _run_agent_loop,
    _select_chat_citations,
    _workflow_error_text,
)
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing import UsageIncrement
from src.config import settings
from src.errors import RouterError
from src.observability import end_workflow_run, start_workflow_run
from src.rag import (
    CHAT_SCOPE_WORKSPACE,
    RagChatMessage,
    RagValidationError,
    append_chat_message,
    create_or_get_workspace_chat_session,
    delete_chat_session as delete_rag_chat_session,
    delete_last_exchange,
    delete_rag_chat_session_attachment,
    get_chat_session as get_rag_chat_session,
    ingest_agent_chat_session_uploads,
    list_chat_messages as list_rag_chat_messages,
    list_chat_sessions as list_rag_chat_sessions,
    list_rag_chat_session_attachments,
    update_chat_session_title as update_rag_chat_session_title,
)
from src.user_memory import enqueue_memory_refresh

logger = logging.getLogger(__name__)

router = APIRouter()


async def _require_workspace_chat_session(
    *,
    session_id: str,
    user_id: str,
) -> None:
    session = await get_rag_chat_session(
        session_id=session_id,
        agent_id=None,
        user_id=user_id,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")


@router.post("/api/rag/chat", tags=["RAG"])
async def rag_chat_workspace(
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    from src.api.rag_chat_helpers import (
        prepare_workspace_rag_chat,
        rag_json_response,
        resolve_suggestions,
        schedule_deferred_suggestions,
    )
    from src.api.rag_chat_timing import RagChatTimings

    await _consume_usage_or_429(current_user.user_id, UsageIncrement(total_questions=1))
    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    timings = RagChatTimings()
    wall_start = time.perf_counter()
    with start_workflow_run(entrypoint="rag_chat_workspace", query=normalized_message) as trace_ctx:
        try:
            prepared = await prepare_workspace_rag_chat(
                user_id=current_user.user_id,
                normalized_message=normalized_message,
                session_id=body.session_id,
                timings=timings,
                tools=body.tools,
            )
            try:
                t_loop = time.perf_counter()
                loop_result = _coerce_agent_loop_result(
                    await _run_agent_loop(
                        messages=prepared.messages,
                        metadata={"user_id": current_user.user_id},
                        bind_tools=prepared.bind_tools,
                        allow_web_search=prepared.allow_web_search,
                        reference_tools=prepared.reference_tools,
                    )
                )
                timings.agent_loop_ms = (time.perf_counter() - t_loop) * 1000
            except Exception as exc:
                logger.exception(
                    "[rag_api] workspace chat loop failed user_id=%s",
                    current_user.user_id,
                )
                raise HTTPException(
                    status_code=503,
                    detail={"code": "agent_loop_error", "error": str(exc)},
                ) from exc

            suggestions = await resolve_suggestions(
                query=normalized_message,
                answer=loop_result.answer,
                context=prepared.rag_context.context or "",
                timings=timings,
            )
            user_msg = RagChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=prepared.chat_session_id,
                agent_id=None,
                owner_id=current_user.user_id,
                role="user",
                content=normalized_message,
                chat_scope=CHAT_SCOPE_WORKSPACE,
            )
            citations = _select_chat_citations(
                prepared.rag_context.chunks,
                loop_result.citations,
                router_action=getattr(getattr(prepared, "router_decision", None), "action", None),
                web_used=loop_result.web_used,
                rag_context_text=prepared.rag_context.context or "",
            )
            assistant_msg = RagChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=prepared.chat_session_id,
                agent_id=None,
                owner_id=current_user.user_id,
                role="assistant",
                content=loop_result.answer,
                citations=citations,
                suggestions=suggestions,
                chat_scope=CHAT_SCOPE_WORKSPACE,
            )
            await append_chat_message(user_msg)
            await append_chat_message(assistant_msg)
            await enqueue_memory_refresh(
                user_id=current_user.user_id,
                source_mode="workspace_chat",
                source_session_id=prepared.chat_session_id,
                user_message=normalized_message,
                assistant_message=loop_result.answer,
                source_user_message_id=user_msg.message_id,
                source_assistant_message_id=assistant_msg.message_id,
            )
            asyncio.create_task(outbox.dispatch_outbox_events(limit=10))
            schedule_deferred_suggestions(
                query=normalized_message,
                answer=loop_result.answer,
                context=prepared.rag_context.context or "",
                assistant_message_id=assistant_msg.message_id,
                session_id=prepared.chat_session_id,
                owner_id=current_user.user_id,
                agent_id=None,
            )
            updated_history = await list_rag_chat_messages(
                prepared.chat_session_id, current_user.user_id
            )
            timings.total_ms = (time.perf_counter() - wall_start) * 1000
            end_workflow_run(
                trace_ctx,
                status="success",
                outputs=_build_chat_trace_outputs(
                    answer=loop_result.answer,
                    session_id=prepared.chat_session_id,
                    citations=citations,
                    suggestions=suggestions,
                    web_used=loop_result.web_used,
                ),
            )
            return rag_json_response(
                {
                    "session_id": prepared.chat_session_id,
                    "agent_id": None,
                    "reply": assistant_msg.to_dict(),
                    "messages": [m.to_dict() for m in updated_history],
                },
                timings,
            )
        except Exception as exc:
            end_workflow_run(trace_ctx, status="error", error=_workflow_error_text(exc))
            if isinstance(exc, RouterError):
                raise HTTPException(
                    status_code=503,
                    detail={"code": "router_error", "message": str(exc)},
                ) from exc
            raise


@router.post("/api/rag/chat/stream", tags=["RAG"])
async def rag_chat_workspace_stream(
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    from src.api.rag_chat_helpers import (
        prepare_workspace_rag_chat,
        resolve_suggestions,
        schedule_deferred_suggestions,
    )
    from src.api.rag_chat_timing import RagChatTimings

    await _consume_usage_or_429(current_user.user_id, UsageIncrement(total_questions=1))
    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    timings = RagChatTimings()
    stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    async def _stream_chat() -> AsyncGenerator[str, None]:
        event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def on_event(event: dict) -> None:
            await event_queue.put(event)

        trace_ctx = None
        try:
            with start_workflow_run(
                entrypoint="rag_chat_workspace_stream", query=normalized_message
            ) as trace_ctx:
                prepared = await prepare_workspace_rag_chat(
                    user_id=current_user.user_id,
                    normalized_message=normalized_message,
                    session_id=body.session_id,
                    timings=timings,
                    tools=body.tools,
                )
                yield f"data: {json.dumps({'type': 'session', 'session_id': prepared.chat_session_id})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer…'})}\n\n"
                loop_task = asyncio.create_task(
                    _run_agent_loop(
                        messages=prepared.messages,
                        metadata={"user_id": current_user.user_id},
                        on_event=on_event,
                        bind_tools=prepared.bind_tools,
                        allow_web_search=prepared.allow_web_search,
                        reference_tools=prepared.reference_tools,
                        stream_answer_chunks=True,
                    )
                )
                last_heartbeat = time.perf_counter()
                while not loop_task.done():
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                        yield f"data: {json.dumps(event)}\n\n"
                        last_heartbeat = time.perf_counter()
                    except asyncio.TimeoutError:
                        if time.perf_counter() - last_heartbeat >= 10.0:
                            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer…'})}\n\n"
                            last_heartbeat = time.perf_counter()
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                loop_result = _coerce_agent_loop_result(loop_task.result())
                if loop_result.web_used:
                    yield f"data: {json.dumps({'type': 'web_used', 'provider': settings.web_search_provider})}\n\n"
                suggestions = await resolve_suggestions(
                    query=normalized_message,
                    answer=loop_result.answer,
                    context=prepared.rag_context.context or "",
                    timings=timings,
                )
                user_msg = RagChatMessage(
                    message_id=str(uuid.uuid4()),
                    session_id=prepared.chat_session_id,
                    agent_id=None,
                    owner_id=current_user.user_id,
                    role="user",
                    content=normalized_message,
                    chat_scope=CHAT_SCOPE_WORKSPACE,
                )
                citations = _select_chat_citations(
                    prepared.rag_context.chunks,
                    loop_result.citations,
                    router_action=getattr(
                        getattr(prepared, "router_decision", None), "action", None
                    ),
                    web_used=loop_result.web_used,
                    rag_context_text=prepared.rag_context.context or "",
                )
                assistant_msg = RagChatMessage(
                    message_id=str(uuid.uuid4()),
                    session_id=prepared.chat_session_id,
                    agent_id=None,
                    owner_id=current_user.user_id,
                    role="assistant",
                    content=loop_result.answer,
                    citations=citations,
                    suggestions=suggestions,
                    chat_scope=CHAT_SCOPE_WORKSPACE,
                )
                await append_chat_message(user_msg)
                await append_chat_message(assistant_msg)
                await enqueue_memory_refresh(
                    user_id=current_user.user_id,
                    source_mode="workspace_chat",
                    source_session_id=prepared.chat_session_id,
                    user_message=normalized_message,
                    assistant_message=loop_result.answer,
                    source_user_message_id=user_msg.message_id,
                    source_assistant_message_id=assistant_msg.message_id,
                )
                asyncio.create_task(outbox.dispatch_outbox_events(limit=10))
                schedule_deferred_suggestions(
                    query=normalized_message,
                    answer=loop_result.answer,
                    context=prepared.rag_context.context or "",
                    assistant_message_id=assistant_msg.message_id,
                    session_id=prepared.chat_session_id,
                    owner_id=current_user.user_id,
                    agent_id=None,
                )
                end_workflow_run(
                    trace_ctx,
                    status="success",
                    outputs=_build_chat_trace_outputs(
                        answer=loop_result.answer,
                        session_id=prepared.chat_session_id,
                        citations=citations,
                        suggestions=suggestions,
                        web_used=loop_result.web_used,
                    ),
                )
                if not loop_result.streamed_answer:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': loop_result.answer})}\n\n"
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
                if suggestions:
                    yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            if trace_ctx is not None:
                end_workflow_run(trace_ctx, status="error", error=_workflow_error_text(exc))
            error_event = {"type": "error", "error": str(exc)}
            if isinstance(exc, RouterError):
                error_event["code"] = "router_error"
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        _stream_chat(),
        media_type="text/event-stream",
        headers=stream_headers,
    )


@router.get("/api/rag/chat/sessions", tags=["RAG"])
async def list_rag_workspace_chat_sessions(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    sessions = await list_rag_chat_sessions(
        agent_id=None,
        user_id=current_user.user_id,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    return {"sessions": sessions}


@router.post("/api/rag/chat/sessions", tags=["RAG"])
async def create_rag_workspace_chat_session(
    body: CreateRagChatSessionRequest | None = None,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    initial_message = None
    if body and body.filename:
        initial_message = f"Attached: {body.filename.strip()}"

    session_id = await create_or_get_workspace_chat_session(
        user_id=current_user.user_id,
        session_id=None,
        initial_message=initial_message,
    )
    return {"session_id": session_id, "agent_id": None}


@router.get("/api/rag/chat/sessions/{session_id}/attachments", tags=["RAG"])
async def list_rag_workspace_chat_session_attachments_endpoint(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _require_workspace_chat_session(
        session_id=session_id,
        user_id=current_user.user_id,
    )
    attachments = await list_rag_chat_session_attachments(
        session_id=session_id,
        owner_id=current_user.user_id,
        agent_id=None,
    )
    return {
        "session_id": session_id,
        "agent_id": None,
        "attachments": [attachment.to_dict() for attachment in attachments],
    }


@router.post("/api/rag/chat/sessions/{session_id}/attachments", tags=["RAG"])
async def upload_rag_workspace_chat_session_attachments(
    session_id: str,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _require_workspace_chat_session(
        session_id=session_id,
        user_id=current_user.user_id,
    )

    form = await request.form()
    files = [
        value
        for key, value in form.multi_items()
        if key == "files" and isinstance(value, StarletteUploadFile)
    ]
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    try:
        attachments = await ingest_agent_chat_session_uploads(
            session_id=session_id,
            agent_id=None,
            user_id=current_user.user_id,
            files=files,
        )
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)

    return {
        "session_id": session_id,
        "agent_id": None,
        "attachments": [attachment.to_dict() for attachment in attachments],
    }


@router.delete(
    "/api/rag/chat/sessions/{session_id}/attachments/{attachment_id}",
    tags=["RAG"],
)
async def delete_rag_workspace_chat_session_attachment_endpoint(
    session_id: str,
    attachment_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _require_workspace_chat_session(
        session_id=session_id,
        user_id=current_user.user_id,
    )

    deleted = await delete_rag_chat_session_attachment(
        session_id=session_id,
        attachment_id=attachment_id,
        owner_id=current_user.user_id,
        agent_id=None,
    )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{attachment_id}' not found.",
        )
    return {"session_id": session_id, "attachment_id": attachment_id, "deleted": True}


@router.get("/api/rag/chat/sessions/{session_id}/messages", tags=["RAG"])
async def list_rag_workspace_chat_session_messages(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    session = await get_rag_chat_session(
        session_id=session_id,
        agent_id=None,
        user_id=current_user.user_id,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    messages = await list_rag_chat_messages(session_id, current_user.user_id)
    return {
        "session_id": session_id,
        "agent_id": None,
        "messages": [m.to_dict() for m in messages],
    }


@router.patch("/api/rag/chat/sessions/{session_id}", tags=["RAG"])
async def update_rag_workspace_chat_session_title(
    session_id: str,
    body: UpdateSessionTitleRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    title = " ".join(body.title.strip().split())
    if not title:
        raise HTTPException(status_code=400, detail="Session title cannot be empty.")
    if len(title) > 120:
        raise HTTPException(status_code=400, detail="Session title is too long.")
    updated = await update_rag_chat_session_title(
        session_id=session_id,
        agent_id=None,
        user_id=current_user.user_id,
        title=title,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    return {"session_id": session_id, "title": title}


@router.delete("/api/rag/chat/sessions/{session_id}", tags=["RAG"])
async def delete_rag_workspace_chat_session(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    deleted = await delete_rag_chat_session(
        session_id=session_id,
        agent_id=None,
        user_id=current_user.user_id,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    return {"session_id": session_id, "deleted": True}


@router.delete("/api/rag/chat/sessions/{session_id}/last-exchange", tags=["RAG"])
async def delete_rag_workspace_chat_last_exchange(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    session = await get_rag_chat_session(
        session_id=session_id,
        agent_id=None,
        user_id=current_user.user_id,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    if session is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    deleted, err = await delete_last_exchange(session_id=session_id, user_id=current_user.user_id)
    if not deleted:
        if err == "empty":
            raise HTTPException(status_code=404, detail="Session has no messages to delete.")
        raise HTTPException(
            status_code=409, detail="Last two messages are not a user/assistant pair."
        )
    return {"session_id": session_id, "deleted": True}
