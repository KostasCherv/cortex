"""RAG agent routes: agent CRUD, resource linking, and per-agent chat sessions."""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from src import outbox
from src.api.deps import (
    CreateRagChatSessionRequest,
    RagChatRequest,
    RagChatTools,
    UpdateSessionTitleRequest,
    _build_chat_trace_outputs,
    _build_rag_citations,
    _coerce_agent_loop_result,
    _consume_usage_or_429,
    _merge_citations,
    _raise_rag_validation_error,
    _run_agent_loop,
    _workflow_error_text,
)
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing import UsageIncrement
from src.config import settings
from src.observability import end_workflow_run, start_workflow_run
from src.rag import (
    CHAT_SCOPE_AGENT,
    RagChatMessage,
    RagValidationError,
    append_chat_message,
    create_agent as create_rag_agent_record,
    create_or_get_chat_session,
    delete_agent as delete_rag_agent_record,
    delete_chat_session as delete_rag_chat_session,
    delete_last_exchange,
    delete_rag_chat_session_attachment,
    delete_rag_chat_session_attachments_and_artifacts,
    get_agent_for_chat,
    get_chat_session as get_rag_chat_session,
    ingest_agent_chat_session_uploads,
    link_resources as link_rag_resources,
    list_agents as list_rag_agents_records,
    list_chat_messages as list_rag_chat_messages,
    list_chat_sessions as list_rag_chat_sessions,
    list_rag_chat_session_attachments,
    suggest_agent_definition as suggest_rag_agent_definition,
    update_agent as update_rag_agent_record,
    update_chat_session_title as update_rag_chat_session_title,
)
from src.user_memory import enqueue_memory_refresh

logger = logging.getLogger(__name__)

router = APIRouter()


class RagAgentCreateRequest(BaseModel):
    name: str
    description: str = ""
    system_instructions: str = ""
    linked_resource_ids: list[str] = []


class RagAgentDraftRequest(BaseModel):
    prompt: str


class RagAgentUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    system_instructions: str | None = None
    linked_resource_ids: list[str] | None = None


class RagAgentLinkRequest(BaseModel):
    resource_ids: list[str]


async def _require_agent_chat_session(
    *,
    agent_id: str,
    session_id: str,
    user_id: str,
) -> None:
    agent_bundle = await get_agent_for_chat(agent_id, user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    session = await get_rag_chat_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=user_id,
    )
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )


async def _files_needing_session_ingest(
    *,
    session_id: str,
    agent_id: str,
    user_id: str,
    files: list,
) -> list:
    if not files:
        return []
    existing = await list_rag_chat_session_attachments(
        session_id=session_id,
        owner_id=user_id,
        agent_id=agent_id,
    )
    ready_filenames = {
        attachment.filename
        for attachment in existing
        if attachment.state == "ready" and attachment.filename
    }
    return [
        file
        for file in files
        if (file.filename or "") not in ready_filenames
    ]


async def _parse_rag_chat_request(
    request: Request,
) -> tuple[RagChatRequest, list]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        files = [
            value
            for key, value in form.multi_items()
            if key == "files" and isinstance(value, StarletteUploadFile)
        ]
        tools_value = form.get("tools")
        if isinstance(tools_value, str) and tools_value:
            try:
                parsed_tools = json.loads(tools_value)
            except json.JSONDecodeError as exc:
                raise RequestValidationError(
                    [
                        {
                            "type": "json_invalid",
                            "loc": ("body", "tools"),
                            "msg": "JSON decode error",
                            "input": tools_value,
                            "ctx": {"error": exc.msg},
                        }
                    ]
                ) from exc
            try:
                tools = RagChatTools.model_validate(parsed_tools)
            except ValidationError as exc:
                raise RequestValidationError(
                    [
                        {
                            "type": "value_error",
                            "loc": ("body", "tools"),
                            "msg": "Invalid tools schema",
                            "input": tools_value,
                            "ctx": {"error": str(exc)},
                        }
                    ]
                ) from exc
        else:
            tools = RagChatTools()
        payload = {
            "message": form.get("message"),
            "session_id": form.get("session_id"),
            "tools": tools,
        }
    else:
        files = []
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body", exc.pos),
                        "msg": "JSON decode error",
                        "input": {},
                        "ctx": {"error": exc.msg},
                    }
                ]
            ) from exc

    try:
        return RagChatRequest.model_validate(payload), files
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@router.post("/api/rag/agents", tags=["RAG"])
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


@router.post("/api/rag/agents/draft", tags=["RAG"])
async def rag_generate_agent_draft(
    body: RagAgentDraftRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        draft = await suggest_rag_agent_definition(body.prompt)
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    return {"draft": draft.to_dict()}


@router.get("/api/rag/agents", tags=["RAG"])
async def rag_list_agents(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agents = await list_rag_agents_records(current_user.user_id)
    return {"agents": [a.to_dict() for a in agents]}


@router.patch("/api/rag/agents/{agent_id}", tags=["RAG"])
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
            description=(
                body.description.strip() if body.description is not None else None
            ),
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


@router.delete("/api/rag/agents/{agent_id}", tags=["RAG"])
async def rag_delete_agent(
    agent_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    deleted = await delete_rag_agent_record(agent_id, current_user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {"agent_id": agent_id, "deleted": True}


@router.post("/api/rag/agents/{agent_id}/resources:link", tags=["RAG"])
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


@router.post("/api/rag/agents/{agent_id}/chat", tags=["RAG"])
async def rag_chat_with_agent(
    agent_id: str,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    from src.api.rag_chat_helpers import (
        ensure_agent_chat_session_id,
        prepare_agent_rag_chat,
        rag_json_response,
        resolve_suggestions,
        schedule_deferred_suggestions,
    )
    from src.api.rag_chat_timing import RagChatTimings

    body, files = await _parse_rag_chat_request(request)

    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(total_questions=1),
    )

    timings = RagChatTimings()
    wall_start = time.perf_counter()

    with start_workflow_run(
        entrypoint="rag_chat", query=normalized_message
    ) as trace_ctx:
        try:
            if files:
                chat_session_id = await ensure_agent_chat_session_id(
                    agent_id=agent_id,
                    user_id=current_user.user_id,
                    session_id=body.session_id,
                    initial_message=normalized_message,
                )
                if chat_session_id is None:
                    raise HTTPException(
                        status_code=404, detail=f"Agent '{agent_id}' not found."
                    )
                files_to_ingest = await _files_needing_session_ingest(
                    session_id=chat_session_id,
                    agent_id=agent_id,
                    user_id=current_user.user_id,
                    files=files,
                )
                if files_to_ingest:
                    await ingest_agent_chat_session_uploads(
                        session_id=chat_session_id,
                        agent_id=agent_id,
                        user_id=current_user.user_id,
                        files=files_to_ingest,
                    )
                prepared = await prepare_agent_rag_chat(
                    agent_id=agent_id,
                    user_id=current_user.user_id,
                    normalized_message=normalized_message,
                    session_id=chat_session_id,
                    timings=timings,
                    tools=body.tools,
                )
            else:
                prepared = await prepare_agent_rag_chat(
                    agent_id=agent_id,
                    user_id=current_user.user_id,
                    normalized_message=normalized_message,
                    session_id=body.session_id,
                    timings=timings,
                    tools=body.tools,
                )
            if prepared is None:
                logger.warning(
                    "[rag_api] agent chat request failed because agent was not found agent_id=%s user_id=%s",
                    agent_id,
                    current_user.user_id,
                )
                raise HTTPException(
                    status_code=404, detail=f"Agent '{agent_id}' not found."
                )
            if not prepared.resource_ids:
                logger.info(
                    "[rag_api] agent chat request proceeding without linked ready resources agent_id=%s user_id=%s",
                    agent_id,
                    current_user.user_id,
                )

            try:
                t_loop = time.perf_counter()
                loop_result = _coerce_agent_loop_result(
                    await _run_agent_loop(
                        messages=prepared.messages,
                        metadata={
                            "agent_id": agent_id,
                            "user_id": current_user.user_id,
                        },
                        bind_tools=prepared.bind_tools,
                        allow_web_search=prepared.allow_web_search,
                        reference_tools=prepared.reference_tools,
                    )
                )
                timings.agent_loop_ms = (time.perf_counter() - t_loop) * 1000
            except Exception as exc:
                logger.exception(
                    "[rag_api] agent chat loop failed agent_id=%s", agent_id
                )
                raise HTTPException(
                    status_code=503,
                    detail={"code": "agent_loop_error", "error": str(exc)},
                ) from exc

            if files:
                suggestions = []
            else:
                suggestions = await resolve_suggestions(
                    query=normalized_message,
                    answer=loop_result.answer,
                    context=prepared.rag_context.context or "",
                    timings=timings,
                )

            t_persist = time.perf_counter()
            user_msg = RagChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=prepared.chat_session_id,
                agent_id=agent_id,
                owner_id=current_user.user_id,
                role="user",
                content=normalized_message,
            )
            citations = _merge_citations(
                _build_rag_citations(prepared.rag_context.chunks),
                loop_result.citations,
            )
            assistant_msg = RagChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=prepared.chat_session_id,
                agent_id=agent_id,
                owner_id=current_user.user_id,
                role="assistant",
                content=loop_result.answer,
                citations=citations,
                suggestions=suggestions,
            )
            await append_chat_message(user_msg)
            await append_chat_message(assistant_msg)
            await enqueue_memory_refresh(
                user_id=current_user.user_id,
                source_mode="agent_chat",
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
                agent_id=agent_id,
                force=bool(files),
            )
            timings.persist_ms = (time.perf_counter() - t_persist) * 1000

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
                    "agent_id": agent_id,
                    "reply": assistant_msg.to_dict(),
                    "messages": [m.to_dict() for m in updated_history],
                },
                timings,
            )
        except Exception as exc:
            end_workflow_run(trace_ctx, status="error", error=_workflow_error_text(exc))
            if isinstance(exc, RagValidationError):
                _raise_rag_validation_error(exc)
            raise


@router.post("/api/rag/agents/{agent_id}/chat/stream", tags=["RAG"])
async def rag_chat_with_agent_stream(
    agent_id: str,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    from src.api.rag_chat_helpers import (
        ensure_agent_chat_session_id,
        prepare_agent_rag_chat,
        resolve_suggestions,
        schedule_deferred_suggestions,
    )
    from src.api.rag_chat_timing import RagChatTimings

    body, files = await _parse_rag_chat_request(request)

    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(total_questions=1),
    )

    timings = RagChatTimings()
    stream_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    async def _stream_chat() -> AsyncGenerator[str, None]:
        event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def on_event(event: dict) -> None:
            await event_queue.put(event)

        trace_ctx = None
        try:
            with start_workflow_run(
                entrypoint="rag_chat_stream", query=normalized_message
            ) as trace_ctx:
                if files:
                    chat_session_id = await ensure_agent_chat_session_id(
                        agent_id=agent_id,
                        user_id=current_user.user_id,
                        session_id=body.session_id,
                        initial_message=normalized_message,
                    )
                    if chat_session_id is None:
                        raise HTTPException(
                            status_code=404, detail=f"Agent '{agent_id}' not found."
                        )
                    yield f"data: {json.dumps({'type': 'session', 'session_id': chat_session_id})}\n\n"
                    files_to_ingest = await _files_needing_session_ingest(
                        session_id=chat_session_id,
                        agent_id=agent_id,
                        user_id=current_user.user_id,
                        files=files,
                    )
                    if files_to_ingest:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing attachment…'})}\n\n"
                        await ingest_agent_chat_session_uploads(
                            session_id=chat_session_id,
                            agent_id=agent_id,
                            user_id=current_user.user_id,
                            files=files_to_ingest,
                        )
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Preparing document context…'})}\n\n"
                    prepared = await prepare_agent_rag_chat(
                        agent_id=agent_id,
                        user_id=current_user.user_id,
                        normalized_message=normalized_message,
                        session_id=chat_session_id,
                        timings=timings,
                        tools=body.tools,
                    )
                else:
                    prepared = await prepare_agent_rag_chat(
                        agent_id=agent_id,
                        user_id=current_user.user_id,
                        normalized_message=normalized_message,
                        session_id=body.session_id,
                        timings=timings,
                        tools=body.tools,
                    )
                    if prepared is None:
                        raise HTTPException(
                            status_code=404, detail=f"Agent '{agent_id}' not found."
                        )
                    yield f"data: {json.dumps({'type': 'session', 'session_id': prepared.chat_session_id})}\n\n"
                if prepared is None:
                    raise HTTPException(
                        status_code=404, detail=f"Agent '{agent_id}' not found."
                    )
                citations = _build_rag_citations(prepared.rag_context.chunks)
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer…'})}\n\n"
                loop_task = asyncio.create_task(
                    _run_agent_loop(
                        messages=prepared.messages,
                        metadata={
                            "agent_id": agent_id,
                            "user_id": current_user.user_id,
                        },
                        on_event=on_event,
                        bind_tools=prepared.bind_tools,
                        allow_web_search=prepared.allow_web_search,
                        reference_tools=prepared.reference_tools,
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
                if files:
                    suggestions = []
                else:
                    suggestions = await resolve_suggestions(
                        query=normalized_message,
                        answer=loop_result.answer,
                        context=prepared.rag_context.context or "",
                        timings=timings,
                    )
                user_msg = RagChatMessage(
                    message_id=str(uuid.uuid4()),
                    session_id=prepared.chat_session_id,
                    agent_id=agent_id,
                    owner_id=current_user.user_id,
                    role="user",
                    content=normalized_message,
                )
                citations = _merge_citations(
                    _build_rag_citations(prepared.rag_context.chunks),
                    loop_result.citations,
                )
                assistant_msg = RagChatMessage(
                    message_id=str(uuid.uuid4()),
                    session_id=prepared.chat_session_id,
                    agent_id=agent_id,
                    owner_id=current_user.user_id,
                    role="assistant",
                    content=loop_result.answer,
                    citations=citations,
                    suggestions=suggestions,
                )
                await append_chat_message(user_msg)
                await append_chat_message(assistant_msg)
                await enqueue_memory_refresh(
                    user_id=current_user.user_id,
                    source_mode="agent_chat",
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
                    agent_id=agent_id,
                    force=bool(files),
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
                yield f"data: {json.dumps({'type': 'chunk', 'text': loop_result.answer})}\n\n"
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
                if suggestions:
                    yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            if trace_ctx is not None:
                end_workflow_run(
                    trace_ctx, status="error", error=_workflow_error_text(exc)
                )
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        _stream_chat(),
        media_type="text/event-stream",
        headers=stream_headers,
    )


@router.get("/api/rag/agents/{agent_id}/chat/sessions", tags=["RAG"])
async def list_rag_agent_chat_sessions(
    agent_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    sessions = await list_rag_chat_sessions(agent_id, current_user.user_id)
    return {"sessions": sessions}


@router.get("/api/rag/agents/{agent_id}/chat/sessions/{session_id}/messages", tags=["RAG"])
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
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )

    messages = await list_rag_chat_messages(session_id, current_user.user_id)
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "messages": [m.to_dict() for m in messages],
    }


@router.post("/api/rag/agents/{agent_id}/chat/sessions", tags=["RAG"])
async def create_rag_agent_chat_session(
    agent_id: str,
    body: CreateRagChatSessionRequest | None = None,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    initial_message = None
    if body and body.filename:
        initial_message = f"Attached: {body.filename.strip()}"

    session_id = await create_or_get_chat_session(
        user_id=current_user.user_id,
        agent_id=agent_id,
        session_id=None,
        initial_message=initial_message,
    )
    return {"session_id": session_id, "agent_id": agent_id}


@router.post(
    "/api/rag/agents/{agent_id}/chat/sessions/{session_id}/attachments",
    tags=["RAG"],
)
async def upload_rag_agent_chat_session_attachments(
    agent_id: str,
    session_id: str,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _require_agent_chat_session(
        agent_id=agent_id,
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
            agent_id=agent_id,
            user_id=current_user.user_id,
            files=files,
        )
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)

    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "attachments": [attachment.to_dict() for attachment in attachments],
    }


@router.delete(
    "/api/rag/agents/{agent_id}/chat/sessions/{session_id}/attachments/{attachment_id}",
    tags=["RAG"],
)
async def delete_rag_agent_chat_session_attachment_endpoint(
    agent_id: str,
    session_id: str,
    attachment_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _require_agent_chat_session(
        agent_id=agent_id,
        session_id=session_id,
        user_id=current_user.user_id,
    )

    deleted = await delete_rag_chat_session_attachment(
        session_id=session_id,
        attachment_id=attachment_id,
        owner_id=current_user.user_id,
        agent_id=agent_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Attachment '{attachment_id}' not found.",
        )
    return {"session_id": session_id, "attachment_id": attachment_id, "deleted": True}


@router.get(
    "/api/rag/agents/{agent_id}/chat/sessions/{session_id}/attachments", tags=["RAG"]
)
async def list_rag_agent_chat_session_attachments_endpoint(
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
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )

    attachments = await list_rag_chat_session_attachments(
        session_id=session_id,
        owner_id=current_user.user_id,
        agent_id=agent_id,
    )
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "attachments": [attachment.to_dict() for attachment in attachments],
    }


@router.patch("/api/rag/agents/{agent_id}/chat/sessions/{session_id}", tags=["RAG"])
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
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )
    return {"session_id": session_id, "title": title}


@router.delete("/api/rag/agents/{agent_id}/chat/sessions/{session_id}", tags=["RAG"])
async def delete_rag_agent_chat_session(
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
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )

    await delete_rag_chat_session_attachments_and_artifacts(
        session_id=session_id,
        owner_id=current_user.user_id,
        agent_id=agent_id,
    )
    deleted = await delete_rag_chat_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=current_user.user_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )
    return {"session_id": session_id, "deleted": True}


@router.delete(
    "/api/rag/agents/{agent_id}/chat/sessions/{session_id}/last-exchange",
    tags=["RAG"],
)
async def delete_rag_agent_chat_last_exchange(
    agent_id: str,
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    session = await get_rag_chat_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=current_user.user_id,
        chat_scope=CHAT_SCOPE_AGENT,
    )
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Chat session '{session_id}' not found."
        )
    deleted, err = await delete_last_exchange(
        session_id=session_id, user_id=current_user.user_id
    )
    if not deleted:
        if err == "empty":
            raise HTTPException(
                status_code=404, detail="Session has no messages to delete."
            )
        raise HTTPException(
            status_code=409, detail="Last two messages are not a user/assistant pair."
        )
    return {"session_id": session_id, "deleted": True}
