"""Shared prepare/finalize helpers for RAG agent chat endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.api.endpoints import RagChatTools

from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.api.rag_chat_timing import RagChatTimings
from src.prompts.registry import prompt_registry
from src.config import settings
from src.rag import (
    RagChatMessage,
    append_chat_message,
    create_or_get_chat_session,
    create_or_get_workspace_chat_session,
    get_agent_for_chat,
    list_chat_messages as list_rag_chat_messages,
    list_workspace_ready_resource_ids,
    retrieve_context_for_query,
)
from src.rag_engine import RagQueryResult
from src.tools.composio_toolset import get_composio_toolset_manager
from src.user_memory import get_user_memory_prompt_block

logger = logging.getLogger(__name__)

_COMPOSIO_META_KEYWORDS = (
    "composio",
    "tool router",
    "connected app",
    "connected apps",
    "do you have access",
    "can you access",
    "can you use",
    "are you connected",
    "which apps",
    "what apps",
    "integrate",
)

_EXTERNAL_INTENT_MARKERS = (
    "http://",
    "https://",
    "www.",
    "latest",
    "current",
    "today",
    "price",
    "stock",
    "weather",
    "search the web",
    "look up",
    "fetch",
    "scrape",
    "email",
    "calendar",
    "slack",
    "github",
    "notion",
    "drive",
)


@dataclass
class RagChatPrepared:
    agent: Any
    resource_ids: list[str]
    rag_context: RagQueryResult
    chat_session_id: str
    messages: list[BaseMessage]
    bind_tools: bool
    tool_skip_reason: str | None
    composio_apps: list[str]
    allow_web_search: bool = True


def build_agent_messages(
    *,
    system_instructions: str,
    history: list,
    rag_context: str,
    user_memory_context: str,
    composio_apps: list[str],
    normalized_message: str,
    bind_tools: bool = True,
) -> list[BaseMessage]:
    template_name = (
        "rag_chat_system" if bind_tools else "rag_chat_system_no_tools"
    )
    system_content, _ = prompt_registry.render(
        template_name,
        {
            "system_instructions": system_instructions,
            "rag_context": rag_context,
            "user_memory_context": user_memory_context,
            "composio_apps": composio_apps,
        },
    )
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    for turn in history:
        role = turn.role if hasattr(turn, "role") else turn.get("role", "")
        content = turn.content if hasattr(turn, "content") else turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=normalized_message))
    return messages


def trim_chat_history(history: list) -> list:
    max_messages = max(2, settings.rag_chat_max_history_messages)
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def should_bind_composio_tools(
    *,
    message: str,
    resource_ids: list[str],
    composio_apps: list[str],
) -> tuple[bool, str | None]:
    normalized = " ".join(message.strip().lower().split())
    if not normalized:
        return False, "empty_message"

    if settings.composio_enabled and any(kw in normalized for kw in _COMPOSIO_META_KEYWORDS):
        return True, "composio_meta_question"

    if not settings.composio_enabled:
        return False, "composio_disabled"
    if not composio_apps:
        return False, "no_connected_apps"
    if not settings.rag_chat_conditional_tools:
        return True, None

    for app in composio_apps:
        if app and app.lower() in normalized:
            return True, "app_mentioned"

    if any(marker in normalized for marker in _EXTERNAL_INTENT_MARKERS):
        return True, "external_intent"

    # Linked documents do not disable Composio — tools stay bound whenever apps are connected.
    return True, "default_bind"


async def prepare_agent_rag_chat(
    *,
    agent_id: str,
    user_id: str,
    normalized_message: str,
    session_id: str | None,
    timings: RagChatTimings,
    tools: "RagChatTools | None" = None,
) -> RagChatPrepared | None:
    t0 = time.perf_counter()
    agent_bundle = await get_agent_for_chat(agent_id, user_id)
    if agent_bundle is None:
        return None
    agent, resource_ids = agent_bundle

    t_session = time.perf_counter()
    chat_session_id = await create_or_get_chat_session(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        initial_message=normalized_message,
    )
    timings.session_ms = (time.perf_counter() - t_session) * 1000

    composio_apps = get_composio_toolset_manager().get_connected_app_names()
    if tools is not None:
        bind_tools = tools.composio and settings.composio_enabled
        if not tools.composio:
            tool_skip_reason = "user_disabled"
        elif not settings.composio_enabled:
            tool_skip_reason = "server_disabled"
        else:
            tool_skip_reason = None
        allow_web_search = tools.web_search
    else:
        bind_tools, tool_skip_reason = should_bind_composio_tools(
            message=normalized_message,
            resource_ids=resource_ids,
            composio_apps=composio_apps,
        )
        allow_web_search = True
    timings.tools_bound = bind_tools
    timings.tool_skip_reason = tool_skip_reason

    retrieve_task = retrieve_context_for_query(
        user_id=user_id,
        resource_ids=resource_ids,
        question=normalized_message,
    )
    memory_task = get_user_memory_prompt_block(user_id, normalized_message)
    history_task = list_rag_chat_messages(chat_session_id, user_id)
    rag_context, user_memory_context, history = await asyncio.gather(
        retrieve_task,
        memory_task,
        history_task,
    )
    history = trim_chat_history(history)

    messages = build_agent_messages(
        system_instructions=agent.system_instructions or "",
        history=history,
        rag_context=rag_context.context or "",
        user_memory_context=user_memory_context,
        composio_apps=composio_apps,
        normalized_message=normalized_message,
        bind_tools=bind_tools,
    )
    timings.prepare_ms = (time.perf_counter() - t0) * 1000
    return RagChatPrepared(
        agent=agent,
        resource_ids=resource_ids,
        rag_context=rag_context,
        chat_session_id=chat_session_id,
        messages=messages,
        bind_tools=bind_tools,
        tool_skip_reason=tool_skip_reason,
        composio_apps=composio_apps,
        allow_web_search=allow_web_search,
    )


async def prepare_workspace_rag_chat(
    *,
    user_id: str,
    normalized_message: str,
    session_id: str | None,
    timings: RagChatTimings,
    tools: "RagChatTools | None" = None,
) -> RagChatPrepared:
    resource_ids = await list_workspace_ready_resource_ids(user_id)

    t_session = time.perf_counter()
    chat_session_id = await create_or_get_workspace_chat_session(
        user_id=user_id,
        session_id=session_id,
        initial_message=normalized_message,
    )
    timings.session_ms = (time.perf_counter() - t_session) * 1000

    composio_apps = get_composio_toolset_manager().get_connected_app_names()
    if tools is not None:
        bind_tools = tools.composio and settings.composio_enabled
        if not tools.composio:
            tool_skip_reason = "user_disabled"
        elif not settings.composio_enabled:
            tool_skip_reason = "server_disabled"
        else:
            tool_skip_reason = None
        allow_web_search = tools.web_search
    else:
        bind_tools, tool_skip_reason = should_bind_composio_tools(
            message=normalized_message,
            resource_ids=resource_ids,
            composio_apps=composio_apps,
        )
        allow_web_search = True
    timings.tools_bound = bind_tools
    timings.tool_skip_reason = tool_skip_reason

    t0 = time.perf_counter()
    rag_context, user_memory_context, history = await asyncio.gather(
        retrieve_context_for_query(
            user_id=user_id,
            resource_ids=resource_ids,
            question=normalized_message,
        ),
        get_user_memory_prompt_block(user_id, normalized_message),
        list_rag_chat_messages(chat_session_id, user_id),
    )
    history = trim_chat_history(history)

    messages = build_agent_messages(
        system_instructions="You are a generic workspace chat assistant.",
        history=history,
        rag_context=rag_context.context or "",
        user_memory_context=user_memory_context,
        composio_apps=composio_apps,
        normalized_message=normalized_message,
        bind_tools=bind_tools,
    )
    timings.prepare_ms = (time.perf_counter() - t0) * 1000
    return RagChatPrepared(
        agent=None,
        resource_ids=resource_ids,
        rag_context=rag_context,
        chat_session_id=chat_session_id,
        messages=messages,
        bind_tools=bind_tools,
        tool_skip_reason=tool_skip_reason,
        composio_apps=composio_apps,
        allow_web_search=allow_web_search,
    )


async def resolve_suggestions(
    *,
    query: str,
    answer: str,
    context: str,
    timings: RagChatTimings,
) -> list[str]:
    if settings.rag_suggestions_deferred:
        timings.suggestions_ms = 0.0
        return []
    from src.api.endpoints import _generate_suggestions

    t0 = time.perf_counter()
    suggestions = await _generate_suggestions(query, answer, context)
    timings.suggestions_ms = (time.perf_counter() - t0) * 1000
    return suggestions


def schedule_deferred_suggestions(
    *,
    query: str,
    answer: str,
    context: str,
    assistant_message_id: str,
    session_id: str,
    owner_id: str,
    agent_id: str | None,
) -> None:
    if not settings.rag_suggestions_deferred:
        return

    async def _run() -> None:
        from src.api.endpoints import _generate_suggestions
        from src.rag import update_chat_message_suggestions

        try:
            suggestions = await _generate_suggestions(query, answer, context)
            if suggestions:
                await update_chat_message_suggestions(
                    message_id=assistant_message_id,
                    session_id=session_id,
                    owner_id=owner_id,
                    agent_id=agent_id,
                    suggestions=suggestions,
                )
        except Exception as exc:
            logger.warning("[rag_api] deferred suggestions failed: %s", exc)

    asyncio.create_task(_run())


def rag_json_response(payload: dict, timings: RagChatTimings) -> JSONResponse:
    timings.total_ms = (
        timings.prepare_ms
        + timings.session_ms
        + timings.agent_loop_ms
        + timings.suggestions_ms
        + timings.persist_ms
    )
    headers: dict[str, str] = {}
    if settings.rag_perf_headers:
        headers["X-Rag-Perf"] = timings.to_header_value()
    return JSONResponse(content=payload, headers=headers)
