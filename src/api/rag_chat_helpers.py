"""Shared prepare/finalize helpers for RAG agent chat endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.api.deps import RagChatTools

from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.api.rag_chat_timing import RagChatTimings
from src.errors import RouterError, StructuredOutputError
from src.llm.factory import get_router_llm
from src.llm.output_parsers import (
    ChatActionDecisionPayload,
    build_validation_retry_prompt,
    parse_chat_action_json,
)
from src.llm.text_utils import extract_llm_text
from src.observability.langsmith import start_step_span
from src.prompts.registry import prompt_registry
from src.config import settings
from src.rag import (
    create_or_get_chat_session,
    create_or_get_workspace_chat_session,
    get_agent_for_chat,
    list_chat_messages as list_rag_chat_messages,
    list_rag_chat_session_attachments,
    list_ready_rag_chat_session_attachment_resource_ids,  # noqa: F401 - patched by tests/conftest.py
    list_workspace_ready_resource_ids,
    retrieve_merged_context_for_agent_chat,
)
from src.rag_engine import RagQueryResult
from src.tools.composio_toolset import get_composio_toolset_manager
from src.tools.registry import (
    default_reference_tool_flags,
    reference_flags_from_tools,
    reference_tool_prompt_lines,
)
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


def should_use_workspace_resources(
    router_decision: ChatActionDecisionPayload | None,
) -> bool:
    """Return whether workspace-wide resources should be retrieved for this turn.

    Only an explicit document/knowledge-base decision may load collection
    resources. Missing or unknown decisions fail closed; explicit session
    attachments are still retrieved separately.
    """
    return getattr(router_decision, "action", None) == "answer_from_rag"


_EXTERNAL_INTENT_MARKERS = (
    "http://",
    "https://",
    "www.",
    "latest",
    "current",
    "today",
    "price",
    "stock",
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
    reference_tools: dict[str, bool] = field(default_factory=default_reference_tool_flags)
    router_decision: ChatActionDecisionPayload | None = None


def build_agent_messages(
    *,
    system_instructions: str,
    history: list,
    rag_context: str,
    user_memory_context: str,
    composio_apps: list[str],
    normalized_message: str,
    session_attachment_files: list[str] | None = None,
    bind_tools: bool = True,
    composio_user_disabled: bool = False,
) -> list[BaseMessage]:
    template_name = "rag_chat_system" if settings.composio_enabled else "rag_chat_system_no_tools"
    system_content, _ = prompt_registry.render(
        template_name,
        {
            "system_instructions": system_instructions,
            "rag_context": rag_context,
            "session_attachment_files": session_attachment_files or [],
            "user_memory_context": user_memory_context,
            # Pass empty apps when not bound so the template's {% if composio_apps %}
            # block never fires; composio_user_disabled then controls whether the
            # "no apps connected" fallback is also suppressed.
            "composio_apps": composio_apps if bind_tools else [],
            "composio_user_disabled": composio_user_disabled,
            "reference_tool_lines": reference_tool_prompt_lines(),
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


_ROUTER_ACTION_SYSTEM_PROMPT = """You are a chat routing assistant. Given a user message and optional context, \
output ONLY a JSON object with no markdown fences or explanation.

JSON schema:
{
  "action": "answer_direct" | "answer_from_rag" | "web_search" | "asset_price" | "search_finance_tools" | "ask_clarifying",
  "reason": "one sentence explaining the routing decision",
  "query": "search query — required for web_search and search_finance_tools, empty string otherwise",
  "symbols": ["TICKER"] — required list for asset_price, empty list otherwise,
  "currency": "ISO currency code if relevant, empty string otherwise"
}

Routing rules:
- answer_direct: question answerable from general knowledge, no live data or documents needed
- answer_from_rag: question about uploaded documents or the knowledge base
- web_search: needs current or live information (news, recent events, live prices)
- asset_price: user wants current price or quote for a specific stock or crypto (symbols required)
- search_finance_tools: user needs financial ratios, statements, or structured data via a tool (query required)
- ask_clarifying: message is too ambiguous, incomplete, or multi-intent to route confidently"""

_ROUTER_ACTION_SCHEMA = None


def _get_router_action_schema() -> str:
    global _ROUTER_ACTION_SCHEMA
    if _ROUTER_ACTION_SCHEMA is None:
        import json

        _ROUTER_ACTION_SCHEMA = json.dumps(ChatActionDecisionPayload.model_json_schema(), indent=2)
    return _ROUTER_ACTION_SCHEMA


def _format_router_user_turn(*, message: str, rag_context: str) -> str:
    if rag_context.strip():
        context_line = f"Available RAG context: yes — {rag_context.strip()}"
    else:
        context_line = "Available RAG context: no"
    return f"User message: {message.strip()}\n{context_line}"


async def classify_chat_action(
    *,
    message: str,
    rag_context: str = "",
) -> ChatActionDecisionPayload:
    """Classify a chat message into a router action using the fast router LLM.

    Routing is mandatory. Provider/model settings default to the main LLM, and
    failures raise RouterError rather than silently enabling document retrieval.
    """
    user_turn = _format_router_user_turn(message=message, rag_context=rag_context)
    prompt = f"{_ROUTER_ACTION_SYSTEM_PROMPT}\n\n{user_turn}"

    try:
        llm = get_router_llm()
        with start_step_span(
            name="rag_chat.classify_chat_action.llm_invoke",
            run_type="llm",
            node_name="classify_chat_action",
            inputs={"prompt": prompt},
            tags=["rag_chat", "router"],
        ):
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=3.0)
        raw_text = extract_llm_text(response)
    except Exception as exc:
        logger.warning("[rag_chat] router LLM call failed: %s", exc)
        raise RouterError("Chat router is unavailable.") from exc

    try:
        return parse_chat_action_json(raw_text)
    except StructuredOutputError as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text=_get_router_action_schema(),
            invalid_response=raw_text,
            validation_error=exc,
        )
        try:
            with start_step_span(
                name="rag_chat.classify_chat_action.llm_repair",
                run_type="llm",
                node_name="classify_chat_action",
                inputs={"prompt": repair_prompt},
                tags=["rag_chat", "router", "repair"],
            ):
                repair_response = await asyncio.wait_for(llm.ainvoke(repair_prompt), timeout=3.0)
            repair_text = extract_llm_text(repair_response)
            return parse_chat_action_json(repair_text)
        except Exception as repair_exc:
            logger.warning("[rag_chat] router decision parse failed after repair: %s", repair_exc)
            raise RouterError("Chat router returned an invalid decision.") from repair_exc


async def ensure_agent_chat_session_id(
    *,
    agent_id: str,
    user_id: str,
    session_id: str | None,
    initial_message: str | None,
) -> str | None:
    agent_bundle = await get_agent_for_chat(agent_id, user_id)
    if agent_bundle is None:
        return None
    return await create_or_get_chat_session(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        initial_message=initial_message,
    )


async def prepare_agent_rag_chat(
    *,
    agent_id: str,
    user_id: str,
    normalized_message: str,
    session_id: str | None,
    timings: RagChatTimings,
    tools: "RagChatTools | None" = None,  # type: ignore[valid-type]
) -> RagChatPrepared | None:
    t0 = time.perf_counter()
    agent_bundle = await get_agent_for_chat(agent_id, user_id)
    if agent_bundle is None:
        return None
    agent, linked_resource_ids = agent_bundle

    t_session = time.perf_counter()
    chat_session_id = await create_or_get_chat_session(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        initial_message=normalized_message,
    )
    timings.session_ms = (time.perf_counter() - t_session) * 1000
    session_attachments = await list_rag_chat_session_attachments(
        session_id=chat_session_id,
        owner_id=user_id,
        agent_id=agent_id,
    )
    session_attachment_resource_ids = [
        attachment.resource_id
        for attachment in session_attachments
        if attachment.state == "ready" and attachment.resource_id
    ]
    session_attachment_files = [
        attachment.filename
        for attachment in session_attachments
        if attachment.state == "ready" and attachment.filename
    ]
    router_decision = await classify_chat_action(message=normalized_message)
    agent_resource_ids = (
        linked_resource_ids if should_use_workspace_resources(router_decision) else []
    )
    merged_resource_ids = list(dict.fromkeys(agent_resource_ids + session_attachment_resource_ids))

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
        reference_tools = reference_flags_from_tools(tools)
    else:
        bind_tools, tool_skip_reason = should_bind_composio_tools(
            message=normalized_message,
            resource_ids=merged_resource_ids,
            composio_apps=composio_apps,
        )
        allow_web_search = True
        reference_tools = default_reference_tool_flags()
    timings.tools_bound = bind_tools
    timings.tool_skip_reason = tool_skip_reason

    retrieve_task = retrieve_merged_context_for_agent_chat(
        user_id=user_id,
        agent_resource_ids=agent_resource_ids,
        session_attachment_resource_ids=session_attachment_resource_ids,
        session_attachment_files=session_attachment_files,
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
        session_attachment_files=session_attachment_files,
        bind_tools=bind_tools,
        composio_user_disabled=(tool_skip_reason == "user_disabled"),
    )
    timings.prepare_ms = (time.perf_counter() - t0) * 1000
    return RagChatPrepared(
        agent=agent,
        resource_ids=merged_resource_ids,
        rag_context=rag_context,
        chat_session_id=chat_session_id,
        messages=messages,
        bind_tools=bind_tools,
        tool_skip_reason=tool_skip_reason,
        composio_apps=composio_apps,
        allow_web_search=allow_web_search,
        reference_tools=reference_tools,
        router_decision=router_decision,
    )


async def prepare_workspace_rag_chat(
    *,
    user_id: str,
    normalized_message: str,
    session_id: str | None,
    timings: RagChatTimings,
    tools: "RagChatTools | None" = None,  # type: ignore[valid-type]
) -> RagChatPrepared:
    t_session = time.perf_counter()
    chat_session_id = await create_or_get_workspace_chat_session(
        user_id=user_id,
        session_id=session_id,
        initial_message=normalized_message,
    )
    timings.session_ms = (time.perf_counter() - t_session) * 1000
    session_attachments = await list_rag_chat_session_attachments(
        session_id=chat_session_id,
        owner_id=user_id,
        agent_id=None,
    )
    session_attachment_resource_ids = [
        attachment.resource_id
        for attachment in session_attachments
        if attachment.state == "ready" and attachment.resource_id
    ]
    session_attachment_files = [
        attachment.filename
        for attachment in session_attachments
        if attachment.state == "ready" and attachment.filename
    ]

    router_decision = await classify_chat_action(message=normalized_message)
    workspace_resource_ids: list[str] = []
    if should_use_workspace_resources(router_decision):
        workspace_resource_ids = await list_workspace_ready_resource_ids(user_id)
    merged_resource_ids = list(
        dict.fromkeys(workspace_resource_ids + session_attachment_resource_ids)
    )

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
        reference_tools = reference_flags_from_tools(tools)
    else:
        bind_tools, tool_skip_reason = should_bind_composio_tools(
            message=normalized_message,
            resource_ids=merged_resource_ids,
            composio_apps=composio_apps,
        )
        allow_web_search = True
        reference_tools = default_reference_tool_flags()
    timings.tools_bound = bind_tools
    timings.tool_skip_reason = tool_skip_reason

    t0 = time.perf_counter()
    rag_context, user_memory_context, history = await asyncio.gather(
        retrieve_merged_context_for_agent_chat(
            user_id=user_id,
            agent_resource_ids=workspace_resource_ids,
            session_attachment_resource_ids=session_attachment_resource_ids,
            session_attachment_files=session_attachment_files,
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
        session_attachment_files=session_attachment_files,
        bind_tools=bind_tools,
        composio_user_disabled=(tool_skip_reason == "user_disabled"),
    )
    timings.prepare_ms = (time.perf_counter() - t0) * 1000
    return RagChatPrepared(
        agent=None,
        resource_ids=merged_resource_ids,
        rag_context=rag_context,
        chat_session_id=chat_session_id,
        messages=messages,
        bind_tools=bind_tools,
        tool_skip_reason=tool_skip_reason,
        composio_apps=composio_apps,
        allow_web_search=allow_web_search,
        reference_tools=reference_tools,
        router_decision=router_decision,
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
    from src.api.deps import _generate_suggestions

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
    force: bool = False,
) -> None:
    if not force and not settings.rag_suggestions_deferred:
        return

    async def _run() -> None:
        from src.api.deps import _generate_suggestions
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
