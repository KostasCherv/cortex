"""FastAPI application — /health, /research (SSE), and session endpoints."""

import asyncio
import json
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import AsyncGenerator, cast

import inngest.fast_api as _inngest_fast_api
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.graph.graph import build_graph
from src.errors import CortexError
from src.observability.langfuse import (
    create_feedback_anchor_for_run,
    create_trace_id_for_workflow,
    submit_user_feedback_score,
)
from src.observability import end_workflow_run, start_workflow_run
from src.config import settings
from src.auth import AuthenticatedUser, get_authenticated_user
from src.cache.client import get_cache
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
    get_session_run,
)
from src.tools.neo4j_graph_store import Neo4jGraphStore
from src.tools.reranker import rerank_chunks
from src.tools.asset_price_provider import (
    get_asset_price_tool,
    validate_asset_price_provider_health,
)
from src.tools.alpha_vantage_mcp_client import (
    call_alpha_vantage_mcp_tool,
    get_alpha_vantage_mcp_tool_definition,
    initialize_alpha_vantage_mcp_client,
    search_alpha_vantage_mcp_tools,
    shutdown_alpha_vantage_mcp_client,
)
from src.tools.web_search import get_web_search_tool, validate_web_search_provider_health
from src.tools.fetcher import fetch_url_content
from src.llm.factory import get_llm
from src.llm.output_parsers import (
    build_validation_retry_prompt,
    parse_chat_action_json,
    parse_finance_tool_call_plan_json,
    parse_finance_tool_selection_json,
)
from src.prompts.registry import prompt_registry
from src.guards import claims_no_web_access
from src import outbox
from src.planner import (
    PlannerValidationError,
    SavedSoftwareDevPlan,
    SavedSoftwareDevPlanListResponse,
    SoftwareDevPlanResponse,
    generate_software_dev_plan,
    get_saved_software_dev_plan,
    list_saved_software_dev_plans,
    save_software_dev_plan,
)
from src.rag import (
    CHAT_SCOPE_AGENT,
    CHAT_SCOPE_WORKSPACE,
    RagChatMessage,
    RagValidationError,
    append_chat_message,
    delete_last_exchange,
    delete_chat_session as delete_rag_chat_session,
    create_agent as create_rag_agent_record,
    create_or_get_chat_session,
    create_or_get_workspace_chat_session,
    create_resource_and_ingest,
    delete_agent as delete_rag_agent_record,
    delete_resource as delete_rag_resource_record,
    get_agent_for_chat,
    get_chat_session as get_rag_chat_session,
    get_resource_status,
    link_resources as link_rag_resources,
    list_agents as list_rag_agents_records,
    list_chat_messages as list_rag_chat_messages,
    list_chat_sessions as list_rag_chat_sessions,
    list_resources as list_rag_resources_records,
    list_workspace_ready_resource_ids,
    retrieve_context_for_query,
    suggest_agent_definition as suggest_rag_agent_definition,
    update_chat_session_title as update_rag_chat_session_title,
    update_agent as update_rag_agent_record,
)
from src.inngest_client import handle_rag_ingestion, handle_research_run, dispatch_outbox_cron, inngest_client
from src.storage import ensure_rag_storage_ready
from src.billing.application import BillingService, UsageIncrement
from src.billing.domain import BillingSyncError, QuotaExceededError
from src.billing.interfaces.http import build_billing_service, usage_summary_to_response
from src.api.planner_chat import router as planner_chat_router
from src.planner_graph.thread_store import planner_thread_store

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

_inngest_fast_api.serve(app, inngest_client, [handle_rag_ingestion, handle_research_run, dispatch_outbox_cron])
app.include_router(planner_chat_router)


@app.on_event("startup")
async def validate_session_store_configuration() -> None:
    """Validate critical runtime dependencies and session persistence wiring."""
    _configure_application_logging()
    validate_web_search_provider_health()
    validate_asset_price_provider_health()
    if (settings.asset_price_provider or "").strip().lower() == "alphavantage_mcp":
        try:
            client = await initialize_alpha_vantage_mcp_client()
            logger.info(
                "[startup] Alpha Vantage MCP catalog loaded with %s tools.",
                len(client.list_available_tools()),
            )
        except Exception as exc:
            logger.warning(
                "[startup] Alpha Vantage MCP catalog initialization failed; "
                "requests may fall back to yfinance: %s",
                exc,
            )
    if not settings.cohere_api_key:
        logger.warning("[startup] Cohere reranking is disabled (COHERE_API_KEY not set).")

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

    async def _evict_planner_threads_periodically() -> None:
        while True:
            await asyncio.sleep(600)  # every 10 minutes
            count = planner_thread_store.evict_expired()
            if count > 0:
                logger.info("[planner] Evicted %d expired planner threads.", count)

    asyncio.ensure_future(_evict_planner_threads_periodically())


@app.on_event("shutdown")
async def shutdown_background_clients() -> None:
    """Stop long-lived background clients gracefully."""
    await shutdown_alpha_vantage_mcp_client()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str


class FollowupRequest(BaseModel):
    question: str
    run_id: str | None = None


class CreateSessionRequest(BaseModel):
    query: str | None = None


class UpdateSessionTitleRequest(BaseModel):
    title: str


class RunFeedbackRequest(BaseModel):
    helpful: bool
    comment: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class RagAgentCreateRequest(BaseModel):
    name: str
    description: str = ""
    system_instructions: str = ""
    linked_resource_ids: list[str] = []


class RagAgentDraftRequest(BaseModel):
    prompt: str


class SoftwareDevPlanRequest(BaseModel):
    prompt: str


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


class BillingCheckoutRequest(BaseModel):
    pass


class PlannerChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(CortexError)
async def cortex_error_handler(request: Request, exc: CortexError):
    raise HTTPException(status_code=500, detail=str(exc))


def _raise_rag_validation_error(exc: RagValidationError) -> None:
    status_by_code = {
        "unsupported_type": 400,
        "size_exceeded": 400,
        "workspace_limit_exceeded": 400,
        "agent_resource_limit_exceeded": 400,
        "agent_prompt_required": 400,
        "agent_draft_generation_failed": 502,
        "processing_failed": 409,
        "unauthorized_linkage": 403,
    }
    raise HTTPException(
        status_code=status_by_code.get(exc.code, 400),
        detail={"code": exc.code, "message": str(exc)},
    )


def _raise_planner_validation_error(exc: PlannerValidationError) -> None:
    status_by_code = {
        "planner_prompt_required": 400,
        "planner_generation_failed": 502,
    }
    raise HTTPException(
        status_code=status_by_code.get(exc.code, 400),
        detail={"code": exc.code, "message": str(exc)},
    )


_billing_service: BillingService | None = None


def _get_billing_service() -> BillingService:
    global _billing_service
    if _billing_service is None:
        _billing_service = build_billing_service()
    return _billing_service


def _raise_quota_exceeded(exc: QuotaExceededError) -> None:
    raise HTTPException(
        status_code=429,
        detail={
            "code": "quota_exceeded",
            "plan": exc.plan,
            "limit_type": exc.limit_type,
            "limit": exc.limit,
            "used": exc.used,
            "resets_at": exc.resets_at,
            "message": exc.message,
        },
    )


async def _consume_usage_or_429(user_id: str, increment: UsageIncrement) -> None:
    try:
        await _get_billing_service().check_and_consume_usage(user_id, increment)
    except QuotaExceededError as exc:
        _raise_quota_exceeded(exc)


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


def _build_web_citations(results: list[dict] | None, provider: str) -> list[dict]:
    if not results:
        return []
    citations: list[dict] = []
    for index, row in enumerate(results):
        source_title = (
            row.get("title")
            or row.get("name")
            or row.get("symbol")
            or f"{provider} result {index + 1}"
        )
        citation_text = row.get("raw_content") or row.get("content") or ""
        if not citation_text and row.get("symbol") and row.get("price") is not None:
            citation_text = (
                f"{row.get('symbol')} price {row.get('price')} "
                f"{row.get('currency') or ''} as of {row.get('as_of') or ''}"
            ).strip()
        citations.append(
            {
                "source_title": source_title,
                "source_url": row.get("url") or "",
                "chunk_id": f"{provider}-web-{index + 1}",
                "text": citation_text,
            }
        )
    return citations


def _build_chat_citations(
    *,
    rag_chunks: list[dict] | None,
    web_results: list[dict] | None,
    web_provider: str,
    prefer_web: bool = False,
) -> list[dict]:
    web_citations = _build_web_citations(web_results, web_provider)
    if prefer_web and web_citations:
        return web_citations
    return _build_rag_citations(rag_chunks) + web_citations


def _build_workspace_fallback_citations(
    rag_context_text: str,
    existing_citations: list[dict],
) -> list[dict]:
    if existing_citations:
        return existing_citations
    cleaned = (rag_context_text or "").strip()
    if not cleaned:
        return existing_citations
    return [
        {
            "source_title": "workspace resources",
            "source_url": None,
            "chunk_id": "workspace-context-fallback",
            "text": cleaned[:1200],
        }
    ]


def _extract_llm_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, (str, dict))
        )
    return str(content)


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://\S+", text or "")


@dataclass(frozen=True)
class _ResolvedWebContext:
    used: bool
    provider: str
    results: list[dict]
    reason: str = "disabled"


@dataclass(frozen=True)
class _ChatActionDecision:
    action: str
    reason: str
    query: str = ""
    url: str = ""
    symbols: list[str] | None = None
    currency: str = ""


def _is_rag_context_insufficient(
    rag_context: str,
    rag_chunks: list[dict] | None,
    message: str,
) -> bool:
    del message
    has_local_text = bool(rag_context.strip()) or any(
        str(chunk.get("text") or "").strip() for chunk in (rag_chunks or [])
    )
    if not has_local_text:
        return True

    relevance_scores: list[float] = []
    for chunk in rag_chunks or []:
        raw_score = chunk.get("rerank_score")
        if raw_score is None:
            continue
        try:
            relevance_scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue

    if relevance_scores:
        return max(relevance_scores) < settings.rerank_relevance_threshold

    return False


async def _search_web(query: str) -> tuple[str, list[dict]]:
    tool = get_web_search_tool()
    results = await asyncio.to_thread(tool.search, query, settings.max_search_results)
    return tool.provider_name, results


async def _quote_asset_prices(symbols: list[str], currency: str = "") -> tuple[str, list[dict]]:
    tool = get_asset_price_tool()
    results = await asyncio.to_thread(tool.quote, symbols, currency or None)
    return tool.provider_name, results


def _tool_match_to_context_row(match: object) -> dict[str, object]:
    name = getattr(match, "name", "") if not isinstance(match, dict) else str(match.get("name") or "")
    description = (
        getattr(match, "description", "")
        if not isinstance(match, dict)
        else str(match.get("description") or "")
    )
    score = getattr(match, "score", None) if not isinstance(match, dict) else match.get("score")
    why = getattr(match, "why", "") if not isinstance(match, dict) else str(match.get("why") or "")
    return {
        "tool_name": name,
        "title": f"Alpha Vantage MCP Tool · {name}",
        "url": "https://www.alphavantage.co/documentation/",
        "content": (
            f"Candidate tool: {name}\n"
            f"Description: {description}\n"
            f"Why matched: {why}\n"
            f"Score: {score}"
        ).strip(),
        "source": "alphavantage_mcp_catalog",
        "raw": {
            "tool_name": name,
            "description": description,
            "score": score,
            "why": why,
        },
    }


def _tool_definition_to_context_row(
    *,
    tool_definition,
    clarifying_question: str = "",
    reason: str = "",
) -> dict[str, object]:
    parameters = tool_definition.parameters if hasattr(tool_definition, "parameters") else {}
    return {
        "tool_name": tool_definition.name,
        "title": f"Alpha Vantage MCP Tool · {tool_definition.name}",
        "url": "https://www.alphavantage.co/documentation/",
        "content": (
            f"Selected tool: {tool_definition.name}\n"
            f"Description: {tool_definition.description}\n"
            f"Schema: {json.dumps(parameters, ensure_ascii=True)}\n"
            f"Reason: {reason}\n"
            f"Clarifying question: {clarifying_question}"
        ).strip(),
        "source": "alphavantage_mcp_catalog",
        "clarifying_question": clarifying_question,
        "raw": {
            "tool_name": tool_definition.name,
            "description": tool_definition.description,
            "parameters": parameters,
        },
    }


def _tool_call_payload_to_context_row(
    *,
    tool_name: str,
    arguments: dict[str, object],
    payload: object,
) -> dict[str, object]:
    serialized_payload = json.dumps(payload, ensure_ascii=True, default=str)
    compact_payload = serialized_payload[:6000]
    return {
        "tool_name": tool_name,
        "title": f"Alpha Vantage MCP · {tool_name}",
        "url": "https://www.alphavantage.co/documentation/",
        "content": (
            f"Source: Alpha Vantage MCP ({tool_name})\n"
            f"Arguments: {json.dumps(arguments, ensure_ascii=True, default=str)}\n"
            f"Payload: {compact_payload}"
        ).strip(),
        "raw_content": compact_payload,
        "source": "alphavantage_mcp",
        "raw": {
            "tool_name": tool_name,
            "arguments": arguments,
            "payload": payload,
        },
    }


async def _select_finance_tool_candidate(
    *,
    message: str,
    matches: list[object],
) -> str:
    llm = get_llm(temperature=0.0)
    shortlist = [
        {
            "name": getattr(match, "name", "") if not isinstance(match, dict) else str(match.get("name") or ""),
            "description": (
                getattr(match, "description", "")
                if not isinstance(match, dict)
                else str(match.get("description") or "")
            ),
            "why": getattr(match, "why", "") if not isinstance(match, dict) else str(match.get("why") or ""),
            "score": getattr(match, "score", None) if not isinstance(match, dict) else match.get("score"),
        }
        for match in matches
    ]
    prompt, _ = prompt_registry.render(
        "finance_tool_selection",
        {
            "message": message,
            "candidate_tools_json": json.dumps(shortlist),
        },
    )
    raw = ""
    try:
        result = await llm.ainvoke(prompt)
        raw = _extract_llm_text(result.content if hasattr(result, "content") else result).strip()
        parsed = parse_finance_tool_selection_json(raw)
    except Exception as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text='{"tool_name":"<tool name from shortlist>","reason":"<snake_case_reason>"}',
            invalid_response=raw,
            validation_error=exc,
        )
        repair_result = await llm.ainvoke(repair_prompt)
        repair_raw = _extract_llm_text(
            repair_result.content if hasattr(repair_result, "content") else repair_result
        ).strip()
        parsed = parse_finance_tool_selection_json(repair_raw)

    valid_names = {item["name"] for item in shortlist if item["name"]}
    if parsed.tool_name not in valid_names:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "finance_tool_selection_invalid",
                "error": f"Selected finance tool '{parsed.tool_name}' was not in the shortlist.",
            },
        )
    return parsed.tool_name


async def _plan_finance_tool_call(
    *,
    message: str,
    tool_definition,
) -> dict[str, object]:
    llm = get_llm(temperature=0.0)
    prompt, _ = prompt_registry.render(
        "finance_tool_call_plan",
        {
            "message": message,
            "tool_name": tool_definition.name,
            "tool_description": tool_definition.description,
            "tool_parameters_json": json.dumps(tool_definition.parameters, ensure_ascii=True),
        },
    )
    raw = ""
    try:
        result = await llm.ainvoke(prompt)
        raw = _extract_llm_text(result.content if hasattr(result, "content") else result).strip()
        parsed = parse_finance_tool_call_plan_json(raw)
    except Exception as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text=(
                '{"should_call":true,"reason":"<snake_case_reason>",'
                '"arguments":{"<arg>":"<value>"},"clarifying_question":"<empty when callable>"}'
            ),
            invalid_response=raw,
            validation_error=exc,
        )
        repair_result = await llm.ainvoke(repair_prompt)
        repair_raw = _extract_llm_text(
            repair_result.content if hasattr(repair_result, "content") else repair_result
        ).strip()
        parsed = parse_finance_tool_call_plan_json(repair_raw)

    return {
        "should_call": parsed.should_call,
        "reason": parsed.reason,
        "arguments": parsed.arguments,
        "clarifying_question": parsed.clarifying_question,
    }


async def _repair_url_access_refusal_if_needed(
    *,
    answer: str,
    normalized_message: str,
    rag_context: str,
    web_results: list[dict],
    llm,
) -> str:
    if not claims_no_web_access(answer):
        return answer

    if not web_results:
        return answer

    repair_prompt, _ = prompt_registry.render(
        "web_search_repair",
        {
            "normalized_message": normalized_message,
            "web_results_json": json.dumps(web_results),
            "rag_context": rag_context,
        },
    )
    repair = await llm.ainvoke(repair_prompt)
    repaired_text = _extract_llm_text(
        repair.content if hasattr(repair, "content") else repair
    ).strip()
    if repaired_text and not claims_no_web_access(repaired_text):
        return repaired_text

    first = web_results[0]
    source = first.get("url") or "provided URL"
    snippet = (first.get("raw_content") or first.get("content") or "").strip()
    snippet = snippet[:1000]
    return (
        f"I fetched content from {source}. "
        f"Here is a concise summary based on the retrieved page text:\n\n{snippet}"
    ).strip()


async def _decide_chat_action(
    *,
    message: str,
    rag_context: str,
    rag_chunks: list[dict] | None = None,
    history_block: str,
) -> _ChatActionDecision:
    llm = get_llm(temperature=0.0)
    rag_is_insufficient = _is_rag_context_insufficient(message=message, rag_context=rag_context, rag_chunks=rag_chunks)
    message_urls = _extract_urls(message)
    history_urls = _extract_urls(history_block)
    decision_prompt, _ = prompt_registry.render(
        "web_search_decision",
        {
            "history_block": history_block,
            "rag_context": rag_context,
            "rag_is_insufficient": rag_is_insufficient,
            "message": message,
            "message_urls": message_urls,
            "history_urls": history_urls,
        },
    )
    try:
        result = await llm.ainvoke(decision_prompt)
        raw = _extract_llm_text(result.content if hasattr(result, "content") else result).strip()
        parsed = parse_chat_action_json(raw)
    except Exception as exc:
        repair_prompt = build_validation_retry_prompt(
            schema_text=(
                '{"action":"answer_direct|answer_from_rag|web_search|asset_price|search_finance_tools|fetch_url|ask_clarifying",'
                '"reason":"<why this action was chosen>","query":"<search query or empty>",'
                '"url":"<url or empty>","symbols":["<asset symbols for asset_price>"],'
                '"currency":"<quote currency or empty>"}'
            ),
            invalid_response=raw if "raw" in locals() else "",
            validation_error=exc,
        )
        try:
            repair_result = await llm.ainvoke(repair_prompt)
            repair_raw = _extract_llm_text(
                repair_result.content if hasattr(repair_result, "content") else repair_result
            ).strip()
            parsed = parse_chat_action_json(repair_raw)
        except Exception:
            return _ChatActionDecision(action="answer_direct", reason="router_parse_failed")

    return _ChatActionDecision(
        action=parsed.action,
        reason=parsed.reason.strip(),
        query=parsed.query.strip(),
        url=parsed.url.strip(),
        symbols=parsed.symbols,
        currency=parsed.currency.strip(),
    )


async def _resolve_web_context(
    *,
    normalized_message: str,
    rag_context: str,
    rag_chunks: list[dict] | None = None,
    history_block: str,
    decision: _ChatActionDecision | None = None,
) -> _ResolvedWebContext:
    web_provider = settings.web_search_provider.lower()

    decision = decision or await _decide_chat_action(
        message=normalized_message,
        rag_context=rag_context,
        rag_chunks=rag_chunks,
        history_block=history_block,
    )

    if decision.action in {"answer_direct", "answer_from_rag", "ask_clarifying"}:
        return _ResolvedWebContext(False, web_provider, [], "not_needed")

    if decision.action == "fetch_url":
        direct_urls = (
            ([decision.url] if decision.url else [])
            or _extract_urls(normalized_message)
            or _extract_urls(history_block)
        )
        if direct_urls:
            url = direct_urls[0]
            fetched = await fetch_url_content(url)
            return _ResolvedWebContext(
                used=True,
                provider="direct_fetch",
                results=[{"url": url, "title": url, "content": fetched, "raw_content": fetched}],
                reason="direct_fetch",
            )
        return _ResolvedWebContext(False, web_provider, [], "not_needed")

    if decision.action == "asset_price":
        provider, results = await _quote_asset_prices(
            decision.symbols or [],
            decision.currency,
        )
        return _ResolvedWebContext(True, provider, results, "asset_price")

    if decision.action == "search_finance_tools":
        matches = await search_alpha_vantage_mcp_tools(decision.query or normalized_message, limit=5)
        if not matches:
            return _ResolvedWebContext(False, "alphavantage_mcp", [], "not_needed")

        selected_tool_name = await _select_finance_tool_candidate(
            message=normalized_message,
            matches=matches,
        )
        tool_definition = await get_alpha_vantage_mcp_tool_definition(selected_tool_name)
        call_plan = await _plan_finance_tool_call(
            message=normalized_message,
            tool_definition=tool_definition,
        )

        if call_plan["should_call"]:
            payload = await call_alpha_vantage_mcp_tool(
                selected_tool_name,
                cast(dict[str, object], call_plan["arguments"]),
            )
            return _ResolvedWebContext(
                True,
                "alphavantage_mcp",
                [
                    _tool_call_payload_to_context_row(
                        tool_name=selected_tool_name,
                        arguments=cast(dict[str, object], call_plan["arguments"]),
                        payload=payload,
                    )
                ],
                "finance_tool_call",
            )

        shortlist_rows = [_tool_match_to_context_row(match) for match in matches[:3]]
        shortlist_rows.insert(
            0,
            _tool_definition_to_context_row(
                tool_definition=tool_definition,
                clarifying_question=str(call_plan["clarifying_question"] or ""),
                reason=str(call_plan["reason"] or ""),
            ),
        )
        return _ResolvedWebContext(
            True,
            "alphavantage_mcp_catalog",
            shortlist_rows,
            "finance_tool_search",
        )

    if decision.action != "web_search":
        return _ResolvedWebContext(False, web_provider, [], "not_needed")

    web_query = decision.query or normalized_message
    direct_urls = _extract_urls(decision.url) if decision.url else []
    if direct_urls:
        url = direct_urls[0]
        fetched = await fetch_url_content(url)
        return _ResolvedWebContext(
            used=True,
            provider="direct_fetch",
            results=[{"url": url, "title": url, "content": fetched, "raw_content": fetched}],
            reason="direct_fetch",
        )

    provider, results = await _search_web(web_query)
    return _ResolvedWebContext(True, provider, results, "model_decision")


def _prefer_web_citations(resolved_web: _ResolvedWebContext) -> bool:
    return resolved_web.reason in {
        "direct_fetch",
        "model_decision",
        "asset_price",
        "finance_tool_call",
        "finance_tool_search",
    }


def _rag_context_for_answer(rag_context: str, resolved_web: _ResolvedWebContext) -> str:
    if resolved_web.reason == "direct_fetch":
        return ""
    return rag_context


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


def _build_chat_messages(
    *,
    system_instructions: str,
    history: list,
    rag_context: str,
    web_results: list[dict],
    normalized_message: str,
    router_action: str | None = None,
) -> list[BaseMessage]:
    system_content, _ = prompt_registry.render(
        "rag_chat_system",
        {
            "system_instructions": system_instructions,
            "rag_context": rag_context,
            "web_results_json": json.dumps(web_results) if web_results else "None",
        },
    )
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    if router_action == "ask_clarifying":
        messages.append(
            SystemMessage(
                content=(
                    "The router determined the user request is underspecified. "
                    "Ask one concise clarifying question and do not answer the original "
                    "request yet."
                )
            )
        )
    elif router_action == "search_finance_tools":
        clarifying_question = ""
        for row in web_results:
            if isinstance(row, dict) and str(row.get("clarifying_question") or "").strip():
                clarifying_question = str(row.get("clarifying_question")).strip()
                break
        if clarifying_question:
            messages.append(
                SystemMessage(
                    content=(
                        "The finance tool workflow found a likely tool but it cannot be "
                        "called confidently yet. Ask this concise clarifying question and do "
                        "not guess missing parameters: "
                        f"{clarifying_question}"
                    )
                )
            )
    for turn in history:
        role = turn.role if hasattr(turn, "role") else turn.get("role", "")
        content = turn.content if hasattr(turn, "content") else turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=normalized_message))
    return messages


def _should_repair_fetched_url_answer(decision: _ChatActionDecision) -> bool:
    return decision.action == "fetch_url"


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
            logger.warning("[session] could not persist web source in graph store: %s", exc)

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
    }

    graph_nodes = {
        "search_and_memory", "rerank", "summarize", "report",
        "abort", "empty",
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

                elif event_type == "on_chat_model_stream" and langgraph_node == "report":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue
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
            run = await get_session_run(run_id=run_id, user_id=user_id, session_id=session_id)
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

    history_block = "\n".join(
        f"{t.role.upper()}: {t.content}" for t in session.conversation[-6:]
    )
    try:
        resolved_web = await _resolve_web_context(
            normalized_message=question,
            rag_context=context_block,
            rag_chunks=chunks,
            history_block=history_block,
        )
    except Exception as exc:
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        return

    answer_context_block = _rag_context_for_answer(context_block, resolved_web)
    prompt, _ = prompt_registry.render(
        "followup_answer",
        {
            "history_block": history_block,
            "answer_context_block": answer_context_block,
            "web_results_json": json.dumps(resolved_web.results) if resolved_web.results else "None",
            "question": question,
            "report_block": report_block,
        },
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

    citations = _build_chat_citations(
        rag_chunks=chunks,
        web_results=resolved_web.results,
        web_provider=resolved_web.provider,
        prefer_web=_prefer_web_citations(resolved_web),
    )
    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

    # Generate suggestions before persisting so they are stored with the turn
    suggestions = await _generate_suggestions(question, full_answer, answer_context_block)

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


@app.post("/internal/dispatch-outbox", tags=["Internal"])
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


@app.get("/api/billing/usage", tags=["Billing"])
async def billing_usage(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    summary = await _get_billing_service().get_usage_summary(current_user.user_id)
    return usage_summary_to_response(summary)


@app.post("/api/billing/checkout-session", tags=["Billing"])
async def create_checkout_session(
    _body: BillingCheckoutRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        checkout_url = await _get_billing_service().start_checkout(
            user_id=current_user.user_id,
            email=current_user.email,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"url": checkout_url}


@app.post("/api/billing/portal-session", tags=["Billing"])
async def create_portal_session(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        portal_url = await _get_billing_service().start_portal(user_id=current_user.user_id)
    except BillingSyncError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"url": portal_url}


@app.post("/api/billing/webhook", tags=["Billing"])
async def stripe_webhook(request: Request):
    signature = request.headers.get("Stripe-Signature", "")
    payload = await request.body()
    try:
        await _get_billing_service().handle_webhook(payload, signature)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"received": True})


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
        "Create a concise title (max 4 words) for this research session.\n"
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
    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(research_queries=1, total_questions=1),
    )

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


@app.get("/sessions/{session_id}/runs/{run_id}/stream", tags=["Sessions"])
async def stream_session_run(
    session_id: str,
    run_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
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


@app.post("/sessions/{session_id}/runs/{run_id}/feedback", tags=["Sessions"])
async def submit_run_feedback(
    session_id: str,
    run_id: str,
    body: RunFeedbackRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    """Record simple user feedback for a completed run in LangFuse."""
    session = await get_session(session_id, current_user.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    run = session.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in session '{session_id}'.",
        )

    if run.feedback_submitted_at:
        raise HTTPException(status_code=409, detail="Feedback has already been submitted for this run.")
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
            raise HTTPException(status_code=502, detail=f"Could not link run to LangFuse: {exc}")
        if not trace_id:
            raise HTTPException(status_code=502, detail="Could not link run to LangFuse.")
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
            raise HTTPException(status_code=400, detail="Feedback comment cannot be empty.")
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
        raise HTTPException(status_code=502, detail=f"Could not submit LangFuse feedback: {exc}")

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


@app.post("/sessions/{session_id}/followup", tags=["Sessions"])
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


@app.post("/api/planner/software-dev", tags=["Planner"])
async def generate_implementation_plan(
    body: SoftwareDevPlanRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SavedSoftwareDevPlan:
    try:
        response = await generate_software_dev_plan(body.prompt)
        return await save_software_dev_plan(current_user.user_id, body.prompt, response)
    except PlannerValidationError as exc:
        _raise_planner_validation_error(exc)
        raise AssertionError("unreachable")


@app.get("/api/planner/software-dev/plans", tags=["Planner"])
async def list_software_dev_plan_history(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SavedSoftwareDevPlanListResponse:
    plans = await list_saved_software_dev_plans(current_user.user_id)
    return SavedSoftwareDevPlanListResponse(plans=plans)


@app.get("/api/planner/software-dev/plans/{plan_id}", tags=["Planner"])
async def get_software_dev_plan_history_detail(
    plan_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
) -> SavedSoftwareDevPlan:
    plan = await get_saved_software_dev_plan(current_user.user_id, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Saved plan '{plan_id}' not found.")
    return plan


@app.post("/api/rag/agents/draft", tags=["RAG"])
async def rag_generate_agent_draft(
    body: RagAgentDraftRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        draft = await suggest_rag_agent_definition(body.prompt)
    except RagValidationError as exc:
        _raise_rag_validation_error(exc)
    return {"draft": draft.to_dict()}


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


@app.delete("/api/rag/agents/{agent_id}", tags=["RAG"])
async def rag_delete_agent(
    agent_id: str,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    deleted = await delete_rag_agent_record(agent_id, current_user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {"agent_id": agent_id, "deleted": True}


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
    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(total_questions=1),
    )

    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        logger.warning(
            "[rag_api] agent chat request failed because agent was not found agent_id=%s user_id=%s",
            agent_id,
            current_user.user_id,
        )
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    agent, resource_ids = agent_bundle
    if not resource_ids:
        logger.info(
            "[rag_api] agent chat request proceeding without linked ready resources agent_id=%s user_id=%s",
            agent_id,
            current_user.user_id,
        )

    rag_context = await retrieve_context_for_query(
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
    try:
        decision = await _decide_chat_action(
            message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
        )
        resolved_web = await _resolve_web_context(
            normalized_message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
            decision=decision,
        )
    except Exception as exc:
        logger.exception(
            "[rag_api] streaming agent chat external tool resolution failed "
            "agent_id=%s user_id=%s action=%s",
            agent_id,
            current_user.user_id,
            decision.action if "decision" in locals() and decision else "unknown",
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "web_search_unavailable",
                "message": (
                    "Web search is required for this request but the provider failed. "
                    "Retry after the web provider recovers."
                ),
                "provider": settings.web_search_provider.lower(),
                "error": str(exc),
            },
        ) from exc

    answer_rag_context = _rag_context_for_answer(rag_context.context or "", resolved_web)
    messages = _build_chat_messages(
        system_instructions=agent.system_instructions or "None",
        history=history,
        rag_context=answer_rag_context,
        web_results=resolved_web.results,
        normalized_message=normalized_message,
        router_action=decision.action,
    )
    llm = get_llm(temperature=0.2)
    result = await llm.ainvoke(messages)
    content = result.content
    if not isinstance(content, str):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    answer = content.strip()
    if resolved_web.used and _should_repair_fetched_url_answer(decision):
        answer = await _repair_url_access_refusal_if_needed(
            answer=answer,
            normalized_message=normalized_message,
            rag_context=answer_rag_context,
            web_results=resolved_web.results,
            llm=llm,
        )
    suggestions = await _generate_suggestions(
        normalized_message,
        answer,
        answer_rag_context,
    )

    user_msg = RagChatMessage(
        message_id=str(uuid.uuid4()),
        session_id=chat_session_id,
        agent_id=agent_id,
        owner_id=current_user.user_id,
        role="user",
        content=normalized_message,
    )
    citations = _build_chat_citations(
        rag_chunks=rag_context.chunks,
        web_results=resolved_web.results,
        web_provider=resolved_web.provider,
        prefer_web=_prefer_web_citations(resolved_web),
    )
    assistant_msg = RagChatMessage(
        message_id=str(uuid.uuid4()),
        session_id=chat_session_id,
        agent_id=agent_id,
        owner_id=current_user.user_id,
        role="assistant",
        content=answer,
        citations=citations,
        suggestions=suggestions,
    )
    await append_chat_message(user_msg)
    await append_chat_message(assistant_msg)

    updated_history = await list_rag_chat_messages(chat_session_id, current_user.user_id)
    return {
        "session_id": chat_session_id,
        "agent_id": agent_id,
        "web_used": resolved_web.used,
        "web_provider": resolved_web.provider if resolved_web.used else None,
        "reply": assistant_msg.to_dict(),
        "messages": [m.to_dict() for m in updated_history],
    }


@app.post("/api/rag/agents/{agent_id}/chat/stream", tags=["RAG"])
async def rag_chat_with_agent_stream(
    agent_id: str,
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _consume_usage_or_429(
        current_user.user_id,
        UsageIncrement(total_questions=1),
    )

    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    agent_bundle = await get_agent_for_chat(agent_id, current_user.user_id)
    if agent_bundle is None:
        logger.warning(
            "[rag_api] streaming agent chat request failed because agent was not found agent_id=%s user_id=%s",
            agent_id,
            current_user.user_id,
        )
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    agent, resource_ids = agent_bundle
    if not resource_ids:
        logger.info(
            "[rag_api] streaming agent chat request proceeding without linked ready resources agent_id=%s user_id=%s",
            agent_id,
            current_user.user_id,
        )

    rag_context = await retrieve_context_for_query(
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
    try:
        decision = await _decide_chat_action(
            message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
        )
        resolved_web = await _resolve_web_context(
            normalized_message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
            decision=decision,
        )
    except Exception as exc:
        logger.exception(
            "[rag_api] agent chat external tool resolution failed "
            "agent_id=%s user_id=%s action=%s",
            agent_id,
            current_user.user_id,
            decision.action if "decision" in locals() and decision else "unknown",
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "web_search_unavailable",
                "message": (
                    "Web search is required for this request but the provider failed. "
                    "Retry after the web provider recovers."
                ),
                "provider": settings.web_search_provider.lower(),
                "error": str(exc),
            },
        ) from exc
    answer_rag_context = _rag_context_for_answer(rag_context.context or "", resolved_web)
    messages = _build_chat_messages(
        system_instructions=agent.system_instructions or "None",
        history=history,
        rag_context=answer_rag_context,
        web_results=resolved_web.results,
        normalized_message=normalized_message,
        router_action=decision.action,
    )

    citations = _build_chat_citations(
        rag_chunks=rag_context.chunks,
        web_results=resolved_web.results,
        web_provider=resolved_web.provider,
        prefer_web=_prefer_web_citations(resolved_web),
    )

    async def _stream_chat() -> AsyncGenerator[str, None]:
        llm = get_llm(temperature=0.2)
        answer_parts: list[str] = []
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': chat_session_id, 'web_used': resolved_web.used, 'web_provider': resolved_web.provider if resolved_web.used else None})}\n\n"
            if resolved_web.used and _should_repair_fetched_url_answer(decision):
                result = await llm.ainvoke(messages)
                answer = _extract_llm_text(
                    result.content if hasattr(result, "content") else result
                ).strip()
                answer = await _repair_url_access_refusal_if_needed(
                    answer=answer,
                    normalized_message=normalized_message,
                    rag_context=answer_rag_context,
                    web_results=resolved_web.results,
                    llm=llm,
                )
                if answer:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': answer})}\n\n"
            else:
                async for chunk in llm.astream(messages):
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
            suggestions = await _generate_suggestions(
                normalized_message,
                answer,
                answer_rag_context,
            )
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
                suggestions=suggestions,
            )
            await append_chat_message(user_msg)
            await append_chat_message(assistant_msg)
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            if suggestions:
                yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"
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


@app.delete(
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
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    deleted, err = await delete_last_exchange(session_id=session_id, user_id=current_user.user_id)
    if not deleted:
        if err == "empty":
            raise HTTPException(status_code=404, detail="Session has no messages to delete.")
        raise HTTPException(status_code=409, detail="Last two messages are not a user/assistant pair.")
    return {"session_id": session_id, "deleted": True}


@app.post("/api/rag/chat", tags=["RAG"])
async def rag_chat_workspace(
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _consume_usage_or_429(current_user.user_id, UsageIncrement(total_questions=1))
    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    resource_ids = await list_workspace_ready_resource_ids(current_user.user_id)
    rag_context = await retrieve_context_for_query(
        user_id=current_user.user_id,
        resource_ids=resource_ids,
        question=normalized_message,
    )
    chat_session_id = await create_or_get_workspace_chat_session(
        user_id=current_user.user_id,
        session_id=body.session_id,
        initial_message=normalized_message,
    )
    history = await list_rag_chat_messages(chat_session_id, current_user.user_id)
    history_block = "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-10:])
    try:
        decision = await _decide_chat_action(
            message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
        )
        resolved_web = await _resolve_web_context(
            normalized_message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
            decision=decision,
        )
    except Exception as exc:
        logger.exception(
            "[rag_api] workspace chat external tool resolution failed "
            "user_id=%s action=%s",
            current_user.user_id,
            decision.action if "decision" in locals() and decision else "unknown",
        )
        raise HTTPException(status_code=503, detail={"code": "web_search_unavailable", "error": str(exc)}) from exc

    answer_rag_context = _rag_context_for_answer(rag_context.context or "", resolved_web)
    messages = _build_chat_messages(
        system_instructions="You are a generic workspace chat assistant.",
        history=history,
        rag_context=answer_rag_context,
        web_results=resolved_web.results,
        normalized_message=normalized_message,
        router_action=decision.action,
    )
    llm = get_llm(temperature=0.2)
    result = await llm.ainvoke(messages)
    answer = _extract_llm_text(result.content if hasattr(result, "content") else result).strip()
    if resolved_web.used and _should_repair_fetched_url_answer(decision):
        answer = await _repair_url_access_refusal_if_needed(
            answer=answer,
            normalized_message=normalized_message,
            rag_context=answer_rag_context,
            web_results=resolved_web.results,
            llm=llm,
        )
    suggestions = await _generate_suggestions(normalized_message, answer, answer_rag_context)
    user_msg = RagChatMessage(
        message_id=str(uuid.uuid4()),
        session_id=chat_session_id,
        agent_id=None,
        owner_id=current_user.user_id,
        role="user",
        content=normalized_message,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    citations = _build_chat_citations(
        rag_chunks=rag_context.chunks,
        web_results=resolved_web.results,
        web_provider=resolved_web.provider,
        prefer_web=_prefer_web_citations(resolved_web),
    )
    citations = _build_workspace_fallback_citations(rag_context.context or "", citations)
    assistant_msg = RagChatMessage(
        message_id=str(uuid.uuid4()),
        session_id=chat_session_id,
        agent_id=None,
        owner_id=current_user.user_id,
        role="assistant",
        content=answer,
        citations=citations,
        suggestions=suggestions,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    await append_chat_message(user_msg)
    await append_chat_message(assistant_msg)
    updated_history = await list_rag_chat_messages(chat_session_id, current_user.user_id)
    return {
        "session_id": chat_session_id,
        "agent_id": None,
        "web_used": resolved_web.used,
        "web_provider": resolved_web.provider if resolved_web.used else None,
        "reply": assistant_msg.to_dict(),
        "messages": [m.to_dict() for m in updated_history],
    }


@app.post("/api/rag/chat/stream", tags=["RAG"])
async def rag_chat_workspace_stream(
    body: RagChatRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    await _consume_usage_or_429(current_user.user_id, UsageIncrement(total_questions=1))
    normalized_message = body.message.strip()
    if not normalized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    resource_ids = await list_workspace_ready_resource_ids(current_user.user_id)
    rag_context = await retrieve_context_for_query(
        user_id=current_user.user_id,
        resource_ids=resource_ids,
        question=normalized_message,
    )
    chat_session_id = await create_or_get_workspace_chat_session(
        user_id=current_user.user_id,
        session_id=body.session_id,
        initial_message=normalized_message,
    )
    history = await list_rag_chat_messages(chat_session_id, current_user.user_id)
    history_block = "\n".join(f"{m.role.upper()}: {m.content}" for m in history[-10:])
    try:
        decision = await _decide_chat_action(
            message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
        )
        resolved_web = await _resolve_web_context(
            normalized_message=normalized_message,
            rag_context=rag_context.context or "",
            rag_chunks=rag_context.chunks,
            history_block=history_block,
            decision=decision,
        )
    except Exception as exc:
        logger.exception(
            "[rag_api] streaming workspace chat external tool resolution failed "
            "user_id=%s action=%s",
            current_user.user_id,
            decision.action if "decision" in locals() and decision else "unknown",
        )
        raise HTTPException(status_code=503, detail={"code": "web_search_unavailable", "error": str(exc)}) from exc
    answer_rag_context = _rag_context_for_answer(rag_context.context or "", resolved_web)
    messages = _build_chat_messages(
        system_instructions="You are a generic workspace chat assistant.",
        history=history,
        rag_context=answer_rag_context,
        web_results=resolved_web.results,
        normalized_message=normalized_message,
        router_action=decision.action,
    )
    citations = _build_chat_citations(
        rag_chunks=rag_context.chunks,
        web_results=resolved_web.results,
        web_provider=resolved_web.provider,
        prefer_web=_prefer_web_citations(resolved_web),
    )
    citations = _build_workspace_fallback_citations(rag_context.context or "", citations)

    async def _stream_chat() -> AsyncGenerator[str, None]:
        llm = get_llm(temperature=0.2)
        answer_parts: list[str] = []
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': chat_session_id, 'web_used': resolved_web.used, 'web_provider': resolved_web.provider if resolved_web.used else None})}\n\n"
            if resolved_web.used and _should_repair_fetched_url_answer(decision):
                result = await llm.ainvoke(messages)
                answer = _extract_llm_text(
                    result.content if hasattr(result, "content") else result
                ).strip()
                answer = await _repair_url_access_refusal_if_needed(
                    answer=answer,
                    normalized_message=normalized_message,
                    rag_context=answer_rag_context,
                    web_results=resolved_web.results,
                    llm=llm,
                )
                if answer:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': answer})}\n\n"
            else:
                async for chunk in llm.astream(messages):
                    text = _extract_llm_text(chunk.content if hasattr(chunk, "content") else chunk)
                    if not text:
                        continue
                    answer_parts.append(text)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"
                answer = "".join(answer_parts).strip()
            suggestions = await _generate_suggestions(normalized_message, answer, answer_rag_context)
            await append_chat_message(
                RagChatMessage(
                    message_id=str(uuid.uuid4()),
                    session_id=chat_session_id,
                    agent_id=None,
                    owner_id=current_user.user_id,
                    role="user",
                    content=normalized_message,
                    chat_scope=CHAT_SCOPE_WORKSPACE,
                )
            )
            await append_chat_message(
                RagChatMessage(
                    message_id=str(uuid.uuid4()),
                    session_id=chat_session_id,
                    agent_id=None,
                    owner_id=current_user.user_id,
                    role="assistant",
                    content=answer,
                    citations=citations,
                    suggestions=suggestions,
                    chat_scope=CHAT_SCOPE_WORKSPACE,
                )
            )
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            if suggestions:
                yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        _stream_chat(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/rag/chat/sessions", tags=["RAG"])
async def list_rag_workspace_chat_sessions(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    sessions = await list_rag_chat_sessions(
        agent_id=None,
        user_id=current_user.user_id,
        chat_scope=CHAT_SCOPE_WORKSPACE,
    )
    return {"sessions": sessions}


@app.get("/api/rag/chat/sessions/{session_id}/messages", tags=["RAG"])
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


@app.patch("/api/rag/chat/sessions/{session_id}", tags=["RAG"])
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


@app.delete("/api/rag/chat/sessions/{session_id}", tags=["RAG"])
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


@app.delete("/api/rag/chat/sessions/{session_id}/last-exchange", tags=["RAG"])
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
        raise HTTPException(status_code=409, detail="Last two messages are not a user/assistant pair.")
    return {"session_id": session_id, "deleted": True}
