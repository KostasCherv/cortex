"""Shared helpers used by more than one router module in ``src/api/routers/``.

Pure code motion from ``src/api/endpoints.py`` — no logic changes.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from fastapi import HTTPException
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.billing import (
    BillingService,
    QuotaExceededError,
    UsageIncrement,
    build_billing_service,
)
from src.config import settings
from src.llm.factory import get_llm
from src.llm.text_utils import extract_llm_text
from src.observability import start_step_span
from src.rag import RagValidationError
from src.tools.arxiv_mcp import (
    ARXIV_MCP_TOOL_NAMES,
    arxiv_mcp_tools_context,
)
from src.tools.composio_toolset import get_composio_toolset_manager
from src.tools.general import (
    GENERAL_WEB_TOOL_NAMES,
    build_agent_tools,
    should_mark_web_used,
)
from src.tools.registry import create_rag_chat_tools_model, is_arxiv_mcp_enabled

logger = logging.getLogger(__name__)

_TOOL_SOURCE_TYPES = frozenset({"web", "wikipedia", "open_library", "arxiv"})


# ---------------------------------------------------------------------------
# Request models shared across RAG chat routers
# ---------------------------------------------------------------------------


RagChatTools = create_rag_chat_tools_model()


class RagChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    tools: RagChatTools = Field(default_factory=RagChatTools)


class CreateRagChatSessionRequest(BaseModel):
    filename: str | None = None


class UpdateSessionTitleRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# Billing usage helpers (billing, sessions, rag_agents, rag_chat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# RAG validation errors (rag_resources, rag_agents, rag_chat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Citation building + agent tool-calling loop (internal, sessions, rag_agents,
# rag_chat)
# ---------------------------------------------------------------------------


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
                "source_type": "rag",
            }
        )
    return citations


def _filter_relevant_rag_chunks(chunks: list[dict] | None) -> list[dict]:
    if not chunks:
        return []

    relevant: list[dict] = []
    for chunk in chunks:
        raw_score = chunk.get("rerank_score")
        if raw_score is None:
            relevant.append(chunk)
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score >= settings.rerank_relevance_threshold:
            relevant.append(chunk)
    return relevant


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
                "source_type": "web",
            }
        )
    return citations


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
            "source_type": "rag_fallback",
        }
    ]


def _normalize_tool_result(raw_result: object) -> tuple[str, object | None]:
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        content, artifact = raw_result
        return str(content), artifact
    return str(raw_result), None


async def _invoke_tool_raw_result(tool: object, tool_args: dict) -> object:
    """Invoke a LangChain tool preserving content+artifact when configured."""
    if getattr(tool, "response_format", None) == "content_and_artifact":
        coroutine = getattr(tool, "coroutine", None)
        if coroutine is not None:
            return await coroutine(**tool_args)
    return await tool.arun(tool_args)


def _merge_citations(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str | None, str, str]] = set()
    for group in groups:
        for citation in group:
            source_title = str(citation.get("source_title") or "source")
            raw_url = citation.get("source_url")
            source_url = str(raw_url) if isinstance(raw_url, str) else None
            chunk_id = str(citation.get("chunk_id") or "")
            text = str(citation.get("text") or "")
            key = (source_title, source_url, chunk_id, text[:160])
            if key in seen:
                continue
            seen.add(key)
            source_type = citation.get("source_type")
            merged.append(
                {
                    "source_title": source_title,
                    "source_url": source_url,
                    "chunk_id": chunk_id,
                    "text": text,
                    **(
                        {"source_type": source_type}
                        if isinstance(source_type, str) and source_type
                        else {}
                    ),
                }
            )
    return merged


def _iter_nested_dicts(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_nested_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _build_wikipedia_citations(results: object) -> list[dict]:
    if not isinstance(results, list):
        return []
    citations: list[dict] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or f"Wikipedia result {index + 1}")
        page_slug = title.replace(" ", "_")
        citations.append(
            {
                "source_title": title,
                "source_url": (
                    f"https://en.wikipedia.org/wiki/{page_slug}" if page_slug else None
                ),
                "chunk_id": f"wikipedia-{index + 1}",
                "text": str(row.get("extract") or ""),
                "source_type": "wikipedia",
            }
        )
    return citations


def _build_open_library_citations(results: object) -> list[dict]:
    if not isinstance(results, list):
        return []
    citations: list[dict] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        text_parts = [
            str(row.get("authors") or "").strip(),
            str(row.get("year") or "").strip(),
        ]
        citations.append(
            {
                "source_title": str(
                    row.get("title") or f"Open Library result {index + 1}"
                ),
                "source_url": str(row.get("url") or "") or None,
                "chunk_id": f"open-library-{index + 1}",
                "text": "\n".join(part for part in text_parts if part),
                "source_type": "open_library",
            }
        )
    return citations


def _build_arxiv_tool_citations(
    tool_name: str,
    tool_args: dict,
    content: str,
    artifact: object | None,
) -> list[dict]:
    paper_id = str(tool_args.get("paper_id") or tool_args.get("id") or "").strip()
    start = tool_args.get("start") or 0
    structured = artifact
    if isinstance(artifact, dict) and "structured_content" in artifact:
        structured = artifact.get("structured_content")

    citations: list[dict] = []
    if tool_name == "search_papers":
        seen_ids: set[str] = set()
        for index, row in enumerate(_iter_nested_dicts(structured)):
            looks_like_paper = any(
                key in row
                for key in (
                    "paper_id",
                    "arxiv_id",
                    "entry_id",
                    "title",
                    "abstract",
                    "summary",
                    "pdf_url",
                    "abs_url",
                )
            )
            if not looks_like_paper:
                continue
            candidate_id = str(
                row.get("paper_id")
                or row.get("arxiv_id")
                or row.get("id")
                or row.get("entry_id")
                or ""
            ).strip()
            if candidate_id and candidate_id in seen_ids:
                continue
            if candidate_id:
                seen_ids.add(candidate_id)
            title = str(
                row.get("title")
                or (
                    f"arXiv:{candidate_id}"
                    if candidate_id
                    else f"arXiv result {index + 1}"
                )
            )
            url = str(row.get("abs_url") or row.get("pdf_url") or "") or (
                f"https://arxiv.org/abs/{candidate_id}" if candidate_id else None
            )
            text = str(
                row.get("abstract") or row.get("summary") or row.get("content") or ""
            )
            citations.append(
                {
                    "source_title": title,
                    "source_url": url or None,
                    "chunk_id": f"arxiv-search-{candidate_id or index + 1}",
                    "text": text,
                    "source_type": "arxiv",
                }
            )
            if len(citations) >= 5:
                break
        return citations

    source_title = f"arXiv:{paper_id}" if paper_id else "arXiv paper"
    for row in _iter_nested_dicts(structured):
        maybe_id = str(row.get("paper_id") or row.get("arxiv_id") or "").strip()
        if maybe_id and not paper_id:
            paper_id = maybe_id
            source_title = str(row.get("title") or f"arXiv:{paper_id}")
            break
        if row.get("title"):
            source_title = str(row.get("title"))
    source_url = f"https://arxiv.org/abs/{paper_id}" if paper_id else None
    if not content.strip():
        return []
    return [
        {
            "source_title": source_title,
            "source_url": source_url,
            "chunk_id": f"{tool_name}:{paper_id or 'unknown'}:{start}",
            "text": content[:1200],
            "source_type": "arxiv",
        }
    ]


def _has_tool_or_web_citations(citations: list[dict]) -> bool:
    return any(
        isinstance(citation.get("source_type"), str)
        and citation["source_type"] in _TOOL_SOURCE_TYPES
        for citation in citations
    )


def _select_chat_citations(
    rag_chunks: list[dict] | None,
    loop_citations: list[dict],
    *,
    web_used: bool,
    rag_context_text: str,
) -> list[dict]:
    """Choose persisted citations based on which evidence actually supported the answer."""
    had_rag_chunks = bool(rag_chunks)
    rag_citations = _build_rag_citations(_filter_relevant_rag_chunks(rag_chunks))

    if web_used or _has_tool_or_web_citations(loop_citations):
        return list(loop_citations)

    if loop_citations and rag_citations:
        return _merge_citations(loop_citations, rag_citations)

    if loop_citations:
        return loop_citations

    if rag_citations:
        return rag_citations

    if had_rag_chunks:
        return []

    if (rag_context_text or "").strip():
        return _build_workspace_fallback_citations(rag_context_text, [])

    return []


def _build_tool_citations(
    tool_name: str, tool_args: dict, raw_result: object
) -> list[dict]:
    content, artifact = _normalize_tool_result(raw_result)
    if tool_name in GENERAL_WEB_TOOL_NAMES:
        results = None
        if isinstance(artifact, dict):
            results = artifact.get("results")
        elif isinstance(raw_result, dict):
            results = raw_result.get("results")
        elif isinstance(raw_result, list):
            results = raw_result
        return _build_web_citations(results, settings.web_search_provider)

    if tool_name == "wikipedia":
        results = artifact.get("results") if isinstance(artifact, dict) else None
        return _build_wikipedia_citations(results)

    if tool_name == "open_library":
        results = artifact.get("results") if isinstance(artifact, dict) else None
        return _build_open_library_citations(results)

    if tool_name in ARXIV_MCP_TOOL_NAMES:
        return _build_arxiv_tool_citations(tool_name, tool_args, content, artifact)

    return []


@dataclass
class AgentLoopResult:
    answer: str
    web_used: bool
    citations: list[dict] = field(default_factory=list)
    streamed_answer: bool = False

    def __iter__(self):
        yield self.answer
        yield self.web_used


def _coerce_agent_loop_result(
    result: AgentLoopResult | tuple[str, bool],
) -> AgentLoopResult:
    if isinstance(result, AgentLoopResult):
        return result
    answer, web_used = result
    return AgentLoopResult(answer=answer, web_used=web_used, citations=[])


def _build_chat_trace_outputs(
    *,
    answer: str,
    session_id: str,
    citations: list[dict],
    suggestions: list[str],
    web_used: bool,
) -> dict[str, object]:
    return {
        "answer": answer,
        "session_id": session_id,
        "citation_count": len(citations),
        "suggestion_count": len(suggestions),
        "web_used": web_used,
    }


def _workflow_error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return f"{exc.status_code}: {exc.detail}"
    return str(exc)


def _extract_stream_chunk_text(chunk: object) -> str:
    """Extract streamed token text without stripping meaningful whitespace."""
    content = chunk.content if hasattr(chunk, "content") else chunk  # type: ignore[union-attr]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


async def _run_agent_loop(
    *,
    messages: list[BaseMessage],
    metadata: dict[str, object],
    on_event: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    bind_tools: bool = True,
    allow_web_search: bool = True,
    reference_tools: dict[str, bool] | None = None,
    stream_answer_chunks: bool = False,
) -> AgentLoopResult:
    """Run an agentic tool-calling loop and return answer, usage, and citations.

    on_event: optional async callable(dict) called for tool_start / tool_end events.
    bind_tools: when False, skip Composio router session and tool schema binding.
    allow_web_search: when True, bind Tavily search + URL extract if TAVILY_API_KEY is set.
    reference_tools: per-tool enable flags for built-in reference lookups.
    """
    llm = get_llm(temperature=0.1)
    max_turns = settings.composio_max_agent_turns
    loop_messages = list(messages)
    last_response_text = ""
    web_used_flag: list[bool] = [False]
    collected_citations: list[dict] = []

    agent_tools = build_agent_tools(
        allow_web=allow_web_search,
        reference_flags=reference_tools,
    )
    arxiv_enabled = is_arxiv_mcp_enabled(reference_tools)

    streamed_answer = False

    async def _invoke_turn(llm_target: Runnable, turn: int) -> BaseMessage:
        nonlocal streamed_answer
        with start_step_span(
            name=f"agent_loop.turn_{turn}",
            run_type="llm",
            node_name="agent_loop",
            inputs={"turn": turn, "bind_tools": bind_tools},
            metadata=metadata,
            tags=["llm", "agent_loop"],
        ):
            if stream_answer_chunks and on_event is not None:
                combined_chunk = None
                async for chunk in llm_target.astream(loop_messages):
                    combined_chunk = chunk if combined_chunk is None else combined_chunk + chunk
                    text = _extract_stream_chunk_text(chunk)
                    if text:
                        streamed_answer = True
                        await on_event({"type": "chunk", "text": text})
                if combined_chunk is not None:
                    return combined_chunk
            return await llm_target.ainvoke(loop_messages)

    async def _run_tool_loop(
        *,
        llm_with_tools: Runnable,
        tool_map: dict[str, object],
        composio_stream: bool,
    ) -> AgentLoopResult:
        nonlocal last_response_text
        for turn in range(max_turns):
            response = await _invoke_turn(llm_with_tools, turn)

            last_response_text = extract_llm_text(
                response.content if hasattr(response, "content") else response
            )
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                break

            loop_messages.append(response)

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_id = tc.get("id", tool_name)
                tool_args = tc.get("args", {})
                input_summary = str(tool_args)[:120]

                if composio_stream and on_event:
                    await on_event(
                        {
                            "type": "tool_start",
                            "tool": tool_name,
                            "input_summary": input_summary,
                        }
                    )

                tool_result = ""
                tool_status = "ok"
                try:
                    span_tags = (
                        ["external", "composio"] if composio_stream else ["external"]
                    )
                    with start_step_span(
                        name=f"agent_loop.tool.{tool_name}",
                        run_type="tool",
                        node_name="agent_loop",
                        inputs={"tool": tool_name, "args": tool_args},
                        metadata=metadata,
                        tags=span_tags,
                    ):
                        matched_tool = tool_map.get(tool_name)
                        if matched_tool is None:
                            raise ValueError(
                                f"Tool '{tool_name}' not found in catalog."
                            )
                        raw_result = await _invoke_tool_raw_result(matched_tool, tool_args)
                        normalized_content, _ = _normalize_tool_result(raw_result)
                        tool_result = normalized_content[:6000]
                        collected_citations[:] = _merge_citations(
                            collected_citations,
                            _build_tool_citations(tool_name, tool_args, raw_result),
                        )
                        if should_mark_web_used(tool_name, raw_result):
                            web_used_flag[0] = True
                except Exception as exc:
                    tool_result = f"Tool '{tool_name}' returned an error: {exc}"
                    tool_status = "error"
                    logger.warning("[agent_loop] tool %s failed: %s", tool_name, exc)

                if composio_stream and on_event:
                    await on_event(
                        {"type": "tool_end", "tool": tool_name, "status": tool_status}
                    )

                loop_messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_id)
                )

        return AgentLoopResult(
            answer=last_response_text,
            web_used=web_used_flag[0],
            citations=collected_citations,
            streamed_answer=streamed_answer,
        )

    async with arxiv_mcp_tools_context(enabled=arxiv_enabled) as arxiv_tools:
        if not bind_tools or not settings.composio_enabled:
            all_tools = agent_tools + arxiv_tools
            base_llm = llm.bind_tools(all_tools) if all_tools else llm
            tool_map = {t.name: t for t in all_tools}
            return await _run_tool_loop(
                llm_with_tools=base_llm,
                tool_map=tool_map,
                composio_stream=False,
            )

        manager = get_composio_toolset_manager()
        user_id = settings.composio_user_id

        async with manager.router_tools_context(user_id) as composio_tools:
            all_tools = list(composio_tools) + agent_tools + arxiv_tools
            llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm
            tool_map = {t.name: t for t in all_tools}
            return await _run_tool_loop(
                llm_with_tools=llm_with_tools,
                tool_map=tool_map,
                composio_stream=True,
            )


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
        content = extract_llm_text(result.content)
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
        logger.warning(
            "[suggestions] failed to generate follow-up suggestions: %s", exc
        )
        return []
