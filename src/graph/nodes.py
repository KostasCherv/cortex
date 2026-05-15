"""All LangGraph nodes for the research pipeline."""

import asyncio
import json
import logging
import re
from datetime import datetime, UTC

from src.errors import SearchError, LLMError
from src.graph.state import ResearchState
from src.llm.factory import get_llm
from src.observability.context import build_trace_metadata, build_trace_tags
from src.observability.langfuse import observe_llm_generation
from src.observability.langsmith import start_step_span
from src.prompts.registry import prompt_registry
from src.tools.neo4j_graph_store import Neo4jGraphStore
from src.tools.search import perform_search_cached

logger = logging.getLogger(__name__)

_RERANK_MODEL = "graph_heuristic_v1"
_RERANK_TOP_K = 5
_RERANK_CANDIDATE_LIMIT = 10
_RERANK_MAX_DOC_CHARS = 1200


def _sanitize_model_name(raw_model: object) -> str:
    if raw_model is None:
        return "unknown"
    model = str(raw_model).strip()
    if not model:
        return "unknown"
    lowered = model.lower()
    if "magicmock" in lowered or "<mock" in lowered:
        return "unknown"
    return model


def _extract_llm_text(response: object) -> str:
    """Extract plain text from provider-specific LLM response shapes."""
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content.strip()
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
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()


def _extract_json_candidate(text: str) -> str:
    """Normalize common LLM wrappers and return best-effort JSON substring."""
    candidate = text.strip()
    if not candidate:
        return ""

    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    start = candidate.find("[")
    end = candidate.rfind("]")
    if start != -1 and end != -1 and end > start:
        return candidate[start : end + 1].strip()
    return candidate


async def _invoke_llm(
    prompt: str,
    *,
    step_name: str,
    llm,
    metadata: dict[str, object] | None = None,
    state: ResearchState | None = None,
):
    metadata = metadata or {}
    trace_metadata = build_trace_metadata(metadata)
    model_name = _sanitize_model_name(
        getattr(llm, "model_name", "")
        or getattr(llm, "model", "")
        or getattr(llm, "model_id", "")
    )
    with start_step_span(
        name=f"{step_name}.llm_invoke",
        run_type="llm",
        node_name=step_name,
        inputs={"prompt": prompt},
        metadata=metadata,
        tags=["llm"],
    ):
        with observe_llm_generation(
            step_name=step_name,
            model=model_name,
            prompt=prompt,
            metadata=trace_metadata,
        ) as generation:
            try:
                response = await llm.ainvoke(
                    prompt,
                    config={
                        "tags": build_trace_tags(["llm"]),
                        "metadata": trace_metadata,
                    },
                )
            except Exception as exc:
                generation.mark_error(exc)
                raise

            generation.mark_output(_extract_llm_text(response))
            if state is not None:
                state["langfuse_trace_id"] = generation.trace_id
                state["langfuse_observation_id"] = generation.observation_id
            return response


def _llm_metadata_for_state(
    state: ResearchState,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if state.get("session_id"):
        metadata["session_id"] = state["session_id"]
    if state.get("run_id"):
        metadata["run_id"] = state["run_id"]
    if state.get("user_id"):
        metadata["user_id"] = state["user_id"]
    if extra:
        metadata.update(extra)
    return metadata


# ---------------------------------------------------------------------------
# Node 1: Search
# ---------------------------------------------------------------------------


async def search_node(state: ResearchState) -> ResearchState:
    """Run a Tavily web search for the user query.

    Populates ``search_results`` or sets ``error`` on failure.
    """
    query = state.get("query", "")
    with start_step_span(
        name="search_node",
        run_type="chain",
        node_name="search",
        inputs={"query": query},
    ):
        logger.info("[search_node] query=%r", query)
        try:
            with start_step_span(
                name="search_node.tavily_search",
                run_type="tool",
                node_name="search",
                inputs={"query": query},
                tags=["external", "tavily"],
            ):
                results = await perform_search_cached(query)
            logger.info("[search_node] got %d results", len(results))
            retrieved_contents = [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "raw_text": r.get("raw_content") or r.get("content", ""),
                }
                for r in results
                if r.get("url")
            ]
            return {
                **state,
                "search_results": results,
                "retrieved_contents": retrieved_contents,
                "error": None,
            }
        except SearchError as exc:
            logger.error("[search_node] %s", exc)
            return {**state, "search_results": [], "retrieved_contents": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Node 2: Rerank
# ---------------------------------------------------------------------------


async def rerank_node(state: ResearchState) -> ResearchState:
    """Rerank retrieved sources with a lightweight lexical relevance heuristic."""
    query = state.get("query", "")
    contents = state.get("retrieved_contents", [])
    with start_step_span(
        name="rerank_node",
        run_type="chain",
        node_name="rerank",
        inputs={"source_count": len(contents), "top_k": _RERANK_TOP_K},
    ):
        if not contents:
            return {
                **state,
                "reranked_contents": [],
                "rerank_metadata": {
                    "fallback": False,
                    "reason": "empty_input",
                    "model": _RERANK_MODEL,
                    "input_count": 0,
                    "output_count": 0,
                },
            }

        limited = contents[:_RERANK_CANDIDATE_LIMIT]
        prepared = []
        for row in limited:
            prepared.append(
                {
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "raw_text": str(row.get("raw_text", ""))[:_RERANK_MAX_DOC_CHARS],
                }
            )

        query_tokens = {token for token in re.split(r"\W+", query.lower()) if token}

        scored = []
        for row in prepared:
            title_tokens = {t for t in re.split(r"\W+", row.get("title", "").lower()) if t}
            text_tokens = {t for t in re.split(r"\W+", row.get("raw_text", "").lower()) if t}
            overlap = len(query_tokens & (title_tokens | text_tokens))
            score = overlap + (2 if query.lower() in row.get("title", "").lower() else 0)
            scored.append({**row, "score": float(score)})

        scored.sort(key=lambda item: item["score"], reverse=True)
        ranked = scored[: min(_RERANK_TOP_K, len(scored))]

        return {
            **state,
            "reranked_contents": ranked,
            "rerank_metadata": {
                "fallback": False,
                "model": _RERANK_MODEL,
                "input_count": len(prepared),
                "output_count": len(ranked),
            },
        }


# ---------------------------------------------------------------------------
# Node 4: Summarize
# ---------------------------------------------------------------------------


async def summarize_node(state: ResearchState) -> ResearchState:
    """Ask the LLM to summarize each retrieved source.

    Populates ``summaries``.
    """
    contents = state.get("reranked_contents") or state.get("retrieved_contents", [])
    query = state.get("query", "")
    with start_step_span(
        name="summarize_node",
        run_type="chain",
        node_name="summarize",
        inputs={"source_count": len(contents)},
    ):
        logger.info("[summarize_node] summarizing %d sources", len(contents))

        llm = get_llm(temperature=0.2)
        prepared_sources: list[dict[str, str]] = []
        source_blocks: list[str] = []
        total_chars = 0
        max_total_chars = 50000
        per_source_limit = 10000

        for item in contents:
            text = item.get("raw_text", "").strip()
            if not text:
                continue

            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip()
            if not url:
                continue

            remaining_budget = max_total_chars - total_chars
            if remaining_budget <= 0:
                break

            clipped_text = text[: min(per_source_limit, remaining_budget)]
            if not clipped_text.strip():
                continue

            prepared_sources.append({"url": url, "title": title})
            source_blocks.append(
                f"SOURCE URL: {url}\nSOURCE TITLE: {title}\nCONTENT:\n{clipped_text}"
            )
            total_chars += len(clipped_text)

        if not source_blocks:
            return {**state, "summaries": []}

        prompt, prompt_version = prompt_registry.render(
            "summarize",
            {
                "query": query,
                "source_blocks": "\n\n---\n\n".join(source_blocks),
                "domain": state.get("domain", ""),
            },
        )

        try:
            response = await _invoke_llm(
                prompt,
                step_name="summarize",
                llm=llm,
                metadata=_llm_metadata_for_state(
                    state,
                    {
                        "source_count": len(source_blocks),
                        "query": query,
                        "prompt_version": prompt_version,
                    },
                ),
                state=state,
            )
            response_text = _extract_llm_text(response)
            parsed = None
            parse_error: Exception | None = None

            for attempt in range(2):
                candidate_text = _extract_json_candidate(response_text)
                try:
                    maybe_parsed = json.loads(candidate_text)
                    if isinstance(maybe_parsed, dict) and isinstance(
                        maybe_parsed.get("summaries"), list
                    ):
                        maybe_parsed = maybe_parsed["summaries"]
                    parsed = maybe_parsed
                    break
                except Exception as exc:
                    parse_error = exc
                    if attempt == 1:
                        break
                    repair_prompt = (
                        "Convert the text below into valid JSON only, with this exact schema:\n"
                        '[{"url":"<source-url>","title":"<source-title>","summary":"<3-5 sentences>"}]\n\n'
                        "Do not add markdown fences or explanations.\n\n"
                        f"TEXT:\n{response_text}"
                    )
                    repair_response = await _invoke_llm(
                        repair_prompt,
                        step_name="summarize_repair",
                        llm=llm,
                        metadata=_llm_metadata_for_state(
                            state,
                            {
                                "source_count": len(source_blocks),
                                "query": query,
                                "prompt_version": prompt_version,
                            },
                        ),
                        state=state,
                    )
                    response_text = _extract_llm_text(repair_response)

            if parsed is None:
                raise ValueError(f"Could not parse summarize JSON: {parse_error}")
            if not isinstance(parsed, list):
                raise ValueError("LLM summarize output must be a JSON list")

            parsed_by_url: dict[str, dict[str, str]] = {}
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                url = str(entry.get("url", "")).strip()
                title = str(entry.get("title", "")).strip()
                summary = str(entry.get("summary", "")).strip()
                if url and summary:
                    parsed_by_url[url] = {"url": url, "title": title, "summary": summary}

            summaries: list[dict[str, str]] = []
            for source in prepared_sources:
                url = source["url"]
                fallback_title = source["title"]
                if url in parsed_by_url:
                    row = parsed_by_url[url]
                    if not row.get("title"):
                        row["title"] = fallback_title
                    summaries.append(row)

            if not summaries:
                raise ValueError("LLM summarize output did not include matched source summaries")

            return {**state, "summaries": summaries}
        except Exception as exc:
            raise LLMError(f"Summarization failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Node 5: Report
# ---------------------------------------------------------------------------


async def report_node(state: ResearchState) -> ResearchState:
    """Generate a final structured markdown report in one LLM call.

    Populates ``report`` and ``report_metadata``.
    """
    query = state.get("query", "")
    summaries = state.get("summaries", [])
    memory_context = state.get("memory_context", "")
    with start_step_span(
        name="report_node",
        run_type="chain",
        node_name="report",
        inputs={"summary_count": len(summaries)},
    ):
        logger.info("[report_node] generating report")

        sources_md = "\n".join(
            f"- [{s['title']}]({s['url']})" for s in summaries if s.get("url")
        )
        summaries_text = "\n\n".join(
            f"Source: {s.get('title', '')} ({s.get('url', '')})\n{s.get('summary', '')}"
            for s in summaries
        )
        prompt, prompt_version = prompt_registry.render(
            "report",
            {
                "query": query,
                "summaries_text": summaries_text,
                "memory_context": memory_context,
                "domain": state.get("domain", ""),
            },
        )

        llm = get_llm(temperature=0.2)
        try:
            chunks: list[str] = []
            trace_metadata = _llm_metadata_for_state(
                state,
                {"summary_count": len(summaries), "prompt_version": prompt_version},
            )
            with start_step_span(
                name="report.llm_stream",
                run_type="llm",
                node_name="report",
                inputs={"prompt": prompt},
                metadata=trace_metadata,
                tags=["llm"],
            ):
                async for chunk in llm.astream(
                    prompt,
                    config={
                        "tags": build_trace_tags(["llm"]),
                        "metadata": build_trace_metadata(trace_metadata),
                    },
                ):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if isinstance(content, str):
                        chunks.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, str):
                                chunks.append(block)
                            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                                chunks.append(block["text"])
            report_text = "".join(chunks).strip()
        except Exception as exc:
            raise LLMError(f"Report generation failed: {exc}") from exc

        metadata = {
            "title": query,
            "sources": [s.get("url", "") for s in summaries],
            "generated_at": datetime.now(UTC).isoformat(),
        }

        # Append references section
        if sources_md:
            report_text += f"\n\n## References\n\n{sources_md}"

        return {**state, "report": report_text, "report_metadata": metadata}


# ---------------------------------------------------------------------------
# Node 6: Vector Store
# ---------------------------------------------------------------------------


async def vector_store_node(state: ResearchState) -> ResearchState:
    """Legacy compatibility node retained for graph-first mode.

    Report persistence now happens in the API run finalization path using Neo4j.
    """
    with start_step_span(
        name="vector_store_node",
        run_type="chain",
        node_name="vector_store",
        inputs={"enabled": bool(state.get("use_vector_store", False))},
    ):
        logger.info("[vector_store_node] graph-first mode active; no-op compatibility node")
        return state


# ---------------------------------------------------------------------------
# Node 7: Memory Context
# ---------------------------------------------------------------------------


async def memory_context_node(state: ResearchState) -> ResearchState:
    """Generate graph-aware memory context from Neo4j."""
    with start_step_span(
        name="memory_context_node",
        run_type="chain",
        node_name="memory_context",
        inputs={},
    ):
        try:
            query = state.get("query", "")
            user_id = state.get("user_id") or ""
            workspace_id = user_id
            if not user_id:
                return {**state, "memory_context": "", "graph_context": "", "graph_chunks": [], "graph_entities": []}
            graph_store = Neo4jGraphStore()
            with start_step_span(
                name="memory_context_node.graph_query",
                run_type="retriever",
                node_name="memory_context",
                inputs={"query": query, "top_k": 3},
                tags=["external", "neo4j"],
            ):
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        graph_store.query_context,
                        query=query,
                        owner_id=user_id,
                        workspace_id=workspace_id,
                        top_k=3,
                    ),
                    timeout=8,
                )
            context = (result.context or "")[:2000]
            return {
                **state,
                "memory_context": context,
                "graph_context": context,
                "graph_chunks": result.chunks,
                "graph_entities": result.entities,
            }
        except asyncio.TimeoutError:
            logger.warning("[memory_context_node] timed out while fetching graph context; continuing without memory context.")
            return {**state, "memory_context": "", "graph_context": "", "graph_chunks": [], "graph_entities": []}
        except Exception as exc:
            logger.warning("[memory_context_node] could not generate context: %s", exc)
            return {**state, "memory_context": "", "graph_context": "", "graph_chunks": [], "graph_entities": []}


# ---------------------------------------------------------------------------
# Combined Node: Search + Memory (parallel)
# ---------------------------------------------------------------------------


async def search_and_memory_node(state: ResearchState) -> ResearchState:
    """Run Tavily search and Pinecone memory lookup concurrently.

    Combines search_node and memory_context_node via asyncio.gather so both
    network calls happen in parallel. Joins their results before reranking.
    """
    search_result, memory_result = await asyncio.gather(
        search_node(state),
        memory_context_node(state),
    )
    return {
        **state,
        "search_results": search_result.get("search_results", []),
        "retrieved_contents": search_result.get("retrieved_contents", []),
        "error": search_result.get("error"),
        "memory_context": memory_result.get("memory_context", ""),
        "graph_context": memory_result.get("graph_context", ""),
        "graph_chunks": memory_result.get("graph_chunks", []),
        "graph_entities": memory_result.get("graph_entities", []),
    }
