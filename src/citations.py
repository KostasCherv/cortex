"""Provenance / citation-selection pipeline — the explainability layer for chat answers.

Builds citation payloads from RAG chunks, web search, and reference tools,
merges/deduplicates them, and selects which citations to persist for an answer.
Moved verbatim from ``src/api/deps.py`` (pure code motion, public names).
"""

from src.config import settings
from src.tools.arxiv_mcp import ARXIV_MCP_TOOL_NAMES
from src.tools.general import GENERAL_WEB_TOOL_NAMES

_TOOL_SOURCE_TYPES = frozenset({"web", "wikipedia", "open_library", "arxiv"})


def build_rag_citations(chunks: list[dict] | None) -> list[dict]:
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


def filter_relevant_rag_chunks(chunks: list[dict] | None) -> list[dict]:
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


def build_web_citations(results: list[dict] | None, provider: str) -> list[dict]:
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


def build_workspace_fallback_citations(
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


def normalize_tool_result(raw_result: object) -> tuple[str, object | None]:
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        content, artifact = raw_result
        return str(content), artifact
    return str(raw_result), None


def merge_citations(*groups: list[dict]) -> list[dict]:
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


def iter_nested_dicts(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_nested_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from iter_nested_dicts(item)


def build_wikipedia_citations(results: object) -> list[dict]:
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
                "source_url": (f"https://en.wikipedia.org/wiki/{page_slug}" if page_slug else None),
                "chunk_id": f"wikipedia-{index + 1}",
                "text": str(row.get("extract") or ""),
                "source_type": "wikipedia",
            }
        )
    return citations


def build_open_library_citations(results: object) -> list[dict]:
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
                "source_title": str(row.get("title") or f"Open Library result {index + 1}"),
                "source_url": str(row.get("url") or "") or None,
                "chunk_id": f"open-library-{index + 1}",
                "text": "\n".join(part for part in text_parts if part),
                "source_type": "open_library",
            }
        )
    return citations


def build_arxiv_tool_citations(
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
        for index, row in enumerate(iter_nested_dicts(structured)):
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
                or (f"arXiv:{candidate_id}" if candidate_id else f"arXiv result {index + 1}")
            )
            url = str(row.get("abs_url") or row.get("pdf_url") or "") or (
                f"https://arxiv.org/abs/{candidate_id}" if candidate_id else None
            )
            text = str(row.get("abstract") or row.get("summary") or row.get("content") or "")
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
    for row in iter_nested_dicts(structured):
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


def has_tool_or_web_citations(citations: list[dict]) -> bool:
    return any(
        isinstance(citation.get("source_type"), str)
        and citation["source_type"] in _TOOL_SOURCE_TYPES
        for citation in citations
    )


def select_chat_citations(
    rag_chunks: list[dict] | None,
    loop_citations: list[dict],
    *,
    router_action: str | None,
    web_used: bool,
    rag_context_text: str,
) -> list[dict]:
    """Choose persisted citations based on which evidence actually supported the answer."""
    if web_used or has_tool_or_web_citations(loop_citations):
        return list(loop_citations)

    if router_action != "answer_from_rag":
        return list(loop_citations)

    had_rag_chunks = bool(rag_chunks)
    rag_citations = build_rag_citations(filter_relevant_rag_chunks(rag_chunks))

    if loop_citations and rag_citations:
        return merge_citations(loop_citations, rag_citations)

    if loop_citations:
        return loop_citations

    if rag_citations:
        return rag_citations

    if had_rag_chunks:
        return []

    if (rag_context_text or "").strip():
        return build_workspace_fallback_citations(rag_context_text, [])

    return []


def build_tool_citations(tool_name: str, tool_args: dict, raw_result: object) -> list[dict]:
    content, artifact = normalize_tool_result(raw_result)
    if tool_name in GENERAL_WEB_TOOL_NAMES:
        results = None
        if isinstance(artifact, dict):
            results = artifact.get("results")
        elif isinstance(raw_result, dict):
            results = raw_result.get("results")
        elif isinstance(raw_result, list):
            results = raw_result
        return build_web_citations(results, settings.web_search_provider)

    if tool_name == "wikipedia":
        results = artifact.get("results") if isinstance(artifact, dict) else None
        return build_wikipedia_citations(results)

    if tool_name == "open_library":
        results = artifact.get("results") if isinstance(artifact, dict) else None
        return build_open_library_citations(results)

    if tool_name in ARXIV_MCP_TOOL_NAMES:
        return build_arxiv_tool_citations(tool_name, tool_args, content, artifact)

    return []
