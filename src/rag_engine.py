"""Shared GraphRAG indexing/query logic used by API and ingestion jobs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.db.supabase_store import SupabaseSessionStore
from src.tools.neo4j_graph_store import Neo4jGraphStore
from src.tools.reranker import rerank_chunks

logger = logging.getLogger(__name__)


@dataclass
class RagQueryResult:
    context: str
    chunks: list[dict]
    entities: list[str] | None = None


async def read_locator_bytes(file_locator: str) -> tuple[bytes, str]:
    parsed = urlparse(file_locator)
    if parsed.scheme in {"http", "https"}:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(file_locator)
        response.raise_for_status()
        return response.content, Path(parsed.path).suffix.lower()

    path = Path(file_locator)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_locator}")
    content = await asyncio.to_thread(path.read_bytes)
    return content, path.suffix.lower()


def extract_text_from_bytes(content: bytes, suffix: str) -> str:
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("pypdf is required for PDF ingestion") from exc
        reader = PdfReader(BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".docx":
        try:
            from docx import Document
        except Exception as exc:
            raise RuntimeError("python-docx is required for DOCX ingestion") from exc
        doc = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    raise RuntimeError(f"Unsupported file type for ingestion: {suffix}")


async def ingest_resource_from_locator(
    *,
    store: SupabaseSessionStore,
    resource_id: str,
    file_locator: str,
    owner_id: str,
    workspace_id: str,
) -> int:
    # ``store`` is kept for compatibility with existing call sites.
    del store
    content, suffix = await read_locator_bytes(file_locator)
    # Parsing PDF/DOCX and chunking can be CPU-bound; keep this off the
    # main event loop so concurrent API streams remain responsive.
    text = await asyncio.to_thread(extract_text_from_bytes, content, suffix)
    source_title = Path(urlparse(file_locator).path).name or resource_id
    graph_store = Neo4jGraphStore()
    return await asyncio.to_thread(
        graph_store.ingest_document,
        document_id=resource_id,
        source_type="resource_upload",
        owner_id=owner_id,
        workspace_id=workspace_id,
        title=source_title,
        source_url=file_locator,
        text=text,
        resource_id=resource_id,
    )


async def query_resource_context(
    *,
    store: SupabaseSessionStore,
    resource_ids: list[str],
    owner_id: str,
    workspace_id: str,
    query: str,
) -> RagQueryResult:
    # ``store`` is kept for compatibility with existing call sites.
    del store
    graph_store = Neo4jGraphStore()
    result = await asyncio.to_thread(
        graph_store.query_context,
        query=query,
        owner_id=owner_id,
        workspace_id=workspace_id,
        resource_ids=resource_ids,
    )
    if not result.chunks:
        logger.warning(
            "[rag_engine] graph query returned no chunks owner_id=%s workspace_id=%s resource_ids=%s",
            owner_id,
            workspace_id,
            resource_ids,
        )
    top = []
    for row in result.chunks:
        top.append(
            {
                "resource_id": row.get("resource_id") or row.get("document_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "text": row.get("text", ""),
                "source_title": row.get("source_title", ""),
                "source_url": row.get("source_url", ""),
            }
        )
    reranked = await asyncio.to_thread(rerank_chunks, query=query, chunks=top)
    context = "\n\n".join(
        f"[source:{row['source_title']} chunk:{row['chunk_id']}]\n{row['text']}"
        for row in reranked
        if row.get("text")
    )
    return RagQueryResult(
        context=context,
        chunks=reranked,
        entities=result.entities,
    )


async def delete_resource_artifacts(
    *,
    store: SupabaseSessionStore,
    resource_id: str,
    owner_id: str,
    workspace_id: str,
) -> bool:
    # Keep sidecar cleanup for backward compatibility.
    await store.delete_rag_sidecar_artifact(resource_id=resource_id)
    graph_store = Neo4jGraphStore()
    return await asyncio.to_thread(
        graph_store.delete_resource_documents,
        resource_id=resource_id,
        owner_id=owner_id,
        workspace_id=workspace_id,
    )
