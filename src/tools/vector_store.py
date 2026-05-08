"""Pinecone vector store manager backed by LlamaIndex."""

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.pinecone import PineconeVectorStore

from src.config import settings
from src.errors import VectorStoreError
from src.llm.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# Characters per chunk when splitting source text for run-scoped retrieval
_CHUNK_SIZE = 1000
# Pinecone namespaces
_NAMESPACE_REPORTS = "reports"
_NAMESPACE_CHUNKS = "source_chunks"
# Pinecone metadata value size limit (bytes); truncate document text to stay under 40KB
_META_TEXT_LIMIT = 38_000


class VectorStoreManager:
    """Thin wrapper around Pinecone for storing and querying research reports."""

    def __init__(self) -> None:
        self._index: Optional[Any] = None
        self._pinecone_client: Optional[Any] = None
        self._embedding_client: Optional[EmbeddingClient] = None
        self._index_dimension_validated = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_index(self):
        """Return a connected Pinecone index, initialising lazily."""
        if self._index is not None:
            return self._index
        if not settings.pinecone_api_key:
            raise VectorStoreError(
                "PINECONE_API_KEY is not set. "
                "Add it to your .env file before using the vector store."
            )
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        self._pinecone_client = pc
        self._index = pc.Index(settings.pinecone_index_name)
        return self._index

    def _get_embed_model(self):
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()
        return self._embedding_client._get_embed_model()

    def _get_index_for_namespace(self, namespace: str) -> VectorStoreIndex:
        vector_store = PineconeVectorStore(
            pinecone_index=self._ensure_index(),
            namespace=namespace,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self._get_embed_model(),
        )

    def _extract_index_dimension(self, index_info: object) -> int | None:
        """Best-effort extraction of a Pinecone index dimension from SDK responses."""
        if hasattr(index_info, "dimension"):
            dimension = getattr(index_info, "dimension")
            if isinstance(dimension, int):
                return dimension

        if isinstance(index_info, dict):
            dimension = index_info.get("dimension")
            if isinstance(dimension, int):
                return dimension

        return None

    def _validate_index_dimension(self) -> None:
        """Ensure the configured Pinecone index matches the embedding dimensions."""
        if self._index_dimension_validated:
            return

        self._ensure_index()
        if self._pinecone_client is None:
            raise VectorStoreError("Pinecone client is not initialized.")

        try:
            index_info = self._pinecone_client.describe_index(settings.pinecone_index_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to inspect Pinecone index: {exc}") from exc

        index_dimension = self._extract_index_dimension(index_info)
        if index_dimension is None:
            raise VectorStoreError(
                "Failed to inspect Pinecone index: index dimension was not available."
            )

        if index_dimension != settings.embedding_dimensions:
            raise VectorStoreError(
                "Pinecone index "
                f"'{settings.pinecone_index_name}' dimension {index_dimension} does not match "
                f"the configured embedding dimensions {settings.embedding_dimensions}. "
                "Use a matching index or reindex existing data for this embedding model."
            )

        self._index_dimension_validated = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save_report(self, query: str, report: str, metadata: dict | None = None) -> str:
        """Persist a research report to Pinecone."""
        try:
            self._validate_index_dimension()
            doc_id = f"report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
            doc_text = report[:_META_TEXT_LIMIT]
            if len(report) > _META_TEXT_LIMIT:
                logger.warning(
                    "[vector_store] report truncated from %d to %d chars for metadata storage.",
                    len(report),
                    _META_TEXT_LIMIT,
                )
            meta = {
                "query": query[:512],
                "generated_at": datetime.now(UTC).isoformat(),
                "document": doc_text,
                **(metadata or {}),
            }
            index = self._get_index_for_namespace(_NAMESPACE_REPORTS)
            index.insert_nodes(
                [
                    TextNode(
                        id_=doc_id,
                        text=report,
                        metadata=meta,
                    )
                ]
            )
            logger.info("Saved report '%s' to vector store.", doc_id)
            return doc_id
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to save report: {exc}") from exc

    def search_reports(self, query: str, n_results: int = 3) -> list[dict]:
        """Semantic search over stored reports."""
        try:
            self._validate_index_dimension()
            retriever = self._get_index_for_namespace(_NAMESPACE_REPORTS).as_retriever(
                similarity_top_k=n_results
            )
            results = retriever.retrieve(query)
            return [
                {
                    "id": getattr(match.node, "id_", ""),
                    "document": (match.node.metadata or {}).get("document", ""),
                    "metadata": match.node.metadata or {},
                }
                for match in results
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to search reports: {exc}") from exc

    # ---------------------------------------------------------------------------
    # Run-scoped source chunks (for session follow-up retrieval)
    # ---------------------------------------------------------------------------

    def save_source_chunks(
        self,
        run_id: str,
        session_id: str,
        sources: list[dict],
    ) -> int:
        """Split and persist source texts as chunks keyed to a run."""
        try:
            self._validate_index_dimension()
            nodes: list[TextNode] = []

            for source in sources:
                text: str = source.get("raw_text") or source.get("summary", "")
                url: str = source.get("url", "")
                title: str = source.get("title", "")

                chunks = [
                    text[start : start + _CHUNK_SIZE]
                    for start in range(0, max(len(text), 1), _CHUNK_SIZE)
                    if text[start : start + _CHUNK_SIZE].strip()
                ]
                for chunk_index, chunk_text in enumerate(chunks):
                    id_seed = f"{run_id}:{url}:{chunk_index}"
                    chunk_id = hashlib.md5(id_seed.encode()).hexdigest()
                    nodes.append(
                        TextNode(
                            id_=chunk_id,
                            text=chunk_text,
                            metadata={
                                "run_id": run_id,
                                "session_id": session_id,
                                "source_url": url[:512],
                                "source_title": title[:256],
                                "chunk_index": chunk_index,
                                "text": chunk_text,
                            },
                        )
                    )

            if not nodes:
                return 0

            self._get_index_for_namespace(_NAMESPACE_CHUNKS).insert_nodes(nodes)
            logger.info("[vector_store] saved %d chunks for run %s", len(nodes), run_id)
            return len(nodes)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to save source chunks: {exc}") from exc

    def search_run_sources(
        self,
        query: str,
        run_id: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Semantic search over source chunks scoped to a specific run."""
        try:
            self._validate_index_dimension()
            filters = MetadataFilters(filters=[ExactMatchFilter(key="run_id", value=run_id)])
            retriever = self._get_index_for_namespace(_NAMESPACE_CHUNKS).as_retriever(
                similarity_top_k=n_results,
                filters=filters,
            )
            results = retriever.retrieve(query)
            return [
                {
                    "text": (match.node.metadata or {}).get("text", ""),
                    "source_url": (match.node.metadata or {}).get("source_url", ""),
                    "source_title": (match.node.metadata or {}).get("source_title", ""),
                    "chunk_index": (match.node.metadata or {}).get("chunk_index", 0),
                }
                for match in results
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to search run sources: {exc}") from exc

    def rerank_documents(
        self,
        *,
        query: str,
        documents: list[dict],
        model: str = "pinecone-rerank-v0",
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank source documents with Pinecone-hosted rerank inference."""
        try:
            self._ensure_index()
            if self._pinecone_client is None:
                raise VectorStoreError("Pinecone client is not initialized.")

            input_docs = [
                {
                    "id": str(i),
                    "text": f"{doc.get('title', '')}\n\n{doc.get('raw_text', '')}",
                }
                for i, doc in enumerate(documents)
            ]
            response = self._pinecone_client.inference.rerank(
                model=model,
                query=query,
                documents=input_docs,
                top_n=top_k,
                return_documents=True,
            )

            ranked: list[dict] = []
            for item in getattr(response, "data", []) or []:
                doc_obj = getattr(item, "document", None)
                doc_id = getattr(doc_obj, "id", None)
                if doc_id is None:
                    continue
                try:
                    index = int(doc_id)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(documents):
                    source = {**documents[index], "score": float(getattr(item, "score", 0.0))}
                    ranked.append(source)

            return ranked
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to rerank documents: {exc}") from exc
