"""Cross-encoder reranking for retrieved RAG chunks."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

RERANK_MODEL = "rerank-v3.5"
_COHERE_ERROR_NAMES = (
    "BadRequestError",
    "ClientClosedRequestError",
    "ForbiddenError",
    "GatewayTimeoutError",
    "InternalServerError",
    "InvalidTokenError",
    "NotFoundError",
    "NotImplementedError",
    "ServiceUnavailableError",
    "TooManyRequestsError",
    "UnauthorizedError",
    "UnprocessableEntityError",
)


@dataclass(frozen=True)
class _CohereClientState:
    api_key: str
    timeout: float
    client: Any


_cohere_client_state: _CohereClientState | None = None
_cohere_client_lock = threading.Lock()


def _value(row: Any, key: str, default: Any = None) -> Any:
    """Read SDK response rows that may be dicts or typed objects."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _fallback(chunks: list[dict], top_k: int) -> list[dict]:
    return [dict(chunk) for chunk in chunks[:top_k]]


def _cohere_error_types(cohere_module: Any) -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = []
    for name in _COHERE_ERROR_NAMES:
        value = getattr(cohere_module, name, None)
        if isinstance(value, type) and issubclass(value, BaseException):
            errors.append(value)
    return tuple(errors)


def _eligible_chunks(chunks: list[dict]) -> list[dict]:
    return [dict(chunk) for chunk in chunks if str(chunk.get("text") or "").strip()]


def _get_cohere_client(api_key: str, timeout: float, cohere_module: Any) -> Any:
    global _cohere_client_state

    state = _cohere_client_state
    if state is not None and state.api_key == api_key and state.timeout == timeout:
        return state.client

    with _cohere_client_lock:
        state = _cohere_client_state
        if state is not None and state.api_key == api_key and state.timeout == timeout:
            return state.client

        client = cohere_module.ClientV2(api_key=api_key, timeout=timeout)
        _cohere_client_state = _CohereClientState(
            api_key=api_key,
            timeout=timeout,
            client=client,
        )
        return client


def _rerank_with_cohere(
    *,
    client: Any,
    query: str,
    chunks: list[dict],
    top_k: int,
    threshold: float,
) -> list[dict]:
    response = client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[chunk["text"] for chunk in chunks],
        top_n=top_k,
    )
    results = _value(response, "results", [])
    ranked: list[dict] = []
    for item in results:
        raw_index = _value(item, "index")
        if raw_index is None:
            continue
        index = int(raw_index)
        score = float(_value(item, "relevance_score", 0.0))
        if index < 0 or index >= len(chunks):
            continue

        chunk = dict(chunks[index])
        chunk["rerank_score"] = score
        if score >= threshold:
            ranked.append(chunk)

    if ranked:
        return ranked[:top_k]
    logger.info(
        "[reranker] no chunks met relevance threshold %.2f; returning no chunks",
        threshold,
    )
    return []


def rerank_chunks(
    query: str,
    chunks: list[dict],
    *,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Rerank chunks by query-document relevance, falling back gracefully."""
    if not chunks:
        return []

    effective_top_k = settings.rerank_top_k if top_k is None else top_k
    if effective_top_k <= 0:
        return []

    effective_threshold = (
        settings.rerank_relevance_threshold if threshold is None else threshold
    )

    if not settings.cohere_api_key:
        return _fallback(chunks, effective_top_k)

    rerankable_chunks = _eligible_chunks(chunks)
    if not rerankable_chunks:
        return []

    try:
        import cohere
    except ImportError as exc:
        logger.warning("[reranker] Cohere import failed; falling back: %s", exc)
        return _fallback(chunks, effective_top_k)

    cohere_errors = _cohere_error_types(cohere)
    client = _get_cohere_client(
        settings.cohere_api_key,
        settings.rerank_timeout_seconds,
        cohere,
    )
    if cohere_errors:
        try:
            return _rerank_with_cohere(
                client=client,
                query=query,
                chunks=rerankable_chunks,
                top_k=effective_top_k,
                threshold=effective_threshold,
            )
        except cohere_errors as exc:
            logger.warning("[reranker] Cohere rerank failed; falling back: %s", exc)
            return _fallback(chunks, effective_top_k)

    return _rerank_with_cohere(
        client=client,
        query=query,
        chunks=rerankable_chunks,
        top_k=effective_top_k,
        threshold=effective_threshold,
    )
