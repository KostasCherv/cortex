import sys
from types import SimpleNamespace

import pytest

from src.config import settings
import src.tools.reranker as reranker


def _chunks():
    return [
        {"chunk_id": "a", "text": "alpha", "source_title": "Doc A"},
        {"chunk_id": "b", "text": "beta", "source_title": "Doc B"},
        {"chunk_id": "c", "text": "gamma", "source_title": "Doc C"},
    ]


def _reset_client(monkeypatch):
    monkeypatch.setattr(reranker, "_cohere_client_state", None)


def test_rerank_chunks_sorts_and_filters_by_relevance(monkeypatch):
    _reset_client(monkeypatch)
    calls = {}

    class FakeClient:
        def __init__(self, api_key, timeout=None):
            calls["api_key"] = api_key

        def rerank(self, **kwargs):
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.91),
                    SimpleNamespace(index=0, relevance_score=0.29),
                    SimpleNamespace(index=2, relevance_score=0.44),
                ]
            )

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setattr(settings, "rerank_relevance_threshold", 0.3)
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClient))

    result = reranker.rerank_chunks("query", _chunks(), top_k=3)

    assert calls["api_key"] == "co-key"
    assert calls["kwargs"]["model"] == "rerank-v3.5"
    assert calls["kwargs"]["documents"] == ["alpha", "beta", "gamma"]
    # Preserve Cohere's descending relevance order after threshold filtering.
    assert [chunk["chunk_id"] for chunk in result] == ["b", "c"]
    assert [chunk["rerank_score"] for chunk in result] == [0.91, 0.44]
    assert result[0]["rerank_score"] == 0.91


def test_rerank_chunks_without_api_key_falls_back_to_original_order(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.setattr(settings, "cohere_api_key", "")
    monkeypatch.setattr(settings, "rerank_top_k", 2)

    result = reranker.rerank_chunks("query", _chunks())

    assert [chunk["chunk_id"] for chunk in result] == ["a", "b"]
    assert all("rerank_score" not in chunk for chunk in result)


def test_rerank_chunks_returns_empty_for_empty_input(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.setattr(settings, "cohere_api_key", "co-key")

    assert reranker.rerank_chunks("query", []) == []


def test_rerank_chunks_falls_back_when_cohere_api_raises(monkeypatch):
    _reset_client(monkeypatch)
    class FakeCohereError(Exception):
        pass

    class FailingClient:
        def __init__(self, api_key, timeout=None):
            pass

        def rerank(self, **kwargs):
            raise FakeCohereError("cohere down")

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setattr(settings, "rerank_top_k", 2)
    monkeypatch.setitem(
        sys.modules,
        "cohere",
        SimpleNamespace(ClientV2=FailingClient, ServiceUnavailableError=FakeCohereError),
    )

    result = reranker.rerank_chunks("query", _chunks())

    assert [chunk["chunk_id"] for chunk in result] == ["a", "b"]


def test_rerank_chunks_propagates_unexpected_errors(monkeypatch):
    _reset_client(monkeypatch)

    class FailingClient:
        def __init__(self, api_key, timeout=None):
            pass

        def rerank(self, **kwargs):
            raise NameError("typo in reranker")

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FailingClient))

    with pytest.raises(NameError, match="typo"):
        reranker.rerank_chunks("query", _chunks())


def test_rerank_chunks_returns_best_chunk_when_all_scores_below_threshold(monkeypatch):
    _reset_client(monkeypatch)
    class FakeClient:
        def __init__(self, api_key, timeout=None):
            pass

        def rerank(self, **kwargs):
            return {
                "results": [
                    {"index": 2, "relevance_score": 0.22},
                    {"index": 0, "relevance_score": 0.19},
                ]
            }

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setattr(settings, "rerank_relevance_threshold", 0.3)
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClient))

    result = reranker.rerank_chunks("query", _chunks(), top_k=2)

    assert [chunk["chunk_id"] for chunk in result] == ["c"]
    assert result[0]["rerank_score"] == 0.22


def test_rerank_chunks_skips_malformed_and_out_of_bounds_results(monkeypatch):
    _reset_client(monkeypatch)

    class FakeClient:
        def __init__(self, api_key, timeout=None):
            pass

        def rerank(self, **kwargs):
            return {
                "results": [
                    {"relevance_score": 0.99},
                    {"index": 99, "relevance_score": 0.98},
                    {"index": 1, "relevance_score": 0.97},
                ]
            }

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setattr(settings, "rerank_relevance_threshold", 0.3)
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClient))

    result = reranker.rerank_chunks("query", _chunks(), top_k=3)

    assert [chunk["chunk_id"] for chunk in result] == ["b"]


def test_rerank_chunks_returns_empty_when_cohere_returns_no_results(monkeypatch):
    _reset_client(monkeypatch)

    class FakeClient:
        def __init__(self, api_key, timeout=None):
            pass

        def rerank(self, **kwargs):
            return {"results": []}

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClient))

    assert reranker.rerank_chunks("query", _chunks()) == []


def test_rerank_chunks_reuses_single_cohere_client_and_sets_timeout(monkeypatch):
    _reset_client(monkeypatch)
    calls = {"created": 0, "timeouts": []}

    class FakeClient:
        def __init__(self, api_key, timeout=None):
            calls["created"] += 1
            calls["timeouts"].append(timeout)

        def rerank(self, **kwargs):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setattr(settings, "rerank_timeout_seconds", 7.5)
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClient))

    reranker.rerank_chunks("query", _chunks())
    reranker.rerank_chunks("query", _chunks())

    assert calls == {"created": 1, "timeouts": [7.5]}


def test_rerank_chunks_honors_zero_top_k(monkeypatch):
    _reset_client(monkeypatch)
    monkeypatch.setattr(settings, "cohere_api_key", "")
    monkeypatch.setattr(settings, "rerank_top_k", 2)

    assert reranker.rerank_chunks("query", _chunks(), top_k=0) == []


def test_rerank_chunks_skips_chunks_without_text_before_cohere(monkeypatch):
    _reset_client(monkeypatch)
    calls = {}

    class FakeClient:
        def __init__(self, api_key, timeout=None):
            pass

        def rerank(self, **kwargs):
            calls["documents"] = kwargs["documents"]
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    chunks = [
        {"chunk_id": "missing"},
        {"chunk_id": "blank", "text": "   "},
        {"chunk_id": "text", "text": "actual text"},
    ]

    monkeypatch.setattr(settings, "cohere_api_key", "co-key")
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=FakeClient))

    result = reranker.rerank_chunks("query", chunks)

    assert calls["documents"] == ["actual text"]
    assert [chunk["chunk_id"] for chunk in result] == ["text"]
