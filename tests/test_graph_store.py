"""Unit tests for Neo4j graph store operations."""

from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.tools.neo4j_graph_store import Neo4jGraphStore


class _Record:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


def _fake_driver_with_side_effect(side_effect):
    driver = MagicMock()

    def _execute_query(query, parameters_=None, database_=None):
        records_payload = side_effect(query, parameters_ or {}, database_)
        return ([_Record(row) for row in records_payload], None, None)

    driver.execute_query.side_effect = _execute_query
    return driver


def test_ingest_document_writes_document_chunk_and_entity_batches():
    calls = []

    def side_effect(query, params, database_):
        calls.append((query, params, database_))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    store._extract_entities_relations = MagicMock(
        return_value=(
            [
                {
                    "id": "entity-1",
                    "name": "OpenAI",
                    "normalized_name": "openai",
                    "entity_type": "Org",
                    "confidence": 0.9,
                }
            ],
            [],
        )
    )

    ingested = store.ingest_document(
        document_id="doc-1",
        source_type="resource_upload",
        owner_id="u-1",
        workspace_id="u-1",
        title="Doc",
        source_url="https://example.com/doc",
        text="Chunk one text. " * 200,
        resource_id="res-1",
    )

    assert ingested >= 1
    assert any("MERGE (d:Document" in query for query, _, _ in calls)
    assert any("MERGE (c)-[m:MENTIONS]->(e)" in query for query, _, _ in calls)


def test_ingest_document_caps_llm_extraction_to_three_chunks():
    driver = _fake_driver_with_side_effect(lambda *_args, **_kwargs: [])
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    store._extract_entities_relations = MagicMock(return_value=([], []))
    store._heuristic_entities_relations = MagicMock(return_value=([], []))

    # Produces >3 chunks with current chunking config.
    long_text = ("Long chunk text segment. " * 260)
    ingested = store.ingest_document(
        document_id="doc-long",
        source_type="web_run",
        owner_id="u-1",
        workspace_id="u-1",
        title="Long Doc",
        source_url="https://example.com/long",
        text=long_text,
        run_id="run-1",
    )

    assert ingested > 3
    assert store._extract_entities_relations.call_count == 3
    assert store._heuristic_entities_relations.call_count == ingested - 3


def test_query_context_fuses_scores_and_applies_scope_filters():
    seen = {"vector": None}

    def side_effect(query, params, database_):
        if "SEARCH node IN" in query or "db.index.vector.queryNodes" in query:
            seen["vector"] = params
            return [
                {
                    "chunk_id": "chunk-a",
                    "text": "OpenAI launches model",
                    "source_url": "https://a.com",
                    "source_title": "A",
                    "chunk_index": 0,
                    "score": 0.4,
                },
                {
                    "chunk_id": "chunk-b",
                    "text": "General industry update",
                    "source_url": "https://b.com",
                    "source_title": "B",
                    "chunk_index": 1,
                    "score": 0.5,
                },
            ]
        if "UNWIND $chunk_ids" in query:
            return [
                {
                    "chunk_id": "chunk-a",
                    "mentions": ["openai", "model"],
                    "neighbors": ["ai"],
                },
                {
                    "chunk_id": "chunk-b",
                    "mentions": [],
                    "neighbors": [],
                },
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    embed = MagicMock()
    embed.embed_texts.return_value = [[0.2] * 4]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    result = store.query_context(
        query="OpenAI model",
        owner_id="u-1",
        workspace_id="u-1",
        run_id="run-1",
        resource_ids=["res-1"],
        top_k=2,
        max_hops=2,
    )

    assert seen["vector"]["run_id"] == "run-1"
    assert seen["vector"]["resource_ids"] == ["res-1"]
    assert result.chunks[0]["chunk_id"] == "chunk-a"
    assert "openai" in result.entities
    assert "source:A" in result.context


def test_query_context_excludes_chunks_below_min_cosine(monkeypatch):
    monkeypatch.setattr(settings, "graph_rag_min_cosine_score", 0.15)

    def side_effect(query, params, database_):
        if "SEARCH node IN" in query or "db.index.vector.queryNodes" in query:
            return [
                {
                    "chunk_id": "low",
                    "text": "Noisy chunk with many entities",
                    "source_url": "https://low.com",
                    "source_title": "Low",
                    "chunk_index": 0,
                    "score": 0.14,
                },
                {
                    "chunk_id": "high",
                    "text": "Relevant chunk",
                    "source_url": "https://high.com",
                    "source_title": "High",
                    "chunk_index": 1,
                    "score": 0.5,
                },
            ]
        if "UNWIND $chunk_ids" in query:
            return [
                {
                    "chunk_id": "low",
                    "mentions": ["openai", "model", "agent", "retrieval"],
                    "neighbors": ["citation", "ranking"],
                },
                {"chunk_id": "high", "mentions": [], "neighbors": []},
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    embed = MagicMock()
    embed.embed_texts.return_value = [[0.2] * 4]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    result = store.query_context(query="OpenAI model", owner_id="u-1", workspace_id="u-1")

    assert [chunk["chunk_id"] for chunk in result.chunks] == ["high"]


def test_query_context_caps_entity_and_neighbor_bonus(monkeypatch):
    monkeypatch.setattr(settings, "graph_rag_min_cosine_score", 0.15)

    def side_effect(query, params, database_):
        if "SEARCH node IN" in query or "db.index.vector.queryNodes" in query:
            return [
                {
                    "chunk_id": "chunk-a",
                    "text": "OpenAI model agent retrieval citation ranking",
                    "source_url": "https://a.com",
                    "source_title": "A",
                    "chunk_index": 0,
                    "score": 0.4,
                }
            ]
        if "UNWIND $chunk_ids" in query:
            return [
                {
                    "chunk_id": "chunk-a",
                    "mentions": ["openai", "model", "agent", "retrieval", "citation", "ranking"],
                    "neighbors": ["n1", "n2", "n3", "n4", "n5", "n6"],
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    embed = MagicMock()
    embed.embed_texts.return_value = [[0.2] * 4]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    result = store.query_context(
        query="OpenAI model agent retrieval",
        owner_id="u-1",
        workspace_id="u-1",
    )

    assert result.chunks[0]["score"] == pytest.approx(0.65)


def test_delete_resource_documents_scopes_delete_by_owner_and_workspace():
    call_params = []

    def side_effect(query, params, database_):
        call_params.append((query, params))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    store = Neo4jGraphStore(driver=driver)

    deleted = store.delete_resource_documents(
        resource_id="res-123",
        owner_id="u-1",
        workspace_id="u-1",
    )

    assert deleted is True
    delete_calls = [params for query, params in call_params if "MATCH (d:Document)" in query]
    assert delete_calls
    assert delete_calls[0]["resource_id"] == "res-123"
