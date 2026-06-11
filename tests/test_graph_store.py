"""Unit tests for Neo4j graph store operations."""

from unittest.mock import MagicMock, patch

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
    entity_tuple = (
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
    store._extract_entities_relations_batched = MagicMock(
        side_effect=lambda chunk_texts: [entity_tuple for _ in chunk_texts]
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


def test_ingest_document_skips_llm_extraction_for_session_attachments():
    driver = _fake_driver_with_side_effect(lambda *_args, **_kwargs: [])
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    store._extract_entities_relations_batched = MagicMock(return_value=[])
    store._heuristic_entities_relations = MagicMock(return_value=([], []))

    ingested = store.ingest_document(
        document_id="doc-session",
        source_type="session_attachment",
        owner_id="u-1",
        workspace_id="u-1",
        title="brief.pdf",
        source_url="supabase://bucket/brief.pdf",
        text="Session attachment body text.",
        resource_id="res-session",
    )

    assert ingested == 1
    store._extract_entities_relations_batched.assert_not_called()
    store._heuristic_entities_relations.assert_not_called()


def test_ingest_document_caps_llm_extraction_to_max_chunks():
    from src.tools.neo4j_graph_store import _MAX_LLM_EXTRACTION_CHUNKS

    driver = _fake_driver_with_side_effect(lambda *_args, **_kwargs: [])
    embed = MagicMock()
    embed.embed_texts.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    store._extract_entities_relations_batched = MagicMock(
        side_effect=lambda chunk_texts: [([], []) for _ in chunk_texts]
    )
    store._heuristic_entities_relations = MagicMock(return_value=([], []))

    # Produces more chunks than the LLM extraction cap.
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

    assert ingested > _MAX_LLM_EXTRACTION_CHUNKS
    store._extract_entities_relations_batched.assert_called_once()
    batched_chunks = store._extract_entities_relations_batched.call_args.args[0]
    assert len(batched_chunks) == _MAX_LLM_EXTRACTION_CHUNKS
    assert store._heuristic_entities_relations.call_count == ingested - _MAX_LLM_EXTRACTION_CHUNKS


def test_query_context_fuses_scores_and_applies_scope_filters():
    seen = {"vector": None}

    def side_effect(query, params, database_):
        if "MATCH (node:Chunk)" in query and "node.resource_id IN $resource_ids" in query:
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


def test_query_context_uses_global_search_without_resource_ids():
    seen = {"global": False}

    def side_effect(query, params, database_):
        if "SEARCH node IN" in query or "db.index.vector.queryNodes" in query:
            seen["global"] = True
            return [
                {
                    "chunk_id": "chunk-a",
                    "text": "Workspace chunk",
                    "source_url": "https://a.com",
                    "source_title": "A",
                    "chunk_index": 0,
                    "score": 0.5,
                }
            ]
        if "UNWIND $chunk_ids" in query:
            return [{"chunk_id": "chunk-a", "mentions": [], "neighbors": []}]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    embed = MagicMock()
    embed.embed_texts.return_value = [[0.2] * 4]

    store = Neo4jGraphStore(driver=driver, embedding_client=embed)
    result = store.query_context(query="workspace", owner_id="u-1", workspace_id="u-1")

    assert seen["global"] is True
    assert result.chunks[0]["chunk_id"] == "chunk-a"


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


def test_extract_entities_relations_parses_validated_envelope():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=(
            '{"entities":[{"name":"OpenAI","entity_type":" Org ","confidence":0.9},'
            '{"name":"GPT-4","entity_type":"Model","confidence":0.8}],'
            '"relations":[{"source":"OpenAI","target":"GPT-4","type":" BUILDS ","confidence":0.7}]}'
        )
    )

    store = Neo4jGraphStore(driver=MagicMock(), embedding_client=MagicMock())

    with patch("src.tools.neo4j_graph_store.get_llm", return_value=mock_llm):
        entities, relations = store._extract_entities_relations("OpenAI builds GPT-4")

    assert [entity["name"] for entity in entities] == ["OpenAI", "GPT-4"]
    assert relations[0]["type"] == "BUILDS"


def test_extract_entities_relations_repairs_invalid_output_once():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(
            content='{"entities":[{"name":"OpenAI","entity_type":"Org","confidence":2.0}],"relations":[]}'
        ),
        MagicMock(
            content='{"entities":[{"name":"OpenAI","entity_type":"Org","confidence":0.9}],"relations":[]}'
        ),
    ]

    store = Neo4jGraphStore(driver=MagicMock(), embedding_client=MagicMock())

    with patch("src.tools.neo4j_graph_store.get_llm", return_value=mock_llm):
        entities, relations = store._extract_entities_relations("OpenAI")

    assert mock_llm.invoke.call_count == 2
    repair_prompt = mock_llm.invoke.call_args_list[1].args[0]
    assert "Validation failed" in repair_prompt
    assert "confidence" in repair_prompt
    assert [entity["name"] for entity in entities] == ["OpenAI"]
    assert relations == []


def test_extract_entities_relations_falls_back_to_heuristics_after_repair_failure():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content='{"entities":[{"name":"","entity_type":"Org","confidence":0.8}],"relations":[]}'),
        MagicMock(content='{"entities":[{"name":"","entity_type":"Org","confidence":0.8}],"relations":[]}'),
    ]

    store = Neo4jGraphStore(driver=MagicMock(), embedding_client=MagicMock())
    store._heuristic_entities_relations = MagicMock(return_value=(["heuristic"], ["rels"]))

    with patch("src.tools.neo4j_graph_store.get_llm", return_value=mock_llm):
        entities, relations = store._extract_entities_relations("OpenAI")

    assert mock_llm.invoke.call_count == 2
    assert entities == ["heuristic"]
    assert relations == ["rels"]


def test_extract_entities_relations_batched_parses_per_chunk_payload():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=(
            '{"chunks":[{"chunk_index":0,"entities":[{"name":"OpenAI","entity_type":"Org","confidence":0.9}],'
            '"relations":[]},{"chunk_index":1,"entities":[{"name":"Anthropic","entity_type":"Org","confidence":0.8}],'
            '"relations":[]}]}'
        )
    )

    store = Neo4jGraphStore(driver=MagicMock(), embedding_client=MagicMock())

    with patch("src.tools.neo4j_graph_store.get_llm", return_value=mock_llm):
        results = store._extract_entities_relations_batched(["OpenAI text", "Anthropic text"])

    assert mock_llm.invoke.call_count == 1
    assert [entity["name"] for entity in results[0][0]] == ["OpenAI"]
    assert [entity["name"] for entity in results[1][0]] == ["Anthropic"]


def test_extract_entities_relations_drops_relations_with_unknown_endpoints():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=(
            '{"entities":[{"name":"OpenAI","entity_type":"Org","confidence":0.9}],'
            '"relations":[{"source":"OpenAI","target":"GPT-4","type":"BUILDS","confidence":0.7}]}'
        )
    )

    store = Neo4jGraphStore(driver=MagicMock(), embedding_client=MagicMock())

    with patch("src.tools.neo4j_graph_store.get_llm", return_value=mock_llm):
        entities, relations = store._extract_entities_relations("OpenAI builds GPT-4")

    assert [entity["name"] for entity in entities] == ["OpenAI"]
    assert relations == []
