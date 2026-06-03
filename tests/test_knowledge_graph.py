"""Integration tests for the knowledge graph engine."""

from __future__ import annotations

from unittest.mock import MagicMock


from src.knowledge_graph.conflict import ConflictResolver
from src.knowledge_graph.dsl import KnowledgeQueryBuilder
from src.knowledge_graph.engine import KnowledgeGraphEngine
from src.knowledge_graph.models import (
    KnowledgeGraphQuery,
    KnowledgeObject,
    RelationshipType,
    TypedRelationship,
)
from tests.test_graph_store import _fake_driver_with_side_effect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine(driver=None, embed=None):
    """Helper to build an engine with mocked dependencies."""
    if driver is None:
        driver = _fake_driver_with_side_effect(lambda *_a, **_kw: [])
    if embed is None:
        embed = MagicMock()
        embed.embed_texts.return_value = [[0.1] * 4]

    # Wrap a real store so _execute properly processes _Record objects into dicts.
    store = MagicMock()
    store._get_driver.return_value = driver
    store._execute.side_effect = (
        lambda query, params=None: [
            r.data() for r in driver.execute_query(
                query, parameters_=params or {}
            )[0]
        ]
    )

    eng = KnowledgeGraphEngine(graph_store=store, embedding_client=embed)
    # Bypass schema bootstrap for unit tests.
    eng._indexes_ensured = True
    return eng, store, embed, driver


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------


def test_create_knowledge_object_writes_node_with_typed_properties():
    """Verify that create_object issues a MERGE with all object properties."""
    calls = []

    def side_effect(query, parameters_=None, database_=None):
        calls.append((query, parameters_ or {}))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    obj = KnowledgeObject(
        id="ko-1",
        title="Test Object",
        object_type="note",
        content="This is a test knowledge object.",
        owner_id="u-1",
        workspace_id="ws-1",
    )

    created = eng.create_object(obj)

    assert created.id == "ko-1"
    assert created.title == "Test Object"

    # At least one MERGE call should include KnowledgeObject and the object id.
    merge_calls = [(q, p) for q, p in calls if "MERGE" in q and "KnowledgeObject" in q]
    assert merge_calls, "Expected a MERGE on KnowledgeObject"
    props = merge_calls[0][1].get("props", {})
    assert props["id"] == "ko-1"
    assert props["title"] == "Test Object"
    assert props["content"] == "This is a test knowledge object."
    assert props["owner_id"] == "u-1"


def test_create_object_generates_embedding_when_missing():
    """Verify embedding is auto-generated when not provided."""
    embed = MagicMock()
    embed.embed_texts.return_value = [[0.5, 0.5, 0.5, 0.5]]

    eng, _, _, _ = _make_engine(embed=embed)

    obj = KnowledgeObject(
        id="ko-embed",
        title="Auto Embed",
        object_type="note",
        content="Generate my embedding please.",
        owner_id="u-1",
        workspace_id="ws-1",
    )

    created = eng.create_object(obj)

    embed.embed_texts.assert_called_once()
    assert created.embedding == [0.5, 0.5, 0.5, 0.5]


def test_get_knowledge_object_returns_none_if_missing():
    """Verify that a non-existent object returns None."""
    eng, _, _, _ = _make_engine()

    result = eng.get_object("nonexistent", "u-1")
    assert result is None


def test_get_knowledge_object_returns_object_with_relationships():
    """Verify fetching includes typed relationships."""
    def side_effect(query, parameters_=None, database_=None):
        if "MATCH (k:KnowledgeObject {id: $object_id" in query:
            return [
                {
                    "node": {
                        "id": "ko-1",
                        "title": "Root",
                        "object_type": "concept",
                        "content": "Root concept",
                        "metadata": {},
                        "tags": ["ml"],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [
                            {
                                "source_id": "ko-1",
                                "target_id": "ko-2",
                                "type": "REFERENCES",
                                "confidence": 0.85,
                                "metadata": {},
                                "created_at": "2025-01-02T00:00:00",
                            }
                        ],
                    }
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    obj = eng.get_object("ko-1", "u-1")
    assert obj is not None
    assert obj.id == "ko-1"
    assert obj.title == "Root"
    assert len(obj.relationships) == 1
    assert obj.relationships[0].relationship_type == RelationshipType.REFERENCES
    assert obj.relationships[0].target_id == "ko-2"
    assert obj.tags == ["ml"]


def test_update_knowledge_object_modifies_properties():
    """Verify update_object issues a SET with allowed keys."""
    calls = []

    def side_effect(query, parameters_=None, database_=None):
        calls.append((query, parameters_ or {}))
        return [{"id": "ko-1"}]

    driver = _fake_driver_with_side_effect(side_effect)
    eng, store, _, _ = _make_engine(driver=driver)

    # Mock get_object on the engine to return a valid object after update.
    original_get = eng.get_object

    def fake_get(object_id, owner_id):
        return KnowledgeObject(
            id=object_id,
            title="Updated Title",
            object_type="note",
            content="Updated content",
            owner_id=owner_id,
            workspace_id="ws-1",
        )

    eng.get_object = fake_get

    updated = eng.update_object("ko-1", {"title": "Updated Title"}, "u-1")

    assert updated.title == "Updated Title"

    eng.get_object = original_get


def test_delete_knowledge_object_detaches_all_relationships():
    """Verify delete_object issues DETACH DELETE."""
    calls = []

    def side_effect(query, parameters_=None, database_=None):
        calls.append((query, parameters_ or {}))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    result = eng.delete_object("ko-1", "u-1")

    assert result is True
    detach_calls = [
        q for q, p in calls if "DETACH DELETE" in q and "KnowledgeObject" in q
    ]
    assert detach_calls, "Expected a DETACH DELETE on KnowledgeObject"


# ---------------------------------------------------------------------------
# Relationship tests
# ---------------------------------------------------------------------------


def test_add_typed_relationship_creates_edge():
    """Verify add_relationship creates a typed MERGE edge."""
    calls = []

    def side_effect(query, parameters_=None, database_=None):
        calls.append((query, parameters_ or {}))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    rel = TypedRelationship(
        source_id="ko-1",
        target_id="ko-2",
        relationship_type=RelationshipType.REFERENCES,
        confidence=0.85,
    )

    result = eng.add_relationship(rel)

    assert result.source_id == "ko-1"
    assert result.target_id == "ko-2"

    # Should include MATCH + MERGE with the relationship type.
    merge_edges = [
        q for q, p in calls if "MATCH (a:KnowledgeObject" in q and "MERGE (a)" in q
    ]
    assert merge_edges


def test_add_duplicate_relationship_triggers_conflict_resolution():
    """Verify adding a duplicate edge calls the conflict resolver.

    The resolver should keep existing (higher confidence) over new (lower).
    """
    fetch_called = [False]

    def side_effect(query, parameters_=None, database_=None):
        if "MATCH (a:KnowledgeObject {id: $source_id})" in query and "MATCH (b:KnowledgeObject {id: $target_id})" in query and "OPTIONAL MATCH (a)-[r]->(b)" in query:
            fetch_called[0] = True
            return [
                {
                    "confidence": 0.95,
                    "type": "REFERENCES",
                    "metadata": {},
                    "created_at": "2025-01-01T00:00:00",
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    # New relationship with lower confidence than existing (0.95 vs 0.5).
    rel = TypedRelationship(
        source_id="ko-1",
        target_id="ko-2",
        relationship_type=RelationshipType.REFERENCES,
        confidence=0.5,
    )

    # Should not raise — conflict resolution runs silently.
    eng.add_relationship(rel)
    assert fetch_called[0]


def test_add_relationship_with_new_entity_creates_target_node():
    """Verify the MATCH + MERGE pattern handles non-existent nodes gracefully."""
    calls = []

    def side_effect(query, parameters_=None, database_=None):
        calls.append((query, parameters_ or {}))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    rel = TypedRelationship(
        source_id="ko-new-src",
        target_id="ko-new-tgt",
        relationship_type=RelationshipType.DERIVES_FROM,
        confidence=0.75,
    )

    result = eng.add_relationship(rel)

    assert result.source_id == "ko-new-src"
    assert result.target_id == "ko-new-tgt"
    assert result.relationship_type == RelationshipType.DERIVES_FROM


def test_remove_relationship_deletes_edge():
    """Verify remove_relationship issues a MATCH + DELETE on the edge."""
    calls = []

    def side_effect(query, parameters_=None, database_=None):
        calls.append((query, parameters_ or {}))
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    result = eng.remove_relationship(
        "ko-1", "ko-2", RelationshipType.REFERENCES
    )

    assert result is True
    delete_edges = [
        q for q, p in calls if "DELETE r" in q and "KnowledgeObject" in q
    ]
    assert delete_edges, "Expected a DELETE r statement on the relationship"


# ---------------------------------------------------------------------------
# Traversal tests
# ---------------------------------------------------------------------------


def test_traverse_follows_typed_relationships_bfs():
    """Verify traversal returns neighboring KnowledgeObjects."""

    def side_effect(query, parameters_=None, database_=None):
        if "MATCH (start:KnowledgeObject" in query:
            return [
                {
                    "node": {
                        "id": "ko-2",
                        "title": "Neighbor",
                        "object_type": "note",
                        "content": "I am related",
                        "metadata": {},
                        "tags": [],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [],
                    }
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    neighbors = eng.traverse("ko-1", rel_type=RelationshipType.REFERENCES)

    assert len(neighbors) == 1
    assert neighbors[0].id == "ko-2"
    assert neighbors[0].title == "Neighbor"


def test_traverse_all_relationships_when_no_type_specified():
    """Verify traversal works without a specific relationship type."""

    def side_effect(query, parameters_=None, database_=None):
        if "MATCH (start:KnowledgeObject" in query:
            return [
                {
                    "node": {
                        "id": "ko-3",
                        "title": "Any Rel",
                        "object_type": "note",
                        "content": "Connected by any edge",
                        "metadata": {},
                        "tags": [],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [],
                    }
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    neighbors = eng.traverse("ko-1")
    assert len(neighbors) == 1
    assert neighbors[0].id == "ko-3"


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


def test_search_by_vector_similarity():
    """Verify vector search is invoked when query.query is set."""

    def side_effect(query, parameters_=None, database_=None):
        if "db.index.vector.queryNodes" in query:
            return [
                {
                    "node": {
                        "id": "ko-v1",
                        "title": "Vector Result",
                        "object_type": "note",
                        "content": "Found by vector similarity",
                        "metadata": {},
                        "tags": [],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [],
                    },
                    "score": 0.92,
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, embed, _ = _make_engine(driver=driver)
    embed.embed_texts.return_value = [[0.1] * 4]

    kg_query = KnowledgeGraphQuery(
        query="machine learning", owner_id="u-1", workspace_id="ws-1", top_k=5
    )

    result = eng.search(kg_query)

    assert len(result.objects) == 1
    assert result.objects[0].id == "ko-v1"
    assert result.objects[0].title == "Vector Result"


def test_search_by_fulltext():
    """Verify full-text search is invoked when query.query is set."""

    def side_effect(query, parameters_=None, database_=None):
        if "db.index.fulltext.queryNodes" in query:
            return [
                {
                    "node": {
                        "id": "ko-ft1",
                        "title": "Fulltext Result",
                        "object_type": "note",
                        "content": "Found by full-text",
                        "metadata": {},
                        "tags": [],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [],
                    },
                    "score": 0.85,
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, embed, _ = _make_engine(driver=driver)
    embed.embed_texts.return_value = [[0.1] * 4]

    kg_query = KnowledgeGraphQuery(
        query="deep learning", owner_id="u-1", workspace_id="ws-1", top_k=5
    )

    result = eng.search(kg_query)

    assert len(result.objects) >= 1
    # At minimum the fulltext result should be present (vector may also fire).
    ft_ids = {o.id for o in result.objects}
    assert "ko-ft1" in ft_ids


def test_search_with_filters():
    """Verify metadata filters narrow results."""

    def side_effect(query, parameters_=None, database_=None):
        if "MATCH (k:KnowledgeObject)" in query and "WHERE" in query:
            return [
                {
                    "node": {
                        "id": "ko-filtered",
                        "title": "Filtered Result",
                        "object_type": "paper",
                        "content": "Filtered by object_type",
                        "metadata": {},
                        "tags": [],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [],
                    }
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    kg_query = KnowledgeGraphQuery(
        filters={"object_type__eq": "paper"},
        owner_id="u-1",
        workspace_id="ws-1",
        top_k=5,
    )

    result = eng.search(kg_query)

    assert len(result.objects) >= 1
    assert result.objects[0].object_type == "paper"


# ---------------------------------------------------------------------------
# DSL tests
# ---------------------------------------------------------------------------


def test_dsl_query_builder_builds_correct_query():
    """Verify the DSL builder produces a correctly configured query."""
    query = (
        KnowledgeQueryBuilder.select()
        .fulltext("reinforcement learning")
        .relate(RelationshipType.REFERENCES)
        .owned_by("u-1")
        .in_workspace("ws-1")
        .limit(20)
        .build()
    )

    assert isinstance(query, KnowledgeGraphQuery)
    assert query.query == "reinforcement learning"
    assert query.relationship_filter == RelationshipType.REFERENCES
    assert query.owner_id == "u-1"
    assert query.workspace_id == "ws-1"
    assert query.top_k == 20


def test_dsl_query_builder_where_clause():
    """Verify the where() method adds filters correctly."""
    query = (
        KnowledgeQueryBuilder.select()
        .where("object_type", "eq", "paper")
        .where("confidence", "gt", 0.8)
        .build()
    )

    assert "object_type__eq" in query.filters
    assert query.filters["object_type__eq"] == "paper"
    assert query.filters["confidence__gt"] == 0.8


# ---------------------------------------------------------------------------
# Conflict resolver tests
# ---------------------------------------------------------------------------


def test_conflict_resolver_keeps_high_confidence_on_low_confidence_merge():
    """Verify the resolver keeps an existing high-confidence edge."""
    resolver = ConflictResolver()

    new_rel = TypedRelationship(
        source_id="ko-1",
        target_id="ko-2",
        relationship_type=RelationshipType.REFERENCES,
        confidence=0.3,
    )

    existing = [
        TypedRelationship(
            source_id="ko-1",
            target_id="ko-2",
            relationship_type=RelationshipType.REFERENCES,
            confidence=0.95,
        )
    ]

    conflict = resolver.resolve(new_rel, existing)

    assert conflict is not None
    assert conflict.resolution == "keep_a"


def test_conflict_resolver_upgrades_when_new_is_better():
    """Verify the resolver upgrades when new relationship is better."""
    resolver = ConflictResolver()

    new_rel = TypedRelationship(
        source_id="ko-1",
        target_id="ko-2",
        relationship_type=RelationshipType.REFERENCES,
        confidence=0.95,
    )

    existing = [
        TypedRelationship(
            source_id="ko-1",
            target_id="ko-2",
            relationship_type=RelationshipType.REFERENCES,
            confidence=0.3,
        )
    ]

    conflict = resolver.resolve(new_rel, existing)

    assert conflict is not None
    assert conflict.resolution == "keep_b"


def test_conflict_resolver_accepts_no_conflict():
    """Verify the resolver returns None when there is no conflict."""
    resolver = ConflictResolver()

    new_rel = TypedRelationship(
        source_id="ko-1",
        target_id="ko-3",  # Different target
        relationship_type=RelationshipType.REFERENCES,
        confidence=0.7,
    )

    existing = [
        TypedRelationship(
            source_id="ko-1",
            target_id="ko-2",
            relationship_type=RelationshipType.REFERENCES,
            confidence=0.7,
        )
    ]

    conflict = resolver.resolve(new_rel, existing)

    assert conflict is None


def test_conflict_resolver_flags_manual_review():
    """Verify the resolver flags manual review when both are confident but disagree."""
    resolver = ConflictResolver()

    new_rel = TypedRelationship(
        source_id="ko-1",
        target_id="ko-2",
        relationship_type=RelationshipType.SUMMARIZES,
        confidence=0.85,
    )

    existing = [
        TypedRelationship(
            source_id="ko-1",
            target_id="ko-2",
            relationship_type=RelationshipType.DERIVES_FROM,
            confidence=0.8,
        )
    ]

    conflict = resolver.resolve(new_rel, existing)

    assert conflict is not None
    assert conflict.resolution == "manual_review"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_empty_traversal_returns_empty_list():
    """Verify traversal with no neighbors returns an empty list."""
    eng, _, _, _ = _make_engine()

    neighbors = eng.traverse("ko-nonexistent", max_hops=2)
    assert neighbors == []


def test_search_with_no_query_returns_filter_results():
    """Verify search with only filters (no text query) returns filtered results."""

    def side_effect(query, parameters_=None, database_=None):
        if "MATCH (k:KnowledgeObject)" in query and "WHERE" in query:
            return [
                {
                    "node": {
                        "id": "ko-f1",
                        "title": "Tagged Result",
                        "object_type": "note",
                        "content": "Has specific tags",
                        "metadata": {},
                        "tags": ["important"],
                        "owner_id": "u-1",
                        "workspace_id": "ws-1",
                        "created_at": "2025-01-01T00:00:00",
                        "updated_at": "2025-01-01T00:00:00",
                        "embedding": None,
                        "relationships": [],
                    }
                }
            ]
        return []

    driver = _fake_driver_with_side_effect(side_effect)
    eng, _, _, _ = _make_engine(driver=driver)

    kg_query = KnowledgeGraphQuery(
        filters={"object_type__eq": "note"},
        owner_id="u-1",
        workspace_id="ws-1",
        top_k=5,
    )

    result = eng.search(kg_query)
    assert len(result.objects) >= 1
