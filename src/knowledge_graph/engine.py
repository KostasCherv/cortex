"""Knowledge graph engine — CRUD, traversal, and search orchestration."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from src.config import settings
from src.errors import VectorStoreError
from src.knowledge_graph.conflict import ConflictResolver
from src.knowledge_graph.models import (
    ConflictRecord,
    KnowledgeGraphQuery,
    KnowledgeGraphResult,
    KnowledgeObject,
    RelationshipType,
    TypedRelationship,
)
from src.llm.embeddings import EmbeddingClient
from src.tools.neo4j_graph_store import Neo4jGraphStore

logger = logging.getLogger(__name__)


class KnowledgeGraphEngine:
    """High-level engine for knowledge objects and typed relationships.

    Wraps ``Neo4jGraphStore`` for Neo4j operations and adds typed relationship
    management, conflict resolution, multi-mode search, and graph traversal.
    """

    def __init__(
        self,
        *,
        graph_store: Neo4jGraphStore | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._embedding_client = embedding_client or EmbeddingClient()
        self._graph_store = graph_store or Neo4jGraphStore(
            embedding_client=self._embedding_client
        )
        self._conflict_resolver = ConflictResolver()
        self._indexes_ensured = False

    # ------------------------------------------------------------------
    # Schema / indexes
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        if self._indexes_ensured:
            return

        try:
            driver = self._graph_store._get_driver()
            queries = [
                "CREATE CONSTRAINT ko_id_unique IF NOT EXISTS FOR (k:KnowledgeObject) REQUIRE k.id IS UNIQUE",
                "CREATE INDEX ko_owner_id IF NOT EXISTS FOR (k:KnowledgeObject) ON (k.owner_id)",
                "CREATE INDEX ko_workspace_id IF NOT EXISTS FOR (k:KnowledgeObject) ON (k.workspace_id)",
                "CREATE INDEX ko_object_type IF NOT EXISTS FOR (k:KnowledgeObject) ON (k.object_type)",
            ]

            try:
                queries.append(
                    "CREATE FULLTEXT INDEX knowledge_ft IF NOT EXISTS "
                    "FOR (k:KnowledgeObject) ON EACH [k.title, k.content]"
                )
            except Exception:
                logger.warning("Full-text index creation not supported on this Neo4j version; skipping.")

            try:
                queries.append(
                    "CREATE VECTOR INDEX knowledge_object_embedding_index IF NOT EXISTS "
                    "FOR (k:KnowledgeObject) ON (k.embedding) "
                    "OPTIONS {indexConfig: {`vector.dimensions`: $dimensions, "
                    "`vector.similarity_function`: 'cosine'}}"
                )
            except Exception:
                logger.warning(
                    "Vector index creation not supported on this Neo4j version; skipping."
                )

            for q in queries:
                try:
                    driver.execute_query(q, parameters_={"dimensions": settings.embedding_dimensions})
                except Exception as exc:
                    logger.debug("Index query skipped: %s", exc)

            self._indexes_ensured = True
        except Exception as exc:
            logger.warning("Could not ensure knowledge graph indexes: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._graph_store._execute(query, params or {})

    def _hash_id(self, prefix: str, raw: str) -> str:
        return hashlib.sha1(f"{prefix}:{raw}".encode("utf-8")).hexdigest()  # nosec B324 — non-security ID generation, not used for auth/crypto

    def _row_to_object(self, row: dict[str, Any]) -> KnowledgeObject:
        """Convert a Neo4j record row to a ``KnowledgeObject``."""
        raw_rels = row.get("relationships", [])
        relationships: list[TypedRelationship] = []
        if isinstance(raw_rels, list):
            for r in raw_rels:
                if isinstance(r, dict):
                    try:
                        relationships.append(
                            TypedRelationship(
                                source_id=r.get("source_id", row.get("id", "")),
                                target_id=r.get("target_id", ""),
                                relationship_type=RelationshipType(
                                    r.get("type", RelationshipType.REFERENCES.value)
                                ),
                                confidence=float(r.get("confidence", 0.7)),
                                metadata=r.get("metadata", {}),
                                created_at=r.get("created_at", ""),
                            )
                        )
                    except (ValueError, TypeError, KeyError):
                        continue

        embedding_raw = row.get("embedding")
        embedding: list[float] | None = None
        if isinstance(embedding_raw, (list, tuple)):
            embedding = [float(v) for v in embedding_raw]

        return KnowledgeObject(
            id=row.get("id", ""),
            title=row.get("title", ""),
            object_type=row.get("object_type", ""),
            content=row.get("content", ""),
            metadata=row.get("metadata", {}),
            relationships=relationships,
            tags=list(row.get("tags", []) or []),
            owner_id=row.get("owner_id", ""),
            workspace_id=row.get("workspace_id", ""),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            embedding=embedding,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_object(self, obj: KnowledgeObject) -> KnowledgeObject:
        """Persist a new KnowledgeObject node.

        If *obj.embedding* is ``None``, an embedding is generated automatically
        from the object's content.
        """
        self._ensure_indexes()

        if obj.embedding is None:
            try:
                vectors = self._embedding_client.embed_texts([obj.content])
                obj.embedding = vectors[0]
            except Exception as exc:
                logger.warning("Embedding generation failed for %s: %s", obj.id, exc)

        now = datetime.now(UTC).isoformat()
        props = {
            "id": obj.id,
            "title": obj.title,
            "object_type": obj.object_type,
            "content": obj.content,
            "metadata": obj.metadata,
            "tags": obj.tags,
            "owner_id": obj.owner_id,
            "workspace_id": obj.workspace_id,
            "created_at": now,
            "updated_at": now,
            "embedding": obj.embedding,
        }

        self._execute(
            """
            MERGE (k:KnowledgeObject {id: $props.id})
            SET k += $props
            """,
            {"props": props},
        )

        return obj.model_copy(update={"created_at": now, "updated_at": now})

    def get_object(self, object_id: str, owner_id: str) -> KnowledgeObject | None:
        """Fetch a KnowledgeObject by id, including its typed relationships.

        Returns ``None`` if not found or owned by a different user.
        """
        rows = self._execute(
            """
            MATCH (k:KnowledgeObject {id: $object_id, owner_id: $owner_id})
            OPTIONAL MATCH (k)-[r]->(target:KnowledgeObject)
            WITH k, collect(CASE WHEN r IS NOT NULL THEN {
                source_id: k.id,
                target_id: target.id,
                type: type(r),
                confidence: r.confidence,
                metadata: r.metadata,
                created_at: r.created_at
            } END) AS relationships
            RETURN k{.*, relationships: relationships} AS node
            """,
            {"object_id": object_id, "owner_id": owner_id},
        )

        if not rows or not rows[0].get("node"):
            return None

        return self._row_to_object(rows[0]["node"])

    def update_object(
        self, object_id: str, updates: dict[str, Any], owner_id: str
    ) -> KnowledgeObject:
        """Partially update a KnowledgeObject.

        Raises ``VectorStoreError`` if the object does not exist or belongs to
        a different owner.
        """
        allowed_keys = {
            "title", "content", "metadata", "tags", "object_type", "embedding"
        }
        safe_updates = {k: v for k, v in updates.items() if k in allowed_keys}
        safe_updates["updated_at"] = datetime.now(UTC).isoformat()

        self._execute(
            """
            MATCH (k:KnowledgeObject {id: $object_id, owner_id: $owner_id})
            SET k += $updates
            RETURN k.id AS id
            """,
            {"object_id": object_id, "owner_id": owner_id, "updates": safe_updates},
        )

        result = self.get_object(object_id, owner_id)
        if result is None:
            raise VectorStoreError(
                f"KnowledgeObject {object_id} not found or not owned by {owner_id}"
            )
        return result

    def delete_object(self, object_id: str, owner_id: str) -> bool:
        """Delete a KnowledgeObject and detach all its relationships.

        Returns ``True`` if deleted, ``False`` if not found.
        """
        self._execute(
            """
            MATCH (k:KnowledgeObject {id: $object_id, owner_id: $owner_id})
            DETACH DELETE k
            """,
            {"object_id": object_id, "owner_id": owner_id},
        )
        return True

    # ------------------------------------------------------------------
    # Relationship management
    # ------------------------------------------------------------------

    def add_relationship(self, rel: TypedRelationship) -> TypedRelationship:
        """Create a typed edge between two KnowledgeObjects.

        If an edge of the same type already exists between the pair, the
        conflict resolver is invoked to decide whether to keep, upgrade, or
        flag the new relationship.
        """
        self._ensure_indexes()

        # Check for existing relationships of the same type between this pair.
        existing = self._fetch_relationships(rel.source_id, rel.target_id)

        conflict: ConflictRecord | None = self._conflict_resolver.resolve(
            rel, existing
        )

        if conflict is not None and conflict.resolution == "keep_a":
            # Keep existing — do not create the new edge.
            logger.info(
                "Conflict: keeping existing relationship between %s and %s (%s)",
                rel.source_id,
                rel.target_id,
                conflict.reason,
            )
            return rel

        now = datetime.now(UTC).isoformat()
        edge_props = {
            "confidence": rel.confidence,
            "metadata": rel.metadata,
            "created_at": now,
        }

        self._execute(
            """
            MATCH (a:KnowledgeObject {id: $source_id})
            MATCH (b:KnowledgeObject {id: $target_id})
            MERGE (a)-[r:$rel_type]->(b)
            SET r += $props
            """.replace("$rel_type", rel.relationship_type.value),
            {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "props": edge_props,
            },
        )

        if conflict is not None and conflict.resolution == "manual_review":
            logger.warning(
                "Manual review flagged: %s -> %s (%s)",
                rel.source_id,
                rel.target_id,
                conflict.reason,
            )

        return rel

    def remove_relationship(
        self, source_id: str, target_id: str, rel_type: RelationshipType
    ) -> bool:
        """Delete a typed edge between two KnowledgeObjects.

        Returns ``True`` whether or not the edge existed.
        """
        self._execute(
            """
            MATCH (a:KnowledgeObject {id: $source_id})
            MATCH (b:KnowledgeObject {id: $target_id})
            MATCH (a)-[r:$rel_type]->(b)
            DELETE r
            """.replace("$rel_type", rel_type.value),
            {"source_id": source_id, "target_id": target_id},
        )
        return True

    def _fetch_relationships(
        self, source_id: str, target_id: str
    ) -> list[TypedRelationship]:
        """Fetch all typed relationships between two nodes (both directions)."""
        rows = self._execute(
            """
            MATCH (a:KnowledgeObject {id: $source_id})
            MATCH (b:KnowledgeObject {id: $target_id})
            OPTIONAL MATCH (a)-[r]->(b)
            RETURN r.confidence AS confidence,
                   type(r) AS type,
                   r.metadata AS metadata,
                   r.created_at AS created_at
            """,
            {"source_id": source_id, "target_id": target_id},
        )

        result: list[TypedRelationship] = []
        for row in rows:
            rel_type_raw = row.get("type")
            if not rel_type_raw:
                continue
            try:
                result.append(
                    TypedRelationship(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=RelationshipType(rel_type_raw),
                        confidence=float(row.get("confidence", 0.7)),
                        metadata=row.get("metadata", {}),
                        created_at=row.get("created_at", ""),
                    )
                )
            except (ValueError, TypeError):
                continue
        return result

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        object_id: str,
        rel_type: RelationshipType | None = None,
        max_hops: int = 2,
    ) -> list[KnowledgeObject]:
        """BFS traversal from a starting KnowledgeObject.

        Follows typed relationships (or all relationships if *rel_type* is
        ``None``) up to *max_hops* deep. Returns all reachable objects.
        """
        if rel_type:
            match_clause = f"(start)-[:{rel_type.value}*1..{max_hops}]->(neighbor)"
        else:
            match_clause = f"(start)-[*1..{max_hops}]-(neighbor)"

        rows = self._execute(
            f"""
            MATCH (start:KnowledgeObject {{id: $object_id}})
            MATCH {match_clause}
            WHERE neighbor.id <> $object_id
            WITH DISTINCT neighbor
            OPTIONAL MATCH (neighbor)-[r]->(rel_target:KnowledgeObject)
            WITH neighbor, collect(CASE WHEN r IS NOT NULL THEN {{
                source_id: neighbor.id,
                target_id: rel_target.id,
                type: type(r),
                confidence: r.confidence,
                metadata: r.metadata,
                created_at: r.created_at
            }} END) AS relationships
            RETURN neighbor{{.*, relationships: relationships}} AS node
            LIMIT 100
            """,
            {"object_id": object_id},
        )

        return [self._row_to_object(row["node"]) for row in rows if row.get("node")]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: KnowledgeGraphQuery) -> KnowledgeGraphResult:
        """Multi-mode search across KnowledgeObjects.

        Supports:
        * Vector similarity search (when *query.query* is set)
        * Full-text search (when *query.query* is set and full-text index exists)
        * Metadata filter-based search

        Results are combined with deduplication and scored by relevance.
        """
        self._ensure_indexes()
        seen_ids: set[str] = set()
        results: list[KnowledgeObject] = []
        details: dict[str, Any] = {}

        # --- Vector similarity search ---
        if query.query and query.owner_id:
            try:
                vec = self._embedding_client.embed_texts([query.query])[0]
                vec_rows = self._execute(
                    """
                    CALL db.index.vector.queryNodes(
                        'knowledge_object_embedding_index',
                        $top_k * 4,
                        $embedding
                    )
                    YIELD node, score
                    WHERE node:KnowledgeObject
                      AND ($owner_id IS NULL OR node.owner_id = $owner_id)
                      AND ($workspace_id IS NULL OR node.workspace_id = $workspace_id)
                    RETURN node{.*} AS node, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """,
                    {
                        "embedding": vec,
                        "top_k": query.top_k,
                        "owner_id": query.owner_id,
                        "workspace_id": query.workspace_id,
                    },
                )
                for row in vec_rows:
                    node_data = row.get("node")
                    if node_data and node_data.get("id") not in seen_ids:
                        seen_ids.add(node_data["id"])
                        results.append(self._row_to_object(node_data))

                details["vector_count"] = len(vec_rows)
            except Exception as exc:
                logger.debug("Vector search skipped: %s", exc)
                details["vector_error"] = str(exc)

        # --- Full-text search ---
        if query.query:
            try:
                ft_rows = self._execute(
                    """
                    CALL db.index.fulltext.queryNodes(
                        'knowledge_ft',
                        $query_text
                    )
                    YIELD node, score
                    WHERE ($owner_id IS NULL OR node.owner_id = $owner_id)
                      AND ($workspace_id IS NULL OR node.workspace_id = $workspace_id)
                    RETURN node{.*} AS node, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """,
                    {
                        "query_text": query.query,
                        "top_k": query.top_k,
                        "owner_id": query.owner_id,
                        "workspace_id": query.workspace_id,
                    },
                )
                for row in ft_rows:
                    node_data = row.get("node")
                    if node_data and node_data.get("id") not in seen_ids:
                        seen_ids.add(node_data["id"])
                        results.append(self._row_to_object(node_data))

                details["fulltext_count"] = len(ft_rows)
            except Exception as exc:
                logger.debug("Full-text search skipped: %s", exc)
                details["fulltext_error"] = str(exc)

        # --- Metadata filter search ---
        if query.filters:
            filter_clauses: list[str] = []
            filter_params: dict[str, Any] = {"owner_id": query.owner_id}
            for idx, (key, value) in enumerate(query.filters.items()):
                param_key = f"filter_{idx}"
                if "__" in key:
                    field, op = key.rsplit("__", 1)
                    if op == "eq":
                        filter_clauses.append(f"k.{field} = ${param_key}")
                    elif op == "in":
                        filter_clauses.append(f"k.{field} IN ${param_key}")
                    elif op == "gt":
                        filter_clauses.append(f"k.{field} > ${param_key}")
                    elif op == "lt":
                        filter_clauses.append(f"k.{field} < ${param_key}")
                    else:
                        filter_clauses.append(f"k.{field} = ${param_key}")
                else:
                    filter_clauses.append(f"k.{key} = ${param_key}")
                filter_params[param_key] = value

            if query.workspace_id:
                filter_clauses.append("k.workspace_id = $workspace_id")
                filter_params["workspace_id"] = query.workspace_id

            where_clause = " AND ".join(filter_clauses) if filter_clauses else "true"

            filter_rows = self._execute(
                f"""
                MATCH (k:KnowledgeObject)
                WHERE {where_clause}
                  AND ($owner_id IS NULL OR k.owner_id = $owner_id)
                RETURN k{{.*}} AS node
                LIMIT $top_k
                """,
                {**filter_params, "top_k": query.top_k},
            )
            for row in filter_rows:
                node_data = row.get("node")
                if node_data and node_data.get("id") not in seen_ids:
                    seen_ids.add(node_data["id"])
                    results.append(self._row_to_object(node_data))

            details["filter_count"] = len(filter_rows)

        # --- Relationship expansion ---
        if query.relationship_filter and results:
            expanded: list[KnowledgeObject] = []
            expanded_ids: set[str] = set(seen_ids)
            for obj in results:
                neighbors = self.traverse(
                    obj.id,
                    rel_type=query.relationship_filter,
                    max_hops=query.max_hops,
                )
                for n in neighbors:
                    if n.id not in expanded_ids:
                        expanded_ids.add(n.id)
                        expanded.append(n)
            results.extend(expanded)
            details["expansion_count"] = len(expanded)

        return KnowledgeGraphResult(
            objects=results[: query.top_k],
            total_count=len(results),
            query_details=details,
        )
