"""Neo4j-backed graph-first retrieval and ingestion primitives."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from neo4j import GraphDatabase

from src.config import settings
from src.errors import ConfigurationError, VectorStoreError
from src.llm.embeddings import EmbeddingClient
from src.llm.factory import get_llm
from src.llm.output_parsers import (
    EntityRelationExtractionEnvelope,
    build_validation_retry_prompt,
    parse_batched_entity_relation_extraction_json,
    parse_entity_relation_extraction_json,
)

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 120
_MAX_LLM_EXTRACTION_CHUNKS = 5
_DEFAULT_MENTION_CONFIDENCE = 0.7
_DEFAULT_RELATION_CONFIDENCE = 0.6
_MENTION_OVERLAP_BONUS = 0.08
_MENTION_COUNT_BONUS = 0.05
_MENTION_BONUS_CAP = 0.15
_NEIGHBOR_COUNT_BONUS = 0.03
_NEIGHBOR_BONUS_CAP = 0.10
_MAX_BONUS_MENTIONS = 3
_MAX_BONUS_NEIGHBORS = 5
_FAST_INGEST_SOURCE_TYPES = frozenset({"session_attachment"})


@dataclass
class GraphQueryResult:
    context: str
    chunks: list[dict[str, Any]]
    entities: list[str]


class Neo4jGraphStore:
    """Graph store for document/chunk/entity ingestion and GraphRAG retrieval."""

    def __init__(
        self,
        *,
        driver: Any | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._driver = driver
        self._embedding_client = embedding_client or EmbeddingClient()
        self._schema_bootstrapped = False

    def _get_driver(self):
        if self._driver is not None:
            return self._driver

        if not settings.neo4j_uri:
            raise ConfigurationError("NEO4J_URI is required for graph-first retrieval.")
        if not settings.neo4j_username:
            raise ConfigurationError(
                "NEO4J_USERNAME is required for graph-first retrieval."
            )
        if not settings.neo4j_password:
            raise ConfigurationError(
                "NEO4J_PASSWORD is required for graph-first retrieval."
            )

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            notifications_min_severity="OFF",
        )
        try:
            self._driver.verify_connectivity()
        except Exception as exc:
            raise VectorStoreError(f"Failed to connect to Neo4j: {exc}") from exc
        return self._driver

    def _execute(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            records, _, _ = self._get_driver().execute_query(
                query,
                parameters_=parameters or {},
                database_=settings.neo4j_database,
            )
        except Exception as exc:
            raise VectorStoreError(f"Neo4j query failed: {exc}") from exc
        return [record.data() for record in records]

    def _ensure_schema(self) -> None:
        if self._schema_bootstrapped:
            return

        self._execute(
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE"
        )
        self._execute(
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        self._execute(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )
        self._execute(
            "CREATE INDEX chunk_run_id_index IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.run_id)"
        )
        self._execute(
            "CREATE INDEX document_resource_id_index IF NOT EXISTS "
            "FOR (d:Document) ON (d.resource_id)"
        )
        self._execute(
            "CREATE INDEX entity_normalized_name_index IF NOT EXISTS "
            "FOR (e:Entity) ON (e.normalized_name)"
        )
        self._execute(
            "CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dimensions, `vector.similarity_function`: 'cosine'}}",
            {"dimensions": settings.embedding_dimensions},
        )
        self._schema_bootstrapped = True

    def _chunk_text(self, text: str) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + _CHUNK_SIZE, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(0, end - _CHUNK_OVERLAP)
        return chunks

    def _heuristic_entities_relations(
        self, text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Fallback extractor used when LLM extraction is unavailable.
        mentions = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)
        unique_names = []
        seen = set()
        for name in mentions:
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_names.append(name)

        entities: list[dict[str, Any]] = []
        for name in unique_names[:12]:
            normalized = name.lower()
            entity_id = hashlib.sha1(normalized.encode("utf-8")).hexdigest()  # nosec B324 — graph node ID, not a security hash
            entities.append(
                {
                    "id": entity_id,
                    "name": name,
                    "normalized_name": normalized,
                    "entity_type": "Unknown",
                    "confidence": _DEFAULT_MENTION_CONFIDENCE,
                }
            )

        relations: list[dict[str, Any]] = []
        for idx in range(max(0, len(entities) - 1)):
            source = entities[idx]
            target = entities[idx + 1]
            relations.append(
                {
                    "source_id": source["id"],
                    "target_id": target["id"],
                    "source_name": source["name"],
                    "target_name": target["name"],
                    "type": "RELATED",
                    "confidence": _DEFAULT_RELATION_CONFIDENCE,
                }
            )

        return entities, relations

    @staticmethod
    def _llm_response_text(response: Any) -> str:
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            return "\n".join(
                part if isinstance(part, str) else str(part.get("text", ""))
                for part in content
            )
        return str(content)

    def _entities_relations_from_payload(
        self, payload: EntityRelationExtractionEnvelope
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities_by_name: dict[str, dict[str, Any]] = {}
        for entity_row in payload.entities:
            name = entity_row.name.strip()
            normalized = name.lower()
            if normalized in entities_by_name:
                continue
            entities_by_name[normalized] = {
                "id": hashlib.sha1(normalized.encode("utf-8")).hexdigest(),  # nosec B324 — graph node ID, not a security hash
                "name": name,
                "normalized_name": normalized,
                "entity_type": entity_row.entity_type,
                "confidence": entity_row.confidence or _DEFAULT_MENTION_CONFIDENCE,
            }

        relations: list[dict[str, Any]] = []
        for relation_row in payload.relations:
            source_name = relation_row.source.strip()
            target_name = relation_row.target.strip()
            source_key = source_name.lower()
            target_key = target_name.lower()
            if source_key not in entities_by_name or target_key not in entities_by_name:
                continue
            relations.append(
                {
                    "source_id": entities_by_name[source_key]["id"],
                    "target_id": entities_by_name[target_key]["id"],
                    "source_name": entities_by_name[source_key]["name"],
                    "target_name": entities_by_name[target_key]["name"],
                    "type": relation_row.type,
                    "confidence": relation_row.confidence or _DEFAULT_RELATION_CONFIDENCE,
                }
            )

        return list(entities_by_name.values()), relations

    def _parse_entity_relation_payload_with_repair(
        self,
        llm: Any,
        *,
        text_out: str,
        schema_text: str,
        parse_payload: Any,
    ) -> Any:
        try:
            return parse_payload(text_out)
        except Exception as exc:
            repair_prompt = build_validation_retry_prompt(
                schema_text=schema_text,
                invalid_response=text_out,
                validation_error=exc,
            )
            repair_response = llm.invoke(repair_prompt)
            repaired_text = self._llm_response_text(repair_response)
            return parse_payload(repaired_text)

    def _extract_entities_relations(
        self, text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results = self._extract_entities_relations_batched([text])
        return results[0]

    def _extract_entities_relations_batched(
        self, chunk_texts: list[str]
    ) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
        if not chunk_texts:
            return []

        single_chunk_schema = (
            '{"entities":[{"name":"<entity name>","entity_type":"<type>","confidence":0.0-1.0}],'
            '"relations":[{"source":"<entity name>","target":"<entity name>",'
            '"type":"<relationship type>","confidence":0.0-1.0}]}'
        )
        batched_schema = (
            '{"chunks":[{"chunk_index":0,"entities":[{"name":"<entity name>",'
            '"entity_type":"<type>","confidence":0.0-1.0}],'
            '"relations":[{"source":"<entity name>","target":"<entity name>",'
            '"type":"<relationship type>","confidence":0.0-1.0}]}]}'
        )

        if len(chunk_texts) == 1:
            prompt = (
                "Extract entities and directed relationships from the text.\n"
                "Return STRICT JSON only with this schema:\n"
                '{"entities":[{"name":str,"entity_type":str,"confidence":float}],'
                '"relations":[{"source":str,"target":str,"type":str,"confidence":float}]}\n'
                "Limit entities to the most relevant 20.\n"
                f"TEXT:\n{chunk_texts[0][:4500]}"
            )
            parse_payload: Any = parse_entity_relation_extraction_json
            schema_text = single_chunk_schema
        else:
            chunk_sections = []
            for idx, chunk_text in enumerate(chunk_texts):
                chunk_sections.append(f"CHUNK {idx}:\n{chunk_text[:4500]}")
            prompt = (
                "Extract entities and directed relationships from each text chunk.\n"
                "Return STRICT JSON only with this schema:\n"
                '{"chunks":[{"chunk_index":int,"entities":[{"name":str,"entity_type":str,'
                '"confidence":float}],"relations":[{"source":str,"target":str,"type":str,'
                '"confidence":float}]}]}\n'
                "Include one object per chunk index. Limit entities to the most relevant 20 per chunk.\n"
                + "\n\n".join(chunk_sections)
            )
            parse_payload = parse_batched_entity_relation_extraction_json
            schema_text = batched_schema

        try:
            llm = get_llm(temperature=0.1)
            response = llm.invoke(prompt)
            text_out = self._llm_response_text(response)
            payload = self._parse_entity_relation_payload_with_repair(
                llm,
                text_out=text_out,
                schema_text=schema_text,
                parse_payload=parse_payload,
            )

            if len(chunk_texts) == 1:
                single_payload = payload
                entities, relations = self._entities_relations_from_payload(single_payload)
                if entities:
                    return [(entities, relations)]
                return [self._heuristic_entities_relations(chunk_texts[0])]

            by_index: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
            for chunk_payload in payload.chunks:
                entities, relations = self._entities_relations_from_payload(chunk_payload)
                by_index[chunk_payload.chunk_index] = (entities, relations)

            results: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
            for idx, chunk_text in enumerate(chunk_texts):
                extracted = by_index.get(idx)
                if extracted and extracted[0]:
                    results.append(extracted)
                else:
                    results.append(self._heuristic_entities_relations(chunk_text))
            return results
        except Exception as exc:
            logger.info("[graph_store] batched llm extraction fallback: %s", exc)

        return [self._heuristic_entities_relations(chunk_text) for chunk_text in chunk_texts]

    def ingest_document(
        self,
        *,
        document_id: str,
        source_type: str,
        owner_id: str,
        workspace_id: str,
        title: str,
        source_url: str,
        text: str,
        session_id: str | None = None,
        run_id: str | None = None,
        resource_id: str | None = None,
    ) -> int:
        self._ensure_schema()
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        embeddings = self._embedding_client.embed_texts(chunks)

        document_row = {
            "id": document_id,
            "source_type": source_type,
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "run_id": run_id,
            "resource_id": resource_id,
            "title": title[:512],
            "source_url": source_url[:1024],
            "created_at": datetime.now(UTC).isoformat(),
        }

        chunk_rows: list[dict[str, Any]] = []
        next_rows: list[dict[str, str]] = []
        mention_rows: list[dict[str, Any]] = []
        relation_rows: list[dict[str, Any]] = []
        skip_graph_llm = source_type in _FAST_INGEST_SOURCE_TYPES
        llm_chunk_extractions: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = (
            None
        )
        if not skip_graph_llm:
            llm_chunks = chunks[:_MAX_LLM_EXTRACTION_CHUNKS]
            if llm_chunks:
                llm_chunk_extractions = self._extract_entities_relations_batched(llm_chunks)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = hashlib.sha1(f"{document_id}:{idx}".encode("utf-8")).hexdigest()  # nosec B324 — chunk ID for graph linking, not a security hash
            if idx > 0:
                prev_chunk_id = hashlib.sha1(  # nosec B324 — chunk ID for graph linking, not a security hash
                    f"{document_id}:{idx - 1}".encode("utf-8")
                ).hexdigest()
                next_rows.append({"from": prev_chunk_id, "to": chunk_id})

            token_count = len(chunk_text.split())
            chunk_rows.append(
                {
                    "id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "embedding": embeddings[idx],
                    "token_count": token_count,
                    "owner_id": owner_id,
                    "workspace_id": workspace_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "resource_id": resource_id,
                    "source_url": source_url[:1024],
                    "source_title": title[:512],
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )

            if skip_graph_llm:
                entities: list[dict[str, Any]] = []
                relations: list[dict[str, Any]] = []
            elif idx < _MAX_LLM_EXTRACTION_CHUNKS and llm_chunk_extractions is not None:
                entities, relations = llm_chunk_extractions[idx]
            else:
                # Keep graph coverage on long documents without paying for LLM extraction
                # on every chunk.
                entities, relations = self._heuristic_entities_relations(chunk_text)
            for entity in entities:
                mention_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "entity_id": entity["id"],
                        "name": entity["name"],
                        "normalized_name": entity["normalized_name"],
                        "entity_type": entity["entity_type"],
                        "confidence": float(
                            entity.get("confidence") or _DEFAULT_MENTION_CONFIDENCE
                        ),
                    }
                )

            for relation in relations:
                relation_rows.append(
                    {
                        "source_id": relation["source_id"],
                        "target_id": relation["target_id"],
                        "source_name": relation["source_name"],
                        "target_name": relation["target_name"],
                        "type": relation["type"],
                        "confidence": float(
                            relation.get("confidence") or _DEFAULT_RELATION_CONFIDENCE
                        ),
                    }
                )

        self._execute(
            """
            MERGE (d:Document {id: $document.id})
            SET d += $document
            WITH d
            UNWIND $chunks AS row
            MERGE (c:Chunk {id: row.id})
            SET c += row
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {"document": document_row, "chunks": chunk_rows},
        )

        if next_rows:
            self._execute(
                """
                UNWIND $pairs AS pair
                MATCH (a:Chunk {id: pair.from})
                MATCH (b:Chunk {id: pair.to})
                MERGE (a)-[:NEXT_CHUNK]->(b)
                """,
                {"pairs": next_rows},
            )

        if mention_rows:
            self._execute(
                """
                UNWIND $mentions AS row
                MATCH (c:Chunk {id: row.chunk_id})
                MERGE (e:Entity {id: row.entity_id})
                SET e.name = row.name,
                    e.normalized_name = row.normalized_name,
                    e.entity_type = row.entity_type
                MERGE (c)-[m:MENTIONS]->(e)
                SET m.confidence = row.confidence
                """,
                {"mentions": mention_rows},
            )

        if relation_rows:
            self._execute(
                """
                UNWIND $rels AS row
                MERGE (s:Entity {id: row.source_id})
                ON CREATE SET s.name = row.source_name,
                              s.normalized_name = toLower(row.source_name),
                              s.entity_type = 'Unknown'
                MERGE (t:Entity {id: row.target_id})
                ON CREATE SET t.name = row.target_name,
                              t.normalized_name = toLower(row.target_name),
                              t.entity_type = 'Unknown'
                MERGE (s)-[r:RELATES_TO {type: row.type}]->(t)
                SET r.confidence = row.confidence
                """,
                {"rels": relation_rows},
            )

        return len(chunk_rows)

    def _fetch_resource_scoped_candidates(
        self,
        *,
        query_vec: list[float],
        owner_id: str,
        workspace_id: str,
        run_id: str | None,
        resource_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        params = {
            "embedding": query_vec,
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "resource_ids": resource_ids,
            "top_k": top_k,
        }
        rows = self._execute(
            """
            MATCH (node:Chunk)
            WHERE node.owner_id = $owner_id
              AND node.workspace_id = $workspace_id
              AND ($run_id IS NULL OR node.run_id = $run_id)
              AND node.resource_id IN $resource_ids
            RETURN node.id AS chunk_id,
                   node.document_id AS document_id,
                   node.resource_id AS resource_id,
                   node.text AS text,
                   node.source_url AS source_url,
                   node.source_title AS source_title,
                   node.chunk_index AS chunk_index,
                   vector.similarity.cosine(node.embedding, $embedding) AS score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            params,
        )
        if rows:
            return rows
        return self._execute(
            """
            MATCH (node:Chunk)
            WHERE node.owner_id = $owner_id
              AND node.workspace_id = $workspace_id
              AND node.resource_id IN $resource_ids
            RETURN node.id AS chunk_id,
                   node.document_id AS document_id,
                   node.resource_id AS resource_id,
                   node.text AS text,
                   node.source_url AS source_url,
                   node.source_title AS source_title,
                   node.chunk_index AS chunk_index,
                   vector.similarity.cosine(node.embedding, $embedding) AS score
            ORDER BY node.chunk_index ASC
            LIMIT $top_k
            """,
            params,
        )

    def _fetch_global_candidates(
        self,
        *,
        query_vec: list[float],
        owner_id: str,
        workspace_id: str,
        run_id: str | None,
        candidate_k: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        params = {
            "candidate_k": candidate_k,
            "embedding": query_vec,
            "owner_id": owner_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "top_k": top_k,
        }
        try:
            return self._execute(
                """
                MATCH (node:Chunk)
                SEARCH node IN (
                  VECTOR INDEX chunk_embedding_index
                  FOR $embedding
                  LIMIT $candidate_k
                )
                WITH node
                WHERE node.owner_id = $owner_id
                  AND node.workspace_id = $workspace_id
                  AND ($run_id IS NULL OR node.run_id = $run_id)
                RETURN node.id AS chunk_id,
                       node.document_id AS document_id,
                       node.resource_id AS resource_id,
                       node.text AS text,
                       node.source_url AS source_url,
                       node.source_title AS source_title,
                       node.chunk_index AS chunk_index,
                       vector.similarity.cosine(node.embedding, $embedding) AS score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                params,
            )
        except VectorStoreError as exc:
            if (
                "invalid syntax" not in str(exc).lower()
                and "search" not in str(exc).lower()
            ):
                raise
            return self._execute(
                """
                CALL db.index.vector.queryNodes('chunk_embedding_index', $candidate_k, $embedding)
                YIELD node, score
                WHERE node:Chunk
                  AND node.owner_id = $owner_id
                  AND node.workspace_id = $workspace_id
                  AND ($run_id IS NULL OR node.run_id = $run_id)
                RETURN node.id AS chunk_id,
                       node.document_id AS document_id,
                       node.resource_id AS resource_id,
                       node.text AS text,
                       node.source_url AS source_url,
                       node.source_title AS source_title,
                       node.chunk_index AS chunk_index,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                params,
            )

    def query_context(
        self,
        *,
        query: str,
        owner_id: str,
        workspace_id: str,
        run_id: str | None = None,
        resource_ids: list[str] | None = None,
        top_k: int | None = None,
        max_hops: int | None = None,
    ) -> GraphQueryResult:
        self._ensure_schema()
        query_vec = self._embedding_client.embed_texts([query])[0]
        effective_top_k = top_k or settings.graph_rag_top_k
        effective_hops = max(1, min(max_hops or settings.graph_rag_max_hops, 2))

        candidate_k = max(12, effective_top_k * 3)
        if resource_ids:
            candidate_rows = self._fetch_resource_scoped_candidates(
                query_vec=query_vec,
                owner_id=owner_id,
                workspace_id=workspace_id,
                run_id=run_id,
                resource_ids=resource_ids,
                top_k=candidate_k,
            )
        else:
            candidate_rows = self._fetch_global_candidates(
                query_vec=query_vec,
                owner_id=owner_id,
                workspace_id=workspace_id,
                run_id=run_id,
                candidate_k=candidate_k,
                top_k=candidate_k,
            )

        if not candidate_rows:
            return GraphQueryResult(context="", chunks=[], entities=[])

        rel_pattern = "*1..1" if effective_hops == 1 else "*1..2"
        enrich_query = f"""
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {{id: chunk_id}})
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[:RELATES_TO{rel_pattern}]-(n:Entity)
            RETURN c.id AS chunk_id,
                   collect(DISTINCT e.normalized_name) AS mentions,
                   collect(DISTINCT n.normalized_name) AS neighbors
        """
        enrichment_rows = self._execute(
            enrich_query,
            {"chunk_ids": [row["chunk_id"] for row in candidate_rows]},
        )

        enrichment = {
            row["chunk_id"]: {
                "mentions": [m for m in (row.get("mentions") or []) if m],
                "neighbors": [n for n in (row.get("neighbors") or []) if n],
            }
            for row in enrichment_rows
        }

        query_tokens = {token for token in re.split(r"\W+", query.lower()) if token}
        scored_rows: list[dict[str, Any]] = []
        for row in candidate_rows:
            cosine = float(row.get("score") or 0.0)
            if cosine < settings.graph_rag_min_cosine_score:
                continue

            chunk_id = row["chunk_id"]
            enrich = enrichment.get(chunk_id, {"mentions": [], "neighbors": []})
            mention_overlap = len(query_tokens & set(enrich["mentions"]))
            entity_bonus = min(
                _MENTION_OVERLAP_BONUS * mention_overlap
                + _MENTION_COUNT_BONUS * min(len(enrich["mentions"]), _MAX_BONUS_MENTIONS),
                _MENTION_BONUS_CAP,
            )
            neighbor_bonus = min(
                _NEIGHBOR_COUNT_BONUS
                * min(len(enrich["neighbors"]), _MAX_BONUS_NEIGHBORS),
                _NEIGHBOR_BONUS_CAP,
            )
            fused_score = cosine + entity_bonus + neighbor_bonus

            scored_rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": row.get("document_id", ""),
                    "resource_id": row.get("resource_id", ""),
                    "text": row.get("text", ""),
                    "source_url": row.get("source_url", ""),
                    "source_title": row.get("source_title", ""),
                    "chunk_index": row.get("chunk_index", 0),
                    "score": fused_score,
                    "mentions": enrich["mentions"],
                    "neighbors": enrich["neighbors"],
                }
            )

        if not scored_rows and resource_ids:
            for row in candidate_rows:
                scored_rows.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row.get("document_id", ""),
                        "resource_id": row.get("resource_id", ""),
                        "text": row.get("text", ""),
                        "source_url": row.get("source_url", ""),
                        "source_title": row.get("source_title", ""),
                        "chunk_index": row.get("chunk_index", 0),
                        "score": float(row.get("score") or 0.0),
                        "mentions": [],
                        "neighbors": [],
                    }
                )

        scored_rows.sort(key=lambda item: item["score"], reverse=True)
        top_rows = scored_rows[:effective_top_k]

        if not top_rows:
            return GraphQueryResult(context="", chunks=[], entities=[])

        entities: list[str] = []
        seen_entities = set()
        for row in top_rows:
            for value in row.get("mentions", []) + row.get("neighbors", []):
                if value in seen_entities:
                    continue
                seen_entities.add(value)
                entities.append(value)

        context = "\n\n".join(
            f"[source:{row['source_title']} chunk:{row['chunk_id']}]\n{row['text']}"
            for row in top_rows
            if row.get("text")
        )
        return GraphQueryResult(context=context, chunks=top_rows, entities=entities)

    def delete_resource_documents(
        self,
        *,
        resource_id: str,
        owner_id: str,
        workspace_id: str,
    ) -> bool:
        self._ensure_schema()
        self._execute(
            """
            MATCH (d:Document)
            WHERE d.resource_id = $resource_id
              AND d.owner_id = $owner_id
              AND d.workspace_id = $workspace_id
            DETACH DELETE d
            """,
            {
                "resource_id": resource_id,
                "owner_id": owner_id,
                "workspace_id": workspace_id,
            },
        )

        self._execute(
            """
            MATCH (e:Entity)
            WHERE NOT ()-[:MENTIONS]->(e)
            DETACH DELETE e
            """
        )
        return True
