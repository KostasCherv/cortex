"""Knowledge graph engine — high-level abstraction for knowledge objects
with typed relationships, conflict resolution, and programmatic query DSL."""

from __future__ import annotations

from src.knowledge_graph.conflict import ConflictResolver
from src.knowledge_graph.dsl import KnowledgeQueryBuilder
from src.knowledge_graph.engine import KnowledgeGraphEngine
from src.knowledge_graph.models import (
    ConflictRecord,
    KnowledgeGraphQuery,
    KnowledgeGraphResult,
    KnowledgeObject,
    RelationshipType,
    TypedRelationship,
)

__all__ = [
    "ConflictRecord",
    "ConflictResolver",
    "KnowledgeGraphEngine",
    "KnowledgeGraphQuery",
    "KnowledgeGraphResult",
    "KnowledgeObject",
    "KnowledgeQueryBuilder",
    "RelationshipType",
    "TypedRelationship",
]
