"""Pydantic models for the knowledge graph engine."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Typed relationship kinds in the knowledge graph.

    Each value is used directly as a Neo4j relationship type.
    """

    REFERENCES = "REFERENCES"
    DERIVES_FROM = "DERIVES_FROM"
    CONTRADICTS = "CONTRADICTS"
    SUMMARIZES = "SUMMARIZES"


class TypedRelationship(BaseModel):
    """A directed, typed edge between two KnowledgeObjects."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class KnowledgeObject(BaseModel):
    """A node in the knowledge graph carrying content and relationships."""

    id: str
    title: str
    object_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    relationships: list[TypedRelationship] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    owner_id: str
    workspace_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    embedding: list[float] | None = None


class KnowledgeGraphQuery(BaseModel):
    """Input parameters for querying the knowledge graph."""

    query: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    relationship_filter: RelationshipType | None = None
    max_hops: int = Field(default=2, ge=1, le=5)
    top_k: int = Field(default=10, ge=1, le=100)
    owner_id: str | None = None
    workspace_id: str | None = None


class KnowledgeGraphResult(BaseModel):
    """Result set from a knowledge graph query."""

    objects: list[KnowledgeObject]
    total_count: int
    query_details: dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    """Records a detected conflict between knowledge relationships."""

    object_a_id: str
    object_b_id: str
    relationship_type: RelationshipType = RelationshipType.CONTRADICTS
    reason: str
    confidence_a: float
    confidence_b: float
    resolution: str | None = None
    resolved_at: str | None = None
