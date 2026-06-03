"""Programmatic query DSL for the knowledge graph."""

from __future__ import annotations

from typing import Any

from src.knowledge_graph.models import KnowledgeGraphQuery, RelationshipType


class KnowledgeQueryBuilder:
    """Builder-style DSL for constructing ``KnowledgeGraphQuery`` instances.

    Usage::

        query = (
            KnowledgeQueryBuilder
            .select()
            .fulltext("machine learning")
            .relate(RelationshipType.REFERENCES)
            .owned_by("user-1")
            .in_workspace("ws-1")
            .limit(20)
            .build()
        )
    """

    def __init__(self) -> None:
        self._query: str | None = None
        self._filters: dict[str, Any] = {}
        self._relationship_filter: RelationshipType | None = None
        self._max_hops: int = 2
        self._top_k: int = 10
        self._owner_id: str | None = None
        self._workspace_id: str | None = None

    @staticmethod
    def select() -> KnowledgeQueryBuilder:
        """Start a new query chain."""
        return KnowledgeQueryBuilder()

    def where(self, field: str, op: str, value: Any) -> KnowledgeQueryBuilder:
        """Add a metadata filter.

        Parameters
        ----------
        field : str
            Metadata field name.
        op : str
            Comparison operator (e.g. ``eq``, ``gt``, ``lt``, ``in``).
        value : Any
            Value to compare against.
        """
        self._filters[f"{field}__{op}"] = value
        return self

    def relate(
        self, rel_type: RelationshipType, direction: str = "out"
    ) -> KnowledgeQueryBuilder:
        """Narrow results by relationship type.

        Parameters
        ----------
        rel_type : RelationshipType
            The typed relationship to follow.
        direction : str
            Relationship direction — ``"out"`` or ``"in"``.
        """
        self._relationship_filter = rel_type
        return self

    def limit(self, n: int) -> KnowledgeQueryBuilder:
        """Maximum results to return."""
        self._top_k = n
        return self

    def offset(self, n: int) -> KnowledgeQueryBuilder:
        """Result offset for pagination."""
        return self

    def fulltext(self, q: str) -> KnowledgeQueryBuilder:
        """Search by full-text query."""
        self._query = q
        return self

    def owned_by(self, owner_id: str) -> KnowledgeQueryBuilder:
        """Scope results to a specific owner."""
        self._owner_id = owner_id
        return self

    def in_workspace(self, workspace_id: str) -> KnowledgeQueryBuilder:
        """Scope results to a specific workspace."""
        self._workspace_id = workspace_id
        return self

    def build(self) -> KnowledgeGraphQuery:
        """Finalize and return the query object."""
        return KnowledgeGraphQuery(
            query=self._query,
            filters=self._filters,
            relationship_filter=self._relationship_filter,
            max_hops=self._max_hops,
            top_k=self._top_k,
            owner_id=self._owner_id,
            workspace_id=self._workspace_id,
        )
