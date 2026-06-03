"""Conflict resolution for ambiguous or conflicting relationships."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Sequence

from src.knowledge_graph.models import ConflictRecord, TypedRelationship

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Resolves conflicts when adding typed relationships.

    Rules
    -----
    * Same source-target-type edge already exists → keep max confidence (no conflict).
    * New confidence >= 0.9 and existing < 0.5 → upgrade to new.
    * Existing confidence >= 0.9 and new < 0.5 → keep existing.
    * Both confidence >= 0.7 but types differ → flag manual review.
    * Otherwise → accept the new relationship (no conflict).
    """

    def resolve(
        self,
        new_rel: TypedRelationship,
        existing: Sequence[TypedRelationship],
    ) -> ConflictRecord | None:
        """Compare *new_rel* against a sequence of existing relationships.

        Returns a ``ConflictRecord`` when action is taken (rejection, upgrade, or
        manual-review flag), or ``None`` when the new relationship can proceed
        without raising a conflict.
        """
        # Check for exact same source-target-type edges.
        same_edges = [
            e
            for e in existing
            if e.source_id == new_rel.source_id
            and e.target_id == new_rel.target_id
            and e.relationship_type == new_rel.relationship_type
        ]
        if same_edges:
            existing_confidence = max(e.confidence for e in same_edges)

            # Same edge — keep max confidence.
            if new_rel.confidence <= existing_confidence:
                return ConflictRecord(
                    object_a_id=new_rel.source_id,
                    object_b_id=new_rel.target_id,
                    relationship_type=new_rel.relationship_type,
                    reason=(
                        f"Existing relationship (confidence={existing_confidence:.2f}) "
                        f"has higher or equal confidence than new (confidence={new_rel.confidence:.2f}). "
                        "Keeping existing."
                    ),
                    confidence_a=existing_confidence,
                    confidence_b=new_rel.confidence,
                    resolution="keep_a",
                    resolved_at=datetime.now(UTC).isoformat(),
                )
            # New has higher confidence — upgrade.
            return ConflictRecord(
                object_a_id=new_rel.source_id,
                object_b_id=new_rel.target_id,
                relationship_type=new_rel.relationship_type,
                reason=(
                    f"Upgrading relationship confidence from existing {existing_confidence:.2f} "
                    f"to new {new_rel.confidence:.2f}."
                ),
                confidence_a=existing_confidence,
                confidence_b=new_rel.confidence,
                resolution="keep_b",
                resolved_at=datetime.now(UTC).isoformat(),
            )

        # Check for same source-target pair with different relationship types.
        same_pair = [
            e
            for e in existing
            if e.source_id == new_rel.source_id and e.target_id == new_rel.target_id
        ]

        for existing_rel in same_pair:
            if existing_rel.relationship_type != new_rel.relationship_type:
                # High-confidence existing relation with low-confidence new → keep existing.
                if existing_rel.confidence >= 0.9 and new_rel.confidence < 0.5:
                    return ConflictRecord(
                        object_a_id=new_rel.source_id,
                        object_b_id=new_rel.target_id,
                        relationship_type=new_rel.relationship_type,
                        reason=(
                            f"Existing {existing_rel.relationship_type.value} (confidence={existing_rel.confidence:.2f}) "
                            f"is well established; new {new_rel.relationship_type.value} "
                            f"(confidence={new_rel.confidence:.2f}) is too low to override."
                        ),
                        confidence_a=existing_rel.confidence,
                        confidence_b=new_rel.confidence,
                        resolution="keep_a",
                        resolved_at=datetime.now(UTC).isoformat(),
                    )

                # High-confidence new relation with low-confidence existing → upgrade.
                if new_rel.confidence >= 0.9 and existing_rel.confidence < 0.5:
                    return ConflictRecord(
                        object_a_id=new_rel.source_id,
                        object_b_id=new_rel.target_id,
                        relationship_type=new_rel.relationship_type,
                        reason=(
                            f"New {new_rel.relationship_type.value} (confidence={new_rel.confidence:.2f}) "
                            f"is strongly supported; existing {existing_rel.relationship_type.value} "
                            f"(confidence={existing_rel.confidence:.2f}) is weak."
                        ),
                        confidence_a=new_rel.confidence,
                        confidence_b=existing_rel.confidence,
                        resolution="keep_b",
                        resolved_at=datetime.now(UTC).isoformat(),
                    )

                # Both moderately confident but types differ → needs human review.
                if existing_rel.confidence >= 0.7 and new_rel.confidence >= 0.7:
                    return ConflictRecord(
                        object_a_id=new_rel.source_id,
                        object_b_id=new_rel.target_id,
                        relationship_type=new_rel.relationship_type,
                        reason=(
                            f"Both existing ({existing_rel.relationship_type.value}, "
                            f"confidence={existing_rel.confidence:.2f}) and new "
                            f"({new_rel.relationship_type.value}, confidence={new_rel.confidence:.2f}) "
                            f"have moderate or high confidence but disagree on relationship type."
                        ),
                        confidence_a=existing_rel.confidence,
                        confidence_b=new_rel.confidence,
                        resolution="manual_review",
                        resolved_at=datetime.now(UTC).isoformat(),
                    )

        # No conflict detected.
        return None
