"""Transactional outbox for reliable Inngest event dispatch."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.db.provider import get_session_store

_MAX_ATTEMPTS = 5


def _backoff_seconds(attempts: int) -> int:
    return min(30 * (2 ** attempts), 3600)


@dataclass
class OutboxEvent:
    id: str
    event_name: str
    payload: dict[str, Any]
    status: str = "pending"
    attempts: int = 0
    last_error: str | None = None
    next_attempt_at: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    sent_at: str | None = None


async def enqueue_event(event_name: str, payload: dict[str, Any]) -> OutboxEvent:
    """Write an event to the outbox. Does not dispatch to Inngest immediately."""
    now = datetime.now(UTC).isoformat()
    event = OutboxEvent(
        id=str(uuid.uuid4()),
        event_name=event_name,
        payload=payload,
        next_attempt_at=now,
        created_at=now,
    )
    await get_session_store().insert_outbox_event(
        {
            "id": event.id,
            "event_name": event.event_name,
            "payload": event.payload,
            "next_attempt_at": event.next_attempt_at,
            "created_at": event.created_at,
        }
    )
    return event


async def dispatch_outbox_events(limit: int = 50) -> int:
    """Send pending outbox events to Inngest. Returns the number successfully dispatched."""
    import inngest
    from src.inngest_client import inngest_client
    import logging

    logger = logging.getLogger(__name__)

    store = get_session_store()

    try:
        # Recover rows stuck in dispatching state from a previous crashed dispatcher
        await store.reset_stuck_dispatching_events()
        rows = await store.fetch_pending_outbox_events(limit=limit)
    except Exception as exc:
        logger.warning("[outbox] pre-dispatch fetch failed: %s", exc)
        return 0
    dispatched = 0

    for row in rows:
        event_id: str = row["id"]
        event_name: str = row["event_name"]
        payload: dict[str, Any] = row["payload"]
        attempts: int = row.get("attempts", 0)

        # Claim atomically — skip if another concurrent dispatcher already grabbed this row
        if not await store.claim_outbox_event(event_id):
            continue

        try:
            await inngest_client.send(inngest.Event(name=event_name, data=payload))
            await store.update_outbox_event(
                event_id,
                {
                    "status": "sent",
                    "sent_at": datetime.now(UTC).isoformat(),
                },
            )
            dispatched += 1
        except Exception as exc:
            new_attempts = attempts + 1
            if new_attempts >= _MAX_ATTEMPTS:
                await store.update_outbox_event(
                    event_id,
                    {
                        "status": "failed",
                        "attempts": new_attempts,
                        "last_error": str(exc),
                    },
                )
            else:
                next_at = (
                    datetime.now(UTC) + timedelta(seconds=_backoff_seconds(new_attempts))
                ).isoformat()
                await store.update_outbox_event(
                    event_id,
                    {
                        "status": "pending",
                        "attempts": new_attempts,
                        "last_error": str(exc),
                        "next_attempt_at": next_at,
                    },
                )

    return dispatched
