"""Session domain models and persistence helpers."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC

from src.db.supabase_store import SupabaseSessionStore


@dataclass
class SessionRun:
    """A single completed research run inside a session."""
    run_id: str
    query: str
    source_urls: list[str] = field(default_factory=list)
    report: str = ""
    status: str = "completed"
    error_details: str | None = None
    latest_node: str | None = None
    latest_event_at: str | None = None
    partial_report: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "source_urls": self.source_urls,
            "report": self.report,
            "status": self.status,
            "error_details": self.error_details,
            "latest_node": self.latest_node,
            "latest_event_at": self.latest_event_at,
            "partial_report": self.partial_report,
            "created_at": self.created_at,
        }


@dataclass
class ConversationTurn:
    """One turn in the session conversation (user question or assistant answer)."""
    role: str            # "user" | "assistant"
    content: str
    run_id: str | None = None
    citations: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "run_id": self.run_id,
            "citations": self.citations,
            "suggestions": self.suggestions,
            "created_at": self.created_at,
        }


@dataclass
class Session:
    """An ongoing research session with runs and conversation history."""
    session_id: str
    title: str = "New session"
    runs: list[SessionRun] = field(default_factory=list)
    conversation: list[ConversationTurn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def latest_run(self) -> SessionRun | None:
        return self.runs[-1] if self.runs else None

    def get_run(self, run_id: str) -> SessionRun | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "runs": [r.to_dict() for r in self.runs],
            "conversation": [t.to_dict() for t in self.conversation],
            "created_at": self.created_at,
        }


_store: SupabaseSessionStore | None = None


def _get_store() -> SupabaseSessionStore:
    global _store
    if _store is None:
        _store = SupabaseSessionStore()
    return _store


def ensure_store_initialized() -> None:
    """Fail fast at startup if Supabase persistence is misconfigured."""
    _get_store()


def suggest_session_title(query: str | None) -> str:
    """Generate a short human-friendly session title from a query."""
    if not query:
        return "New session"
    normalized = " ".join(query.strip().split())
    if not normalized:
        return "New session"
    words = normalized.split(" ")
    if len(words) <= 8:
        return normalized
    return " ".join(words[:8]) + "..."


async def create_session(user_id: str, title: str | None = None) -> Session:
    """Create a new user-owned session and persist it."""
    return await _get_store().create_session(user_id=user_id, title=title or "New session")


async def list_sessions(user_id: str) -> list[dict[str, str]]:
    """List user-owned sessions ordered by newest first."""
    return await _get_store().list_sessions(user_id=user_id)


async def get_session(session_id: str, user_id: str) -> Session | None:
    """Return a session by ID scoped to the owning user."""
    return await _get_store().get_session(session_id=session_id, user_id=user_id)


async def append_run(user_id: str, session_id: str, run: SessionRun) -> None:
    """Persist a completed run for a given session and user."""
    await _get_store().append_run(user_id=user_id, session_id=session_id, run=run)


async def create_session_run(user_id: str, session_id: str, run: SessionRun) -> None:
    """Persist a new run in running/pending state."""
    await _get_store().create_session_run(user_id=user_id, session_id=session_id, run=run)


async def update_session_run(
    *,
    run_id: str,
    user_id: str,
    session_id: str,
    patch: dict,
) -> bool:
    """Patch status or metadata for an existing run."""
    return await _get_store().update_session_run(
        run_id=run_id,
        user_id=user_id,
        session_id=session_id,
        patch=patch,
    )


async def get_session_run(*, run_id: str, user_id: str, session_id: str) -> SessionRun | None:
    """Return a single run by ID scoped to the owning user/session."""
    return await _get_store().get_session_run(
        run_id=run_id,
        user_id=user_id,
        session_id=session_id,
    )


async def append_turn(user_id: str, session_id: str, turn: ConversationTurn) -> None:
    """Persist a conversation turn for a given session and user."""
    await _get_store().append_turn(user_id=user_id, session_id=session_id, turn=turn)


async def update_session_title(user_id: str, session_id: str, title: str) -> bool:
    """Update a user-owned session title."""
    return await _get_store().update_session_title(
        user_id=user_id,
        session_id=session_id,
        title=title,
    )


async def delete_session(user_id: str, session_id: str) -> bool:
    """Delete a user-owned session."""
    return await _get_store().delete_session(
        user_id=user_id,
        session_id=session_id,
    )


def generate_run_id() -> str:
    return str(uuid.uuid4())
