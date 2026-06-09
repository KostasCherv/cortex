"""Cross-session user memory service."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import UTC, datetime

import httpx

from src import outbox
from src.db.provider import get_session_store

logger = logging.getLogger(__name__)


def _workspace_id_for_user(user_id: str) -> str:
    return user_id


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_line(value: str) -> str:
    return _normalize_text(value).rstrip(".").lower()


def _extract_memory_candidates(user_message: str, assistant_message: str) -> list[str]:
    del assistant_message
    user_text = _normalize_text(user_message)
    lowered = user_text.lower()
    candidates: list[str] = []
    patterns = (
        r"\bi prefer ([^.!?]+)",
        r"\bi like ([^.!?]+)",
        r"\bi love ([^.!?]+)",
        r"\bi usually ([^.!?]+)",
        r"\bi work in ([^.!?]+)",
        r"\bi'm ([^.!?]+)",
        r"\bi am ([^.!?]+)",
        r"\bmy role is ([^.!?]+)",
        r"\bi need ([^.!?]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            fragment = match.group(0).strip(" .")
            if len(fragment) >= 10:
                candidates.append(fragment[0].upper() + fragment[1:])
    if "prefer" in lowered and "concise" in lowered:
        candidates.append("Prefers concise answers.")
    return _dedupe_lines(candidates)


def _dedupe_lines(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in values:
        cleaned = _normalize_text(raw)
        if not cleaned:
            continue
        key = _normalize_line(cleaned)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned if cleaned.endswith(".") else f"{cleaned}.")
    return deduped


def _split_memory_lines(content: str) -> list[str]:
    if not content.strip():
        return []
    raw_lines = [line for line in content.splitlines() if line.strip()]
    if raw_lines:
        return _dedupe_lines(raw_lines)
    return _dedupe_lines([content])


def _merge_memory_content(existing_content: str, candidates: list[str]) -> str:
    merged = _split_memory_lines(existing_content)
    seen = {_normalize_line(line) for line in merged}
    for candidate in candidates:
        key = _normalize_line(candidate)
        if key in seen:
            continue
        merged.append(candidate if candidate.endswith(".") else f"{candidate}.")
        seen.add(key)
    return "\n".join(merged)


def _event_key(
    *,
    user_id: str,
    source_mode: str,
    source_session_id: str,
    source_user_message_id: str | None,
    source_assistant_message_id: str | None,
    user_message: str,
    assistant_message: str,
) -> str:
    raw = "|".join(
        [
            user_id,
            source_mode,
            source_session_id,
            source_user_message_id or "",
            source_assistant_message_id or "",
            _normalize_text(user_message),
            _normalize_text(assistant_message),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()  # nosec B324


def _is_non_retryable_store_error(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError):
        return "supabase persistence is not configured" in str(exc).lower()
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    return 400 <= exc.response.status_code < 500


async def get_user_memory(user_id: str) -> dict:
    try:
        row = await get_session_store().get_user_memory(
            owner_id=user_id,
            workspace_id=_workspace_id_for_user(user_id),
        )
    except Exception as exc:
        logger.warning("[user_memory] failed to load memory for user_id=%s: %s", user_id, exc)
        row = None
    if row is None:
        return {"content": "", "updated_at": None, "last_refreshed_at": None}
    return {
        "content": str(row.get("content") or ""),
        "updated_at": row.get("updated_at"),
        "last_refreshed_at": row.get("last_refreshed_at"),
    }


async def update_user_memory(user_id: str, content: str) -> dict:
    cleaned = content.strip()
    now = datetime.now(UTC).isoformat()
    payload = {
        "owner_id": user_id,
        "workspace_id": _workspace_id_for_user(user_id),
        "content": cleaned,
        "updated_at": now,
    }
    current = await get_session_store().get_user_memory(
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )
    if current and current.get("last_refreshed_at") is not None:
        payload["last_refreshed_at"] = current["last_refreshed_at"]
    await get_session_store().upsert_user_memory(payload=payload)
    return {
        "content": cleaned,
        "updated_at": now,
        "last_refreshed_at": current.get("last_refreshed_at") if current else None,
    }


async def delete_user_memory(user_id: str) -> dict:
    deleted = await get_session_store().delete_user_memory(
        owner_id=user_id,
        workspace_id=_workspace_id_for_user(user_id),
    )
    return {"deleted": deleted}


async def get_user_memory_prompt_block(user_id: str, prompt: str) -> str:
    del prompt
    memory = await get_user_memory(user_id)
    content = str(memory.get("content") or "").strip()
    if not content:
        return ""
    return f"User memory for personalization:\n{content}"


async def enqueue_memory_refresh(
    *,
    user_id: str,
    source_mode: str,
    source_session_id: str,
    user_message: str,
    assistant_message: str,
    source_user_message_id: str | None = None,
    source_assistant_message_id: str | None = None,
) -> bool:
    if not _normalize_text(user_message) or not _normalize_text(assistant_message):
        return False
    payload = {
        "user_id": user_id,
        "workspace_id": _workspace_id_for_user(user_id),
        "source_mode": source_mode,
        "source_session_id": source_session_id,
        "source_user_message_id": source_user_message_id,
        "source_assistant_message_id": source_assistant_message_id,
        "user_message": _normalize_text(user_message),
        "assistant_message": _normalize_text(assistant_message),
        "event_key": _event_key(
            user_id=user_id,
            source_mode=source_mode,
            source_session_id=source_session_id,
            source_user_message_id=source_user_message_id,
            source_assistant_message_id=source_assistant_message_id,
            user_message=user_message,
            assistant_message=assistant_message,
        ),
    }
    await outbox.enqueue_event("memory/refresh.requested", payload)
    return True


async def refresh_user_memory(
    *,
    user_id: str,
    source_mode: str,
    source_session_id: str,
    user_message: str,
    assistant_message: str,
    event_key: str,
    workspace_id: str | None = None,
    source_user_message_id: str | None = None,
    source_assistant_message_id: str | None = None,
) -> str:
    try:
        store = get_session_store()
        workspace_id = workspace_id or _workspace_id_for_user(user_id)
        now = datetime.now(UTC).isoformat()
        claimed = await store.claim_user_memory_refresh_event(
            {
                "id": str(uuid.uuid4()),
                "owner_id": user_id,
                "workspace_id": workspace_id,
                "event_key": event_key,
                "source_mode": source_mode,
                "source_session_id": source_session_id,
                "source_user_message_id": source_user_message_id,
                "source_assistant_message_id": source_assistant_message_id,
                "processed_at": now,
                "created_at": now,
            }
        )
        if not claimed:
            return "skipped"

        candidates = _extract_memory_candidates(user_message, assistant_message)
        if not candidates:
            return "noop"

        current = await store.get_user_memory(owner_id=user_id, workspace_id=workspace_id)
        current_content = str(current.get("content") or "") if current else ""
        merged_content = _merge_memory_content(current_content, candidates)
        if merged_content == current_content and current is not None:
            await store.upsert_user_memory(
                payload={
                    "owner_id": user_id,
                    "workspace_id": workspace_id,
                    "content": current_content,
                    "updated_at": current.get("updated_at") or now,
                    "last_refreshed_at": now,
                }
            )
            return "noop"

        await store.upsert_user_memory(
            payload={
                "owner_id": user_id,
                "workspace_id": workspace_id,
                "content": merged_content,
                "updated_at": now,
                "last_refreshed_at": now,
            }
        )
        return "updated"
    except Exception as exc:
        if not _is_non_retryable_store_error(exc):
            raise
        logger.warning(
            "[user_memory] refresh skipped after non-retryable store error "
            "user_id=%s workspace_id=%s source_mode=%s source_session_id=%s event_key=%s error=%s",
            user_id,
            workspace_id or _workspace_id_for_user(user_id),
            source_mode,
            source_session_id,
            event_key,
            exc,
        )
        return "error"
