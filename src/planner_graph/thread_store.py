"""In-memory thread store for planner chat sessions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

MAX_THREADS: int = 1000
THREAD_TTL_SECONDS: float = 7200.0  # 2 hours


@dataclass
class ThreadEntry:
    thread_id: str
    user_id: str
    messages: list[dict]
    created_at: float
    last_active: float


class PlannerThreadStore:
    def __init__(self, max_threads: int = MAX_THREADS, ttl: float = THREAD_TTL_SECONDS) -> None:
        self._max_threads = max_threads
        self._ttl = ttl
        self._threads: dict[str, ThreadEntry] = {}

    def create_thread(self, user_id: str) -> str:
        self._enforce_capacity()
        thread_id = str(uuid.uuid4())
        now = time.monotonic()
        self._threads[thread_id] = ThreadEntry(
            thread_id=thread_id,
            user_id=user_id,
            messages=[],
            created_at=now,
            last_active=now,
        )
        return thread_id

    def get_thread(self, thread_id: str, user_id: str) -> ThreadEntry | None:
        entry = self._threads.get(thread_id)
        if entry is None:
            return None
        if entry.user_id != user_id:
            return None
        if self._is_expired(entry):
            del self._threads[thread_id]
            return None
        return entry

    def append_message(self, thread_id: str, message: dict) -> None:
        entry = self._threads.get(thread_id)
        if entry is None:
            return
        entry.messages.append(message)
        entry.last_active = time.monotonic()

    def delete_last_exchange(self, thread_id: str) -> None:
        entry = self._threads.get(thread_id)
        if entry is None or len(entry.messages) < 2:
            return
        entry.messages = entry.messages[:-2]
        entry.last_active = time.monotonic()

    def evict_expired(self) -> int:
        expired = [tid for tid, e in self._threads.items() if self._is_expired(e)]
        for tid in expired:
            del self._threads[tid]
        return len(expired)

    def _is_expired(self, entry: ThreadEntry) -> bool:
        return (time.monotonic() - entry.last_active) > self._ttl

    def _enforce_capacity(self) -> None:
        if len(self._threads) < self._max_threads:
            return
        # Evict the LRU entry (least recently active)
        lru_id = min(self._threads, key=lambda tid: self._threads[tid].last_active)
        del self._threads[lru_id]


planner_thread_store = PlannerThreadStore()
