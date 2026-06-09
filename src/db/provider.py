from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.supabase_store import SupabaseSessionStore
    from src.storage import SupabaseStorageAdapter

_session_store: "SupabaseSessionStore | None" = None
_storage_adapter: "SupabaseStorageAdapter | None" = None


def get_session_store() -> "SupabaseSessionStore":
    global _session_store
    if _session_store is None:
        from src.db.supabase_store import SupabaseSessionStore
        _session_store = SupabaseSessionStore()
    return _session_store


def set_session_store(store: "SupabaseSessionStore | None") -> None:
    global _session_store
    _session_store = store


def get_storage_adapter() -> "SupabaseStorageAdapter":
    global _storage_adapter
    if _storage_adapter is None:
        from src.storage import SupabaseStorageAdapter
        _storage_adapter = SupabaseStorageAdapter()
    return _storage_adapter


def set_storage_adapter(adapter: "SupabaseStorageAdapter | None") -> None:
    global _storage_adapter
    _storage_adapter = adapter
