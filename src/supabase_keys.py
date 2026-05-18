"""Supabase API key helpers (publishable/secret and legacy JWT keys)."""

from __future__ import annotations


def supabase_api_headers(
    server_key: str,
    *,
    user_access_token: str | None = None,
    content_type: str | None = None,
) -> dict[str, str]:
    """Build headers for Supabase REST, Storage, and Auth HTTP APIs.

    New ``sb_secret_...`` keys must be sent only in ``apikey``. Legacy JWT
    ``service_role`` keys also require ``Authorization: Bearer <same value>``.
    When validating a user, pass their access token as ``user_access_token``.
    """
    if not server_key:
        raise ValueError("Supabase server key is required")

    headers: dict[str, str] = {"apikey": server_key}
    if user_access_token:
        headers["Authorization"] = f"Bearer {user_access_token}"
    elif server_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {server_key}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers
