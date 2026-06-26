# scripts/finetune/teacher_client.py
"""Pluggable teacher-model client for dataset generation.

Selects the backend via the ``TEACHER_API`` environment variable:

- ``ollama`` (default): Ollama-native API at ``{base}/api/chat``.
- ``openai``: OpenAI-compatible API at ``{base}/chat/completions``
  (e.g. LM Studio's local server, default base ``http://localhost:1234/v1``).

Base URL resolution (first match wins):
  1. ``TEACHER_BASE_URL`` (explicit, applies to either backend)
  2. ``OLLAMA_BASE_URL`` (only when ``TEACHER_API=ollama``, for back-compat)
  3. backend default

For the OpenAI backend, ``TEACHER_API_KEY`` is sent as a bearer token;
it defaults to a dummy value since LM Studio does not require auth.
"""

from __future__ import annotations

import os
import re

import httpx

_DEFAULT_BASE = {
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:1234/v1",
}

# Reasoning models (o1, o3, o4-mini, ...) reject a custom temperature; only the
# server default is allowed. Matches an optional "provider/" prefix then o<digit>.
_REASONING_RE = re.compile(r"^(?:[a-z0-9_-]+/)?o\d", re.IGNORECASE)


def _omit_temperature(model: str) -> bool:
    """Whether to drop the temperature field for this model.

    Override with TEACHER_OMIT_TEMPERATURE=true/false; otherwise auto-detect
    OpenAI o-series reasoning models by name.
    """
    override = os.getenv("TEACHER_OMIT_TEMPERATURE")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes"}
    return bool(_REASONING_RE.match(model.strip()))


def teacher_api() -> str:
    """Resolve the configured backend name (lowercased)."""
    return os.getenv("TEACHER_API", "ollama").strip().lower()


def teacher_base_url() -> str:
    """Resolve the teacher base URL for the active backend (no trailing slash)."""
    explicit = os.getenv("TEACHER_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    api = teacher_api()
    if api == "ollama":
        return os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE["ollama"]).rstrip("/")
    return _DEFAULT_BASE.get(api, _DEFAULT_BASE["ollama"]).rstrip("/")


def call_teacher(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    json_mode: bool = False,
    timeout: float = 120.0,
) -> str:
    """Send a chat request to the configured teacher and return raw text content.

    Raises ``httpx.HTTPError`` on transport/HTTP failures and ``KeyError`` if the
    response shape is unexpected; callers are expected to handle both.
    """
    base = teacher_base_url()
    api = teacher_api()

    if api == "openai":
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # Reasoning models (o4-mini, etc.) only accept the default temperature.
        if not _omit_temperature(model):
            payload["temperature"] = temperature
        # NOTE: we intentionally do NOT send response_format={"type":"json_object"}.
        # Many OpenAI-compatible servers (e.g. LM Studio) reject it with
        # "'response_format.type' must be 'json_schema' or 'text'". Callers rely on
        # the prompt instructions + extract_json_candidate() to parse JSON instead.
        api_key = os.getenv("TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY") or "lm-studio"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = httpx.post(
            f"{base}/chat/completions", json=payload, headers=headers, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # Default: Ollama-native API.
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    resp = httpx.post(f"{base}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]
