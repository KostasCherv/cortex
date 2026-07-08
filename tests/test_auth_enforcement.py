"""Integration tests verifying all protected endpoints enforce authentication.

Every parametrized endpoint must return 401 when called with no token or an
invalid JWT.  Public endpoints (GET /health, POST /api/billing/webhook,
POST /internal/dispatch-outbox) are intentionally excluded because they use
their own auth mechanisms or are truly public.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure inngest is in dev-mode before importing the app
os.environ.setdefault("INNGEST_DEV", "1")

from src.api.endpoints import app  # noqa: E402

INVALID_JWT = "Bearer thisisinvalidjwt.notavalidtoken.atall"

# Placeholder UUIDs for path-param endpoints
_UUID = "00000000-0000-0000-0000-000000000001"
_UUID2 = "00000000-0000-0000-0000-000000000002"

# ---------------------------------------------------------------------------
# Protected endpoints discovered by reading router files.
# Excluded:
#   GET  /health                        — public liveness probe
#   POST /api/billing/webhook           — Stripe signature auth, no JWT
#   POST /internal/dispatch-outbox      — internal bearer-secret auth, no JWT
# ---------------------------------------------------------------------------
PROTECTED_ENDPOINTS: list[tuple[str, str]] = [
    # --- Billing ---
    ("GET", "/api/billing/usage"),
    ("POST", "/api/billing/checkout-session"),
    ("POST", "/api/billing/portal-session"),
    # --- Sessions ---
    ("POST", "/sessions"),
    ("GET", "/sessions"),
    ("GET", f"/sessions/{_UUID}"),
    ("PATCH", f"/sessions/{_UUID}"),
    ("DELETE", f"/sessions/{_UUID}"),
    ("POST", f"/sessions/{_UUID}/research"),
    ("GET", f"/sessions/{_UUID}/runs/{_UUID2}/stream"),
    ("POST", f"/sessions/{_UUID}/runs/{_UUID2}/feedback"),
    ("POST", f"/sessions/{_UUID}/followup"),
    # --- RAG resources ---
    ("GET", "/api/rag/resources"),
    ("DELETE", f"/api/rag/resources/{_UUID}"),
    ("GET", f"/api/rag/resources/{_UUID}/status"),
    # --- RAG agents ---
    ("POST", "/api/rag/agents"),
    ("POST", "/api/rag/agents/draft"),
    ("GET", "/api/rag/agents"),
    ("PATCH", f"/api/rag/agents/{_UUID}"),
    ("DELETE", f"/api/rag/agents/{_UUID}"),
    ("POST", f"/api/rag/agents/{_UUID}/resources:link"),
    ("POST", f"/api/rag/agents/{_UUID}/chat"),
    ("POST", f"/api/rag/agents/{_UUID}/chat/stream"),
    ("GET", f"/api/rag/agents/{_UUID}/chat/sessions"),
    ("GET", f"/api/rag/agents/{_UUID}/chat/sessions/{_UUID2}/messages"),
    ("PATCH", f"/api/rag/agents/{_UUID}/chat/sessions/{_UUID2}"),
    ("DELETE", f"/api/rag/agents/{_UUID}/chat/sessions/{_UUID2}"),
    ("DELETE", f"/api/rag/agents/{_UUID}/chat/sessions/{_UUID2}/last-exchange"),
    # --- RAG workspace chat ---
    ("POST", "/api/rag/chat"),
    ("POST", "/api/rag/chat/stream"),
    ("GET", "/api/rag/chat/sessions"),
    ("GET", f"/api/rag/chat/sessions/{_UUID}/messages"),
    ("PATCH", f"/api/rag/chat/sessions/{_UUID}"),
    ("DELETE", f"/api/rag/chat/sessions/{_UUID}"),
    ("DELETE", f"/api/rag/chat/sessions/{_UUID}/last-exchange"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
async def test_returns_401_without_auth(method: str, path: str) -> None:
    """Calling a protected endpoint with no Authorization header must return 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await getattr(client, method.lower())(path)
    assert response.status_code == 401, (
        f"{method} {path} should return 401 without auth, got {response.status_code}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
async def test_returns_401_with_invalid_jwt(method: str, path: str) -> None:
    """Calling a protected endpoint with a malformed JWT must return 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await getattr(client, method.lower())(
            path, headers={"Authorization": INVALID_JWT}
        )
    assert response.status_code == 401, (
        f"{method} {path} with invalid JWT should return 401, got {response.status_code}"
    )
