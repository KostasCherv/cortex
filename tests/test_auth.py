import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.auth import get_authenticated_user, _verify_with_supabase_userinfo


def test_get_authenticated_user_rejects_missing_token():
    try:
        asyncio.run(get_authenticated_user(None))
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected HTTPException for missing token")


def test_get_authenticated_user_accepts_valid_jwt_payload():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.c2ln",
    )
    mock_key = MagicMock()
    mock_key.key = "public-key"

    with (
        patch("src.auth._jwks_url", return_value="https://example.supabase.co/auth/v1/.well-known/jwks.json"),
        patch("src.auth.jwt.PyJWKClient") as mock_jwk_client_cls,
        patch("src.auth.jwt.decode") as mock_decode,
    ):
        mock_jwk_client = MagicMock()
        mock_jwk_client.get_signing_key_from_jwt.return_value = mock_key
        mock_jwk_client_cls.return_value = mock_jwk_client
        mock_decode.return_value = {"sub": "user-1", "email": "user@example.com"}

        user = asyncio.run(get_authenticated_user(credentials))

    assert user.user_id == "user-1"
    assert user.email == "user@example.com"


def test_get_authenticated_user_rejects_invalid_jwt():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.bad-signature",
    )
    with (
        patch("src.auth._jwks_url", return_value="https://example.supabase.co/auth/v1/.well-known/jwks.json"),
        patch("src.auth.jwt.PyJWKClient") as mock_jwk_client_cls,
        patch("src.auth._verify_with_supabase_userinfo", side_effect=HTTPException(status_code=401, detail="Invalid or expired token.")),
    ):
        mock_jwk_client = MagicMock()
        mock_jwk_client.get_signing_key_from_jwt.side_effect = jwt.InvalidTokenError("invalid")
        mock_jwk_client_cls.return_value = mock_jwk_client

        try:
            asyncio.run(get_authenticated_user(credentials))
        except HTTPException as exc:
            assert exc.status_code == 401
        else:
            raise AssertionError("Expected HTTPException for invalid token")


def test_get_authenticated_user_handles_jwks_client_error_with_fallback():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyLTEifQ.bad-kid",
    )
    with (
        patch("src.auth._jwks_url", return_value="https://example.supabase.co/auth/v1/.well-known/jwks.json"),
        patch("src.auth.jwt.PyJWKClient") as mock_jwk_client_cls,
        patch("src.auth._verify_with_supabase_userinfo", return_value=MagicMock(user_id="user-1", email="u@example.com")) as mock_fallback,
    ):
        mock_jwk_client = MagicMock()
        mock_jwk_client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError("jwks unavailable")
        mock_jwk_client_cls.return_value = mock_jwk_client

        user = asyncio.run(get_authenticated_user(credentials))

    assert user.user_id == "user-1"
    mock_fallback.assert_called_once_with(credentials.credentials)


def test_verify_with_supabase_userinfo_returns_cached_user_without_http_call():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "auth:key"
    mock_cache.get.return_value = {"user_id": "cached-user", "email": "cached@example.com"}

    with (
        patch("src.auth.settings") as mock_settings,
        patch("src.auth.get_cache", return_value=mock_cache),
        patch("src.auth.httpx.AsyncClient") as mock_http_client,
    ):
        mock_settings.supabase_url = "https://example.supabase.co"
        user = asyncio.run(_verify_with_supabase_userinfo("token-1"))

    assert user.user_id == "cached-user"
    mock_http_client.assert_not_called()


def test_verify_with_supabase_userinfo_miss_calls_supabase_and_writes_cache():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "auth:key"
    mock_cache.get.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "user-1", "email": "user@example.com"}
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = False

    with (
        patch("src.auth.settings") as mock_settings,
        patch("src.auth.get_cache", return_value=mock_cache),
        patch("src.auth.httpx.AsyncClient", return_value=mock_ctx),
    ):
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_secret_key = "service-key"
        mock_settings.redis_cache_ttl_auth_seconds = 300
        user = asyncio.run(_verify_with_supabase_userinfo("token-1"))

    assert user.user_id == "user-1"
    mock_cache.set.assert_awaited_once()


def test_verify_with_supabase_userinfo_without_cache_calls_supabase_directly():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "user-1", "email": "user@example.com"}
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = False

    with (
        patch("src.auth.settings") as mock_settings,
        patch("src.auth.get_cache", return_value=None),
        patch("src.auth.httpx.AsyncClient", return_value=mock_ctx),
    ):
        mock_settings.supabase_url = "https://example.supabase.co"
        mock_settings.supabase_secret_key = "service-key"
        user = asyncio.run(_verify_with_supabase_userinfo("token-1"))

    assert user.user_id == "user-1"
