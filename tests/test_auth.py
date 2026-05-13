import asyncio
from unittest.mock import MagicMock, patch

import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.auth import get_authenticated_user


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
