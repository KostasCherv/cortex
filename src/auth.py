"""Authentication helpers for Supabase JWT validation."""

from dataclasses import dataclass

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.cache.client import get_cache
from src.config import settings
from src.supabase_keys import supabase_api_headers

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthenticatedUser:
    user_id: str
    email: str | None = None


def _jwks_url() -> str:
    if settings.supabase_jwks_url:
        return settings.supabase_jwks_url
    if settings.supabase_url:
        return f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    return ""


async def _verify_with_supabase_userinfo(token: str) -> AuthenticatedUser:
    """Validate token against Supabase Auth API and return user claims."""
    if not settings.supabase_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase URL is not configured.",
        )

    cache = get_cache()
    cache_key = ""
    if cache is not None:
        cache_key = cache.hash_key("auth:userinfo", token.strip())
        cached = await cache.get(cache_key)
        if isinstance(cached, dict) and cached.get("user_id"):
            return AuthenticatedUser(
                user_id=str(cached["user_id"]),
                email=cached.get("email"),
            )

    server_key = settings.supabase_secret_key or ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
                headers=supabase_api_headers(server_key, user_access_token=token),
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    data = response.json()
    user_id = data.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject.",
        )
    if cache is not None and cache_key:
        await cache.set(
            cache_key,
            {"user_id": user_id, "email": data.get("email")},
            settings.redis_cache_ttl_auth_seconds,
        )
    return AuthenticatedUser(user_id=user_id, email=data.get("email"))


async def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    """Validate bearer token and return authenticated user claims."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    jwks_url = _jwks_url()
    if not jwks_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase JWT verification is not configured.",
        )

    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc

    algorithm = str(header.get("alg", "")).upper()
    payload: dict
    if algorithm.startswith("HS"):
        if settings.supabase_jwt_secret:
            try:
                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience=settings.supabase_jwt_audience,
                    options={"require": ["exp", "sub"]},
                )
            except jwt.InvalidTokenError:
                return await _verify_with_supabase_userinfo(token)
        else:
            return await _verify_with_supabase_userinfo(token)
    else:
        try:
            jwk_client = jwt.PyJWKClient(jwks_url)
            signing_key = jwk_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=settings.supabase_jwt_audience,
                options={"require": ["exp", "sub"]},
            )
        except (jwt.InvalidTokenError, jwt.PyJWKClientError):
            return await _verify_with_supabase_userinfo(token)
        except Exception:
            return await _verify_with_supabase_userinfo(token)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject.",
        )

    return AuthenticatedUser(user_id=user_id, email=payload.get("email"))
