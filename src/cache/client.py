"""Optional Redis cache with JSON serialisation. Disabled when REDIS_URL is unset."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

_cache: "RedisCache | None" = None


def get_cache() -> "RedisCache | None":
    """Return (or lazily create) singleton RedisCache, or None when disabled."""
    global _cache
    if _cache is None and settings.redis_url:
        _cache = RedisCache(url=settings.redis_url)
    return _cache


class RedisCache:
    """Thin async wrapper over redis.asyncio with JSON serialisation."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("[cache] get failed key=%r: %s", key, exc)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            await self._client.setex(key, ttl_seconds, json.dumps(value))
        except Exception as exc:
            logger.warning("[cache] set failed key=%r: %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except Exception as exc:
            logger.warning("[cache] delete failed key=%r: %s", key, exc)

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception as exc:
            logger.warning("[cache] ping failed: %s", exc)
            return False

    @staticmethod
    def hash_key(prefix: str, value: str) -> str:
        """Produce a stable, collision-resistant cache key."""
        digest = hashlib.sha256(value.encode()).hexdigest()
        return f"{prefix}:{digest}"
