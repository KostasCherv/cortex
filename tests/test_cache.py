"""Tests for src/cache/client.py"""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_returns_none_when_key_missing():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        result = await cache.get("missing-key")
    assert result is None


@pytest.mark.asyncio
async def test_get_returns_parsed_json_on_hit():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps({"user_id": "u1", "email": "a@b.com"})
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        result = await cache.get("auth:abc")
    assert result == {"user_id": "u1", "email": "a@b.com"}


@pytest.mark.asyncio
async def test_set_calls_setex_with_ttl():
    mock_redis = AsyncMock()
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        await cache.set("some-key", {"data": 1}, ttl_seconds=60)
    mock_redis.setex.assert_called_once_with("some-key", 60, json.dumps({"data": 1}))


@pytest.mark.asyncio
async def test_delete_calls_redis_delete():
    mock_redis = AsyncMock()
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        await cache.delete("doomed-key")
    mock_redis.delete.assert_called_once_with("doomed-key")


@pytest.mark.asyncio
async def test_get_degrades_gracefully_on_redis_error():
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = ConnectionError("Redis down")
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        result = await cache.get("any-key")
    assert result is None


@pytest.mark.asyncio
async def test_set_degrades_gracefully_on_redis_error():
    mock_redis = AsyncMock()
    mock_redis.setex.side_effect = ConnectionError("Redis down")
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        # Must not raise — caching is best-effort
        await cache.set("any-key", {"x": 1}, ttl_seconds=60)


@pytest.mark.asyncio
async def test_ping_returns_false_on_error():
    mock_redis = AsyncMock()
    mock_redis.ping.side_effect = ConnectionError("Redis down")
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        assert await cache.ping() is False


@pytest.mark.asyncio
async def test_ping_returns_true_on_success():
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        from src.cache.client import RedisCache

        cache = RedisCache(url="redis://localhost:6379/0")
        assert await cache.ping() is True


def test_hash_key_is_deterministic():
    from src.cache.client import RedisCache

    key1 = RedisCache.hash_key("auth", "token-abc")
    key2 = RedisCache.hash_key("auth", "token-abc")
    assert key1 == key2
    assert key1.startswith("auth:")


def test_hash_key_differs_for_different_inputs():
    from src.cache.client import RedisCache

    assert RedisCache.hash_key("auth", "a") != RedisCache.hash_key("auth", "b")
    assert RedisCache.hash_key("auth", "x") != RedisCache.hash_key("search", "x")


def test_get_cache_returns_none_when_redis_url_empty():
    with patch("src.cache.client.settings") as mock_settings:
        mock_settings.redis_url = ""
        import src.cache.client as cache_module

        cache_module._cache = None  # reset singleton
        result = cache_module.get_cache()
    assert result is None


def test_get_cache_constructs_when_redis_url_set():
    with (
        patch("src.cache.client.settings") as mock_settings,
        patch("redis.asyncio.from_url") as mock_from_url,
    ):
        mock_settings.redis_url = "redis://localhost:6379/0"
        import src.cache.client as cache_module

        cache_module._cache = None  # reset singleton
        try:
            result = cache_module.get_cache()
            assert result is not None
            mock_from_url.assert_called_once_with(
                "redis://localhost:6379/0", decode_responses=True
            )
        finally:
            cache_module._cache = None  # reset for other tests
