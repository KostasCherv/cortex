"""Tests for src/tools/search.py"""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from src.errors import SearchError
from src.tools.search import perform_search_cached


def _make_tavily_response(n: int = 2) -> dict:
    return {
        "results": [
            {"url": f"https://example.com/{i}", "title": f"Title {i}", "content": f"Content {i}"}
            for i in range(n)
        ]
    }


def test_perform_search_returns_results():
    mock_client = MagicMock()
    mock_client.search.return_value = _make_tavily_response(3)

    with (
        patch("src.tools.search.settings") as mock_settings,
        # Patch at the tavily package level so the real network call is skipped
        patch("tavily.TavilyClient", return_value=mock_client),
    ):
        mock_settings.tavily_api_key = "tvly-test"
        mock_settings.max_search_results = 5

        from src.tools.search import perform_search

        results = perform_search("LangGraph tutorial")

    assert len(results) == 3
    assert results[0]["url"] == "https://example.com/0"
    assert "title" in results[0]
    assert "content" in results[0]


def test_perform_search_raises_without_api_key():
    with patch("src.tools.search.settings") as mock_settings:
        mock_settings.tavily_api_key = ""
        mock_settings.max_search_results = 5

        from src.tools.search import perform_search

        with pytest.raises(SearchError):
            perform_search("anything")


def test_perform_search_retries_on_failure():
    mock_client = MagicMock()
    mock_client.search.side_effect = [
        RuntimeError("network error"),
        RuntimeError("network error"),
        _make_tavily_response(1),
    ]

    with (
        patch("src.tools.search.settings") as mock_settings,
        patch("tavily.TavilyClient", return_value=mock_client),
        patch("src.tools.search.time.sleep"),  # suppress actual sleep
    ):
        mock_settings.tavily_api_key = "tvly-test"
        mock_settings.max_search_results = 5

        from src.tools.search import perform_search

        results = perform_search("LangGraph")

    assert len(results) == 1
    assert mock_client.search.call_count == 3


def test_perform_search_raises_after_all_retries():
    mock_client = MagicMock()
    mock_client.search.side_effect = RuntimeError("always fails")

    with (
        patch("src.tools.search.settings") as mock_settings,
        patch("tavily.TavilyClient", return_value=mock_client),
        patch("src.tools.search.time.sleep"),
    ):
        mock_settings.tavily_api_key = "tvly-test"
        mock_settings.max_search_results = 5

        from src.tools.search import perform_search

        with pytest.raises(SearchError):
            perform_search("fail query")


@pytest.mark.asyncio
async def test_perform_search_cached_returns_cached_result_without_uncached_call():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "search:key"
    mock_cache.get.return_value = [{"url": "https://cached.com", "title": "Cached", "content": "Old"}]

    with (
        patch("src.tools.search.get_cache", return_value=mock_cache),
        patch("src.tools.search._perform_search_uncached") as mock_uncached,
    ):
        results = await perform_search_cached("LangGraph tutorial")

    assert results[0]["url"] == "https://cached.com"
    mock_uncached.assert_not_called()


@pytest.mark.asyncio
async def test_perform_search_cached_miss_calls_uncached_and_stores():
    mock_cache = AsyncMock()
    mock_cache.hash_key.return_value = "search:key"
    mock_cache.get.return_value = None
    uncached = [{"url": "https://fresh.com", "title": "Fresh", "content": "New"}]

    with (
        patch("src.tools.search.get_cache", return_value=mock_cache),
        patch("src.tools.search.settings") as mock_settings,
        patch("src.tools.search.asyncio.to_thread", new=AsyncMock(return_value=uncached)),
    ):
        mock_settings.max_search_results = 5
        mock_settings.redis_cache_ttl_search_seconds = 1800
        results = await perform_search_cached("LangGraph tutorial")

    assert results == uncached
    mock_cache.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_perform_search_cached_without_cache_uses_uncached():
    uncached = [{"url": "https://fresh.com", "title": "Fresh", "content": "New"}]
    with (
        patch("src.tools.search.get_cache", return_value=None),
        patch("src.tools.search.asyncio.to_thread", new=AsyncMock(return_value=uncached)),
    ):
        results = await perform_search_cached("LangGraph tutorial")

    assert results == uncached


@pytest.mark.asyncio
async def test_perform_search_cached_normalizes_query_for_hash_key():
    mock_cache = AsyncMock()
    mock_cache.get.return_value = None

    with (
        patch("src.tools.search.get_cache", return_value=mock_cache),
        patch("src.tools.search.settings") as mock_settings,
        patch("src.tools.search.asyncio.to_thread", new=AsyncMock(return_value=[])),
    ):
        mock_settings.max_search_results = 5
        mock_settings.redis_cache_ttl_search_seconds = 1800
        await perform_search_cached("  LangGraph  ")
        await perform_search_cached("langgraph")

    key_left = mock_cache.hash_key.call_args_list[0].args[1]
    key_right = mock_cache.hash_key.call_args_list[1].args[1]
    assert key_left == key_right
