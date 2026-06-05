"""Tests for general-purpose Tavily LangChain tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.general import (
    GENERAL_WEB_TOOL_NAMES,
    build_general_tools,
    should_mark_web_used,
)


def test_general_web_tool_names():
    assert GENERAL_WEB_TOOL_NAMES == frozenset({"tavily_search", "tavily_extract"})


def test_should_mark_web_used_rejects_tavily_error_payload():
    assert should_mark_web_used("tavily_extract", {"error": "bad key"}) is False
    assert should_mark_web_used("tavily_search", {"error": ValueError("boom")}) is False


def test_should_mark_web_used_accepts_extract_results():
    assert should_mark_web_used(
        "tavily_extract",
        {"results": [{"url": "https://example.com", "raw_content": "hi"}]},
    ) is True
    assert should_mark_web_used("tavily_extract", {"results": []}) is False


def test_should_mark_web_used_accepts_search_text():
    assert should_mark_web_used("tavily_search", "Title\ncontent") is True
    assert should_mark_web_used("tavily_search", "   ") is False


def test_should_mark_web_used_ignores_unrelated_tools():
    assert should_mark_web_used("github_create_issue", {"results": []}) is False


def test_build_general_tools_returns_empty_without_api_key():
    with patch("src.tools.general.settings") as mock_settings:
        mock_settings.tavily_api_key = ""
        assert build_general_tools(allow_web=True) == []


def test_build_general_tools_returns_empty_when_disabled():
    with patch("src.tools.general.settings") as mock_settings:
        mock_settings.tavily_api_key = "tvly-test"
        assert build_general_tools(allow_web=False) == []


def test_build_general_tools_binds_search_and_extract_together():
    with patch("src.tools.general.settings") as mock_settings:
        mock_settings.tavily_api_key = "tvly-test"
        with patch("src.tools.general._make_cached_tavily_search_tool") as mock_search, patch(
            "src.tools.general.TavilyExtract"
        ) as mock_extract:
            mock_search.return_value = MagicMock(name="tavily_search")
            mock_extract.return_value = MagicMock(name="tavily_extract")
            tools = build_general_tools(allow_web=True)
    assert len(tools) == 2
    mock_search.assert_called_once_with()
    mock_extract.assert_called_once_with(tavily_api_key="tvly-test")


@pytest.mark.asyncio
async def test_cached_tavily_search_uses_perform_search_cached():
    from src.tools.general import _cached_tavily_search

    with patch(
        "src.tools.general.perform_search_cached",
        new_callable=AsyncMock,
        return_value=[
            {"title": "Example", "url": "https://example.com", "content": "body"}
        ],
    ) as mock_search, patch("src.tools.general.settings") as mock_settings:
        mock_settings.max_search_results = 5
        text = await _cached_tavily_search("latest news")

    mock_search.assert_awaited_once_with("latest news", max_results=5)
    assert "Example" in text
    assert "https://example.com" in text
