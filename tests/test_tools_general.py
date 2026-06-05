"""Tests for general-purpose Tavily LangChain tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.general import (
    GENERAL_WEB_TOOL_NAMES,
    _wikipedia_lookup,
    build_agent_tools,
    build_general_tools,
    build_reference_tools,
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


def test_build_reference_tools_includes_wikipedia_when_enabled():
    tools = build_reference_tools()
    assert len(tools) == 1
    assert tools[0].name == "wikipedia"


def test_build_reference_tools_respects_disable_flag():
    assert build_reference_tools(allow_wikipedia=False) == []


def test_build_agent_tools_combines_web_and_reference_tools():
    with patch("src.tools.general.settings") as mock_settings:
        mock_settings.tavily_api_key = "tvly-test"
        with patch("src.tools.general.build_general_tools") as mock_web, patch(
            "src.tools.general.build_reference_tools"
        ) as mock_ref:
            mock_web.return_value = [MagicMock(name="tavily_search")]
            mock_ref.return_value = [MagicMock(name="wikipedia")]
            tools = build_agent_tools(allow_web=True)
    assert len(tools) == 2
    mock_web.assert_called_once_with(allow_web=True)
    mock_ref.assert_called_once_with(allow_wikipedia=True)


@pytest.mark.asyncio
async def test_wikipedia_lookup_handles_empty_response_body():
    response = MagicMock()
    response.text = ""
    response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.general.httpx.AsyncClient", return_value=mock_client):
        result = await _wikipedia_lookup("Zurich")

    assert "empty response" in result.lower()


@pytest.mark.asyncio
async def test_wikipedia_lookup_handles_invalid_json():
    response = MagicMock()
    response.text = "<html>blocked</html>"
    response.raise_for_status = MagicMock()
    response.json.side_effect = ValueError("bad json")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.general.httpx.AsyncClient", return_value=mock_client):
        result = await _wikipedia_lookup("Zurich")

    assert "unexpected response" in result.lower()


@pytest.mark.asyncio
async def test_wikipedia_lookup_formats_page_summaries():
    response = MagicMock()
    response.text = '{"query":{"pages":{"1":{"title":"Zurich","extract":"City in Switzerland."}}}}'
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "query": {"pages": {"1": {"title": "Zurich", "extract": "City in Switzerland."}}}
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("src.tools.general.httpx.AsyncClient", return_value=mock_client):
        result = await _wikipedia_lookup("Zurich")

    assert "Page: Zurich" in result
    assert "City in Switzerland." in result


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
