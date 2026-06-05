"""Tests for arxiv-mcp-server integration."""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException

from src.tools.arxiv_mcp import (
    ARXIV_MCP_TOOL_NAMES,
    _resolve_arxiv_mcp_command,
    arxiv_mcp_tools_context,
    filter_arxiv_mcp_tools,
    format_arxiv_mcp_error,
    wrap_arxiv_mcp_tool,
)


def _mock_tool(name: str) -> MagicMock:
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = name

    async def _coro(**kwargs: Any) -> str:
        return "ok"

    tool.coroutine = _coro
    return tool


def test_filter_arxiv_mcp_tools_keeps_core_workflow_only():
    tools = [
        _mock_tool("search_papers"),
        _mock_tool("semantic_search"),
        _mock_tool("download_paper"),
    ]
    filtered = filter_arxiv_mcp_tools(tools)
    assert [tool.name for tool in filtered] == ["search_papers", "download_paper"]


@pytest.mark.asyncio
async def test_arxiv_mcp_tools_context_yields_empty_when_disabled():
    async with arxiv_mcp_tools_context(enabled=False) as tools:
        assert tools == []


@pytest.mark.asyncio
async def test_arxiv_mcp_tools_context_loads_filtered_tools():
    mock_tools = [
        _mock_tool("search_papers"),
        _mock_tool("watch_topic"),
        _mock_tool("read_paper"),
    ]

    @asynccontextmanager
    async def fake_session(_name):
        yield MagicMock()

    mock_client = MagicMock()
    mock_client.session = fake_session

    with patch("src.tools.arxiv_mcp.get_arxiv_mcp_client", return_value=mock_client), patch(
        "src.tools.arxiv_mcp.load_mcp_tools", new_callable=AsyncMock, return_value=mock_tools
    ):
        async with arxiv_mcp_tools_context(enabled=True) as tools:
            names = [tool.name for tool in tools]

    assert names == ["search_papers", "read_paper"]
    assert "watch_topic" not in names
    assert set(names).issubset(ARXIV_MCP_TOOL_NAMES)


def test_format_arxiv_mcp_error_handles_empty_json_message():
    raw = '{"status": "error", "message": ""}'
    result = format_arxiv_mcp_error("read_paper", raw)
    assert "download_paper" in result


def test_format_arxiv_mcp_error_extracts_message():
    raw = '{"status": "error", "message": "Paper 1234.5678 not found"}'
    assert format_arxiv_mcp_error("read_paper", raw) == "Paper 1234.5678 not found"


@pytest.mark.asyncio
async def test_wrap_arxiv_mcp_tool_converts_tool_exception_to_text():
    mock_tool = _mock_tool("search_papers")

    async def failing_coroutine(**kwargs: Any) -> str:
        raise ToolException('{"status": "error", "message": ""}')

    mock_tool.coroutine = failing_coroutine

    wrapped = wrap_arxiv_mcp_tool(mock_tool)
    result = await wrapped.coroutine(query="transformer", max_results=2)

    assert "rate-limit" in result.lower() or "search failed" in result.lower()


@pytest.mark.asyncio
async def test_wrap_arxiv_mcp_tool_preserves_content_and_artifact_response_format():
    mock_tool = _mock_tool("search_papers")
    mock_tool.response_format = "content_and_artifact"

    async def successful_coroutine(**kwargs: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
        return ([{"type": "text", "text": "abstract text"}], {"structured_content": {"paper_id": "1234.5678"}})

    mock_tool.coroutine = successful_coroutine

    wrapped = wrap_arxiv_mcp_tool(mock_tool)
    result = await wrapped.coroutine(query="transformer", max_results=2)

    assert result == (
        "abstract text",
        {"structured_content": {"paper_id": "1234.5678"}},
    )


@pytest.mark.asyncio
async def test_wrap_arxiv_mcp_tool_returns_tuple_on_handled_error_when_content_and_artifact():
    mock_tool = _mock_tool("search_papers")
    mock_tool.response_format = "content_and_artifact"

    async def failing_coroutine(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        raise ToolException('{"status": "error", "message": ""}')

    mock_tool.coroutine = failing_coroutine

    wrapped = wrap_arxiv_mcp_tool(mock_tool)
    result = await wrapped.coroutine(query="transformer", max_results=2)

    assert isinstance(result, tuple)
    assert "rate-limit" in result[0].lower() or "search failed" in result[0].lower()
    assert result[1] is None


def test_resolve_arxiv_mcp_command_prefers_backend_environment_script(tmp_path: pytest.TempPathFactory):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_path = bin_dir / "python"
    python_path.write_text("", encoding="utf-8")
    sibling_script = bin_dir / "arxiv-mcp-server"
    sibling_script.write_text("", encoding="utf-8")

    with patch("src.tools.arxiv_mcp.settings.arxiv_mcp_command", ""), patch(
        "src.tools.arxiv_mcp.sys.executable", str(python_path)
    ), patch("src.tools.arxiv_mcp.shutil.which", return_value="/usr/local/bin/arxiv-mcp-server"):
        assert _resolve_arxiv_mcp_command() == str(sibling_script)


def test_resolve_arxiv_mcp_command_raises_when_executable_missing(tmp_path: pytest.TempPathFactory):
    python_path = tmp_path / "bin" / "python"
    python_path.parent.mkdir()
    python_path.write_text("", encoding="utf-8")

    with patch("src.tools.arxiv_mcp.settings.arxiv_mcp_command", ""), patch(
        "src.tools.arxiv_mcp.sys.executable", str(python_path)
    ), patch("src.tools.arxiv_mcp.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="arxiv-mcp-server"):
            _resolve_arxiv_mcp_command()
