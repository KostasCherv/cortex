"""Long-lived arXiv MCP server integration via arxiv-mcp-server."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src.config import settings

logger = logging.getLogger(__name__)

_ARXIV_MCP_INSTALL_HINT = (
    "arxiv-mcp-server is required in the backend runtime. Install backend dependencies "
    "(for example `uv sync` locally or rebuild the deployment image) so the "
    "`arxiv-mcp-server` console script is available, or set ARXIV_MCP_COMMAND."
)

# Core workflow from https://github.com/blazickjp/arxiv-mcp-server
ARXIV_MCP_TOOL_NAMES = frozenset(
    {
        "search_papers",
        "download_paper",
        "read_paper",
        "list_papers",
        "get_abstract",
    }
)

_ARXIV_MCP_ERROR_HINTS = {
    "read_paper": "Paper not downloaded yet. Call download_paper first, then read_paper.",
    "download_paper": "Download failed. Check the paper ID or wait if arXiv rate-limited this IP.",
    "search_papers": "Search failed. arXiv may be rate-limiting this IP — wait 60 seconds and retry.",
    "get_abstract": "Could not fetch abstract. Check the paper ID or wait if rate limited.",
    "list_papers": "Could not list downloaded papers.",
}

_client_singleton: MultiServerMCPClient | None = None


def format_arxiv_mcp_error(tool_name: str, raw: str) -> str:
    """Turn MCP/arxiv-mcp-server failures into LLM-readable tool messages."""
    text = (raw or "").strip()
    if text.startswith("Error:"):
        return text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if text:
            return f"{tool_name} failed: {text}"
        return _arxiv_mcp_error_hint(tool_name)

    if not isinstance(payload, dict):
        return text or _arxiv_mcp_error_hint(tool_name)

    if payload.get("status") != "error":
        return text or _arxiv_mcp_error_hint(tool_name)

    message = (payload.get("message") or "").strip()
    if message:
        if "rate limit" in message.lower():
            return f"{message} Wait 60 seconds before retrying."
        return message

    return _arxiv_mcp_error_hint(tool_name)


def _arxiv_mcp_error_hint(tool_name: str) -> str:
    return _ARXIV_MCP_ERROR_HINTS.get(tool_name, f"{tool_name} failed.")


def _normalize_mcp_tool_result(result: Any) -> str:
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return "No content returned."
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict):
            if payload.get("status") == "error":
                return format_arxiv_mcp_error("arxiv", text)
            if "content" in payload and payload.get("status") == "success":
                return str(payload["content"])
        return text

    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, str):
                parts.append(_normalize_mcp_tool_result(item))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(_normalize_mcp_tool_result(str(item.get("text", ""))))
        return "\n".join(part for part in parts if part) or str(result)

    return str(result)


def _preserve_response_format(tool: BaseTool, content: str, artifact: Any = None) -> Any:
    if getattr(tool, "response_format", None) == "content_and_artifact":
        return (content, artifact)
    return content


def wrap_arxiv_mcp_tool(tool: BaseTool) -> BaseTool:
    """Catch ToolException from langchain-mcp-adapters and return plain text instead."""
    original_coroutine = getattr(tool, "coroutine", None)
    if original_coroutine is None:
        return tool

    async def safe_coroutine(*args: Any, **kwargs: Any) -> Any:
        try:
            result = await original_coroutine(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                return _preserve_response_format(
                    tool,
                    _normalize_mcp_tool_result(content),
                    artifact,
                )
            return _preserve_response_format(tool, _normalize_mcp_tool_result(result))
        except ToolException as exc:
            message = format_arxiv_mcp_error(tool.name, str(exc))
            logger.warning("[arxiv-mcp] %s: %s", tool.name, message)
            return _preserve_response_format(tool, message)
        except Exception as exc:
            message = format_arxiv_mcp_error(tool.name, str(exc))
            logger.warning("[arxiv-mcp] %s unexpected error: %s", tool.name, message)
            return _preserve_response_format(tool, message)

    setattr(tool, "coroutine", safe_coroutine)
    return tool


def _resolve_arxiv_mcp_command() -> str:
    if settings.arxiv_mcp_command.strip():
        return settings.arxiv_mcp_command.strip()

    bundled_script = Path(sys.executable).with_name("arxiv-mcp-server")
    if bundled_script.exists():
        return str(bundled_script)

    discovered = shutil.which("arxiv-mcp-server")
    if discovered:
        return discovered

    raise RuntimeError(_ARXIV_MCP_INSTALL_HINT)


def _resolve_storage_path() -> Path:
    return Path(settings.arxiv_mcp_storage_path).expanduser()


def get_arxiv_mcp_client() -> MultiServerMCPClient:
    global _client_singleton
    storage_path = _resolve_storage_path()
    storage_path.mkdir(parents=True, exist_ok=True)
    command = _resolve_arxiv_mcp_command()
    connection: Any = {
        "arxiv": {
            "transport": "stdio",
            "command": command,
            "args": ["--storage-path", str(storage_path)],
        }
    }
    cache_key = (command, str(storage_path))
    if _client_singleton is not None and getattr(_client_singleton, "_cache_key", None) == cache_key:
        return _client_singleton

    client = MultiServerMCPClient(connection)
    client._cache_key = cache_key  # type: ignore[attr-defined]
    _client_singleton = client
    return _client_singleton


def filter_arxiv_mcp_tools(tools: list[BaseTool]) -> list[BaseTool]:
    return [tool for tool in tools if tool.name in ARXIV_MCP_TOOL_NAMES]


@asynccontextmanager
async def arxiv_mcp_tools_context(*, enabled: bool = True) -> AsyncGenerator[list[BaseTool], None]:
    """Yield LangChain tools backed by a single arxiv-mcp-server session."""
    if not enabled:
        yield []
        return

    client = get_arxiv_mcp_client()
    async with client.session("arxiv") as session:
        tools = filter_arxiv_mcp_tools(await load_mcp_tools(session))
        wrapped = [wrap_arxiv_mcp_tool(tool) for tool in tools]
        logger.debug("[arxiv-mcp] Loaded %d tools: %s", len(wrapped), [t.name for t in wrapped])
        yield wrapped


async def ensure_arxiv_mcp_available() -> None:
    """Raise when the required arxiv-mcp-server runtime is unavailable."""
    try:
        async with arxiv_mcp_tools_context(enabled=True) as tools:
            if tools:
                return
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"arxiv-mcp-server startup validation failed: {exc}") from exc

    raise RuntimeError(
        "arxiv-mcp-server started but did not expose any supported tools. "
        f"{_ARXIV_MCP_INSTALL_HINT}"
    )


async def verify_arxiv_mcp_available() -> bool:
    """Best-effort startup probe; returns False when the MCP server cannot start."""
    try:
        await ensure_arxiv_mcp_available()
        return True
    except Exception as exc:
        logger.warning("[arxiv-mcp] Startup probe failed: %s", exc)
        return False
