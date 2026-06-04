"""Tests for RAG chat prepare helpers and tool-binding heuristics."""

from unittest.mock import MagicMock

import pytest

from src.api.rag_chat_helpers import (
    should_bind_composio_tools,
    trim_chat_history,
)


def test_trim_chat_history_keeps_tail():
    history = [MagicMock() for _ in range(30)]
    trimmed = trim_chat_history(history)
    assert len(trimmed) == 20


def test_should_bind_composio_tools_with_linked_docs_still_binds():
    bind, reason = should_bind_composio_tools(
        message="Hello, summarize my resume.",
        resource_ids=["res-1"],
        composio_apps=["gmail"],
    )
    assert bind is True
    assert reason == "default_bind"


def test_should_bind_composio_tools_greeting_with_linked_docs_still_binds():
    bind, reason = should_bind_composio_tools(
        message="Hello",
        resource_ids=["res-1"],
        composio_apps=["gmail"],
    )
    assert bind is True
    assert reason == "default_bind"


def test_should_bind_composio_tools_external_intent():
    bind, reason = should_bind_composio_tools(
        message="What is the latest AAPL stock price?",
        resource_ids=["res-1"],
        composio_apps=["gmail"],
    )
    assert bind is True
    assert reason == "external_intent"


def test_should_bind_composio_tools_composio_access_question_with_resources():
    bind, reason = should_bind_composio_tools(
        message="do you have access to composio?",
        resource_ids=["res-1"],
        composio_apps=["gmail"],
    )
    assert bind is True
    assert reason == "composio_meta_question"


def test_should_bind_composio_tools_disabled():
    from src.config import settings

    original = settings.composio_enabled
    settings.composio_enabled = False
    try:
        bind, reason = should_bind_composio_tools(
            message="Check my email",
            resource_ids=[],
            composio_apps=["gmail"],
        )
    finally:
        settings.composio_enabled = original
    assert bind is False
    assert reason == "composio_disabled"


@pytest.mark.asyncio
async def test_run_agent_loop_skips_router_when_bind_tools_false():
    from langchain_core.messages import AIMessage, HumanMessage

    from src.api.endpoints import _run_agent_loop

    llm_response = AIMessage(content="Done.")
    mock_llm = MagicMock()
    from unittest.mock import AsyncMock

    mock_llm.ainvoke = AsyncMock(return_value=llm_response)

    from unittest.mock import patch

    with patch("src.api.endpoints.get_llm", return_value=mock_llm):
        answer = await _run_agent_loop(
            messages=[HumanMessage(content="Hi")],
            metadata={},
            bind_tools=False,
        )

    assert answer == "Done."
    mock_llm.bind_tools.assert_not_called()
