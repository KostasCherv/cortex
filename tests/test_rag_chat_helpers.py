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
        answer, web_used = await _run_agent_loop(
            messages=[HumanMessage(content="Hi")],
            metadata={},
            bind_tools=False,
            allow_web_search=False,
        )

    assert answer == "Done."
    assert web_used is False
    mock_llm.bind_tools.assert_not_called()


def test_rag_chat_prepared_has_allow_web_search():
    from src.api.rag_chat_helpers import RagChatPrepared
    from langchain_core.messages import HumanMessage
    from src.rag_engine import RagQueryResult
    prepared = RagChatPrepared(
        agent=None,
        resource_ids=[],
        rag_context=MagicMock(spec=RagQueryResult),
        chat_session_id="sess-1",
        messages=[HumanMessage(content="hi")],
        bind_tools=False,
        tool_skip_reason=None,
        composio_apps=[],
        allow_web_search=True,
    )
    assert prepared.allow_web_search is True


@pytest.mark.asyncio
async def test_prepare_workspace_respects_composio_false():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.api.endpoints import RagChatTools
    from unittest.mock import AsyncMock, patch

    tools = RagChatTools(web_search=True, composio=False)

    with patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new_callable=AsyncMock, return_value=[]), \
         patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new_callable=AsyncMock, return_value="sess-1"), \
         patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr, \
         patch("src.api.rag_chat_helpers.retrieve_context_for_query", new_callable=AsyncMock, return_value=MagicMock(context="", chunks=[])), \
         patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new_callable=AsyncMock, return_value=""), \
         patch("src.api.rag_chat_helpers.list_rag_chat_messages", new_callable=AsyncMock, return_value=[]):
        mock_mgr.return_value.get_connected_app_names.return_value = ["slack"]
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="latest news",
            session_id=None,
            timings=RagChatTimings(),
            tools=tools,
        )
        assert result.bind_tools is False
        assert result.allow_web_search is True


@pytest.mark.asyncio
async def test_prepare_workspace_respects_composio_true():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.api.endpoints import RagChatTools
    from unittest.mock import AsyncMock, patch

    tools = RagChatTools(web_search=True, composio=True)

    with patch("src.api.rag_chat_helpers.list_workspace_ready_resource_ids", new_callable=AsyncMock, return_value=[]), \
         patch("src.api.rag_chat_helpers.create_or_get_workspace_chat_session", new_callable=AsyncMock, return_value="sess-1"), \
         patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr, \
         patch("src.api.rag_chat_helpers.settings") as mock_settings, \
         patch("src.api.rag_chat_helpers.retrieve_context_for_query", new_callable=AsyncMock, return_value=MagicMock(context="", chunks=[])), \
         patch("src.api.rag_chat_helpers.get_user_memory_prompt_block", new_callable=AsyncMock, return_value=""), \
         patch("src.api.rag_chat_helpers.list_rag_chat_messages", new_callable=AsyncMock, return_value=[]):
        mock_mgr.return_value.get_connected_app_names.return_value = ["slack"]
        mock_settings.composio_enabled = True
        mock_settings.rag_chat_max_history_messages = 10
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="send slack message",
            session_id=None,
            timings=RagChatTimings(),
            tools=tools,
        )
        assert result.bind_tools is True
        assert result.allow_web_search is True
        assert result.tool_skip_reason is None
