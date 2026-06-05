"""Tests for _run_agent_loop and _build_agent_messages in endpoints.py."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _make_tool(name: str, result: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.arun = AsyncMock(return_value=result)
    return tool


def _ai_message_with_tool_call(tool_name: str, tool_id: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "id": tool_id, "args": args, "type": "tool_call"}],
    )


def _patch_router_tools(mock_mgr: MagicMock, tools: list) -> None:
    @asynccontextmanager
    async def _router_context(_user_id: str):
        yield tools

    mock_mgr.return_value.router_tools_context = _router_context


@pytest.mark.asyncio
async def test_run_agent_loop_no_tool_calls_returns_answer():
    from src.api.endpoints import _run_agent_loop

    llm_response = AIMessage(content="The answer is 42.")
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(return_value=llm_response)

    with patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr:
        _patch_router_tools(mock_mgr, [])
        with patch("src.api.endpoints.get_llm", return_value=mock_llm):
            answer, web_used = await _run_agent_loop(
                messages=[HumanMessage(content="What is 6 times 7?")],
                metadata={},
            )

    assert answer == "The answer is 42."
    assert web_used is False


@pytest.mark.asyncio
async def test_run_agent_loop_executes_tool_and_returns_final_answer():
    from src.api.endpoints import _run_agent_loop

    tool_call_response = _ai_message_with_tool_call("TAVILY_SEARCH", "call_1", {"query": "AAPL"})
    final_response = AIMessage(content="AAPL is trading at $200.")

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

    fake_tool = _make_tool("TAVILY_SEARCH", '{"price": 200}')

    with patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr:
        _patch_router_tools(mock_mgr, [fake_tool])
        with patch("src.api.endpoints.get_llm", return_value=mock_llm):
            answer, web_used = await _run_agent_loop(
                messages=[HumanMessage(content="What is AAPL price?")],
                metadata={},
            )

    assert answer == "AAPL is trading at $200."
    assert web_used is False
    fake_tool.arun.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_loop_router_search_then_execute():
    from src.api.endpoints import _run_agent_loop

    search_response = _ai_message_with_tool_call(
        "COMPOSIO_SEARCH_TOOLS",
        "call_search",
        {"use_case": "search the web for latest AAPL price"},
    )
    execute_response = _ai_message_with_tool_call(
        "COMPOSIO_MULTI_EXECUTE_TOOL",
        "call_exec",
        {"tools": [{"tool_slug": "TAVILY_SEARCH", "arguments": {"query": "AAPL price"}}]},
    )
    final_response = AIMessage(content="AAPL is trading at $200.")

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(
        side_effect=[search_response, execute_response, final_response]
    )

    search_tool = _make_tool("COMPOSIO_SEARCH_TOOLS", '{"tools": ["TAVILY_SEARCH"]}')
    execute_tool = _make_tool("COMPOSIO_MULTI_EXECUTE_TOOL", '{"price": 200}')

    with patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr:
        _patch_router_tools(mock_mgr, [search_tool, execute_tool])
        with patch("src.api.endpoints.get_llm", return_value=mock_llm):
            answer, web_used = await _run_agent_loop(
                messages=[HumanMessage(content="What is AAPL price?")],
                metadata={},
            )

    assert answer == "AAPL is trading at $200."
    assert web_used is False
    search_tool.arun.assert_awaited_once()
    execute_tool.arun.assert_awaited_once()
    assert mock_llm.ainvoke.call_count == 3


@pytest.mark.asyncio
async def test_run_agent_loop_emits_tool_events_via_on_event():
    from src.api.endpoints import _run_agent_loop

    events: list[dict] = []

    async def capture_event(event: dict) -> None:
        events.append(event)

    tool_call_response = _ai_message_with_tool_call("GITHUB_CREATE_ISSUE", "call_2", {"title": "Bug"})
    final_response = AIMessage(content="Issue created.")

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

    fake_tool = _make_tool("GITHUB_CREATE_ISSUE", '{"id": 42}')

    with patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr:
        _patch_router_tools(mock_mgr, [fake_tool])
        with patch("src.api.endpoints.get_llm", return_value=mock_llm):
            await _run_agent_loop(
                messages=[HumanMessage(content="Create an issue.")],
                metadata={},
                on_event=capture_event,
            )

    tool_start_events = [e for e in events if e["type"] == "tool_start"]
    tool_end_events = [e for e in events if e["type"] == "tool_end"]
    assert len(tool_start_events) == 1
    assert tool_start_events[0]["tool"] == "GITHUB_CREATE_ISSUE"
    assert len(tool_end_events) == 1
    assert tool_end_events[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_run_agent_loop_tool_error_emits_error_event_and_continues():
    from src.api.endpoints import _run_agent_loop

    events: list[dict] = []

    async def capture_event(event: dict) -> None:
        events.append(event)

    tool_call_response = _ai_message_with_tool_call("GITHUB_CREATE_ISSUE", "call_3", {"title": "Bug"})
    final_response = AIMessage(content="I could not create the issue due to an error.")

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])

    broken_tool = MagicMock()
    broken_tool.name = "GITHUB_CREATE_ISSUE"
    broken_tool.arun = AsyncMock(side_effect=Exception("Unauthorized"))

    with patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr:
        _patch_router_tools(mock_mgr, [broken_tool])
        with patch("src.api.endpoints.get_llm", return_value=mock_llm):
            answer, web_used = await _run_agent_loop(
                messages=[HumanMessage(content="Create an issue.")],
                metadata={},
                on_event=capture_event,
            )

    error_events = [e for e in events if e["type"] == "tool_end" and e["status"] == "error"]
    assert len(error_events) == 1
    assert "could not create" in answer
    assert web_used is False


@pytest.mark.asyncio
async def test_run_agent_loop_respects_max_turns():
    from src.api.endpoints import _run_agent_loop

    tool_call_response = _ai_message_with_tool_call("TAVILY_SEARCH", "call_n", {"query": "x"})

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(return_value=tool_call_response)

    fake_tool = _make_tool("TAVILY_SEARCH", "result")

    with patch("src.api.endpoints.get_composio_toolset_manager") as mock_mgr:
        _patch_router_tools(mock_mgr, [fake_tool])
        with patch("src.api.endpoints.get_llm", return_value=mock_llm):
            with patch("src.api.endpoints.settings") as mock_settings:
                mock_settings.composio_max_agent_turns = 3
                mock_settings.composio_enabled = True
                await _run_agent_loop(
                    messages=[HumanMessage(content="Loop forever.")],
                    metadata={},
                )

    assert mock_llm.ainvoke.call_count == 3


def test_build_agent_messages_constructs_correct_structure():
    from src.api.endpoints import _build_agent_messages

    class FakeMsg:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    history = [FakeMsg("user", "Hello"), FakeMsg("assistant", "Hi there")]
    messages = _build_agent_messages(
        system_instructions="Be concise",
        history=history,
        rag_context="Some doc text",
        user_memory_context="Prefers concise answers and works in fintech.",
        composio_apps=["github", "tavily"],
        normalized_message="What is the price?",
    )

    assert isinstance(messages[0], SystemMessage)
    assert "github" in messages[0].content
    assert "COMPOSIO_SEARCH_TOOLS" in messages[0].content
    assert "COMPOSIO_MULTI_EXECUTE_TOOL" in messages[0].content
    assert "Some doc text" in messages[0].content
    assert "Prefers concise answers" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Hello"
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "What is the price?"
