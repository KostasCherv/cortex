"""Tests for RAG chat prepare helpers and tool-binding heuristics."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.rag_chat_helpers import (
    classify_chat_action,
    should_bind_composio_tools,
    trim_chat_history,
)
from src.llm.output_parsers import ChatActionDecisionPayload


@pytest.fixture(autouse=True)
def default_router_decision():
    """Keep preparation tests deterministic now that routing is mandatory."""
    decision = ChatActionDecisionPayload(
        action="answer_direct",
        reason="general knowledge",
    )
    with patch(
        "src.api.rag_chat_helpers.classify_chat_action",
        new=AsyncMock(return_value=decision),
    ):
        yield decision


def test_build_agent_messages_hides_composio_apps_when_bind_tools_false():
    """When the user disables Composio, the system prompt must neither instruct
    the LLM to call Composio tools nor falsely claim no apps are connected.
    """
    from unittest.mock import patch
    from src.api.rag_chat_helpers import build_agent_messages

    with patch("src.api.rag_chat_helpers.settings") as mock_settings:
        mock_settings.composio_enabled = True  # server has Composio enabled
        messages = build_agent_messages(
            system_instructions="",
            history=[],
            rag_context="",
            user_memory_context="",
            composio_apps=["slack", "gmail"],
            normalized_message="latest news",
            bind_tools=False,
            composio_user_disabled=True,  # tool_skip_reason == "user_disabled"
        )

    system_content = messages[0].content
    # App names must not appear — LLM must not be told to call bound tools
    assert "slack" not in system_content.lower()
    assert "gmail" not in system_content.lower()
    # Fallback "no apps connected" text must also be absent — apps exist, they
    # were just disabled by the user for this session.
    assert "no external apps are currently connected" not in system_content.lower()


def test_build_agent_messages_shows_no_apps_connected_when_apps_empty():
    """'No apps connected' fallback must still appear when composio is enabled
    server-side but no apps are linked — i.e. not user-disabled."""
    from unittest.mock import patch
    from src.api.rag_chat_helpers import build_agent_messages

    with patch("src.api.rag_chat_helpers.settings") as mock_settings:
        mock_settings.composio_enabled = True
        messages = build_agent_messages(
            system_instructions="",
            history=[],
            rag_context="",
            user_memory_context="",
            composio_apps=[],
            normalized_message="check my email",
            bind_tools=False,
            composio_user_disabled=False,  # no_connected_apps path, not user-disabled
        )

    assert "no external apps are currently connected" in messages[0].content.lower()


def test_build_agent_messages_includes_composio_when_bind_tools_true():
    """System prompt must list connected apps when Composio is actually bound."""
    from unittest.mock import patch
    from src.api.rag_chat_helpers import build_agent_messages

    with patch("src.api.rag_chat_helpers.settings") as mock_settings:
        mock_settings.composio_enabled = True
        messages = build_agent_messages(
            system_instructions="",
            history=[],
            rag_context="",
            user_memory_context="",
            composio_apps=["slack"],
            normalized_message="send a slack message",
            bind_tools=True,
        )

    system_content = messages[0].content
    assert "slack" in system_content.lower()


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

    from src.api.deps import _run_agent_loop

    llm_response = AIMessage(content="Done.")
    mock_llm = MagicMock()
    from unittest.mock import AsyncMock

    mock_llm.ainvoke = AsyncMock(return_value=llm_response)

    from unittest.mock import patch

    with (
        patch("src.api.deps.get_llm", return_value=mock_llm),
        patch("src.api.deps.build_agent_tools", return_value=[]),
    ):
        result = await _run_agent_loop(
            messages=[HumanMessage(content="Hi")],
            metadata={},
            bind_tools=False,
            allow_web_search=False,
        )

    assert result.answer == "Done."
    assert result.web_used is False
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
    from src.api.deps import RagChatTools
    from unittest.mock import AsyncMock, patch

    tools = RagChatTools(web_search=True, composio=False)

    with (
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=MagicMock(context="", chunks=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
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
        assert result.reference_tools["wikipedia"] is True


@pytest.mark.asyncio
async def test_prepare_workspace_respects_reference_tool_toggles():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.api.deps import RagChatTools
    from unittest.mock import AsyncMock, patch

    tools = RagChatTools(
        web_search=False,
        wikipedia=False,
        arxiv=False,
        open_library=False,
        composio=False,
    )

    with (
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=MagicMock(context="", chunks=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="tell me about Athens",
            session_id=None,
            timings=RagChatTimings(),
            tools=tools,
        )
        assert result.allow_web_search is False
        assert result.reference_tools == {
            "wikipedia": False,
            "arxiv": False,
            "open_library": False,
        }


@pytest.mark.asyncio
async def test_prepare_agent_merges_session_attachment_resource_ids():
    from src.api.rag_chat_helpers import prepare_agent_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from unittest.mock import AsyncMock, patch

    agent = MagicMock(system_instructions="system")
    rag_context = MagicMock(context="", chunks=[])
    router_decision = ChatActionDecisionPayload(
        action="answer_from_rag",
        reason="document question",
    )

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new=AsyncMock(return_value=router_decision),
        ),
        patch(
            "src.api.rag_chat_helpers.get_agent_for_chat",
            new_callable=AsyncMock,
            return_value=(agent, ["agent-res-1"]),
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[
                MagicMock(resource_id="attachment-res-1", filename="brief.pdf", state="ready"),
                MagicMock(resource_id="attachment-res-2", filename="notes.txt", state="ready"),
            ],
        ) as mock_list_attachments,
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=rag_context,
        ) as mock_retrieve,
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []

        result = await prepare_agent_rag_chat(
            agent_id="agent-1",
            user_id="user-1",
            normalized_message="What changed?",
            session_id=None,
            timings=RagChatTimings(),
        )

    assert result is not None
    assert result.resource_ids == [
        "agent-res-1",
        "attachment-res-1",
        "attachment-res-2",
    ]
    mock_list_attachments.assert_awaited_once_with(
        session_id="sess-1",
        owner_id="user-1",
        agent_id="agent-1",
    )
    mock_retrieve.assert_awaited_once_with(
        user_id="user-1",
        agent_resource_ids=["agent-res-1"],
        session_attachment_resource_ids=["attachment-res-1", "attachment-res-2"],
        session_attachment_files=["brief.pdf", "notes.txt"],
        question="What changed?",
    )


@pytest.mark.asyncio
async def test_prepare_agent_with_explicit_tools_deduplicates_merged_resource_ids():
    from src.api.rag_chat_helpers import prepare_agent_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.api.deps import RagChatTools
    from unittest.mock import AsyncMock, patch

    agent = MagicMock(system_instructions="system")
    rag_context = MagicMock(context="", chunks=[])
    tools = RagChatTools(web_search=False, composio=False)
    router_decision = ChatActionDecisionPayload(
        action="answer_from_rag",
        reason="document question",
    )

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new=AsyncMock(return_value=router_decision),
        ),
        patch(
            "src.api.rag_chat_helpers.get_agent_for_chat",
            new_callable=AsyncMock,
            return_value=(agent, ["shared-res", "agent-res-1"]),
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[
                MagicMock(resource_id="shared-res", filename="shared.pdf", state="ready"),
                MagicMock(resource_id="attachment-res-1", filename="brief.pdf", state="ready"),
            ],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=rag_context,
        ) as mock_retrieve,
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = ["slack"]

        result = await prepare_agent_rag_chat(
            agent_id="agent-1",
            user_id="user-1",
            normalized_message="What changed?",
            session_id=None,
            timings=RagChatTimings(),
            tools=tools,
        )

    assert result is not None
    assert result.resource_ids == ["shared-res", "agent-res-1", "attachment-res-1"]
    mock_retrieve.assert_awaited_once_with(
        user_id="user-1",
        agent_resource_ids=["shared-res", "agent-res-1"],
        session_attachment_resource_ids=["shared-res", "attachment-res-1"],
        session_attachment_files=["shared.pdf", "brief.pdf"],
        question="What changed?",
    )


@pytest.mark.asyncio
async def test_prepare_agent_loads_linked_resources_even_for_direct_router_decision():
    from src.api.rag_chat_helpers import prepare_agent_rag_chat
    from src.api.rag_chat_timing import RagChatTimings

    agent = MagicMock(system_instructions="system")
    agent.name = "CV Agent"
    agent.description = "Resume advice grounded in the linked CV."
    rag_context = MagicMock(context="attachment context", chunks=[])
    router_decision = ChatActionDecisionPayload(
        action="answer_direct",
        reason="test decision",
    )

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new=AsyncMock(return_value=router_decision),
        ) as mock_router,
        patch(
            "src.api.rag_chat_helpers.get_agent_for_chat",
            new=AsyncMock(return_value=(agent, ["agent-res-1"])),
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_chat_session",
            new=AsyncMock(return_value="sess-1"),
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new=AsyncMock(
                return_value=[
                    MagicMock(
                        resource_id="attachment-res-1",
                        filename="brief.pdf",
                        state="ready",
                    )
                ]
            ),
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new=AsyncMock(return_value=rag_context),
        ) as mock_retrieve,
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new=AsyncMock(return_value=""),
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new=AsyncMock(return_value=[]),
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        result = await prepare_agent_rag_chat(
            agent_id="agent-1",
            user_id="user-1",
            normalized_message="hi",
            session_id=None,
            timings=RagChatTimings(),
        )

    assert result is not None
    assert result.resource_ids == ["agent-res-1", "attachment-res-1"]
    mock_router.assert_awaited_once_with(
        message="hi",
        resource_scope=(
            'Custom agent "CV Agent" has 1 linked uploaded resource and '
            "1 ready session attachment. Agent description: Resume advice grounded in the linked CV."
        ),
    )
    mock_retrieve.assert_awaited_once_with(
        user_id="user-1",
        agent_resource_ids=["agent-res-1"],
        session_attachment_resource_ids=["attachment-res-1"],
        session_attachment_files=["brief.pdf"],
        question="hi",
    )


@pytest.mark.asyncio
async def test_prepare_workspace_respects_composio_true():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.api.deps import RagChatTools
    from unittest.mock import AsyncMock, patch

    tools = RagChatTools(web_search=True, composio=True)

    with (
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch("src.api.rag_chat_helpers.settings") as mock_settings,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=MagicMock(context="", chunks=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
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


def test_router_has_no_disable_flag():
    from src.config import Settings

    assert "router_enabled" not in Settings.model_fields


def test_router_turn_does_not_treat_missing_context_as_missing_resources():
    from src.api.rag_chat_helpers import _format_router_user_turn

    turn = _format_router_user_turn(
        message="Using only my uploaded resources, summarize the key findings.",
        rag_context="",
    )

    assert "Resource availability: unknown" in turn
    assert "context is retrieved after routing" in turn
    assert "Available RAG context: no" not in turn


@pytest.mark.asyncio
async def test_classify_chat_action_parses_valid_response():
    from unittest.mock import AsyncMock, patch

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content='{"action": "answer_direct", "reason": "greeting"}')
    )
    with patch("src.api.rag_chat_helpers.get_router_llm", return_value=mock_llm):
        result = await classify_chat_action(message="hello")
        assert result.action == "answer_direct"


@pytest.mark.asyncio
async def test_classify_chat_action_uses_configured_router_timeout():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content='{"action": "answer_direct", "reason": "greeting"}')
    )

    async def passthrough(awaitable, *, timeout):
        assert timeout == 10.0
        return await awaitable

    with (
        patch("src.api.rag_chat_helpers.get_router_llm", return_value=mock_llm),
        patch("src.api.rag_chat_helpers.asyncio.wait_for", side_effect=passthrough),
        patch("src.api.rag_chat_helpers.settings") as mock_settings,
    ):
        mock_settings.router_prompt_path = ""
        mock_settings.router_timeout_seconds = 10.0
        result = await classify_chat_action(message="hello")

    assert result.action == "answer_direct"


@pytest.mark.asyncio
async def test_classify_chat_action_repairs_invalid_json_once():
    from unittest.mock import AsyncMock, patch

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content="not json"),
            MagicMock(
                content='{"action": "web_search", "reason": "needs current info", "query": "latest news"}'
            ),
        ]
    )
    with patch("src.api.rag_chat_helpers.get_router_llm", return_value=mock_llm):
        result = await classify_chat_action(message="what's the latest news")
        assert result.action == "web_search"
        assert mock_llm.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_classify_chat_action_raises_router_error_after_failed_repair():
    from src.errors import RouterError
    from unittest.mock import AsyncMock, patch

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content="not json"),
            MagicMock(content="still not json"),
        ]
    )
    with patch("src.api.rag_chat_helpers.get_router_llm", return_value=mock_llm):
        with pytest.raises(RouterError, match="invalid decision"):
            await classify_chat_action(message="hello")


@pytest.mark.asyncio
async def test_classify_chat_action_raises_router_error_on_llm_exception():
    from src.errors import RouterError
    from unittest.mock import AsyncMock, patch

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("ollama unreachable"))
    with patch("src.api.rag_chat_helpers.get_router_llm", return_value=mock_llm):
        with pytest.raises(RouterError, match="unavailable"):
            await classify_chat_action(message="hello")


@pytest.mark.asyncio
async def test_prepare_workspace_router_error_never_lists_or_retrieves_resources():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.errors import RouterError

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new=AsyncMock(side_effect=RouterError("Chat router is unavailable.")),
        ),
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new=AsyncMock(return_value="sess-1"),
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new=AsyncMock(return_value=["res-1"]),
        ) as list_resources,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new=AsyncMock(),
        ) as retrieve,
    ):
        with pytest.raises(RouterError, match="unavailable"):
            await prepare_workspace_rag_chat(
                user_id="u1",
                normalized_message="hi",
                session_id=None,
                timings=RagChatTimings(),
            )

    list_resources.assert_not_awaited()
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_workspace_uses_mandatory_router_decision(default_router_decision):
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from unittest.mock import AsyncMock, patch

    with (
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=[],
        ) as list_resources,
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch("src.api.rag_chat_helpers.settings") as mock_settings,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=MagicMock(context="", chunks=[]),
        ),
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        mock_settings.composio_enabled = True
        mock_settings.rag_chat_max_history_messages = 10
        mock_settings.rag_chat_conditional_tools = True
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="hello",
            session_id=None,
            timings=RagChatTimings(),
        )
        assert result.router_decision == default_router_decision
        list_resources.assert_not_awaited()


@pytest.mark.parametrize(
    ("action", "should_list_workspace_resources"),
    [
        ("web_search", False),
        ("asset_price", False),
        ("search_finance_tools", False),
        ("answer_from_rag", True),
        ("answer_direct", False),
    ],
)
def test_should_use_workspace_resources(action, should_list_workspace_resources):
    from src.api.rag_chat_helpers import should_use_workspace_resources
    from src.llm.output_parsers import ChatActionDecisionPayload

    kwargs = {"action": action, "reason": "routing"}
    if action == "web_search":
        kwargs["query"] = "latest news"
    if action == "search_finance_tools":
        kwargs["query"] = "AAPL ratios"
    if action == "asset_price":
        kwargs["symbols"] = ["AAPL"]

    decision = ChatActionDecisionPayload(**kwargs)
    assert should_use_workspace_resources(decision) is should_list_workspace_resources
    assert should_use_workspace_resources(None) is False


@pytest.mark.asyncio
async def test_prepare_workspace_live_query_skips_workspace_rag():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.llm.output_parsers import ChatActionDecisionPayload
    from unittest.mock import AsyncMock, patch

    router_decision = ChatActionDecisionPayload(
        action="web_search",
        reason="needs current info",
        query="latest crypto news",
    )
    rag_context = MagicMock(context="", chunks=[])

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new_callable=AsyncMock,
            return_value=router_decision,
        ),
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=["saas-pdf", "playbook-pdf"],
        ) as mock_list_resources,
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=rag_context,
        ) as mock_retrieve,
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="latest crypto news",
            session_id=None,
            timings=RagChatTimings(),
        )

    mock_list_resources.assert_not_awaited()
    mock_retrieve.assert_awaited_once_with(
        user_id="u1",
        agent_resource_ids=[],
        session_attachment_resource_ids=[],
        session_attachment_files=[],
        question="latest crypto news",
    )
    assert result.router_decision == router_decision
    assert result.resource_ids == []


@pytest.mark.asyncio
async def test_prepare_workspace_rag_query_uses_workspace_resources():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.llm.output_parsers import ChatActionDecisionPayload
    from unittest.mock import AsyncMock, patch

    router_decision = ChatActionDecisionPayload(
        action="answer_from_rag",
        reason="document question",
    )
    rag_context = MagicMock(context="doc context", chunks=[])

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new_callable=AsyncMock,
            return_value=router_decision,
        ),
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=["saas-pdf", "playbook-pdf"],
        ) as mock_list_resources,
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=rag_context,
        ) as mock_retrieve,
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="summarize my uploaded playbook",
            session_id=None,
            timings=RagChatTimings(),
        )

    mock_list_resources.assert_awaited_once_with("u1")
    mock_retrieve.assert_awaited_once_with(
        user_id="u1",
        agent_resource_ids=["saas-pdf", "playbook-pdf"],
        session_attachment_resource_ids=[],
        session_attachment_files=[],
        question="summarize my uploaded playbook",
    )
    assert result.resource_ids == ["saas-pdf", "playbook-pdf"]


@pytest.mark.asyncio
async def test_prepare_workspace_live_query_keeps_session_attachments():
    from src.api.rag_chat_helpers import prepare_workspace_rag_chat
    from src.api.rag_chat_timing import RagChatTimings
    from src.llm.output_parsers import ChatActionDecisionPayload
    from unittest.mock import AsyncMock, patch

    router_decision = ChatActionDecisionPayload(
        action="web_search",
        reason="needs current info",
        query="latest crypto news",
    )
    rag_context = MagicMock(context="attachment context", chunks=[])

    with (
        patch(
            "src.api.rag_chat_helpers.classify_chat_action",
            new_callable=AsyncMock,
            return_value=router_decision,
        ),
        patch(
            "src.api.rag_chat_helpers.list_workspace_ready_resource_ids",
            new_callable=AsyncMock,
            return_value=["saas-pdf"],
        ) as mock_list_resources,
        patch(
            "src.api.rag_chat_helpers.create_or_get_workspace_chat_session",
            new_callable=AsyncMock,
            return_value="sess-1",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_session_attachments",
            new_callable=AsyncMock,
            return_value=[
                MagicMock(resource_id="attachment-res-1", filename="brief.pdf", state="ready"),
            ],
        ),
        patch("src.api.rag_chat_helpers.get_composio_toolset_manager") as mock_mgr,
        patch(
            "src.api.rag_chat_helpers.retrieve_merged_context_for_agent_chat",
            new_callable=AsyncMock,
            return_value=rag_context,
        ) as mock_retrieve,
        patch(
            "src.api.rag_chat_helpers.get_user_memory_prompt_block",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch(
            "src.api.rag_chat_helpers.list_rag_chat_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_mgr.return_value.get_connected_app_names.return_value = []
        result = await prepare_workspace_rag_chat(
            user_id="u1",
            normalized_message="latest crypto news",
            session_id="sess-1",
            timings=RagChatTimings(),
        )

    mock_list_resources.assert_not_awaited()
    mock_retrieve.assert_awaited_once_with(
        user_id="u1",
        agent_resource_ids=[],
        session_attachment_resource_ids=["attachment-res-1"],
        session_attachment_files=["brief.pdf"],
        question="latest crypto news",
    )
    assert result.resource_ids == ["attachment-res-1"]


class TestGetRouterSystemPrompt:
    """ROUTER_PROMPT_PATH activation path for optimized router prompts."""

    def _clear_cache(self):
        from src.api import rag_chat_helpers

        rag_chat_helpers._ROUTER_PROMPT_CACHE.clear()

    def test_defaults_to_builtin_prompt(self):
        from src.api.rag_chat_helpers import (
            _ROUTER_ACTION_SYSTEM_PROMPT,
            get_router_system_prompt,
        )

        self._clear_cache()
        with patch("src.api.rag_chat_helpers.settings") as settings_mock:
            settings_mock.router_prompt_path = ""
            assert get_router_system_prompt() == _ROUTER_ACTION_SYSTEM_PROMPT

    def test_loads_optimized_artifact(self, tmp_path):
        import json

        from src.api.rag_chat_helpers import get_router_system_prompt

        self._clear_cache()
        artifact = tmp_path / "router.json"
        artifact.write_text(json.dumps({"system_prompt": "OPTIMIZED ROUTER PROMPT"}))
        with patch("src.api.rag_chat_helpers.settings") as settings_mock:
            settings_mock.router_prompt_path = str(artifact)
            assert get_router_system_prompt() == "OPTIMIZED ROUTER PROMPT"

    def test_falls_back_on_bad_artifact(self, tmp_path):
        from src.api.rag_chat_helpers import (
            _ROUTER_ACTION_SYSTEM_PROMPT,
            get_router_system_prompt,
        )

        self._clear_cache()
        artifact = tmp_path / "broken.json"
        artifact.write_text("{not json")
        with patch("src.api.rag_chat_helpers.settings") as settings_mock:
            settings_mock.router_prompt_path = str(artifact)
            assert get_router_system_prompt() == _ROUTER_ACTION_SYSTEM_PROMPT

    def test_falls_back_on_missing_key(self, tmp_path):
        import json

        from src.api.rag_chat_helpers import (
            _ROUTER_ACTION_SYSTEM_PROMPT,
            get_router_system_prompt,
        )

        self._clear_cache()
        artifact = tmp_path / "empty.json"
        artifact.write_text(json.dumps({"system_prompt": "  "}))
        with patch("src.api.rag_chat_helpers.settings") as settings_mock:
            settings_mock.router_prompt_path = str(artifact)
            assert get_router_system_prompt() == _ROUTER_ACTION_SYSTEM_PROMPT
