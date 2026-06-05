"""Tests for the reference tool registry."""

from src.tools.registry import (
    REFERENCE_TOOL_IDS,
    REFERENCE_TOOL_SPECS,
    build_reference_tools,
    create_rag_chat_tools_model,
    default_reference_tool_flags,
    reference_flags_from_tools,
    reference_tool_prompt_lines,
)


def test_reference_tool_specs_include_expected_tools():
    assert REFERENCE_TOOL_IDS == frozenset({"wikipedia", "arxiv", "open_library"})
    assert [spec.id for spec in REFERENCE_TOOL_SPECS] == [
        "wikipedia",
        "arxiv",
        "open_library",
    ]


def test_default_reference_tool_flags_match_registry_defaults():
    assert default_reference_tool_flags() == {
        "wikipedia": True,
        "arxiv": False,
        "open_library": False,
    }


def test_reference_tool_prompt_lines_include_all_tools():
    lines = reference_tool_prompt_lines()
    assert len(lines) == 3
    assert all(line.startswith("- ") for line in lines)
    assert "wikipedia:" in lines[0]
    assert "search_papers" in lines[1]
    assert "open_library:" in lines[2]


def test_create_rag_chat_tools_model_includes_registry_fields():
    RagChatTools = create_rag_chat_tools_model()
    tools = RagChatTools()
    assert tools.web_search is True
    assert tools.composio is False
    assert tools.wikipedia is True
    assert tools.arxiv is False
    assert tools.open_library is False


def test_reference_flags_from_tools_reads_registry_fields():
    RagChatTools = create_rag_chat_tools_model()
    tools = RagChatTools(wikipedia=False, arxiv=True, open_library=True)
    assert reference_flags_from_tools(tools) == {
        "wikipedia": False,
        "arxiv": True,
        "open_library": True,
    }


def test_build_reference_tools_returns_only_enabled_tools():
    tools = build_reference_tools(
        reference_flags={"wikipedia": True, "arxiv": False, "open_library": False}
    )
    assert len(tools) == 1
    assert tools[0].name == "wikipedia"


def test_build_reference_tools_includes_all_free_tools_by_default():
    tools = build_reference_tools()
    names = {tool.name for tool in tools}
    assert names == {"wikipedia"}


def test_build_reference_tools_can_enable_open_library():
    tools = build_reference_tools(
        reference_flags={"wikipedia": False, "arxiv": True, "open_library": True}
    )
    names = [tool.name for tool in tools]
    assert names == ["open_library"]
