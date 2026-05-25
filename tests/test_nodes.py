"""Tests for graph nodes (src/graph/nodes.py)"""

import asyncio
from unittest.mock import patch, MagicMock
from unittest.mock import AsyncMock


# ---------------------------------------------------------------------------
# search_node
# ---------------------------------------------------------------------------

def test_search_node_populates_results():
    with patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {"url": "https://a.com", "title": "A", "content": "snippet", "raw_content": "full text"}
        ]
        from src.graph.nodes import search_node
        state = asyncio.run(search_node({"query": "LangGraph", "error": None}))

    assert state["error"] is None
    assert len(state["search_results"]) == 1
    assert len(state["retrieved_contents"]) == 1
    assert state["retrieved_contents"][0]["raw_text"] == "full text"


def test_search_node_falls_back_to_content_when_no_raw_content():
    with patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {"url": "https://a.com", "title": "A", "content": "snippet", "raw_content": ""}
        ]
        from src.graph.nodes import search_node
        state = asyncio.run(search_node({"query": "LangGraph", "error": None}))

    assert state["retrieved_contents"][0]["raw_text"] == "snippet"


def test_search_node_sets_error_on_failure():
    from src.errors import SearchError
    with patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock, side_effect=SearchError("boom")):
        from src.graph.nodes import search_node
        state = asyncio.run(search_node({"query": "fail", "error": None}))

    assert state["error"] == "boom"
    assert state["search_results"] == []
    assert state["retrieved_contents"] == []


# ---------------------------------------------------------------------------
# search_and_memory_node
# ---------------------------------------------------------------------------

def test_search_and_memory_node_runs_both_concurrently():
    with (
        patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock) as mock_search,
        patch("src.graph.nodes.Neo4jGraphStore") as mock_graph_cls,
    ):
        mock_search.return_value = [
            {"url": "https://a.com", "title": "A", "content": "c", "raw_content": "full"}
        ]
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(
            context="old report",
            chunks=[],
            entities=[],
        )
        mock_graph_cls.return_value = mock_graph

        from src.graph.nodes import search_and_memory_node
        state = asyncio.run(search_and_memory_node({"query": "LangGraph", "user_id": "u-1"}))

    assert len(state["search_results"]) == 1
    assert len(state["retrieved_contents"]) == 1
    assert "old report" in state["memory_context"]
    assert state["error"] is None


def test_search_and_memory_node_propagates_search_error():
    from src.errors import SearchError
    with (
        patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock, side_effect=SearchError("no search")),
        patch("src.graph.nodes.Neo4jGraphStore") as mock_graph_cls,
    ):
        mock_graph = MagicMock()
        mock_graph.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_graph_cls.return_value = mock_graph

        from src.graph.nodes import search_and_memory_node
        state = asyncio.run(search_and_memory_node({"query": "fail", "user_id": "u-1"}))

    assert state["error"] == "no search"
    assert state["search_results"] == []


def test_search_and_memory_node_continues_when_memory_fails():
    with (
        patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock) as mock_search,
        patch("src.graph.nodes.Neo4jGraphStore") as mock_graph_cls,
    ):
        mock_search.return_value = [
            {"url": "https://a.com", "title": "A", "content": "c", "raw_content": "text"}
        ]
        mock_graph = MagicMock()
        mock_graph.query_context.side_effect = RuntimeError("neo4j down")
        mock_graph_cls.return_value = mock_graph

        from src.graph.nodes import search_and_memory_node
        state = asyncio.run(search_and_memory_node({"query": "LangGraph", "user_id": "u-1"}))

    assert len(state["search_results"]) == 1
    assert state["memory_context"] == ""
    assert state["error"] is None


# ---------------------------------------------------------------------------
# rerank_node
# ---------------------------------------------------------------------------

def test_rerank_node_uses_shared_reranker_results():
    from src.graph.nodes import rerank_node

    with patch("src.graph.nodes.rerank_chunks") as mock_rerank:
        mock_rerank.return_value = [
            {
                "url": "https://b.com",
                "title": "B",
                "raw_text": "Beta",
                "text": "Beta",
                "rerank_score": 0.91,
            },
            {
                "url": "https://a.com",
                "title": "A",
                "raw_text": "Alpha",
                "text": "Alpha",
                "rerank_score": 0.12,
            },
        ]
        state = asyncio.run(
            rerank_node(
                {
                    "query": "beta",
                    "retrieved_contents": [
                        {"url": "https://a.com", "title": "A", "raw_text": "Alpha"},
                        {"url": "https://b.com", "title": "B", "raw_text": "Beta"},
                    ],
                }
            )
        )

    assert [row["url"] for row in state["reranked_contents"]] == [
        "https://b.com",
        "https://a.com",
    ]
    mock_rerank.assert_called_once()
    assert state["rerank_metadata"]["fallback"] is False


def test_rerank_node_handles_empty_input():
    from src.graph.nodes import rerank_node

    state = asyncio.run(rerank_node({"query": "LangGraph", "retrieved_contents": []}))

    assert state["reranked_contents"] == []
    assert state["rerank_metadata"]["reason"] == "empty_input"


# ---------------------------------------------------------------------------
# summarize_node
# ---------------------------------------------------------------------------


def test_summarize_node_prefers_reranked_contents_when_present():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='[{"url":"https://b.com","title":"B","summary":"Summary B"}]'
        )
    )

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import summarize_node

        state = asyncio.run(
            summarize_node(
                {
                    "query": "LangGraph",
                    "retrieved_contents": [
                        {"url": "https://a.com", "title": "A", "raw_text": "Alpha text"}
                    ],
                    "reranked_contents": [
                        {"url": "https://b.com", "title": "B", "raw_text": "Beta text"}
                    ],
                }
            )
        )

    assert [row["url"] for row in state["summaries"]] == ["https://b.com"]

def test_summarize_node_calls_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='[{"url":"https://a.com","title":"A","summary":"Nice summary."}]'
        )
    )

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import summarize_node
        state = asyncio.run(summarize_node({
            "query": "LangGraph",
            "retrieved_contents": [{"url": "https://a.com", "title": "A", "raw_text": "Some text content here."}],
        }))

    assert len(state["summaries"]) == 1
    assert state["summaries"][0]["summary"] == "Nice summary."


def test_summarize_node_makes_single_call_for_multiple_sources():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '[{"url":"https://a.com","title":"A","summary":"Summary A"},'
                '{"url":"https://b.com","title":"B","summary":"Summary B"}]'
            )
        )
    )

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import summarize_node

        state = asyncio.run(
            summarize_node(
                {
                    "query": "LangGraph",
                    "retrieved_contents": [
                        {"url": "https://a.com", "title": "A", "raw_text": "Alpha text"},
                        {"url": "https://b.com", "title": "B", "raw_text": "Beta text"},
                    ],
                }
            )
        )

    assert mock_llm.ainvoke.await_count == 1
    assert len(state["summaries"]) == 2
    prompt_arg = mock_llm.ainvoke.await_args.args[0]
    metadata = mock_llm.ainvoke.await_args.kwargs["config"]["metadata"]
    assert "Create high-coverage source summaries relevant to the query 'LangGraph'." in prompt_arg
    assert metadata["prompt_version"]


def test_summarize_node_parses_markdown_fenced_json():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='```json\n[{"url":"https://a.com","title":"A","summary":"Summary A"}]\n```'
        )
    )

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import summarize_node

        state = asyncio.run(
            summarize_node(
                {
                    "query": "LangGraph",
                    "retrieved_contents": [
                        {"url": "https://a.com", "title": "A", "raw_text": "Alpha text"}
                    ],
                }
            )
        )

    assert len(state["summaries"]) == 1
    assert state["summaries"][0]["summary"] == "Summary A"


def test_summarize_node_repairs_non_json_output_once():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content="Here is the summary in plain text."),
            MagicMock(content='[{"url":"https://a.com","title":"A","summary":"Summary A"}]'),
        ]
    )

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import summarize_node

        state = asyncio.run(
            summarize_node(
                {
                    "query": "LangGraph",
                    "retrieved_contents": [
                        {"url": "https://a.com", "title": "A", "raw_text": "Alpha text"}
                    ],
                }
            )
        )

    assert mock_llm.ainvoke.await_count == 2
    assert len(state["summaries"]) == 1


def test_summarize_node_repairs_schema_invalid_output_once():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            MagicMock(content='[{"url":"https://a.com","title":"A"}]'),
            MagicMock(content='[{"url":"https://a.com","title":"A","summary":"Summary A"}]'),
        ]
    )

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import summarize_node

        state = asyncio.run(
            summarize_node(
                {
                    "query": "LangGraph",
                    "retrieved_contents": [
                        {"url": "https://a.com", "title": "A", "raw_text": "Alpha text"}
                    ],
                }
            )
        )

    assert mock_llm.ainvoke.await_count == 2
    assert len(state["summaries"]) == 1
    assert state["summaries"][0]["summary"] == "Summary A"


# ---------------------------------------------------------------------------
# report_node
# ---------------------------------------------------------------------------

def test_report_node_generates_report():
    async def _fake_astream(*args, **kwargs):
        yield MagicMock(content="# My Report\nContent here.")

    mock_llm = MagicMock()
    mock_llm.astream = _fake_astream

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import report_node
        state = asyncio.run(report_node({
            "query": "LangGraph",
            "summaries": [{"url": "https://a.com", "title": "A", "summary": "x"}],
            "memory_context": "older context",
        }))

    assert "# My Report" in state["report"]
    assert "report_metadata" in state
    assert state["report_metadata"]["title"] == "LangGraph"


def test_report_node_omits_memory_context_section_when_empty():
    captured = {}

    async def _fake_astream(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["config"] = kwargs.get("config", {})
        yield MagicMock(content="# My Report\nContent here.")

    mock_llm = MagicMock()
    mock_llm.astream = _fake_astream

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import report_node

        asyncio.run(
            report_node(
                {
                    "query": "LangGraph",
                    "summaries": [{"url": "https://a.com", "title": "A", "summary": "x"}],
                    "memory_context": "",
                }
            )
        )

    assert "Prior context from past internal reports" not in captured["prompt"]
    assert captured["config"]["metadata"]["prompt_version"]


def test_report_node_includes_domain_section_when_present():
    captured = {}

    async def _fake_astream(prompt, **kwargs):
        captured["prompt"] = prompt
        yield MagicMock(content="# My Report\nContent here.")

    mock_llm = MagicMock()
    mock_llm.astream = _fake_astream

    with patch("src.graph.nodes.get_llm", return_value=mock_llm):
        from src.graph.nodes import report_node

        asyncio.run(
            report_node(
                {
                    "query": "LangGraph",
                    "summaries": [{"url": "https://a.com", "title": "A", "summary": "x"}],
                    "memory_context": "",
                    "domain": "enterprise software",
                }
            )
        )

    assert "Domain focus: enterprise software" in captured["prompt"]


# ---------------------------------------------------------------------------
# memory_context_node
# ---------------------------------------------------------------------------


def test_memory_context_node_builds_truncated_context():
    with patch("src.graph.nodes.Neo4jGraphStore") as mock_cls:
        mock_manager = MagicMock()
        long_doc = "A" * 2500
        mock_manager.query_context.return_value = MagicMock(
            context=long_doc,
            chunks=[{"chunk_id": "c1"}],
            entities=["alpha"],
        )
        mock_cls.return_value = mock_manager

        from src.graph.nodes import memory_context_node
        state = asyncio.run(memory_context_node({"query": "LangGraph", "user_id": "u-1"}))

    assert "memory_context" in state
    assert state["memory_context"].startswith("A")
    assert len(state["memory_context"]) == 2000


def test_memory_context_node_returns_empty_context_when_no_results():
    with patch("src.graph.nodes.Neo4jGraphStore") as mock_cls:
        mock_manager = MagicMock()
        mock_manager.query_context.return_value = MagicMock(context="", chunks=[], entities=[])
        mock_cls.return_value = mock_manager

        from src.graph.nodes import memory_context_node
        state = asyncio.run(memory_context_node({"query": "LangGraph", "user_id": "u-1"}))

    assert state["memory_context"] == ""


def test_memory_context_node_handles_lookup_failure():
    with patch("src.graph.nodes.Neo4jGraphStore") as mock_cls:
        mock_manager = MagicMock()
        mock_manager.query_context.side_effect = RuntimeError("neo4j unavailable")
        mock_cls.return_value = mock_manager

        from src.graph.nodes import memory_context_node
        state = asyncio.run(memory_context_node({"query": "LangGraph", "user_id": "u-1"}))

    assert state["memory_context"] == ""
