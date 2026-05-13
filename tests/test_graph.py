"""Tests for the compiled LangGraph (src/graph/graph.py)"""

import asyncio
from unittest.mock import patch, MagicMock
from unittest.mock import AsyncMock


def _make_mock_nodes():
    """Return a dict of no-op node mocks that pass state through."""
    def passthrough(state):
        return state

    return passthrough


def test_build_graph_returns_compiled_graph():
    from src.graph.graph import build_graph

    graph = build_graph()
    # A compiled LangGraph has an .invoke method
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")


def test_graph_invoke_with_error_reaches_abort(monkeypatch):
    from src.errors import SearchError

    with (
        patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock, side_effect=SearchError("no search")),
        patch("src.graph.nodes.VectorStoreManager") as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs.search_reports.return_value = []
        mock_vs_cls.return_value = mock_vs

        from src.graph.graph import build_graph
        graph = build_graph()
        final = asyncio.run(
            graph.ainvoke({"query": "test", "use_vector_store": False, "error": None})
        )

    # Pipeline should abort and set an error
    assert final.get("error") is not None


def test_graph_invoke_happy_path(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='[{"url":"https://example.com","title":"Example","summary":"Source summary."}]'
        )
    )

    async def _fake_report_stream(*args, **kwargs):
        yield MagicMock(content="# Report\nFinal output.")

    mock_llm.astream = _fake_report_stream

    search_result = [
        {"url": "https://example.com", "title": "Example", "content": "Content", "raw_content": "Full text"}
    ]

    with (
        patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock, return_value=search_result),
        patch("src.graph.nodes.get_llm", return_value=mock_llm),
        patch("src.graph.nodes.VectorStoreManager") as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs.search_reports.return_value = []
        mock_vs.rerank_documents.return_value = [
            {
                "url": "https://example.com",
                "title": "Example",
                "raw_text": "Full text",
                "score": 0.9,
            }
        ]
        mock_vs_cls.return_value = mock_vs
        from src.graph.graph import build_graph
        graph = build_graph()
        final = asyncio.run(
            graph.ainvoke({"query": "LangGraph", "use_vector_store": False, "error": None})
        )

    assert "report" in final
    assert len(final["report"]) > 0


def test_graph_invoke_continues_when_memory_lookup_fails():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='[{"url":"https://example.com","title":"Example","summary":"Source summary."}]'
        )
    )

    async def _fake_report_stream(*args, **kwargs):
        yield MagicMock(content="# Report\nFinal output.")

    mock_llm.astream = _fake_report_stream

    search_result = [
        {"url": "https://example.com", "title": "Example", "content": "Content", "raw_content": "Full text"}
    ]

    with (
        patch("src.graph.nodes.perform_search_cached", new_callable=AsyncMock, return_value=search_result),
        patch("src.graph.nodes.get_llm", return_value=mock_llm),
        patch("src.graph.nodes.VectorStoreManager") as mock_vs_cls,
    ):
        mock_vs = MagicMock()
        mock_vs.search_reports.side_effect = RuntimeError("pinecone unavailable")
        mock_vs.rerank_documents.return_value = [
            {
                "url": "https://example.com",
                "title": "Example",
                "raw_text": "Full text",
                "score": 0.9,
            }
        ]
        mock_vs_cls.return_value = mock_vs

        from src.graph.graph import build_graph
        graph = build_graph()
        final = asyncio.run(
            graph.ainvoke({"query": "LangGraph", "use_vector_store": False, "error": None})
        )

    assert "report" in final
    assert len(final["report"]) > 0
