"""Tests for src/tools/vector_store.py"""

from unittest.mock import MagicMock, patch

import pytest

from src.errors import VectorStoreError
from src.config import settings


def _make_manager():
    from src.tools.vector_store import VectorStoreManager

    return VectorStoreManager()


def _make_manager_with_mocks():
    """Return (manager, mock_index, mock_pinecone_client)."""
    manager = _make_manager()

    mock_index = MagicMock()
    mock_pinecone = MagicMock()
    mock_index_info = MagicMock()
    mock_index_info.dimension = settings.embedding_dimensions
    mock_pinecone.describe_index.return_value = mock_index_info

    manager._index = mock_index
    manager._pinecone_client = mock_pinecone

    return manager, mock_index, mock_pinecone


def test_save_report_calls_index_insert_nodes():
    manager, _, _ = _make_manager_with_mocks()
    mock_llama_index = MagicMock()
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    doc_id = manager.save_report(query="LangGraph", report="# Report")

    mock_llama_index.insert_nodes.assert_called_once()
    nodes = mock_llama_index.insert_nodes.call_args.args[0]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.id_ == doc_id
    assert node.metadata["query"] == "LangGraph"
    assert node.metadata["document"] == "# Report"
    assert doc_id.startswith("report_")
    manager._get_index_for_namespace.assert_called_once_with("reports")


def test_save_report_raises_vector_store_error_on_failure():
    manager, _, _ = _make_manager_with_mocks()
    mock_llama_index = MagicMock()
    mock_llama_index.insert_nodes.side_effect = RuntimeError("pinecone down")
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    with pytest.raises(VectorStoreError, match="pinecone down"):
        manager.save_report(query="test", report="# Report")


def test_search_reports_returns_structured_list():
    manager, _, _ = _make_manager_with_mocks()

    node = MagicMock()
    node.id_ = "report_001"
    node.metadata = {
        "query": "LangGraph",
        "generated_at": "2026-01-01",
        "document": "# Report content",
    }
    result = MagicMock()
    result.node = node

    retriever = MagicMock()
    retriever.retrieve.return_value = [result]
    mock_llama_index = MagicMock()
    mock_llama_index.as_retriever.return_value = retriever
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    results = manager.search_reports("LangGraph")

    assert len(results) == 1
    assert results[0]["id"] == "report_001"
    assert results[0]["document"] == "# Report content"
    assert results[0]["metadata"]["query"] == "LangGraph"


def test_search_reports_raises_on_failure():
    manager, _, _ = _make_manager_with_mocks()
    mock_llama_index = MagicMock()
    mock_llama_index.as_retriever.side_effect = RuntimeError("query failed")
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    with pytest.raises(VectorStoreError, match="query failed"):
        manager.search_reports("anything")


def test_save_source_chunks_calls_insert_nodes():
    manager, _, _ = _make_manager_with_mocks()
    mock_llama_index = MagicMock()
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    sources = [
        {"url": "https://a.com", "title": "A", "raw_text": "Some content about topic A."},
        {"url": "https://b.com", "title": "B", "raw_text": "Content about topic B."},
    ]
    count = manager.save_source_chunks(run_id="run-1", session_id="sess-1", sources=sources)

    assert count > 0
    mock_llama_index.insert_nodes.assert_called_once()
    inserted_nodes = mock_llama_index.insert_nodes.call_args.args[0]
    assert len(inserted_nodes) == count
    for n in inserted_nodes:
        assert n.metadata["run_id"] == "run-1"
        assert n.metadata["session_id"] == "sess-1"
        assert "text" in n.metadata


def test_save_source_chunks_returns_zero_for_empty_sources():
    manager, _, _ = _make_manager_with_mocks()
    manager._get_index_for_namespace = MagicMock()
    count = manager.save_source_chunks(run_id="run-1", session_id="sess-1", sources=[])
    assert count == 0
    manager._get_index_for_namespace.assert_not_called()


def test_search_run_sources_returns_structured_list():
    manager, _, _ = _make_manager_with_mocks()

    node = MagicMock()
    node.metadata = {
        "run_id": "run-1",
        "session_id": "sess-1",
        "source_url": "https://a.com",
        "source_title": "A",
        "chunk_index": 0,
        "text": "Relevant text here.",
    }
    result = MagicMock()
    result.node = node

    retriever = MagicMock()
    retriever.retrieve.return_value = [result]
    mock_llama_index = MagicMock()
    mock_llama_index.as_retriever.return_value = retriever
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    results = manager.search_run_sources("query text", run_id="run-1")

    assert len(results) == 1
    assert results[0]["text"] == "Relevant text here."
    assert results[0]["source_url"] == "https://a.com"
    assert results[0]["source_title"] == "A"


def test_search_run_sources_returns_empty_when_no_chunks():
    manager, _, _ = _make_manager_with_mocks()

    retriever = MagicMock()
    retriever.retrieve.return_value = []
    mock_llama_index = MagicMock()
    mock_llama_index.as_retriever.return_value = retriever
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    results = manager.search_run_sources("query text", run_id="run-1")

    assert results == []
    retriever.retrieve.assert_called_once()


def test_search_run_sources_passes_run_id_filter():
    manager, _, _ = _make_manager_with_mocks()

    retriever = MagicMock()
    retriever.retrieve.return_value = []
    mock_llama_index = MagicMock()
    mock_llama_index.as_retriever.return_value = retriever
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    manager.search_run_sources("query text", run_id="run-2")

    filters = mock_llama_index.as_retriever.call_args.kwargs["filters"]
    assert len(filters.filters) == 1
    assert filters.filters[0].key == "run_id"
    assert filters.filters[0].value == "run-2"


def test_search_run_sources_raises_on_failure():
    manager, _, _ = _make_manager_with_mocks()
    mock_llama_index = MagicMock()
    mock_llama_index.as_retriever.side_effect = RuntimeError("query error")
    manager._get_index_for_namespace = MagicMock(return_value=mock_llama_index)

    with pytest.raises(VectorStoreError, match="query error"):
        manager.search_run_sources("anything", run_id="run-1")


def test_dimension_mismatch_raises_clear_error():
    manager, _, mock_pinecone = _make_manager_with_mocks()
    mock_index_info = MagicMock()
    mock_index_info.dimension = 768
    mock_pinecone.describe_index.return_value = mock_index_info

    with patch("src.tools.vector_store.settings") as mock_settings:
        mock_settings.pinecone_index_name = "research-agent-ollama-nomic"
        mock_settings.embedding_dimensions = 1536
        with pytest.raises(VectorStoreError, match="does not match the configured embedding dimensions"):
            manager.save_report(query="LangGraph", report="# Report")
