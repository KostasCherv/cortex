"""Tests for LangFuse dataset artifact loading and sync helpers."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_load_golden_queries_reads_expected_shape(tmp_path: Path):
    from src.observability.langfuse_datasets import load_golden_queries

    fixture = tmp_path / "golden.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "id": "langgraph-overview",
                    "input": {"query": "What is LangGraph?"},
                    "rubric": {"must_include": ["stateful workflows"], "must_not": []},
                    "tags": ["baseline"],
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_golden_queries(fixture)

    assert len(records) == 1
    assert records[0].id == "langgraph-overview"
    assert records[0].input == {"query": "What is LangGraph?"}


def test_load_golden_queries_rejects_missing_required_fields(tmp_path: Path):
    from src.observability.langfuse_datasets import load_golden_queries

    fixture = tmp_path / "golden.json"
    fixture.write_text(
        json.dumps([{"id": "broken", "input": {"query": "x"}, "tags": ["baseline"]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rubric"):
        load_golden_queries(fixture)


def test_sync_dataset_upserts_items_with_rubric_metadata():
    from src.observability.langfuse_datasets import GoldenQueryRecord, sync_dataset

    client = MagicMock()
    client.get_dataset.side_effect = Exception("missing")

    sync_dataset(
        client=client,
        dataset_name="cortex/golden-queries",
        records=[
            GoldenQueryRecord(
                id="langgraph-overview",
                input={"query": "What is LangGraph?"},
                rubric={"must_include": ["stateful workflows"], "must_not": ["confuse with LangSmith"]},
                tags=["baseline", "definition"],
                difficulty="easy",
                notes="seed example",
            )
        ],
        source_path="tests/fixtures/langfuse_golden_queries.json",
    )

    client.create_dataset.assert_called_once()
    client.create_dataset_item.assert_called_once()
