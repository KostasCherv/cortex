# tests/finetune/test_generate_router_dataset.py
import json
from pathlib import Path

import pytest

from scripts.generate_router_dataset import deduplicate_inputs, split_records, write_jsonl

_ACTIONS = [
    "answer_direct", "answer_from_rag", "web_search",
    "asset_price", "search_finance_tools", "ask_clarifying",
]


def _make_records(n_per_class: int = 10) -> list[dict]:
    records = []
    for action in _ACTIONS:
        for i in range(n_per_class):
            records.append({
                "messages": [{"role": "user", "content": f"msg {i}"}],
                "_action_label": action,
            })
    return records


def test_split_preserves_total_count():
    records = _make_records(10)
    train, held_out = split_records(records, held_out_fraction=0.2, seed=42)
    assert len(train) + len(held_out) == len(records)


def test_split_held_out_fraction_approximately_correct():
    records = _make_records(20)
    train, held_out = split_records(records, held_out_fraction=0.2, seed=42)
    ratio = len(held_out) / len(records)
    assert 0.15 <= ratio <= 0.25


def test_split_all_classes_represented_in_held_out():
    records = _make_records(10)
    _, held_out = split_records(records, held_out_fraction=0.2, seed=42)
    actions_in_held_out = {r["_action_label"] for r in held_out}
    assert actions_in_held_out == set(_ACTIONS)


def test_write_jsonl_train_strips_internal_keys(tmp_path: Path):
    records = [{"messages": [{"role": "user", "content": "hi"}], "_action_label": "answer_direct"}]
    out = tmp_path / "train.jsonl"
    write_jsonl(records, out, keep_meta=False)

    row = json.loads(out.read_text().strip())
    assert "messages" in row
    assert "_action_label" not in row
    assert "action_label" not in row


def test_write_jsonl_held_out_keeps_expected_action(tmp_path: Path):
    records = [{"messages": [{"role": "user", "content": "hi"}], "_action_label": "answer_direct"}]
    out = tmp_path / "held_out.jsonl"
    write_jsonl(records, out, keep_meta=True)

    row = json.loads(out.read_text().strip())
    assert "messages" in row
    assert row["action_label"] == "answer_direct"
    assert "_action_label" not in row


def test_write_jsonl_creates_parent_dirs(tmp_path: Path):
    records = [{"messages": [], "_action_label": "web_search"}]
    nested = tmp_path / "a" / "b" / "out.jsonl"
    write_jsonl(records, nested, keep_meta=False)
    assert nested.exists()


def test_deduplicate_inputs_removes_exact_duplicates():
    inputs = [
        {"message": "What is AAPL?", "rag_context": ""},
        {"message": "What is AAPL?", "rag_context": ""},
        {"message": "Tell me about Tesla", "rag_context": ""},
    ]
    result = deduplicate_inputs(inputs)
    assert len(result) == 2


def test_deduplicate_inputs_case_and_whitespace_insensitive():
    inputs = [
        {"message": "What is AAPL?", "rag_context": ""},
        {"message": "  what is aapl?  ", "rag_context": ""},
    ]
    result = deduplicate_inputs(inputs)
    assert len(result) == 1


def test_deduplicate_inputs_preserves_first_occurrence():
    inputs = [
        {"message": "Hello", "rag_context": "", "action": "answer_direct"},
        {"message": "Hello", "rag_context": "", "action": "web_search"},
    ]
    result = deduplicate_inputs(inputs)
    assert result[0]["action"] == "answer_direct"


def test_deduplicate_inputs_keeps_distinct_rag_context():
    inputs = [
        {"message": "Summarise this", "rag_context": "doc A"},
        {"message": "Summarise this", "rag_context": "doc B"},
    ]
    result = deduplicate_inputs(inputs)
    assert len(result) == 2
