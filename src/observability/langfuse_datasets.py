"""LangFuse dataset helpers for checked-in golden query sync."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.observability.langfuse import require_client


DEFAULT_DATASET_NAME = "cortex/golden-queries"
DEFAULT_GOLDEN_QUERIES_PATH = Path("tests/fixtures/langfuse_golden_queries.json")


@dataclass
class GoldenQueryRecord:
    id: str
    input: dict[str, Any]
    rubric: dict[str, Any]
    tags: list[str]
    difficulty: str | None = None
    notes: str | None = None


def load_golden_queries(path: Path | str = DEFAULT_GOLDEN_QUERIES_PATH) -> list[GoldenQueryRecord]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Golden query artifact must contain a JSON list.")

    records: list[GoldenQueryRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Golden query item at index {index} must be an object.")
        missing = [key for key in ("id", "input", "rubric", "tags") if key not in item]
        if missing:
            raise ValueError(f"Golden query item '{item.get('id', index)}' is missing {', '.join(missing)}.")
        if not isinstance(item["input"], dict):
            raise ValueError(f"Golden query item '{item['id']}' input must be an object.")
        if not isinstance(item["rubric"], dict):
            raise ValueError(f"Golden query item '{item['id']}' rubric must be an object.")
        if not isinstance(item["tags"], list):
            raise ValueError(f"Golden query item '{item['id']}' tags must be a list.")

        records.append(
            GoldenQueryRecord(
                id=str(item["id"]),
                input=item["input"],
                rubric=item["rubric"],
                tags=[str(tag) for tag in item["tags"]],
                difficulty=str(item["difficulty"]) if item.get("difficulty") is not None else None,
                notes=str(item["notes"]) if item.get("notes") is not None else None,
            )
        )
    return records


def sync_dataset(
    *,
    client: Any,
    dataset_name: str,
    records: list[GoldenQueryRecord],
    source_path: str,
) -> int:
    try:
        client.get_dataset(name=dataset_name)
    except Exception:
        client.create_dataset(
            name=dataset_name,
            description="Production golden queries for Cortex.",
            metadata={"source_path": source_path},
        )

    for record in records:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=record.id,
            input=record.input,
            metadata={
                "rubric": record.rubric,
                "tags": record.tags,
                "difficulty": record.difficulty,
                "notes": record.notes,
                "source_path": source_path,
            },
        )
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()
    return len(records)


def sync_golden_queries_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    source_path: Path | str = DEFAULT_GOLDEN_QUERIES_PATH,
) -> int:
    client = require_client()
    fixture_path = Path(source_path)
    records = load_golden_queries(fixture_path)
    return sync_dataset(
        client=client,
        dataset_name=dataset_name,
        records=records,
        source_path=str(fixture_path),
    )
