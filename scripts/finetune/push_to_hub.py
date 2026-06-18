# scripts/finetune/push_to_hub.py
"""Push train / held-out JSONL files to a private HuggingFace Dataset repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def push_to_hub(
    *,
    train_path: Path,
    held_out_path: Path,
    repo_id: str,
    private: bool = True,
) -> None:
    """Push train + held-out splits to HF Hub as a DatasetDict."""
    train_records = _load_jsonl(train_path)
    held_out_records = _load_jsonl(held_out_path)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "held_out": Dataset.from_list(held_out_records),
        }
    )
    dataset_dict.push_to_hub(repo_id, private=private)
    print(
        f"Pushed {len(train_records)} train + {len(held_out_records)} held-out rows to {repo_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/cortex-router-dataset")
    parser.add_argument("--train", default="data/router_dataset/train.jsonl")
    parser.add_argument("--held-out", default="data/router_dataset/held_out.jsonl")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    push_to_hub(
        train_path=Path(args.train),
        held_out_path=Path(getattr(args, "held_out")),
        repo_id=args.repo_id,
        private=not args.public,
    )
