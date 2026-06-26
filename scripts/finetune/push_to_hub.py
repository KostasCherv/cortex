# scripts/finetune/push_to_hub.py
"""Push train / held-out JSONL files to a private HuggingFace Dataset repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


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
    """Push train + held-out to HF Hub as two configs.

    The two files have different schemas (train is messages-only; held_out also
    carries action_label + verified for scoring), so they are pushed as separate
    configs rather than splits of a single config -- a DatasetDict requires all
    splits to share identical features. The train data uses config_name="default"
    so that load_dataset(repo)["train"] resolves without passing a config name
    (the training notebook depends on this).
    """
    train_records = _load_jsonl(train_path)
    held_out_records = _load_jsonl(held_out_path)

    train_ds = Dataset.from_list(train_records)
    held_out_ds = Dataset.from_list(held_out_records)

    train_ds.push_to_hub(repo_id, config_name="default", split="train", private=private)
    held_out_ds.push_to_hub(repo_id, config_name="held_out", split="held_out", private=private)
    print(
        f"Pushed {len(train_records)} train (config=default) + "
        f"{len(held_out_records)} held-out (config=held_out) rows to {repo_id}"
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
