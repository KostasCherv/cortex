#!/usr/bin/env python3
"""Sync checked-in golden queries into a LangFuse dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.observability.langfuse_datasets import (
    DEFAULT_DATASET_NAME,
    DEFAULT_GOLDEN_QUERIES_PATH,
    sync_golden_queries_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync golden queries into LangFuse.")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="LangFuse dataset name to create or update.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_GOLDEN_QUERIES_PATH,
        help="Path to the checked-in golden-query artifact.",
    )
    args = parser.parse_args()
    synced = sync_golden_queries_dataset(
        dataset_name=args.dataset_name,
        source_path=args.source,
    )
    print(f"Synced {synced} golden-query item(s) to LangFuse dataset {args.dataset_name}")


if __name__ == "__main__":
    main()
