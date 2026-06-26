# scripts/generate_router_dataset.py
"""Orchestrate: expand seeds → label → split → export train.jsonl + held_out.jsonl."""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from scripts.finetune.action_seeds import ACTION_SEEDS
from scripts.finetune.label_inputs import label_input
from scripts.finetune.teacher_client import call_teacher
from src.llm.output_parsers import extract_json_candidate

load_dotenv()

OUTPUT_DIR = Path("data/router_dataset")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
HELD_OUT_PATH = OUTPUT_DIR / "held_out.jsonl"

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "qwen3:30b")
# Number of concurrent teacher requests (HTTP is I/O-bound; the backend may still
# serialize on a single GPU, so keep this modest).
TEACHER_CONCURRENCY = max(1, int(os.getenv("TEACHER_CONCURRENCY", "4")))

_EXPAND_SYSTEM = (
    "You are a data augmentation assistant. Generate {n} realistic paraphrases "
    "of the seed message that should route to the same action: {action}. "
    "Return ONLY a JSON array of strings."
)


def _expand_seed(message: str, action: str, rag_context: str, n: int) -> list[dict[str, str]]:
    try:
        content = call_teacher(
            messages=[
                {"role": "system", "content": _EXPAND_SYSTEM.format(n=n, action=action)},
                {"role": "user", "content": f"Seed: {message}\nRAG context: {rag_context or 'none'}"},
            ],
            model=TEACHER_MODEL,
            temperature=0.7,
            json_mode=True,
        )
        variants = json.loads(extract_json_candidate(content))
        if not isinstance(variants, list):
            return []
        return [
            {"message": str(v).strip(), "rag_context": rag_context, "action": action}
            for v in variants
            if isinstance(v, str) and v.strip()
        ]
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
        print(f"[expand_seed] failed for action={action!r}: {exc}", flush=True)
        return []


def generate_inputs(*, variants_per_seed: int = 6) -> list[dict[str, str]]:
    """Expand all seeds (concurrently) into the full input list."""
    seed_tasks = [(seed, action) for action, seeds in ACTION_SEEDS.items() for seed in seeds]

    # Always keep the verbatim seeds, regardless of expansion success.
    inputs: list[dict[str, str]] = [{**seed, "action": action} for seed, action in seed_tasks]

    with ThreadPoolExecutor(max_workers=TEACHER_CONCURRENCY) as pool:
        futures = [
            pool.submit(
                _expand_seed, seed["message"], action, seed.get("rag_context", ""), variants_per_seed
            )
            for seed, action in seed_tasks
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Expanding seeds", unit="seed"
        ):
            inputs.extend(fut.result())
    return inputs


def deduplicate_inputs(inputs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop inputs whose (message, rag_context) pair has already been seen.

    Without this, the same message can appear in both train and held-out if
    _expand_seed happens to regenerate a variant that matches a seed verbatim,
    or if two seeds produce overlapping paraphrases.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for inp in inputs:
        key = (inp["message"].strip().lower(), inp.get("rag_context", "").strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(inp)
    return unique


def _label_one(inp: dict[str, str]) -> dict | None:
    """Label a single input, tagging the source action on success."""
    record = label_input(
        message=inp["message"],
        rag_context=inp.get("rag_context", ""),
        ollama_model=TEACHER_MODEL,
    )
    if record is not None:
        record["_action_label"] = inp.get("action", "")
    return record


def label_all(inputs: list[dict[str, str]]) -> list[dict]:
    """Label all inputs concurrently; silently drop validation failures."""
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=TEACHER_CONCURRENCY) as pool:
        futures = [pool.submit(_label_one, inp) for inp in inputs]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Labelling", unit="ex"
        ):
            record = fut.result()
            if record is not None:
                records.append(record)
    return records


def split_records(
    records: list[dict],
    *,
    held_out_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified split by action class into train and held-out sets."""
    by_action: dict[str, list[dict]] = {}
    for r in records:
        key = r.get("_action_label", "unknown")
        by_action.setdefault(key, []).append(r)

    train: list[dict] = []
    held_out: list[dict] = []
    rng = random.Random(seed)
    for class_records in by_action.values():
        rng.shuffle(class_records)
        n_held_out = max(1, int(len(class_records) * held_out_fraction))
        held_out.extend(class_records[:n_held_out])
        train.extend(class_records[n_held_out:])

    return train, held_out


def write_jsonl(records: list[dict], path: Path, *, keep_meta: bool) -> None:
    """Write records to JSONL.

    keep_meta=True renames _action_label → action_label for the held-out scoring set.
    keep_meta=False strips all underscore-prefixed internal keys for the training set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            if keep_meta:
                row = {(k.lstrip("_") if k.startswith("_") else k): v for k, v in r.items()}
            else:
                row = {k: v for k, v in r.items() if not k.startswith("_")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(*, variants_per_seed: int = 6) -> None:
    print(f"Teacher: {TEACHER_MODEL} (concurrency={TEACHER_CONCURRENCY})")
    print("Step 1/4: Expanding seeds...")
    inputs = generate_inputs(variants_per_seed=variants_per_seed)
    before = len(inputs)
    inputs = deduplicate_inputs(inputs)
    print(f"  {len(inputs)} inputs ({before - len(inputs)} duplicates removed)")

    print("Step 2/4: Labelling with teacher model (this takes a while)...")
    records = label_all(inputs)
    dropped = len(inputs) - len(records)
    print(f"  {len(records)} valid records, {dropped} dropped by validator")

    print("Step 3/4: Splitting train / held-out...")
    train, held_out = split_records(records)
    print(f"  train={len(train)}, held_out={len(held_out)}")

    print("Step 4/4: Writing JSONL...")
    write_jsonl(train, TRAIN_PATH, keep_meta=False)
    for r in held_out:
        r["_verified"] = False
    write_jsonl(held_out, HELD_OUT_PATH, keep_meta=True)
    print(f"  -> {TRAIN_PATH}")
    print(f"  -> {HELD_OUT_PATH}")


if __name__ == "__main__":
    main()
