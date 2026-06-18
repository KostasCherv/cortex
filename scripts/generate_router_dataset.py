# scripts/generate_router_dataset.py
"""Orchestrate: expand seeds → label → split → export train.jsonl + held_out.jsonl."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import httpx

from scripts.finetune.action_seeds import ACTION_SEEDS
from scripts.finetune.label_inputs import label_input

OUTPUT_DIR = Path("data/router_dataset")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
HELD_OUT_PATH = OUTPUT_DIR / "held_out.jsonl"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "qwen3:30b")

_EXPAND_SYSTEM = (
    "You are a data augmentation assistant. Generate {n} realistic paraphrases "
    "of the seed message that should route to the same action: {action}. "
    "Return ONLY a JSON array of strings."
)


def _expand_seed(message: str, action: str, rag_context: str, n: int) -> list[dict[str, str]]:
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": _EXPAND_SYSTEM.format(n=n, action=action)},
            {"role": "user", "content": f"Seed: {message}\nRAG context: {rag_context or 'none'}"},
        ],
        "stream": False,
        "options": {"temperature": 0.7},
        "format": "json",
    }
    try:
        resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        variants = json.loads(resp.json()["message"]["content"])
        if not isinstance(variants, list):
            return []
        return [
            {"message": str(v).strip(), "rag_context": rag_context, "action": action}
            for v in variants
            if isinstance(v, str) and v.strip()
        ]
    except Exception:
        return []


def generate_inputs(*, variants_per_seed: int = 6) -> list[dict[str, str]]:
    """Expand all seeds into the full input list."""
    inputs: list[dict[str, str]] = []
    for action, seeds in ACTION_SEEDS.items():
        for seed in seeds:
            inputs.append({**seed, "action": action})
            inputs.extend(
                _expand_seed(seed["message"], action, seed.get("rag_context", ""), variants_per_seed)
            )
    return inputs


def label_all(inputs: list[dict[str, str]]) -> list[dict]:
    """Label all inputs; silently drop validation failures."""
    records: list[dict] = []
    for i, inp in enumerate(inputs, 1):
        if i % 50 == 0:
            print(f"  labelling {i}/{len(inputs)}...")
        record = label_input(message=inp["message"], rag_context=inp.get("rag_context", ""))
        if record is not None:
            record["_action_label"] = inp.get("action", "")
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
    print("Step 1/4: Expanding seeds...")
    inputs = generate_inputs(variants_per_seed=variants_per_seed)
    print(f"  {len(inputs)} inputs total")

    print("Step 2/4: Labelling with teacher model (this takes a while)...")
    records = label_all(inputs)
    dropped = len(inputs) - len(records)
    print(f"  {len(records)} valid records, {dropped} dropped by validator")

    print("Step 3/4: Splitting train / held-out...")
    train, held_out = split_records(records)
    print(f"  train={len(train)}, held_out={len(held_out)}")

    print("Step 4/4: Writing JSONL...")
    write_jsonl(train, TRAIN_PATH, keep_meta=False)
    write_jsonl(held_out, HELD_OUT_PATH, keep_meta=True)
    print(f"  -> {TRAIN_PATH}")
    print(f"  -> {HELD_OUT_PATH}")


if __name__ == "__main__":
    main()
