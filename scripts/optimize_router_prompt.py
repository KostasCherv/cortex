#!/usr/bin/env python3
"""Optimize the production router system prompt with DSPy MIPROv2.

Unlike the generic template pipeline, this targets the actual production
router (`classify_chat_action` in src/api/rag_chat_helpers.py) and uses the
real teacher-labeled router dataset:

- trains on a stratified subsample of data/router_dataset/train.jsonl
- scores BEFORE and AFTER on data/router_dataset/held_out.jsonl through the
  production prompt + parse_chat_action_json path (not DSPy's adapter), so
  the reported delta is exactly what production would see
- writes optimized_prompts/router_action_optimized.json, loadable in
  production via ROUTER_PROMPT_PATH

CAVEAT (always repeat when reporting numbers): held-out labels are generated
by the teacher model (Qwen3-30B), not humans. Scores measure teacher
agreement, not ground-truth accuracy, until records carry verified=true.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TRAIN_PATH = Path("data/router_dataset/train.jsonl")
HELD_OUT_PATH = Path("data/router_dataset/held_out.jsonl")
OUTPUT_PATH = Path("optimized_prompts/router_action_optimized.json")
SEED = 13
CONCURRENCY = 8


def load_records(path: Path) -> list[dict]:
    """Return [{user_turn, gold_action, gold_json}] from a router JSONL file."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        user_turn = next(m["content"] for m in raw["messages"] if m["role"] == "user")
        gold_json = next(m["content"] for m in raw["messages"] if m["role"] == "assistant")
        action = raw.get("action_label") or json.loads(gold_json)["action"]
        records.append({"user_turn": user_turn, "gold_action": action, "gold_json": gold_json})
    return records


def stratified_sample(records: list[dict], per_label: int) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_label[rec["gold_action"]].append(rec)
    rng = random.Random(SEED)
    sample: list[dict] = []
    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        sample.extend(group[:per_label])
    rng.shuffle(sample)
    return sample


async def score_prompt(system_prompt: str, records: list[dict]) -> dict:
    """Score a system prompt on records through the production LLM + parser path."""
    from src.llm.factory import get_router_llm
    from src.llm.output_parsers import parse_chat_action_json
    from src.llm.text_utils import extract_llm_text

    llm = get_router_llm()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(rec: dict) -> dict:
        prompt = f"{system_prompt}\n\n{rec['user_turn']}"
        async with semaphore:
            try:
                response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=30.0)
                predicted = parse_chat_action_json(extract_llm_text(response)).action
            except Exception as exc:
                return {"gold": rec["gold_action"], "predicted": None, "error": type(exc).__name__}
        return {"gold": rec["gold_action"], "predicted": predicted, "error": None}

    results = await asyncio.gather(*(one(rec) for rec in records))
    hits = sum(1 for r in results if r["predicted"] == r["gold"])
    per_label: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in results:
        per_label[r["gold"]]["total"] += 1
        if r["predicted"] == r["gold"]:
            per_label[r["gold"]]["hits"] += 1
    return {
        "agreement": round(hits / len(results), 4),
        "n": len(results),
        "parse_failures": sum(1 for r in results if r["error"]),
        "per_label": {
            label: round(v["hits"] / v["total"], 4) for label, v in sorted(per_label.items())
        },
    }


def optimize(train_records: list[dict], seed_prompt: str, auto: str) -> tuple[str, object]:
    """Run MIPROv2 over the trainset; return (composed_system_prompt, program)."""
    import dspy
    from dspy.teleprompt import MIPROv2

    from src.llm.output_parsers import parse_chat_action_json
    from src.prompts.dspy_optimizer import create_lm_from_settings

    lm = create_lm_from_settings()
    if lm is None:
        raise SystemExit("No LM configured (set OPENAI_API_KEY / LLM_PROVIDER).")
    dspy.configure(lm=lm)

    signature = dspy.Signature(
        {"user_turn": dspy.InputField(desc="User message plus available context lines"),
         "decision_json": dspy.OutputField(desc="The routing decision as a single JSON object")},
        seed_prompt,
    )
    module = dspy.Predict(signature)

    def metric(example, prediction, trace=None) -> float:
        try:
            action = parse_chat_action_json(prediction.decision_json).action
        except Exception:
            return 0.0
        return 1.0 if action == example.gold_action else 0.0

    trainset = [
        dspy.Example(
            user_turn=rec["user_turn"],
            decision_json=rec["gold_json"],
            gold_action=rec["gold_action"],
        ).with_inputs("user_turn")
        for rec in train_records
    ]

    optimizer = MIPROv2(metric=metric, auto=auto, num_threads=CONCURRENCY, seed=SEED)
    program = optimizer.compile(
        module, trainset=trainset, max_bootstrapped_demos=3, max_labeled_demos=3
    )

    instructions = program.signature.instructions.strip()
    demo_blocks = []
    for demo in program.demos:
        user_turn = getattr(demo, "user_turn", None)
        decision = getattr(demo, "decision_json", None)
        if user_turn and decision:
            demo_blocks.append(f"{user_turn.strip()}\n{decision.strip()}")
    composed = instructions
    if demo_blocks:
        composed += "\n\nExamples:\n\n" + "\n\n".join(demo_blocks)
    composed += "\n\nOutput ONLY the JSON object for the next user message."
    return composed, program


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--per-label", type=int, default=20, help="Train cases per action label")
    parser.add_argument("--eval-only", action="store_true", help="Only score the current prompt")
    args = parser.parse_args()

    from src.api.rag_chat_helpers import _ROUTER_ACTION_SYSTEM_PROMPT

    held_out = load_records(HELD_OUT_PATH)
    print(f"Held-out: {len(held_out)} cases (teacher labels, unverified)")

    print("Scoring current production prompt on held-out…")
    before = asyncio.run(score_prompt(_ROUTER_ACTION_SYSTEM_PROMPT, held_out))
    print(f"  before: {before['agreement']:.4f} ({before['parse_failures']} parse failures)")
    print(f"  per label: {before['per_label']}")

    if args.eval_only:
        return

    train = stratified_sample(load_records(TRAIN_PATH), args.per_label)
    print(f"Optimizing with MIPROv2 (auto={args.auto}) on {len(train)} train cases…")
    start = time.perf_counter()
    composed_prompt, _ = optimize(train, _ROUTER_ACTION_SYSTEM_PROMPT, args.auto)
    elapsed = time.perf_counter() - start

    print("Scoring optimized prompt on held-out…")
    after = asyncio.run(score_prompt(composed_prompt, held_out))
    print(f"  after: {after['agreement']:.4f} ({after['parse_failures']} parse failures)")
    print(f"  per label: {after['per_label']}")

    improvement = round(after["agreement"] - before["agreement"], 4)
    artifact = {
        "system_prompt": composed_prompt,
        "metric": "held_out_teacher_agreement",
        "caveat": "Labels are Qwen3-30B teacher labels, not human-verified ground truth.",
        "score_before": before["agreement"],
        "score_after": after["agreement"],
        "improvement": improvement,
        "per_label_before": before["per_label"],
        "per_label_after": after["per_label"],
        "parse_failures_before": before["parse_failures"],
        "parse_failures_after": after["parse_failures"],
        "n_held_out": len(held_out),
        "n_train": len(train),
        "optimizer": f"MIPROv2 auto={args.auto} seed={SEED}",
        "duration_s": round(elapsed, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nArtifact: {OUTPUT_PATH}  (improvement: {improvement:+.4f})")
    verdict = (
        f"Activate with ROUTER_PROMPT_PATH={OUTPUT_PATH}"
        if improvement > 0
        else "No improvement — keep the built-in prompt (artifact retained for the record)."
    )
    print(verdict)


if __name__ == "__main__":
    main()
