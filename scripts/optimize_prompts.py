#!/usr/bin/env python3
"""Run DSPy prompt optimization against the golden set and print results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent / "src" / "evals"
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.json"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize prompt templates with DSPy MIPROv2"
    )
    parser.add_argument(
        "--module",
        choices=["summarize", "report", "rag_chat_system", "followup_answer", "web_search_decision"],
        default="summarize",
        help="Prompt module to optimize",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 optimization budget (default: light)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_prompts",
        help="Directory to save optimized programs",
    )
    parser.add_argument(
        "--max-bootstrapped",
        type=int,
        default=2,
        help="Max bootstrapped demos (default: 2)",
    )
    parser.add_argument(
        "--max-labeled",
        type=int,
        default=2,
        help="Max labeled demos (default: 2)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Optimization thread count (default: 4)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run A/B comparison after optimization",
    )
    args = parser.parse_args()

    from src.prompts.dspy_optimizer import (
        DspyPromptOptimizer,
        FollowupAnswerModule,
        RagChatSystemModule,
        ReportModule,
        SummarizeModule,
        WebSearchDecisionModule,
        routing_accuracy_metric,
    )

    golden_set = load_golden_set()
    print(f"Loaded {len(golden_set)} golden cases from {GOLDEN_SET_PATH}")

    MODULES = {
        "summarize": SummarizeModule,
        "report": ReportModule,
        "rag_chat_system": RagChatSystemModule,
        "followup_answer": FollowupAnswerModule,
        "web_search_decision": WebSearchDecisionModule,
    }
    MODULE_METRICS = {
        "web_search_decision": routing_accuracy_metric,
    }
    module_cls = MODULES[args.module]
    module = module_cls()
    optimizer = DspyPromptOptimizer(output_dir=args.output)
    metric = MODULE_METRICS.get(args.module)

    print(f"\nOptimizing {args.module} module (auto={args.auto}, "
          f"bootstrapped={args.max_bootstrapped}, labeled={args.max_labeled})...")
    start = time.perf_counter()

    result = optimizer.optimize(
        module,
        golden_set,
        module_type=args.module,
        metric=metric,
        max_bootstrapped_demos=args.max_bootstrapped,
        max_labeled_demos=args.max_labeled,
        auto=args.auto,
        num_threads=args.threads,
    )

    elapsed = time.perf_counter() - start
    output_name = f"{args.module}_optimized"

    if result.optimized_program is not None:
        saved = optimizer.save(result, output_name)
    else:
        saved = Path(args.output) / f"{output_name}.json"

    print(f"\n{'='*50}")
    print(f"  Module:       {result.module_type}")
    print(f"  Score before: {result.before_score:.4f}")
    print(f"  Score after:  {result.after_score:.4f}")
    print(f"  Improvement:  {result.improvement:+.4f}")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Saved to:     {saved}")
    print(f"{'='*50}")

    if args.compare:
        print("\nA/B comparison per golden case:")
        print(f"{'Query':<50} {'Original':>10} {'Optimized':>10}")
        print("-" * 72)
        comparisons = optimizer.compare(
            module,
            saved,
            golden_set,
            args.module,
            metric=metric,
        )
        for c in comparisons:
            query = c["query"][:47] + "..." if len(c["query"]) > 50 else c["query"]
            print(f"{query:<50} {c['original_score']:>10.4f} {c['optimized_score']:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
