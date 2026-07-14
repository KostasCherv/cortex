"""Fast, credential-free regression checks for deterministic AI boundaries."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.deps import _select_chat_citations
from src.llm.output_parsers import (
    parse_chat_action_json,
    parse_finance_tool_call_plan_json,
    parse_finance_tool_selection_json,
)

DEFAULT_DATASET = Path(__file__).with_name("ai_regression_set.json")


def _parse_json(payload: dict[str, Any], parser):
    try:
        return parser(json.dumps(payload)), None
    except Exception as exc:  # the contract is pass/fail, regardless of validation subtype
        return None, type(exc).__name__


def _evaluate_router(case: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    parsed, error = _parse_json(case["response"], parse_chat_action_json)
    actual: dict[str, Any] = {"valid": parsed is not None}
    if parsed is not None:
        actual.update(action=parsed.action, query=parsed.query, symbols=parsed.symbols)
    if error:
        actual["error_type"] = error
    expected = case["expected"]
    return all(actual.get(key) == value for key, value in expected.items()), actual


def _evaluate_citation(case: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    data = case["input"]
    citations = _select_chat_citations(
        data["rag_chunks"],
        data["loop_citations"],
        web_used=data["web_used"],
        rag_context_text=data["rag_context_text"],
    )
    actual = {
        "chunk_ids": [item.get("chunk_id") for item in citations],
        "source_types": [item.get("source_type") for item in citations],
    }
    return actual == case["expected"], actual


def _evaluate_tool_selection(case: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    parser = (
        parse_finance_tool_selection_json
        if case["stage"] == "selection"
        else parse_finance_tool_call_plan_json
    )
    parsed, error = _parse_json(case["response"], parser)
    actual: dict[str, Any] = {"valid": parsed is not None}
    if parsed is not None:
        for field in ("tool_name", "should_call", "arguments"):
            if hasattr(parsed, field):
                actual[field] = getattr(parsed, field)
    if error:
        actual["error_type"] = error
    expected = case["expected"]
    return all(actual.get(key) == value for key, value in expected.items()), actual


EVALUATORS = {
    "router": _evaluate_router,
    "citation": _evaluate_citation,
    "tool_selection": _evaluate_tool_selection,
}


def _git_sha() -> str:
    return os.getenv("GITHUB_SHA", "local")


def evaluate(dataset_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = []
    for case in dataset["cases"]:
        case_passed, actual = EVALUATORS[case["category"]](case)
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": case_passed,
                "actual": actual,
            }
        )

    totals = Counter(item["category"] for item in results)
    passed_counts = Counter(item["category"] for item in results if item["passed"])
    category_scores = {
        category: round(passed_counts[category] / total, 4)
        for category, total in sorted(totals.items())
    }
    return {
        "schema_version": 1,
        "dataset_version": dataset["version"],
        "commit_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "passed_count": sum(item["passed"] for item in results),
        "score": round(sum(item["passed"] for item in results) / len(results), 4),
        "category_scores": category_scores,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("reports/ai-regression.json"))
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    report = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AI regression score: {report['passed_count']}/{report['case_count']} "
        f"({report['score']:.0%}); report: {args.output}"
    )
    failures = [item["id"] for item in report["results"] if not item["passed"]]
    if failures:
        print("Failed cases: " + ", ".join(failures))
    category_failed = any(score < args.threshold for score in report["category_scores"].values())
    return 1 if report["score"] < args.threshold or category_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
