"""Run summarize-node model comparisons against a local golden set."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.config import settings
from src.graph.nodes import summarize_node
from src.graph.state import ResearchState

MODEL_SLUGS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
]

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.json"
RESULTS_PATH = EVALS_DIR / "results.csv"


@contextmanager
def temporary_openrouter_settings(model_slug: str) -> Iterator[None]:
    original_provider = settings.llm_provider
    original_model = settings.openrouter_model
    settings.llm_provider = "openrouter"
    settings.openrouter_model = model_slug
    try:
        yield
    finally:
        settings.llm_provider = original_provider
        settings.openrouter_model = original_model


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_test_case(case: dict, actual_output: str) -> LLMTestCase:
    retrieval_context = [
        str(item.get("raw_text", "")).strip()
        for item in case["retrieved_contents"]
        if str(item.get("raw_text", "")).strip()
    ]
    return LLMTestCase(
        input=case["query"],
        actual_output=actual_output,
        expected_output=case["expected_answer"],
        retrieval_context=retrieval_context,
    )


async def run_case(model_slug: str, case: dict) -> dict[str, object]:
    state: ResearchState = {
        "query": case["query"],
        "retrieved_contents": case["retrieved_contents"],
    }

    with temporary_openrouter_settings(model_slug):
        started_at = time.perf_counter()
        result = await summarize_node(state)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    actual_output = "\n\n".join(
        row["summary"] for row in result.get("summaries", []) if row.get("summary")
    )
    test_case = build_test_case(case, actual_output)

    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()
    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)

    return {
        "model": model_slug,
        "query": case["query"],
        "faithfulness": faithfulness.score,
        "relevancy": answer_relevancy.score,
        "latency_ms": latency_ms,
    }


async def main() -> None:
    golden_set = load_golden_set()
    rows: list[dict[str, object]] = []

    for model_slug in MODEL_SLUGS:
        for case in golden_set:
            rows.append(await run_case(model_slug, case))

    dataframe = pd.DataFrame(
        rows,
        columns=["model", "query", "faithfulness", "relevancy", "latency_ms"],
    )
    dataframe.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {len(rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
