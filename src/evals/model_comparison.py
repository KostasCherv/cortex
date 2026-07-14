"""Run summarize-node model comparisons against a local golden set."""

# ruff: noqa: E402

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
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.graph.nodes import summarize_node
from src.graph.state import ResearchState

MODEL_CONFIGS = [
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "ollama", "model": "gemma4:31b-cloud"},
]

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.json"
RESULTS_PATH = EVALS_DIR / "results.csv"


PROVIDER_MODEL_SETTING = {
    "openai": "openai_model",
    "ollama": "ollama_model",
}


@contextmanager
def temporary_provider_settings(provider: str, model: str) -> Iterator[None]:
    if provider not in PROVIDER_MODEL_SETTING:
        supported = ", ".join(sorted(PROVIDER_MODEL_SETTING))
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose one of: {supported}."
        )

    model_setting = PROVIDER_MODEL_SETTING[provider]
    original_provider = settings.llm_provider
    original_model = getattr(settings, model_setting)
    settings.llm_provider = provider
    setattr(settings, model_setting, model)
    try:
        yield
    finally:
        settings.llm_provider = original_provider
        setattr(settings, model_setting, original_model)


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


async def run_case(model_config: dict[str, str], case: dict) -> dict[str, object]:
    state: ResearchState = {
        "query": case["query"],
        "retrieved_contents": case["retrieved_contents"],
    }
    provider = model_config["provider"]
    model = model_config["model"]

    with temporary_provider_settings(provider, model):
        started_at = time.perf_counter()
        try:
            result = await summarize_node(state)
        except Exception as exc:
            # Hard cases (e.g. context that cannot answer the query) can make
            # summarize_node raise; record the failure instead of aborting the run.
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            return {
                "model": model,
                "query": case["query"],
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "latency_ms": latency_ms,
                "error": type(exc).__name__,
            }
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
        "model": model,
        "query": case["query"],
        "faithfulness": faithfulness.score,
        "relevancy": answer_relevancy.score,
        "latency_ms": latency_ms,
        "error": "",
    }


async def main() -> None:
    golden_set = load_golden_set()
    rows: list[dict[str, object]] = []

    for model_config in MODEL_CONFIGS:
        for case in golden_set:
            rows.append(await run_case(model_config, case))

    dataframe = pd.DataFrame(
        rows,
        columns=["model", "query", "faithfulness", "relevancy", "latency_ms", "error"],
    )
    dataframe.to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {len(rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
