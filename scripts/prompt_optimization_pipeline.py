#!/usr/bin/env python3
"""Production-grade prompt optimization pipeline.

Scans every Jinja2 template in src/prompts/, analyzes its purpose and
inputs, creates DSPy modules, runs MIPROv2 optimization with DeepEval
metrics, and generates a before/after comparison report.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "prompts"
EVALS_DIR = Path(__file__).resolve().parent.parent / "src" / "evals"
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.json"
OUTPUT_DIR = Path("optimized_prompts")


@dataclass
class PromptSpec:
    """Specification for a prompt template — its purpose, inputs, output."""

    template_name: str  # e.g. "summarize"
    subject: str  # e.g. "summarize retrieved sources"
    goal: str  # what the prompt is supposed to achieve
    input_fields: list[tuple[str, str, str]]  # (name, type_description, description)
    output_fields: list[tuple[str, str, str]]  # (name, type_description, description)
    golden_mapping: str  # how golden set maps: "summarize" | "report" | "rag" | "router"
    template_source: str = ""


def analyze_template(path: Path) -> PromptSpec | None:
    """Read a Jinja2 template and build its PromptSpec."""
    source = path.read_text(encoding="utf-8")
    name = path.stem  # removes .j2
    # Extract Jinja2 variable references (handles {{ var }} and {{ var or 'default' }})
    variables = set(re.findall(r"{{\s*(\w+)", source))

    specs: dict[str, PromptSpec] = {
        "summarize": PromptSpec(
            template_name="summarize",
            subject="summarize retrieved sources",
            goal="Generate structured JSON summaries of retrieved sources relevant to a research query, preserving source URLs and focusing on factual claims.",
            input_fields=[
                ("query", "str", "The user's research question"),
                ("source_blocks", "str", "Formatted source blocks with URL, title, and content"),
                ("domain", "str", "Optional domain or topic hint"),
            ],
            output_fields=[
                ("summaries", "str", "JSON array of summaries, each with url, title, and summary"),
            ],
            golden_mapping="summarize",
            template_source=source,
        ),
        "report": PromptSpec(
            template_name="report",
            subject="generate research report",
            goal="Produce a polished markdown research report with executive summary, key findings, and conclusion consolidated from all source summaries.",
            input_fields=[
                ("query", "str", "The user's research question"),
                ("summaries_text", "str", "Consolidated summaries text from all sources"),
                ("memory_context", "str", "Context from previous research sessions"),
                ("domain", "str", "Optional domain or topic hint"),
            ],
            output_fields=[
                ("report", "str", "Polished markdown research report"),
            ],
            golden_mapping="report",
            template_source=source,
        ),
        "rag_chat_system": PromptSpec(
            template_name="rag_chat_system",
            subject="RAG chat answer",
            goal="Answer a user question grounded in retrieved document context and web search results, staying faithful to the provided sources.",
            input_fields=[
                ("system_instructions", "str", "Custom system instructions for the assistant"),
                ("rag_context", "str", "Retrieved document context relevant to the question"),
                ("web_results_json", "str", "Web search results in JSON format"),
            ],
            output_fields=[
                ("answer", "str", "Clear answer grounded in the provided context"),
            ],
            golden_mapping="summarize",
            template_source=source,
        ),
        "followup_answer": PromptSpec(
            template_name="followup_answer",
            subject="follow-up research answer",
            goal="Answer a follow-up question grounded in the existing research report, conversation history, retrieved passages, and web search context.",
            input_fields=[
                ("report_block", "str", "The research report's main findings"),
                ("history_block", "str", "Conversation history so far"),
                ("answer_context_block", "str", "Retrieved source passages"),
                ("web_results_json", "str", "Web search results in JSON format"),
                ("question", "str", "The user's follow-up question"),
            ],
            output_fields=[
                ("answer", "str", "Concise answer grounded in report, passages, and web context"),
            ],
            golden_mapping="summarize",
            template_source=source,
        ),
        "web_search_decision": PromptSpec(
            template_name="web_search_decision",
            subject="route assistant action",
            goal="Route the next assistant action (answer_direct, answer_from_rag, web_search, fetch_url, ask_clarifying) based on conversation history, RAG context, and the user message.",
            input_fields=[
                ("history_block", "str", "Conversation history"),
                ("rag_context", "str", "Retrieved RAG context"),
                ("rag_is_insufficient", "str", "Whether RAG context is insufficient (true/false)"),
                ("message_urls", "str", "URLs in the current user message"),
                ("history_urls", "str", "URLs in conversation history"),
                ("message", "str", "The user's current message"),
            ],
            output_fields=[
                ("action", "str", "One of: answer_direct, answer_from_rag, web_search, fetch_url, ask_clarifying"),
                ("reason", "str", "Short snake_case explanation"),
                ("query", "str", "Search query (only for web_search)"),
                ("url", "str", "URL to fetch (only for fetch_url)"),
            ],
            golden_mapping="router",
            template_source=source,
        ),
        "web_search_repair": PromptSpec(
            template_name="web_search_repair",
            subject="repair URL access refusal",
            goal="Repair when the model falsely claims it cannot access URLs — rewrite the answer using already-retrieved web content.",
            input_fields=[
                ("normalized_message", "str", "The user's original request"),
                ("web_results_json", "str", "Already-retrieved web content"),
                ("rag_context", "str", "Retrieved RAG context"),
            ],
            output_fields=[
                ("answer", "str", "Helpful answer using the retrieved content"),
            ],
            golden_mapping="summarize",
            template_source=source,
        ),
    }

    result = specs.get(name)
    if result is None:
        return None
    result.template_source = source
    return result


# ── Metrics ──────────────────────────────────────────────────────────────────


def _safe_deepeval_measure(metric, input_text: str, output: str, context: list[str] | None = None) -> float:
    """Call a DeepEval metric using LLMTestCase and return 0 on any failure."""
    try:
        from deepeval.test_case import LLMTestCase

        tc = LLMTestCase(input=input_text, actual_output=output)
        if context is not None:
            tc.retrieval_context = context
        metric.measure(tc)
        return float(metric.score)
    except Exception:
        return 0.0


def build_deepeval_metrics() -> tuple[Callable | None, str]:
    """Build combined DeepEval metric if deepeval is available and configured."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

        faith = FaithfulnessMetric()
        relev = AnswerRelevancyMetric()
        name = "DeepEval Faithfulness+Relevancy"

        def combined(example, prediction, trace=None):
            output = (
                getattr(prediction, "summaries", None)
                or getattr(prediction, "report", None)
                or getattr(prediction, "answer", None)
                or ""
            )
            input_text = (
                getattr(example, "query", "")
                or getattr(example, "question", "")
                or getattr(example, "message", "")
                or ""
            )
            context = []
            for attr in ("source_blocks", "rag_context", "answer_context_block"):
                val = getattr(example, attr, None)
                if val and val.strip() and val != "None":
                    context.append(val)
            f = _safe_deepeval_measure(faith, input_text, output, context if context else None)
            r = _safe_deepeval_measure(relev, input_text, output)
            if f == 0 and r == 0:
                return 0.0
            return (f + r) / 2

        return combined, name
    except ImportError:
        return None, "deepeval_not_installed"
    except Exception:
        return None, "deepeval_error"


def routing_accuracy_metric(example, prediction, trace=None) -> float:
    """Score router output: valid action + proper field usage."""
    VALID_ACTIONS = {"answer_direct", "answer_from_rag", "web_search", "fetch_url", "ask_clarifying"}
    action = getattr(prediction, "action", None) or ""
    reason = getattr(prediction, "reason", None) or ""
    query = getattr(prediction, "query", None) or ""
    url = getattr(prediction, "url", None) or ""

    if action not in VALID_ACTIONS:
        return 0.0
    score = 0.4
    if reason.strip():
        score += 0.2
    if action == "web_search" and query.strip():
        score += 0.4
    elif action == "fetch_url" and url.strip():
        score += 0.4
    elif action in ("answer_direct", "answer_from_rag", "ask_clarifying") and not query.strip() and not url.strip():
        score += 0.4
    return score


# ── Pipeline ─────────────────────────────────────────────────────────────────


@dataclass
class PromptAnalysis:
    """Analysis result for a single prompt template."""

    spec: PromptSpec
    variables_found: set[str]
    matches_spec: bool
    missing_vars: list[str]
    extra_vars: list[str]


@dataclass
class OptimizationRun:
    """Result of optimizing a single prompt."""

    template_name: str
    subject: str
    goal: str
    score_before: float
    score_after: float
    improvement: float
    metric_name: str
    duration_seconds: float
    optimized_path: Path | None
    per_case: list[dict[str, Any]]


def analyze_prompt(spec: PromptSpec) -> PromptAnalysis:
    """Check if the actual Jinja2 template matches its spec."""
    source = spec.template_source
    variables = set(re.findall(r"{{\s*(\w+)", source))
    spec_inputs = {name for name, _, _ in spec.input_fields}
    missing = sorted(spec_inputs - variables)
    extra = sorted(variables - spec_inputs)
    return PromptAnalysis(
        spec=spec,
        variables_found=variables,
        matches_spec=len(missing) == 0 and len(extra) == 0,
        missing_vars=missing,
        extra_vars=extra,
    )


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_metric_for(module_type: str, use_deepeval: bool = True) -> tuple[Callable, str]:
    """Pick the best available metric for a module type. Returns (metric_fn, name)."""
    if module_type == "web_search_decision":
        return routing_accuracy_metric, "routing_accuracy"

    if use_deepeval:
        metric, name = build_deepeval_metrics()
        if metric is not None:
            return metric, name

    return _default_overlap_metric, "word_overlap"


def _default_overlap_metric(example, prediction, trace=None) -> float:
    predicted = (
        getattr(prediction, "summaries", None)
        or getattr(prediction, "report", None)
        or getattr(prediction, "answer", None)
        or getattr(prediction, "action", None)
        or ""
    )
    expected = getattr(example, "expected_output", None) or ""
    if not predicted or not expected:
        return 0.0
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    return min(overlap * 2.0, 1.0)


def build_module(spec: PromptSpec):
    """Dynamically create a DSPy module for a prompt spec."""
    import dspy

    input_str = ", ".join(name for name, _, _ in spec.input_fields)
    output_str = ", ".join(name for name, _, _ in spec.output_fields)
    signature_def = f"{input_str} -> {output_str}"

    class DynamicModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(signature_def)

        def forward(self, **kwargs) -> dspy.Prediction:
            return self.generate(**kwargs)

    DynamicModule.__name__ = f"{spec.template_name.title().replace('_', '')}Module"
    return DynamicModule()


def build_examples(spec: PromptSpec, golden_set: list[dict]) -> list[Any]:
    """Build DSPy training examples from the golden set."""
    import dspy

    examples: list[dspy.Example] = []
    mapping = spec.golden_mapping

    for case in golden_set:
        kwargs = {}
        input_keys = [name for name, _, _ in spec.input_fields]
        source_text = "\n\n".join(
            f"SOURCE URL: {s.get('url', '')}\n"
            f"SOURCE TITLE: {s.get('title', '')}\n"
            f"CONTENT:\n{s.get('raw_text', '')}"
            for s in case.get("retrieved_contents", [])
        )

        if mapping == "summarize":
            for key in input_keys:
                if key == "query":
                    kwargs["query"] = case["query"]
                elif key == "source_blocks":
                    kwargs["source_blocks"] = source_text
                elif key == "domain":
                    kwargs["domain"] = ""
                elif key == "system_instructions":
                    kwargs["system_instructions"] = ""
                elif key == "rag_context":
                    kwargs["rag_context"] = source_text
                elif key == "web_results_json":
                    kwargs["web_results_json"] = "[]"
                elif key == "report_block":
                    kwargs["report_block"] = ""
                elif key == "history_block":
                    kwargs["history_block"] = ""
                elif key == "answer_context_block":
                    kwargs["answer_context_block"] = source_text
                elif key == "question":
                    kwargs["question"] = case["query"]
                elif key == "normalized_message":
                    kwargs["normalized_message"] = case["query"]
                elif key == "summaries_text":
                    kwargs["summaries_text"] = "\n\n".join(
                        str(s.get("raw_text", "")) for s in case.get("retrieved_contents", [])
                    )
                elif key == "memory_context":
                    kwargs["memory_context"] = ""
                else:
                    kwargs[key] = ""

            output_keys = [name for name, _, _ in spec.output_fields]
            for key in output_keys:
                kwargs[key] = case.get("expected_answer", "")

        elif mapping == "router":
            for key in input_keys:
                if key == "message":
                    kwargs["message"] = case["query"]
                elif key == "rag_context":
                    kwargs["rag_context"] = "None"
                elif key == "history_block":
                    kwargs["history_block"] = "None"
                elif key == "rag_is_insufficient":
                    kwargs["rag_is_insufficient"] = "false"
                elif key == "message_urls":
                    kwargs["message_urls"] = "None"
                elif key == "history_urls":
                    kwargs["history_urls"] = "None"
                else:
                    kwargs[key] = ""

            kwargs["action"] = "answer_from_rag"
            kwargs["reason"] = "context_is_relevant"
            kwargs["query"] = ""
            kwargs["url"] = ""

        kwargs["expected_output"] = case.get("expected_answer", "")
        example = dspy.Example(**kwargs).with_inputs(*input_keys)
        examples.append(example)

    return examples


def estimate_quality(spec: PromptSpec, golden_set: list[dict], metric: Callable, module) -> dict[str, Any]:
    """Score a module against the golden set without optimization."""
    import dspy

    examples = build_examples(spec, golden_set)
    total = 0.0
    per_case = []
    for i, ex in enumerate(examples):
        try:
            pred = module(**ex.inputs())
            score = metric(ex, pred)
        except Exception:
            score = 0.0
        total += score
        per_case.append({
            "case": i,
            "query": ex.get("query", ex.get("question", ex.get("message", ""))),
            "score": round(score, 4),
        })
    return {
        "average_score": round(total / len(examples), 4) if examples else 0.0,
        "per_case": per_case,
    }


def run_pipeline(
    use_deepeval: bool = True,
    auto: str = "light",
    skip_optimization: bool = False,
    template_name: str | None = None,
) -> list[OptimizationRun]:
    """Run the full prompt optimization pipeline for all templates."""
    import dspy
    from dspy.teleprompt import MIPROv2

    from src.prompts.dspy_optimizer import create_lm_from_settings

    lm = create_lm_from_settings()
    if lm is None:
        print("  ⚠️  No LM configured — skipping DSPy optimization")
        return []
    dspy.configure(lm=lm)

    golden_set = load_golden_set()
    results: list[OptimizationRun] = []

    template_paths = sorted(PROMPTS_DIR.glob("*.j2"))
    if template_name:
        template_paths = [path for path in template_paths if path.stem == template_name]
    print(f"Found {len(template_paths)} Jinja2 templates in {PROMPTS_DIR}")
    print(f"Golden set: {len(golden_set)} cases\n")

    for tpl_path in template_paths:
        spec = analyze_template(tpl_path)
        if spec is None:
            print(f"  ⏭️  {tpl_path.name} — no spec defined, skipping")
            continue

        # ── Phase 1: Analyze ──
        analysis = analyze_prompt(spec)
        print(f"\n{'='*60}")
        print(f"📄 {spec.template_name}.j2")
        print(f"   Subject: {spec.subject}")
        print(f"   Goal: {spec.goal}")
        print(f"   Template vars: {sorted(analysis.variables_found)}")
        if not analysis.matches_spec:
            if analysis.missing_vars:
                print(f"   ⚠️  Missing expected vars: {analysis.missing_vars}")
            if analysis.extra_vars:
                print(f"   ℹ️  Extra vars found: {analysis.extra_vars}")

        # ── Phase 2: Build module ──
        module = build_module(spec)
        metric, metric_name = build_metric_for(spec.template_name, use_deepeval)

        # ── Phase 3: Score before optimization ──
        before = estimate_quality(spec, golden_set, metric, module)
        print(f"   📊 Before optimization: {before['average_score']:.4f} ({metric_name})")

        if skip_optimization:
            results.append(OptimizationRun(
                template_name=spec.template_name,
                subject=spec.subject,
                goal=spec.goal,
                score_before=before["average_score"],
                score_after=before["average_score"],
                improvement=0.0,
                metric_name=metric_name,
                duration_seconds=0.0,
                optimized_path=None,
                per_case=before["per_case"],
            ))
            continue

        # ── Phase 4: Optimize ──
        print(f"   🔧 Running MIPROv2 (auto={auto})...")
        examples = build_examples(spec, golden_set)
        start = time.perf_counter()

        optimizer = MIPROv2(metric=metric, auto=auto, num_threads=4)
        optimized_program = optimizer.compile(
            module,
            trainset=examples,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
        )
        elapsed = time.perf_counter() - start

        # ── Phase 5: Score after optimization ──
        after = estimate_quality(spec, golden_set, metric, optimized_program)
        improvement = after["average_score"] - before["average_score"]

        # ── Phase 6: Save ──
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / f"{spec.template_name}_optimized.json"
        optimized_program.save(str(save_path))

        print(f"   ✅ After optimization: {after['average_score']:.4f} ({improvement:+.4f})")
        print(f"   ⏱  {elapsed:.1f}s  →  {save_path}")

        results.append(OptimizationRun(
            template_name=spec.template_name,
            subject=spec.subject,
            goal=spec.goal,
            score_before=before["average_score"],
            score_after=after["average_score"],
            improvement=improvement,
            metric_name=metric_name,
            duration_seconds=elapsed,
            optimized_path=save_path,
            per_case=after["per_case"],
        ))

    return results


def print_report(results: list[OptimizationRun]) -> None:
    """Print a formatted optimization report."""
    print(f"\n\n{'='*60}")
    print("📊 PROMPT OPTIMIZATION REPORT")
    print(f"{'='*60}\n")

    print(f"{'Template':<25} {'Before':>8} {'After':>8} {'Δ':>8} {'Metric':<30} {'Time':>7}")
    print("-" * 88)
    for r in results:
        delta = f"{r.improvement:+.4f}"
        print(f"{r.template_name:<25} {r.score_before:>8.4f} {r.score_after:>8.4f} {delta:>8} {r.metric_name:<30} {r.duration_seconds:>6.1f}s")

    improved = [r for r in results if r.improvement > 0]
    regressed = [r for r in results if r.improvement < 0]
    unchanged = [r for r in results if r.improvement == 0]

    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} total")
    print(f"  ✅ Improved: {len(improved)}")
    print(f"  ❌ Regressed: {len(regressed)}")
    print(f"  ➡️  Unchanged: {len(unchanged)}")

    if improved:
        print(f"\n{'='*60}")
        print("Top improvements:")
        print("-" * 60)
        for r in sorted(improved, key=lambda x: x.improvement, reverse=True)[:5]:
            print(f"  +{r.improvement:.4f}  {r.template_name}  ({r.subject})")

    if regressed:
        print(f"\n{'='*60}")
        print("Regressions (investigate):")
        print("-" * 60)
        for r in sorted(regressed, key=lambda x: x.improvement):
            print(f"  {r.improvement:.4f}  {r.template_name}")

    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    import argparse

    parser = argparse.ArgumentParser(
        description="Production-grade prompt optimization pipeline."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip optimization — only evaluate current prompts",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 optimization budget (default: light)",
    )
    parser.add_argument(
        "--no-deepeval",
        action="store_true",
        help="Skip DeepEval metrics, use word overlap fallback",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Only process a specific template (e.g. 'summarize')",
    )
    args = parser.parse_args()

    print("🚀 Cortex Prompt Optimization Pipeline")
    print(f"{'='*60}")
    print(f"DeepEval metrics: {'OFF' if args.no_deepeval else 'ON'}")
    print(f"Mode: {'Evaluation only' if args.eval_only else f'Optimization (auto={args.auto})'}")
    if args.template:
        print(f"Template filter: {args.template}")

    results = run_pipeline(
        use_deepeval=not args.no_deepeval,
        auto=args.auto,
        skip_optimization=args.eval_only,
        template_name=args.template,
    )

    print_report(results)

    # Save machine-readable report
    report_path = OUTPUT_DIR / "pipeline_report.json"
    report_data = []
    for r in results:
        report_data.append({
            "template": r.template_name,
            "subject": r.subject,
            "goal": r.goal,
            "score_before": r.score_before,
            "score_after": r.score_after,
            "improvement": r.improvement,
            "metric": r.metric_name,
            "duration_s": r.duration_seconds,
            "optimized_path": str(r.optimized_path) if r.optimized_path else None,
        })
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
