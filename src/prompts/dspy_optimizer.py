"""DSPy-based prompt optimization for the prompt registry.

Wraps existing Jinja2 prompt templates as DSPy modules and uses MIPROv2
to algorithmically optimize prompts against golden set metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import dspy
from dspy.teleprompt import MIPROv2

from src.config import settings
from src.prompts.registry import PromptRegistry


@dataclass(frozen=True)
class OptimizedPrompt:
    """Metadata for an optimized prompt version."""

    name: str
    version: str
    score_before: float
    score_after: float
    improvement: float
    config: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a DSPy optimization run."""

    module_type: str
    optimized_program: dspy.Module
    before_score: float
    after_score: float
    improvement: float
    config: dict[str, Any]
    optimized_path: Path | None = None


class RagChatSystemSignature(dspy.Signature):
    """Answer a user question grounded in RAG context and web search results.

    Given system instructions, retrieved document context, and web search results,
    produce a clear answer that stays faithful to the provided sources.
    """

    system_instructions: str = dspy.InputField(
        desc="Custom system instructions for the assistant"
    )
    rag_context: str = dspy.InputField(
        desc="Retrieved document context relevant to the question"
    )
    web_results_json: str = dspy.InputField(
        desc="Web search results in JSON format"
    )
    answer: str = dspy.OutputField(
        desc="Clear answer grounded in the provided context"
    )


class FollowupAnswerSignature(dspy.Signature):
    """Answer a follow-up question grounded in a research report, conversation history, and retrieved passages."""

    report_block: str = dspy.InputField(
        desc="The research report's main findings and analysis"
    )
    history_block: str = dspy.InputField(
        desc="Conversation history so far"
    )
    answer_context_block: str = dspy.InputField(
        desc="Retrieved source passages relevant to the question"
    )
    web_results_json: str = dspy.InputField(
        desc="Web search results in JSON format"
    )
    question: str = dspy.InputField(desc="The user's follow-up question")
    answer: str = dspy.OutputField(
        desc="Concise answer grounded in report, passages, and web context"
    )


class WebSearchDecisionSignature(dspy.Signature):
    """Route the next assistant action based on the user message and available context.

    Given conversation history, RAG context, and the user message, decide whether to:
    answer directly, answer from RAG, search the web, fetch a URL, or ask a clarifying question.
    Return strict JSON with action, reason, query, and url fields.
    """

    history_block: str = dspy.InputField(
        desc="Conversation history"
    )
    rag_context: str = dspy.InputField(
        desc="Retrieved RAG context"
    )
    rag_is_insufficient: str = dspy.InputField(
        desc="Whether RAG context is insufficient (true/false)"
    )
    message_urls: str = dspy.InputField(
        desc="URLs found in the current user message"
    )
    history_urls: str = dspy.InputField(
        desc="URLs found in conversation history"
    )
    message: str = dspy.InputField(desc="The user's current message")
    action: str = dspy.OutputField(
        desc="One of: answer_direct, answer_from_rag, web_search, fetch_url, ask_clarifying"
    )
    reason: str = dspy.OutputField(
        desc="Short snake_case explanation for the chosen action"
    )
    query: str = dspy.OutputField(
        desc="Search query (only for web_search action)"
    )
    url: str = dspy.OutputField(
        desc="URL to fetch (only for fetch_url action)"
    )


class SummarizeSignature(dspy.Signature):
    """Generate structured JSON summaries of retrieved sources for a research query.

    Given a user query and a set of source blocks (each with URL, title, and content),
    produce concise factual summaries that capture the key information relevant
    to the query. Output as structured JSON.
    """

    query: str = dspy.InputField(desc="The user's research question")
    source_blocks: str = dspy.InputField(
        desc="Formatted source blocks, each with SOURCE URL, SOURCE TITLE, and CONTENT"
    )
    domain: str = dspy.InputField(desc="Optional domain or topic hint")
    summaries: str = dspy.OutputField(
        desc="JSON array of summaries, each with title, url, and summary text"
    )


class ReportSignature(dspy.Signature):
    """Generate a polished markdown research report from query and summaries.

    Consolidates all source summaries into a coherent, well-structured
    markdown report with an executive summary, key findings, and conclusion.
    """

    query: str = dspy.InputField(desc="The user's research question")
    summaries_text: str = dspy.InputField(
        desc="Consolidated summaries text from all retrieved sources"
    )
    memory_context: str = dspy.InputField(
        desc="Context from previous research sessions, if any"
    )
    domain: str = dspy.InputField(desc="Optional domain or topic hint")
    report: str = dspy.OutputField(
        desc="Polished markdown research report with executive summary, key findings, and conclusion"
    )


class RagChatSystemModule(dspy.Module):
    """DSPy module wrapping the RAG chat system prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(RagChatSystemSignature)

    def forward(
        self,
        system_instructions: str,
        rag_context: str,
        web_results_json: str,
    ) -> dspy.Prediction:
        return self.generate(
            system_instructions=system_instructions,
            rag_context=rag_context,
            web_results_json=web_results_json,
        )


class FollowupAnswerModule(dspy.Module):
    """DSPy module wrapping the follow-up answer prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(FollowupAnswerSignature)

    def forward(
        self,
        report_block: str,
        history_block: str,
        answer_context_block: str,
        web_results_json: str,
        question: str,
    ) -> dspy.Prediction:
        return self.generate(
            report_block=report_block,
            history_block=history_block,
            answer_context_block=answer_context_block,
            web_results_json=web_results_json,
            question=question,
        )


class WebSearchDecisionModule(dspy.Module):
    """DSPy module wrapping the web search decision router prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(WebSearchDecisionSignature)

    def forward(
        self,
        history_block: str,
        rag_context: str,
        rag_is_insufficient: str,
        message_urls: str,
        history_urls: str,
        message: str,
    ) -> dspy.Prediction:
        return self.generate(
            history_block=history_block,
            rag_context=rag_context,
            rag_is_insufficient=rag_is_insufficient,
            message_urls=message_urls,
            history_urls=history_urls,
            message=message,
        )


class SummarizeModule(dspy.Module):
    """DSPy module wrapping the summarize prompt workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(SummarizeSignature)

    def forward(
        self, query: str, source_blocks: str, domain: str
    ) -> dspy.Prediction:
        return self.generate(
            query=query, source_blocks=source_blocks, domain=domain
        )


class ReportModule(dspy.Module):
    """DSPy module wrapping the report prompt workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(ReportSignature)

    def forward(
        self,
        query: str,
        summaries_text: str,
        memory_context: str,
        domain: str,
    ) -> dspy.Prediction:
        return self.generate(
            query=query,
            summaries_text=summaries_text,
            memory_context=memory_context,
            domain=domain,
        )


def routing_accuracy_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Metric for web_search_decision routing.

    Scores based on action validity, proper field usage (query for web_search,
    url for fetch_url), and reason presence. Returns 0-1.
    """
    VALID_ACTIONS = {"answer_direct", "answer_from_rag", "web_search", "fetch_url", "ask_clarifying"}
    action = getattr(prediction, "action", None) or ""
    reason = getattr(prediction, "reason", None) or ""
    query = getattr(prediction, "query", None) or ""
    url = getattr(prediction, "url", None) or ""

    if action not in VALID_ACTIONS:
        return 0.0

    score = 0.4  # valid action
    if reason.strip():
        score += 0.2
    if action == "web_search" and query.strip():
        score += 0.4
    elif action == "fetch_url" and url.strip():
        score += 0.4
    elif action in ("answer_direct", "answer_from_rag", "ask_clarifying") and not query.strip() and not url.strip():
        score += 0.4
    else:
        score += 0.0
    return score


def default_overlap_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Simple word-overlap metric as a fallback when DeepEval is unavailable.

    Compares the predicted output against example.expected_output using
    normalized word overlap, scaled so that full overlap = 1.0.
    """
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


def create_lm_from_settings() -> dspy.LM | None:
    """Create a DSPy LM from the application settings, if credentials exist.

    Returns None if no suitable provider is configured (e.g., Ollama in tests).
    """
    provider = settings.llm_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return dspy.LM(
            f"openai/{settings.openai_model}",
            api_key=settings.openai_api_key,
        )
    if provider == "openrouter" and settings.openrouter_api_key:
        return dspy.LM(
            f"openai/{settings.openrouter_model}",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    if provider == "ollama":
        return dspy.LM(
            f"openai/{settings.ollama_model}",
            api_key="",
            base_url=settings.ollama_base_url,
        )
    return None


class DspyPromptOptimizer:
    """Optimize prompt templates using DSPy's MIPROv2.

    Wraps prompts from the registry as DSPy modules and optimizes
    them against a golden evaluation set. Optimized programs can be
    saved to disk and loaded at inference time.

    Usage:
        optimizer = DspyPromptOptimizer()
        module = SummarizeModule()
        result = optimizer.optimize(module, golden_set, "summarize")
        optimizer.save(result, "optimized_summarize")
    """

    def __init__(
        self,
        metric: Callable | None = None,
        output_dir: str | Path = "optimized_prompts",
        registry: PromptRegistry | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric or default_overlap_metric
        self.registry = registry or PromptRegistry()

        lm = create_lm_from_settings()
        if lm is not None:
            dspy.configure(lm=lm)

    def _build_example(
        self,
        case: dict,
        module_type: str,
    ) -> dspy.Example | None:
        """Build a single DSPy Example from a golden set case."""
        if module_type == "summarize":
            source_blocks = "\n\n".join(
                f"SOURCE URL: {s.get('url', '')}\n"
                f"SOURCE TITLE: {s.get('title', '')}\n"
                f"CONTENT:\n{s.get('raw_text', '')}"
                for s in case.get("retrieved_contents", [])
            )
            return dspy.Example(
                query=case["query"],
                source_blocks=source_blocks,
                domain="",
                summaries=case.get("expected_answer", ""),
                expected_output=case.get("expected_answer", ""),
            ).with_inputs("query", "source_blocks", "domain")

        if module_type == "report":
            summaries_text = "\n\n".join(
                str(s.get("raw_text", ""))
                for s in case.get("retrieved_contents", [])
            )
            return dspy.Example(
                query=case["query"],
                summaries_text=summaries_text,
                memory_context="",
                domain="",
                report=case.get("expected_answer", ""),
                expected_output=case.get("expected_answer", ""),
            ).with_inputs("query", "summaries_text", "memory_context", "domain")

        if module_type == "rag_chat_system":
            source_text = "\n\n".join(
                str(s.get("raw_text", ""))
                for s in case.get("retrieved_contents", [])
            )
            return dspy.Example(
                system_instructions="",
                rag_context=source_text,
                web_results_json="[]",
                answer=case.get("expected_answer", ""),
                expected_output=case.get("expected_answer", ""),
            ).with_inputs("system_instructions", "rag_context", "web_results_json")

        if module_type == "followup_answer":
            source_text = "\n\n".join(
                str(s.get("raw_text", ""))
                for s in case.get("retrieved_contents", [])
            )
            return dspy.Example(
                report_block="",
                history_block="",
                answer_context_block=source_text,
                web_results_json="[]",
                question=case["query"],
                answer=case.get("expected_answer", ""),
                expected_output=case.get("expected_answer", ""),
            ).with_inputs(
                "report_block", "history_block", "answer_context_block",
                "web_results_json", "question",
            )

        if module_type == "web_search_decision":
            return dspy.Example(
                history_block="None",
                rag_context="None",
                rag_is_insufficient="false",
                message_urls="None",
                history_urls="None",
                message=case["query"],
                action="answer_from_rag",
                reason="context_is_sufficient",
                query="",
                url="",
                expected_output=case.get("expected_answer", ""),
            ).with_inputs(
                "history_block", "rag_context", "rag_is_insufficient",
                "message_urls", "history_urls", "message",
            )

        return None

    def optimize(
        self,
        module: dspy.Module,
        golden_set: list[dict],
        module_type: str = "summarize",
        *,
        metric: Callable | None = None,
        max_bootstrapped_demos: int = 2,
        max_labeled_demos: int = 2,
        auto: str = "light",
        num_threads: int = 4,
    ) -> OptimizationResult:
        """Run MIPROv2 optimization on a DSPy module.

        Args:
            module: DSPy module to optimize (SummarizeModule or ReportModule).
            golden_set: List of golden evaluation cases with query,
                retrieved_contents, and expected_answer keys.
            module_type: One of 'summarize' or 'report'.
            max_bootstrapped_demos: Max bootstrapped demos for MIPROv2.
            max_labeled_demos: Max labeled demos for MIPROv2.
            auto: Optimization mode ('light', 'medium', 'heavy').
            num_threads: Number of parallel threads for optimization.

        Returns:
            OptimizationResult with before/after scores and the optimized program.
        """
        trainset: list[dspy.Example] = []
        for case in golden_set:
            example = self._build_example(case, module_type)
            if example is not None:
                trainset.append(example)

        if not trainset:
            raise ValueError(
                f"No training examples could be built from the golden set "
                f"for module_type='{module_type}'"
            )

        active_metric = metric or self.metric
        before_scores = [
            active_metric(ex, module(**ex.inputs()))
            for ex in trainset
        ]
        before_score = (
            sum(before_scores) / len(before_scores) if before_scores else 0.0
        )

        optimizer = MIPROv2(
            metric=active_metric,
            auto=auto,
            num_threads=num_threads,
        )

        optimized_program = optimizer.compile(
            module,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )

        after_scores = [
            active_metric(ex, optimized_program(**ex.inputs()))
            for ex in trainset
        ]
        after_score = (
            sum(after_scores) / len(after_scores) if after_scores else 0.0
        )

        return OptimizationResult(
            module_type=module_type,
            optimized_program=optimized_program,
            before_score=round(before_score, 4),
            after_score=round(after_score, 4),
            improvement=round(after_score - before_score, 4),
            config={
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "auto": auto,
                "num_threads": num_threads,
            },
        )

    def save(self, result: OptimizationResult, name: str) -> Path:
        """Save an optimized DSPy program to disk as JSON."""
        path = self.output_dir / f"{name}.json"
        result.optimized_program.save(str(path))
        return path

    def load(
        self, module_template: dspy.Module, path: str | Path
    ) -> dspy.Module:
        """Load an optimized DSPy program from a saved JSON file.

        Args:
            module_template: An instance of the DSPy module type to load into.
            path: Path to the saved JSON file.

        Returns:
            The loaded DSPy module with optimized prompt instructions.
        """
        module_template.load(str(path))
        return module_template

    def compare(
        self,
        original_module: dspy.Module,
        optimized_path: str | Path,
        golden_set: list[dict],
        module_type: str = "summarize",
        *,
        metric: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Run A/B comparison between original and optimized modules.

        Args:
            original_module: The unoptimized DSPy module.
            optimized_path: Path to saved optimized program JSON.
            golden_set: Golden evaluation cases.
            module_type: Module type ('summarize' or 'report').

        Returns:
            List of comparison dicts with query, original_score,
            optimized_score, and the outputs from each.
        """
        optimized_module = self.load(type(original_module)(), optimized_path)
        active_metric = metric or self.metric
        results: list[dict[str, Any]] = []

        for case in golden_set:
            example = self._build_example(case, module_type)
            if example is None:
                continue

            inputs = example.inputs()
            original_pred = original_module(**inputs)
            optimized_pred = optimized_module(**inputs)

            results.append(
                {
                    "query": case["query"],
                    "original_score": active_metric(example, original_pred),
                    "optimized_score": active_metric(example, optimized_pred),
                    "expected": case.get("expected_answer", ""),
                }
            )
        return results
