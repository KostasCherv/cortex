"""Tests for DSPy prompt optimizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pytest

from src.prompts.dspy_optimizer import (
    DspyPromptOptimizer,
    FollowupAnswerModule,
    OptimizationResult,
    RagChatSystemModule,
    ReportModule,
    SummarizeModule,
    WebSearchDecisionModule,
    default_overlap_metric,
)

GOLDEN_SET = [
    {
        "query": "What refund window does the store offer for shoes that do not fit?",
        "retrieved_contents": [
            {
                "url": "https://example.com/returns",
                "title": "Store Returns Policy",
                "raw_text": "Customers can return unworn shoes within 30 days of delivery for a full refund. Exchanges are free when inventory is available.",
            },
            {
                "url": "https://example.com/help",
                "title": "Support FAQ",
                "raw_text": "The support team recommends starting a return from the orders page. Refunds are sent to the original payment method after the warehouse receives the shoes.",
            },
        ],
        "expected_answer": "The store offers a 30-day refund window for unworn shoes.",
    },
    {
        "query": "What causes the sky to appear blue during the day?",
        "retrieved_contents": [
            {
                "url": "https://example.com/rayleigh-scattering",
                "title": "Rayleigh Scattering Explained",
                "raw_text": "Molecules in Earth's atmosphere scatter shorter blue wavelengths more strongly than longer red wavelengths, which makes the sky appear blue to human observers.",
            },
            {
                "url": "https://example.com/sunset-colors",
                "title": "Why Sunsets Look Red",
                "raw_text": "At sunrise and sunset, sunlight passes through more atmosphere, so more blue light is scattered away before it reaches your eyes.",
            },
        ],
        "expected_answer": "The sky appears blue because atmospheric molecules scatter blue light more strongly.",
    },
]


class TestDefaultOverlapMetric:
    def test_returns_1_for_full_overlap(self):
        example = dspy.Example(expected_output="the quick brown fox")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = "the quick brown fox"

        score = default_overlap_metric(example, prediction)
        assert score == 1.0

    def test_returns_0_for_no_overlap(self):
        example = dspy.Example(expected_output="alpha beta gamma")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = "delta epsilon zeta"

        score = default_overlap_metric(example, prediction)
        assert score == 0.0

    def test_returns_0_for_empty_output(self):
        example = dspy.Example(expected_output="something")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = ""

        score = default_overlap_metric(example, prediction)
        assert score == 0.0

    def test_returns_0_for_empty_expected(self):
        example = dspy.Example(expected_output="")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = "something"

        score = default_overlap_metric(example, prediction)
        assert score == 0.0

    def test_falls_back_to_answer_field(self):
        example = dspy.Example(expected_output="hello world")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = None
        prediction.report = None
        prediction.answer = "hello world"

        score = default_overlap_metric(example, prediction)
        assert score == 1.0

    def test_falls_back_to_action_field(self):
        example = dspy.Example(expected_output="answer_from_rag")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = None
        prediction.report = None
        prediction.answer = None
        prediction.action = "answer_from_rag"

        score = default_overlap_metric(example, prediction)
        assert score == 1.0

    def test_falls_back_to_report_field(self):
        example = dspy.Example(expected_output="hello world")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = None
        prediction.report = "hello world"

        score = default_overlap_metric(example, prediction)
        assert score == 1.0

    def test_scales_partial_overlap(self):
        example = dspy.Example(expected_output="a b c d")
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.summaries = "a b"

        # overlap = 2/4 = 0.5, scaled = min(1.0, 0.5*2) = 1.0
        score = default_overlap_metric(example, prediction)
        assert score == 1.0


class TestSummarizeModule:
    def test_module_initializes(self):
        module = SummarizeModule()
        assert isinstance(module, dspy.Module)

    def test_forward_calls_generate(self):
        module = SummarizeModule()
        with patch.object(module, "generate") as mock_generate:
            mock_generate.return_value = dspy.Prediction(
                summaries='[{"title": "Test", "summary": "Content"}]'
            )
            result = module.forward(
                query="test query",
                source_blocks="SOURCE URL: https://example.com",
                domain="",
            )
            mock_generate.assert_called_once_with(
                query="test query",
                source_blocks="SOURCE URL: https://example.com",
                domain="",
            )
            assert result.summaries is not None


class TestReportModule:
    def test_module_initializes(self):
        module = ReportModule()
        assert isinstance(module, dspy.Module)

    def test_forward_calls_generate(self):
        module = ReportModule()
        with patch.object(module, "generate") as mock_generate:
            mock_generate.return_value = dspy.Prediction(
                report="# Executive Summary\nContent here"
            )
            result = module.forward(
                query="test query",
                summaries_text="Summary content",
                memory_context="",
                domain="",
            )
            mock_generate.assert_called_once_with(
                query="test query",
                summaries_text="Summary content",
                memory_context="",
                domain="",
            )
            assert result.report is not None


class TestRagChatSystemModule:
    def test_module_initializes(self):
        module = RagChatSystemModule()
        assert isinstance(module, dspy.Module)

    def test_forward_calls_generate(self):
        module = RagChatSystemModule()
        with patch.object(module, "generate") as mock_generate:
            mock_generate.return_value = dspy.Prediction(
                answer="30-day return policy."
            )
            result = module.forward(
                system_instructions="",
                rag_context="30 days return",
                web_results_json="[]",
            )
            mock_generate.assert_called_once_with(
                system_instructions="",
                rag_context="30 days return",
                web_results_json="[]",
            )
            assert result.answer is not None


class TestFollowupAnswerModule:
    def test_module_initializes(self):
        module = FollowupAnswerModule()
        assert isinstance(module, dspy.Module)

    def test_forward_calls_generate(self):
        module = FollowupAnswerModule()
        with patch.object(module, "generate") as mock_generate:
            mock_generate.return_value = dspy.Prediction(answer="Based on the report...")
            result = module.forward(
                report_block="Report content",
                history_block="Q: Hi",
                answer_context_block="Retrieved passages",
                web_results_json="[]",
                question="Tell me more",
            )
            mock_generate.assert_called_once_with(
                report_block="Report content",
                history_block="Q: Hi",
                answer_context_block="Retrieved passages",
                web_results_json="[]",
                question="Tell me more",
            )
            assert result.answer is not None


class TestWebSearchDecisionModule:
    def test_module_initializes(self):
        module = WebSearchDecisionModule()
        assert isinstance(module, dspy.Module)

    def test_forward_calls_generate(self):
        module = WebSearchDecisionModule()
        with patch.object(module, "generate") as mock_generate:
            mock_generate.return_value = dspy.Prediction(
                action="answer_from_rag",
                reason="context_is_sufficient",
                query="",
                url="",
            )
            result = module.forward(
                history_block="None",
                rag_context="Some context",
                rag_is_insufficient="false",
                message_urls="None",
                history_urls="None",
                message="What is the return policy?",
            )
            mock_generate.assert_called_once_with(
                history_block="None",
                rag_context="Some context",
                rag_is_insufficient="false",
                message_urls="None",
                history_urls="None",
                message="What is the return policy?",
            )
            assert result.action == "answer_from_rag"


class TestDspyPromptOptimizer:
    def test_initializes_with_defaults(self, tmp_path: Path):
        optimizer = DspyPromptOptimizer(output_dir=str(tmp_path))
        assert optimizer.output_dir == tmp_path
        assert optimizer.metric is default_overlap_metric
        assert optimizer.registry is not None

    def test_initializes_with_custom_metric(self, tmp_path: Path):
        def custom_metric(ex, pred, trace=None):
            return 0.5

        optimizer = DspyPromptOptimizer(
            metric=custom_metric, output_dir=str(tmp_path)
        )
        assert optimizer.metric is custom_metric

    def test_build_example_summarize(self):
        optimizer = DspyPromptOptimizer(output_dir="/tmp")
        example = optimizer._build_example(GOLDEN_SET[0], "summarize")

        assert example is not None
        assert example.query == GOLDEN_SET[0]["query"]
        assert "SOURCE URL: https://example.com/returns" in example.source_blocks
        assert "30 days" in example.source_blocks
        assert example.domain == ""
        assert (
            example.expected_output
            == GOLDEN_SET[0]["expected_answer"]
        )
        assert example._input_keys == {"query", "source_blocks", "domain"}

    def test_build_example_report(self):
        optimizer = DspyPromptOptimizer(output_dir="/tmp")
        example = optimizer._build_example(GOLDEN_SET[0], "report")

        assert example is not None
        assert example.query == GOLDEN_SET[0]["query"]
        assert "30 days" in example.summaries_text
        assert example.memory_context == ""
        assert example.domain == ""
        assert example._input_keys == {
            "query",
            "summaries_text",
            "memory_context",
            "domain",
        }

    def test_build_example_rag_chat_system(self):
        optimizer = DspyPromptOptimizer(output_dir="/tmp")
        example = optimizer._build_example(GOLDEN_SET[0], "rag_chat_system")
        assert example is not None
        assert example.system_instructions == ""
        assert "30 days" in example.rag_context
        assert example.web_results_json == "[]"
        assert example._input_keys == {
            "system_instructions", "rag_context", "web_results_json",
        }

    def test_build_example_followup_answer(self):
        optimizer = DspyPromptOptimizer(output_dir="/tmp")
        example = optimizer._build_example(GOLDEN_SET[0], "followup_answer")
        assert example is not None
        assert example.question == GOLDEN_SET[0]["query"]
        assert example.report_block == ""
        assert example.history_block == ""
        assert "30 days" in example.answer_context_block
        assert example.web_results_json == "[]"
        assert example._input_keys == {
            "report_block", "history_block", "answer_context_block",
            "web_results_json", "question",
        }

    def test_build_example_web_search_decision(self):
        optimizer = DspyPromptOptimizer(output_dir="/tmp")
        example = optimizer._build_example(GOLDEN_SET[0], "web_search_decision")
        assert example is not None
        assert example.message == GOLDEN_SET[0]["query"]
        assert example.rag_context == "None"
        assert example.rag_is_insufficient == "false"
        assert example._input_keys == {
            "history_block", "rag_context", "rag_is_insufficient",
            "message_urls", "history_urls", "message",
        }

    def test_build_example_unknown_type(self):
        optimizer = DspyPromptOptimizer(output_dir="/tmp")
        example = optimizer._build_example(GOLDEN_SET[0], "unknown")
        assert example is None

    def test_optimize_raises_on_empty_trainset(self, tmp_path: Path):
        optimizer = DspyPromptOptimizer(output_dir=str(tmp_path))
        module = SummarizeModule()
        with pytest.raises(
            ValueError, match="No training examples could be built"
        ):
            optimizer.optimize(module, [], "summarize")

    @patch("src.prompts.dspy_optimizer.MIPROv2")
    @patch("src.prompts.dspy_optimizer.dspy.configure")
    def test_optimize_runs_optimization(
        self, mock_configure, mock_miprov2_cls, tmp_path: Path
    ):
        mock_optimizer = MagicMock()
        mock_miprov2_cls.return_value = mock_optimizer

        mock_optimized = MagicMock(spec=dspy.Module)
        mock_optimized.return_value = dspy.Prediction(
            summaries="store offers 30-day refund"
        )
        mock_optimizer.compile.return_value = mock_optimized

        optimizer = DspyPromptOptimizer(
            metric=lambda ex, pred, trace=None: 1.0,
            output_dir=str(tmp_path),
        )

        module = SummarizeModule()
        module.return_value = dspy.Prediction(summaries="30 day refund policy")

        with patch.object(
            module, "forward", return_value=dspy.Prediction(summaries="30 day refund policy")
        ):
            result = optimizer.optimize(module, GOLDEN_SET, "summarize")

        assert result.module_type == "summarize"
        assert result.before_score >= 0.0
        assert result.after_score >= 0.0
        assert isinstance(result.improvement, float)
        assert result.optimized_program is mock_optimized

        mock_miprov2_cls.assert_called_once_with(
            metric=optimizer.metric,
            auto="light",
            num_threads=4,
        )

    def test_save_and_load_optimized(self, tmp_path: Path):
        optimizer = DspyPromptOptimizer(output_dir=str(tmp_path))
        module = SummarizeModule()

        result = OptimizationResult(
            module_type="summarize",
            optimized_program=module,
            before_score=0.5,
            after_score=0.8,
            improvement=0.3,
            config={"auto": "light"},
        )

        saved_path = optimizer.save(result, "test_module")
        assert saved_path.exists()
        assert saved_path.name == "test_module.json"
        assert saved_path.parent == tmp_path

    def test_compare_returns_comparison_results(self, tmp_path: Path):
        optimizer = DspyPromptOptimizer(output_dir=str(tmp_path))

        original = SummarizeModule()
        mock_pred = dspy.Prediction(summaries="30 day refund store policy")

        with patch.object(original, "forward", return_value=mock_pred):
            with patch.object(
                optimizer, "load", return_value=original
            ):
                saved_path = tmp_path / "dummy.json"
                saved_path.write_text('{"dummy": true}')

                results = optimizer.compare(
                    original, saved_path, GOLDEN_SET[:1], "summarize"
                )

        assert len(results) == 1
        assert results[0]["query"] == GOLDEN_SET[0]["query"]
        assert "original_score" in results[0]
        assert "optimized_score" in results[0]

    def test_compare_uses_explicit_metric_when_provided(self, tmp_path: Path):
        optimizer = DspyPromptOptimizer(output_dir=str(tmp_path))
        original = SummarizeModule()
        optimized = SummarizeModule()

        original_pred = dspy.Prediction(summaries="no overlap here")
        optimized_pred = dspy.Prediction(summaries="still no overlap")

        def explicit_metric(example, prediction, trace=None):
            text = prediction.summaries
            return 0.25 if "still" not in text else 0.75

        with patch.object(original, "forward", return_value=original_pred):
            with patch.object(optimized, "forward", return_value=optimized_pred):
                with patch.object(optimizer, "load", return_value=optimized):
                    saved_path = tmp_path / "dummy.json"
                    saved_path.write_text('{"dummy": true}')

                    results = optimizer.compare(
                        original,
                        saved_path,
                        GOLDEN_SET[:1],
                        "summarize",
                        metric=explicit_metric,
                    )

        assert len(results) == 1
        assert results[0]["original_score"] == 0.25
        assert results[0]["optimized_score"] == 0.75
