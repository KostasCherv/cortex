"""Tests for the prompt optimization pipeline script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

PIPELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "prompt_optimization_pipeline.py"
)


def load_pipeline_module():
    module_name = "test_prompt_optimization_pipeline_module"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PIPELINE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_run_pipeline_filters_templates_before_execution(tmp_path, monkeypatch):
    pipeline = load_pipeline_module()

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for name in ("summarize", "report"):
        (prompts_dir / f"{name}.j2").write_text("{{ query }}", encoding="utf-8")

    output_dir = tmp_path / "optimized"
    seen_templates: list[str] = []

    class FakeProgram:
        def __init__(self, template_name: str):
            self.template_name = template_name

        def save(self, path: str) -> None:
            Path(path).write_text(self.template_name, encoding="utf-8")

    class FakeMIPROv2:
        def __init__(self, metric, auto, num_threads):
            self.metric = metric
            self.auto = auto
            self.num_threads = num_threads

        def compile(
            self,
            module,
            trainset,
            max_bootstrapped_demos,
            max_labeled_demos,
        ):
            return FakeProgram(module.template_name)

    def fake_build_module(spec):
        return SimpleNamespace(template_name=spec.template_name)

    def fake_estimate_quality(spec, golden_set, metric, module):
        seen_templates.append(spec.template_name)
        is_optimized = isinstance(module, FakeProgram)
        return {
            "average_score": 0.6 if is_optimized else 0.3,
            "per_case": [],
        }

    monkeypatch.setattr(pipeline, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(pipeline, "load_golden_set", lambda: [{"query": "Q"}])
    monkeypatch.setattr(pipeline, "build_examples", lambda spec, golden_set: ["example"])
    monkeypatch.setattr(pipeline, "build_module", fake_build_module)
    monkeypatch.setattr(pipeline, "estimate_quality", fake_estimate_quality)
    monkeypatch.setattr(
        "src.prompts.dspy_optimizer.create_lm_from_settings",
        lambda: object(),
    )

    import dspy
    import dspy.teleprompt

    monkeypatch.setattr(dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(dspy.teleprompt, "MIPROv2", FakeMIPROv2)

    results = pipeline.run_pipeline(
        use_deepeval=False,
        auto="light",
        skip_optimization=False,
        template_name="summarize",
    )

    assert [r.template_name for r in results] == ["summarize"]
    assert seen_templates == ["summarize", "summarize"]
    assert (output_dir / "summarize_optimized.json").exists()
    assert not (output_dir / "report_optimized.json").exists()


def test_main_passes_template_filter_to_run_pipeline(tmp_path, monkeypatch):
    pipeline = load_pipeline_module()
    captured: dict[str, object] = {}

    def fake_run_pipeline(*, use_deepeval, auto, skip_optimization, template_name=None):
        captured["template_name"] = template_name
        return []

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(pipeline, "print_report", lambda results: None)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prompt_optimization_pipeline.py",
            "--template",
            "summarize",
            "--eval-only",
            "--no-deepeval",
        ],
    )

    pipeline.main()

    assert captured["template_name"] == "summarize"
    assert (tmp_path / "pipeline_report.json").exists()
