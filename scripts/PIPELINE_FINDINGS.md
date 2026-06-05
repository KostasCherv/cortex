# DSPy Prompt Optimization Pipeline — Findings

## What Was Built

### Core: `src/prompts/dspy_optimizer.py`
DSPy integration covering all 6 prompt templates:
- 5 concrete modules (`SummarizeModule`, `ReportModule`, `RagChatSystemModule`, `FollowupAnswerModule`, `WebSearchDecisionModule`)
- 1 dynamic module constructor (`build_module`) in the pipeline
- `DspyPromptOptimizer` class wrapping MIPROv2
- `create_lm_from_settings()` bridging app config to `dspy.LM`
- `routing_accuracy_metric` for structured router output

### CLI: `scripts/optimize_prompts.py`
Working single-template optimizer. Usage:
```
uv run python scripts/optimize_prompts.py --module summarize --auto light --compare
```
Supports 5 modules, `--auto` (light/medium/heavy), `--compare` for A/B per-case, `--output` dir.

### Pipeline: `scripts/prompt_optimization_pipeline.py`
Batch pipeline for all 6 templates. Features:
- Auto-discovers `.j2` files in `src/prompts/`
- Analyzes template variables vs spec (catches `{{ var or default }}` patterns)
- Dynamic DSPy module construction from spec (no manual classes needed)
- Pluggable metrics: DeepEval (Faithfulness + AnswerRelevancy), routing_accuracy, word_overlap
- Saves JSON report + optimized programs

### Tests: `tests/test_dspy_optimizer.py`
30 tests covering metrics, modules, optimizer lifecycle.

### Optimized Programs: `optimized_prompts/*.json`
Previously saved from successful optimize_prompts.py runs.

---

## What Works

| Area | Status |
|---|---|
| Template variable extraction | Correct — handles `{{ var or default }}` |
| DSPy module inference | Works for all 6 templates |
| optimize_prompts.py (single template) | Produces real improvements |
| web_search_decision routing_accuracy | 0.0000 → 0.6000 in pipeline |
| Pipeline template analysis | All 6 templates pass |
| Pipeline save/load/report | Works |
| load_dotenv() for .env | Configured |

## What Is Broken or Missing

### 1. DeepEval Integration (pipeline only)
DeepEval 3.9.9 changed its API — requires `LLMTestCase` object instead of kwargs. Fixed, but baseline scores still 0.0 for text templates. The metrics make LLM calls that may fail due to insufficient context or formatting issues in the golden examples.

### 2. word_overlap Metric Mismatch (pipeline only)
| Template | Before | Problem |
|---|---|---|
| report | 0.2495 | Module outputs markdown, golden expected_answer is plain text |
| web_search_decision | 0.0000 | Router outputs structured fields (action/reason/query/url), golden has no labels |

The pipeline's `_default_overlap_metric` compares word sets of predicted vs expected output — structural differences (markdown vs text, JSON vs text) produce misleadingly low scores.

### 3. Pipeline Golden Set Mapping
`build_examples()` in the pipeline uses two mappings (`"summarize"` and `"router"`). The "summarize" mapping handles all input fields via if/elif chains, but some templates may not get correct inputs for fields not present in the golden set (`memory_context`, `answer_context_block`, `report_block`). This may cause DSPy to optimize against empty/placeholder inputs.

### 4. Missing: Labeled Router Golden Cases
The golden set has no `action`/`reason`/`query`/`url` labels. `routing_accuracy_metric` checks action validity + field usage, but without ground-truth router decisions it can only score structural correctness, not decision quality.

### 5. Missing: Production Guardrails
- No budget/rate-limit protection for API calls during optimization
- No checkpoint/resume for long `--auto medium` or `--auto heavy` runs
- No diff view showing what the optimized prompt changed vs original
- No integration test running the full pipeline end-to-end


## Key Decisions

- **Opt-in by design**: DSPy is a separate toolchain — existing Jinja2 rendering + `llm.invoke()` remains the default path
- **Dynamic modules in pipeline**: `build_module()` creates modules from spec at runtime, avoiding manual classes per prompt
- **word_overlap as fallback**: When DeepEval isn't available or fails, word overlap gives a coarse quality signal. Good enough for regression detection, not for absolute quality measurement.

---

## How to Run

```bash
# Single template (working):
uv run python scripts/optimize_prompts.py --module web_search_decision --auto light --compare

# All templates, eval only (pipeline):
uv run python scripts/prompt_optimization_pipeline.py --eval-only --no-deepeval

# All templates, full optimization (pipeline, web_search_decision only works):
uv run python scripts/prompt_optimization_pipeline.py --no-deepeval

# Specific template (pipeline):
uv run python scripts/prompt_optimization_pipeline.py --template summarize --no-deepeval

# With DeepEval (needs OPENAI_API_KEY in .env):
uv run python scripts/prompt_optimization_pipeline.py --template summarize

# Tests:
uv run python -m pytest tests/test_dspy_optimizer.py -v
```
