# Model evaluation

Cortex separates deterministic AI contracts, which run in CI, from model-backed semantic evaluation and prompt optimization, which run manually with provider credentials.

## Golden set

`src/evals/golden_set.json` holds 20 cases shared by the DeepEval comparison and the DSPy pipeline. Beyond straightforward retrieval questions, it deliberately includes hard classes — conflicting sources, context that cannot answer the query, distractor-heavy retrievals, partial answers, and numeric/temporal synthesis — so metrics can actually discriminate. An evaluation where every score is 1.0 is evidence about the dataset, not the model; keep adding cases until some fail.

## Deterministic regression gate

The credential-free gate protects routing, citation provenance, and finance-tool planning on every pull request:

```bash
uv run python -m src.evals.regression_gate
```

See [Testing and quality](testing-and-quality.md#ai-regression-gate) for its CI policy.

## Model comparison with DeepEval

Install evaluation dependencies:

```bash
uv sync --extra evals
```

`src/evals/model_comparison.py`:

- Loads cases from `src/evals/golden_set.json`
- Runs `summarize_node` for each configured provider/model pair
- Scores faithfulness and answer relevancy with DeepEval
- Writes results to `src/evals/results.csv`

Edit `MODEL_CONFIGS` in the script, then run:

```bash
uv run python src/evals/model_comparison.py
```

Per-case failures (some hard cases can make `summarize_node` raise) are recorded as zero-score rows with an `error` column rather than aborting the run.

Latest run (July 2026, 20-case golden set):

| Model | Faithfulness | Relevancy | p50 latency | Sub-1.0 cases |
|---|---|---|---|---|
| gemma4:31b-cloud | 1.000 | 0.972 | 1.6 s | 3 |
| gpt-4o-mini | 0.953 | 0.917 | 2.6 s | 6 |

The hard cases separate the models where the original 5-case set scored everything 1.0: gpt-4o-mini loses faithfulness on distractor-heavy and temporal cases, and both models drop relevancy on unanswerable-from-context questions.

## Router prompt optimization

The production chat router is a hardcoded prompt in `src/api/rag_chat_helpers.py`, optimized separately from the Jinja2 templates because it has a real labeled dataset: `data/router_dataset/` (429 train / 103 held-out cases labeled by a Qwen3-30B teacher).

```bash
uv run python scripts/optimize_router_prompt.py --eval-only   # score current prompt
uv run python scripts/optimize_router_prompt.py --auto medium # full MIPROv2 run
```

The script scores the current prompt on the held-out set through the exact production path (`get_router_llm` + `parse_chat_action_json`), runs MIPROv2 over a stratified train subsample, composes a candidate production prompt from the optimized instructions and demos, re-scores it on held-out, and writes `optimized_prompts/router_action_optimized.json` with both numbers.

**Caveat that applies to every number here:** held-out labels are teacher-model labels, not human-verified ground truth. Scores measure teacher agreement until records carry `verified=true`.

Latest run (July 2026, gpt-4o-mini): the current prompt scores **92.2% held-out agreement** (0 parse failures). MIPROv2 at both `light` and `medium` budgets returned the existing prompt as the best program — every candidate instruction and demo set it proposed scored worse. The residual disagreement concentrates in `search_finance_tools` (78.9%) and `asset_price` (89.5%), the two actions with genuinely overlapping intent, which is consistent with teacher-label ambiguity rather than prompt deficiency. The honest conclusion: the baseline prompt is at this optimizer's ceiling on this dataset, and the next real gain requires verified labels, not prompt search.

If a future run does beat the baseline, activate the artifact without a code change:

```dotenv
ROUTER_PROMPT_PATH=optimized_prompts/router_action_optimized.json
```

A bad or missing artifact logs a warning and falls back to the built-in prompt.

## Prompt optimization with DSPy

Cortex uses DSPy MIPROv2 to search for prompt instructions and examples against the golden set. The existing Jinja2 prompt pipeline remains the production default; optimization is an opt-in toolchain.

Relevant implementation:

- `SummarizeSignature` and `ReportSignature` define typed inputs and outputs.
- `SummarizeModule` and `ReportModule` wrap structured chain-of-thought generation.
- `DspyPromptOptimizer` builds training data, runs optimization, compares scores, and persists the optimized program.

Run optimization:

```python
from src.evals.model_comparison import load_golden_set
from src.prompts.dspy_optimizer import DspyPromptOptimizer, SummarizeModule

optimizer = DspyPromptOptimizer()
result = optimizer.optimize(SummarizeModule(), load_golden_set(), "summarize")
print(result.before_score, result.after_score, result.improvement)
optimizer.save(result, "optimized_summarize")
```

Compare an optimized program with the original:

```python
from src.evals.model_comparison import load_golden_set
from src.prompts.dspy_optimizer import DspyPromptOptimizer, SummarizeModule

optimizer = DspyPromptOptimizer()
results = optimizer.compare(
    SummarizeModule(),
    "optimized_prompts/optimized_summarize.json",
    load_golden_set(),
    "summarize",
)
for result in results:
    print(result["query"], result["original_score"], result["optimized_score"])
```

Load the persisted program:

```python
from src.prompts.dspy_optimizer import DspyPromptOptimizer, SummarizeModule

optimizer = DspyPromptOptimizer()
module = optimizer.load(
    SummarizeModule(),
    "optimized_prompts/optimized_summarize.json",
)
prediction = module(query="your question", source_blocks="...", domain="")
print(prediction.summaries)
```

Tests:

```bash
uv run pytest tests/test_dspy_optimizer.py -v
```

Expand `src/evals/golden_set.json` with representative cases before treating an optimization score as evidence of general improvement.

